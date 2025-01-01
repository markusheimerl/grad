#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, ADD, SUB, RESHAPE, SOFTMAX, PERMUTE, RMSNORM, HADAMARD } OpType;
typedef struct Tensor { float *data, *grad; int *dims, ndims, size, requires_grad; } Tensor;
typedef struct { OpType op; Tensor *result, *input1, *input2; int *aux_data; } TapeEntry;

static TapeEntry tape[MAX_TAPE]; static int tape_len = 0;
static Tensor* registry[MAX_TENSORS]; static int registry_len = 0;

static int get_index(int idx, const int* dims, int ndims, const int* ref_dims, int ref_ndims) {
    int result = 0, stride = 1;
    for (int d = ndims - 1; d >= 0; d--) {
        result += ((idx / stride) % ref_dims[d + ref_ndims - ndims]) * (dims[d] == 1 ? 0 : stride);
        stride *= dims[d];
    }
    return result;
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims; t->dims = malloc(ndims * sizeof(int)); t->size = 1;
    for (int i = 0; i < ndims; i++) t->size *= dims[i];
    memcpy(t->dims, dims, ndims * sizeof(int));
    t->data = malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    if ((t->requires_grad = requires_grad)) t->grad = calloc(t->size, sizeof(float));
    registry[registry_len++] = t;
    return t;
}

void clean_registry() {
    while (registry_len > 0) {
        Tensor* t = registry[--registry_len];
        free(t->data); free(t->grad); free(t->dims); free(t);
    }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!a || !b) return NULL;
    int max_d = fmax(a->ndims, b->ndims), rd[32];
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        rd[max_d-1-i] = fmax(d1, d2);
    }
    Tensor* r = tensor_new(max_d, rd, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < r->size; i++) {
        float av = a->data[get_index(i, a->dims, a->ndims, rd, max_d)];
        float bv = b->data[get_index(i, b->dims, b->ndims, rd, max_d)];
        r->data[i] = op == HADAMARD ? av * bv : (op == ADD ? av + bv : av - bv);
    }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){op, r, a, b, NULL};
    return r;
}

Tensor* tensor_add(Tensor* a, Tensor* b) { return tensor_op(a, b, ADD); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return tensor_op(a, b, SUB); }
Tensor* tensor_hadamard(Tensor* a, Tensor* b) { return tensor_op(a, b, HADAMARD); }

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b || a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
    int max_d = fmax(a->ndims, b->ndims), dims[32];
    memcpy(dims, (a->ndims > b->ndims ? a : b)->dims, (max_d - 2) * sizeof(int));
    dims[max_d-2] = a->dims[a->ndims-2]; dims[max_d-1] = b->dims[b->ndims-1];
    Tensor* r = tensor_new(max_d, dims, NULL, a->requires_grad || b->requires_grad);
    int M = a->dims[a->ndims-2], N = b->dims[b->ndims-1], K = a->dims[a->ndims-1], batch = r->size / (M * N);
    for (int n = 0; n < batch; n++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += a->data[n*M*K + i*K + k] * b->data[n*K*N + k*N + j];
                r->data[n*M*N + i*N + j] = sum;
            }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, r, a, b, NULL};
    return r;
}

Tensor* tensor_softmax(Tensor* a) {
    if (!a || a->ndims < 1) return NULL;
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    int last_dim = a->dims[a->ndims - 1], outer_size = a->size / last_dim;
    for (int i = 0; i < outer_size; i++) {
        float max_val = a->data[i * last_dim];
        for (int j = 1; j < last_dim; j++) max_val = fmaxf(max_val, a->data[i * last_dim + j]);
        float sum = 0;
        for (int j = 0; j < last_dim; j++) sum += (r->data[i * last_dim + j] = expf(a->data[i * last_dim + j] - max_val));
        for (int j = 0; j < last_dim; j++) r->data[i * last_dim + j] /= sum;
    }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){SOFTMAX, r, a, NULL, NULL};
    return r;
}

Tensor* tensor_rms_norm(Tensor* x, float eps) {
    if (!x || x->ndims < 1) return NULL;
    Tensor* out = tensor_new(x->ndims, x->dims, NULL, x->requires_grad);
    int last_dim = x->dims[x->ndims - 1], batch_size = x->size / last_dim;
    for (int b = 0; b < batch_size; b++) {
        float ms = 0.0f;
        for (int i = 0; i < last_dim; i++) ms += x->data[b * last_dim + i] * x->data[b * last_dim + i];
        float scale = 1.0f / sqrt(ms/last_dim + eps);
        for (int i = 0; i < last_dim; i++) out->data[b * last_dim + i] = x->data[b * last_dim + i] * scale;
    }
    if (out->requires_grad) {
        float* eps_ptr = malloc(sizeof(float)); *eps_ptr = eps;
        tape[tape_len++] = (TapeEntry){RMSNORM, out, x, NULL, (int*)eps_ptr};
    }
    return out;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1; for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL, NULL};
    return r;
}

Tensor* tensor_permute(Tensor* a, const int* perm, int perm_size) {
    if (!a || !perm || perm_size != a->ndims) return NULL;
    int* used = calloc(perm_size, sizeof(int));
    for (int i = 0; i < perm_size; i++) if (perm[i] < 0 || perm[i] >= perm_size || used[perm[i]]) {free(used); return NULL;} else used[perm[i]] = 1;
    free(used);
    int* new_dims = malloc(a->ndims * sizeof(int));
    for (int i = 0; i < a->ndims; i++) new_dims[i] = a->dims[perm[i]];
    Tensor* r = tensor_new(a->ndims, new_dims, NULL, a->requires_grad);
    int *a_strides = malloc(a->ndims * sizeof(int)), *r_strides = malloc(r->ndims * sizeof(int));
    a_strides[a->ndims - 1] = r_strides[r->ndims - 1] = 1;
    for (int i = a->ndims - 2; i >= 0; i--) {
        a_strides[i] = a_strides[i + 1] * a->dims[i + 1];
        r_strides[i] = r_strides[i + 1] * r->dims[i + 1];
    }
    for (int i = 0; i < r->size; i++) {
        int temp = i, old_idx = 0;
        for (int d = 0; d < r->ndims; d++) {
            int coord = temp / r_strides[d]; temp %= r_strides[d];
            old_idx += coord * a_strides[perm[d]];
        }
        r->data[i] = a->data[old_idx];
    }
    free(a_strides); free(r_strides); free(new_dims);
    if (r->requires_grad) {
        int* stored_perm = malloc(perm_size * sizeof(int));
        memcpy(stored_perm, perm, perm_size * sizeof(int));
        tape[tape_len++] = (TapeEntry){PERMUTE, r, a, NULL, stored_perm};
    }
    return r;
}

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* e = &tape[t]; Tensor *r = e->result, *a = e->input1, *b = e->input2;
        switch(e->op) {
            case ADD: case SUB: case HADAMARD:
                for (int i = 0; i < r->size; i++) {
                    float grad = r->grad[i];
                    if (a->requires_grad) {
                        int a_idx = get_index(i, a->dims, a->ndims, r->dims, r->ndims);
                        a->grad[a_idx] += e->op == HADAMARD ? grad * b->data[get_index(i, b->dims, b->ndims, r->dims, r->ndims)] : grad;
                    }
                    if (b->requires_grad) {
                        int b_idx = get_index(i, b->dims, b->ndims, r->dims, r->ndims);
                        b->grad[b_idx] += e->op == HADAMARD ? grad * a->data[get_index(i, a->dims, a->ndims, r->dims, r->ndims)] : (e->op == ADD ? 1 : -1) * grad;
                    }
                }
                break;
            case MATMUL: {
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1], batch = r->size/(M*N);
                for (int n = 0; n < batch; n++)
                    for (int i = 0; i < M; i++)
                        for (int j = 0; j < N; j++) {
                            float g = r->grad[n*M*N + i*N + j];
                            for (int k = 0; k < K; k++) {
                                if (a->requires_grad) a->grad[n*M*K + i*K + k] += g * b->data[n*K*N + k*N + j];
                                if (b->requires_grad) b->grad[n*K*N + k*N + j] += g * a->data[n*M*K + i*K + k];
                            }
                        }
                break;
            }
            case SOFTMAX:
                if (a->requires_grad) {
                    int last_dim = a->dims[a->ndims - 1], outer_size = a->size / last_dim;
                    for (int i = 0; i < outer_size; i++) {
                        float sum = 0;
                        for (int j = 0; j < last_dim; j++) sum += r->grad[i*last_dim+j] * r->data[i*last_dim+j];
                        for (int j = 0; j < last_dim; j++) a->grad[i*last_dim+j] += r->data[i*last_dim+j] * (r->grad[i*last_dim+j] - sum);
                    }
                }
                break;
            case RESHAPE:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += r->grad[i];
                break;
            case PERMUTE:
                if (a->requires_grad) {
                    int *inv_perm = malloc(a->ndims * sizeof(int)), *a_strides = malloc(a->ndims * sizeof(int)), *r_strides = malloc(r->ndims * sizeof(int));
                    for (int i = 0; i < a->ndims; i++) inv_perm[e->aux_data[i]] = i;
                    a_strides[a->ndims-1] = r_strides[r->ndims-1] = 1;
                    for (int i = a->ndims-2; i >= 0; i--) {
                        a_strides[i] = a_strides[i+1] * a->dims[i+1];
                        r_strides[i] = r_strides[i+1] * r->dims[i+1];
                    }
                    for (int i = 0; i < r->size; i++) {
                        int temp = i, old_idx = 0;
                        for (int d = 0; d < r->ndims; d++) {
                            int coord = temp / r_strides[d]; temp %= r_strides[d];
                            old_idx += coord * a_strides[inv_perm[d]];
                        }
                        a->grad[old_idx] += r->grad[i];
                    }
                    free(a_strides); free(r_strides); free(inv_perm);
                }
                break;
            case RMSNORM:
                if (a->requires_grad) {
                    float eps = *(float*)e->aux_data;
                    int last_dim = a->dims[a->ndims-1], batch_size = a->size/last_dim;
                    for (int b = 0; b < batch_size; b++) {
                        float ms = 0.0f;
                        for (int i = 0; i < last_dim; i++) ms += a->data[b*last_dim+i] * a->data[b*last_dim+i];
                        ms /= last_dim;
                        float scale = 1.0f/sqrt(ms+eps), sum_grad_times_val = 0.0f;
                        for (int i = 0; i < last_dim; i++) sum_grad_times_val += r->grad[b*last_dim+i] * a->data[b*last_dim+i];
                        for (int i = 0; i < last_dim; i++)
                            a->grad[b*last_dim+i] += scale * r->grad[b*last_dim+i] - (scale*scale*scale) * a->data[b*last_dim+i] * sum_grad_times_val/last_dim;
                    }
                }
                break;
        }
        free(e->aux_data); e->aux_data = NULL;
    }
    tape_len = 0;
}

Tensor* tensor_masked_multihead_attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* mask, int num_heads) {
    if (!Q || !K || !V || !mask || Q->ndims != 3 || K->ndims != 3 || V->ndims != 3 || mask->ndims != 4) return NULL;
    int batch_size = Q->dims[0], seq_len_q = Q->dims[1], seq_len_k = K->dims[1], d_model = Q->dims[2];
    if (d_model % num_heads != 0 || K->dims[2] != d_model || V->dims[2] != d_model || 
        seq_len_k != V->dims[1] || batch_size != K->dims[0] || batch_size != V->dims[0] ||
        mask->dims[0] != batch_size || mask->dims[1] != num_heads || 
        mask->dims[2] != seq_len_q || mask->dims[3] != seq_len_k) return NULL;
    
    int d_head = d_model/num_heads; tape_len = 0;
    int reshape_dims[] = {batch_size, -1, num_heads, d_head}, perm[] = {0, 2, 1, 3};
    
    reshape_dims[1] = seq_len_q;
    Tensor* Q_perm = tensor_permute(tensor_reshape(Q, 4, reshape_dims), perm, 4);
    reshape_dims[1] = seq_len_k;
    Tensor* K_perm = tensor_permute(tensor_reshape(K, 4, reshape_dims), perm, 4);
    Tensor* V_perm = tensor_permute(tensor_reshape(V, 4, reshape_dims), perm, 4);
    if (!Q_perm || !K_perm || !V_perm) return NULL;

    Tensor* K_transpose = tensor_permute(K_perm, (int[]){0,1,3,2}, 4);
    if (!K_transpose) return NULL;
    
    Tensor* scores = tensor_matmul(Q_perm, K_transpose);
    if (!scores) return NULL;
    
    float scale = 1.0f/sqrt(d_head);
    Tensor* scaled_scores = tensor_hadamard(scores, tensor_new(4, (int[]){1,1,1,1}, (float[]){scale}, 0));
    if (!scaled_scores) return NULL;

    // Apply mask
    Tensor* masked_scores = tensor_hadamard(scaled_scores, mask);
    if (!masked_scores) return NULL;

    Tensor* attention = tensor_matmul(tensor_softmax(masked_scores), V_perm);
    if (!attention) return NULL;

    return tensor_reshape(tensor_permute(attention, (int[]){0,2,1,3}, 4), 3, (int[]){batch_size,seq_len_q,d_model});
}

int main() {
    {
        int batch_size = 1, seq_len = 2, d_model = 4, num_heads = 2;
        int qkv_dims[] = {batch_size, seq_len, d_model};
        int mask_dims[] = {batch_size, num_heads, seq_len, seq_len};
        
        float q_data[] = {1,1,0,0, 0,0,1,1};
        float k_data[] = {1,1,0,0, 0,0,1,1};
        float v_data[] = {1,1,2,2, 3,3,4,4};
        // Causal mask: upper triangle is 0
        float mask_data[] = {1,0, 1,1};
        
        Tensor *Q = tensor_new(3, qkv_dims, q_data, 1);
        Tensor *K = tensor_new(3, qkv_dims, k_data, 1);
        Tensor *V = tensor_new(3, qkv_dims, v_data, 1);
        Tensor *mask = tensor_new(4, mask_dims, mask_data, 0);

        printf("\nTest 1: Masked Multi-Head Attention\n\nInput values:\nQ:"); 
        for(int i = 0; i < 8; i++) printf(" %f", q_data[i]);
        printf("\nK:"); for(int i = 0; i < 8; i++) printf(" %f", k_data[i]);
        printf("\nV:"); for(int i = 0; i < 8; i++) printf(" %f", v_data[i]);
        printf("\nMask:"); for(int i = 0; i < 4; i++) printf(" %f", mask_data[i]);

        Tensor* output = tensor_masked_multihead_attention(Q, K, V, mask, num_heads);
        printf("\n\nFinal output values (with causal masking):\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d:", i);
            for (int j = 0; j < d_model; j++) printf(" %6.3f", output->data[i * d_model + j]);
            printf("\n");
        }

        // Test gradients
        for (int i = 0; i < output->size; i++) output->grad[i] = 1.0f;
        backward();
        printf("\nQ gradients:\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d:", i);
            for (int j = 0; j < d_model; j++) printf(" %6.3f", Q->grad[i * d_model + j]);
            printf("\n");
        }
    }

    {
        // Test with different mask patterns
        int batch_size = 1, seq_len = 3, d_model = 4, num_heads = 2;
        int qkv_dims[] = {batch_size, seq_len, d_model};
        int mask_dims[] = {batch_size, num_heads, seq_len, seq_len};
        
        float q_data[12], k_data[12], v_data[12];
        for (int i = 0; i < seq_len * d_model; i++) {
            q_data[i] = k_data[i] = 1.0f;
            v_data[i] = (i/d_model) + 1.0f;
        }
        
        // Create a mask that only allows attention to even positions
        float mask_data[] = {1,0,1, 1,0,1, 1,0,1};
        
        Tensor *Q = tensor_new(3, qkv_dims, q_data, 1);
        Tensor *K = tensor_new(3, qkv_dims, k_data, 1);
        Tensor *V = tensor_new(3, qkv_dims, v_data, 1);
        Tensor *mask = tensor_new(4, mask_dims, mask_data, 0);

        Tensor* output = tensor_masked_multihead_attention(Q, K, V, mask, num_heads);
        printf("\nTest 2: Custom Mask Pattern\nOutput:\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d:", i);
            for (int j = 0; j < d_model; j++) printf(" %6.3f", output->data[i * d_model + j]);
            printf("\n");
        }
    }

    clean_registry();
    return 0;
}