#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, ADD, SUB, RESHAPE, SOFTMAX, PERMUTE, RMSNORM, HADAMARD } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int *aux_data;
} TapeEntry;

static TapeEntry tape[MAX_TAPE];
static int tape_len = 0;
static Tensor* registry[MAX_TENSORS];
static int registry_len = 0;

static int get_index(int idx, const int* dims, int ndims, const int* ref_dims, int ref_ndims) {
    int result = 0, stride = 1;
    for (int d = ndims - 1; d >= 0; d--) {
        int coord = (idx / stride) % ref_dims[d + ref_ndims - ndims];
        result += (dims[d] == 1 ? 0 : coord) * stride;
        stride *= dims[d];
    }
    return result;
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    t->size = 1;
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

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
    int max_d = fmax(a->ndims, b->ndims), dims[32];
    memcpy(dims, (a->ndims > b->ndims ? a : b)->dims, (max_d - 2) * sizeof(int));
    dims[max_d-2] = a->dims[a->ndims-2];
    dims[max_d-1] = b->dims[b->ndims-1];
    
    Tensor* r = tensor_new(max_d, dims, NULL, a->requires_grad || b->requires_grad);
    int M = a->dims[a->ndims-2], N = b->dims[b->ndims-1], K = a->dims[a->ndims-1];
    int batch = r->size / (M * N);
    
    for (int n = 0; n < batch; n++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a->data[n*M*K + i*K + k] * b->data[n*K*N + k*N + j];
                r->data[n*M*N + i*N + j] = sum;
            }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, r, a, b, NULL};
    return r;
}

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    
    // Similar broadcasting rules as ADD/SUB
    int max_d = fmax(a->ndims, b->ndims);
    int rd[32];
    
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        rd[max_d-1-i] = fmax(d1, d2);
    }
    
    Tensor* r = tensor_new(max_d, rd, NULL, a->requires_grad || b->requires_grad);
    
    // Element-wise multiplication with broadcasting
    for (int i = 0; i < r->size; i++) {
        float av = a->data[get_index(i, a->dims, a->ndims, rd, max_d)];
        float bv = b->data[get_index(i, b->dims, b->ndims, rd, max_d)];
        r->data[i] = av * bv;
    }
    
    if (r->requires_grad) {
        tape[tape_len++] = (TapeEntry){HADAMARD, r, a, b, NULL};
    }
    return r;
}

Tensor* tensor_permute(Tensor* a, const int* perm, int perm_size) {
    if (!a || !perm || perm_size != a->ndims) return NULL;
    
    int* used = calloc(perm_size, sizeof(int));
    for (int i = 0; i < perm_size; i++) {
        if (perm[i] < 0 || perm[i] >= perm_size || used[perm[i]]) {
            free(used);
            return NULL;
        }
        used[perm[i]] = 1;
    }
    free(used);
    
    int* new_dims = malloc(a->ndims * sizeof(int));
    for (int i = 0; i < a->ndims; i++) new_dims[i] = a->dims[perm[i]];
    
    Tensor* r = tensor_new(a->ndims, new_dims, NULL, a->requires_grad);
    
    int* a_strides = malloc(a->ndims * sizeof(int));
    int* r_strides = malloc(r->ndims * sizeof(int));
    
    a_strides[a->ndims - 1] = r_strides[r->ndims - 1] = 1;
    
    for (int i = a->ndims - 2; i >= 0; i--) {
        a_strides[i] = a_strides[i + 1] * a->dims[i + 1];
        r_strides[i] = r_strides[i + 1] * r->dims[i + 1];
    }
    
    for (int i = 0; i < r->size; i++) {
        int temp = i, old_idx = 0;
        for (int d = 0; d < r->ndims; d++) {
            int coord = temp / r_strides[d];
            temp %= r_strides[d];
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

Tensor* tensor_rms_norm(Tensor* x, float eps) {
    if (!x || x->ndims < 1) return NULL;
    
    // Create output tensor with same shape
    Tensor* out = tensor_new(x->ndims, x->dims, NULL, x->requires_grad);
    
    // Calculate the size of the last dimension (normalization dimension)
    int last_dim = x->dims[x->ndims - 1];
    int batch_size = x->size / last_dim;
    
    // For each batch
    for (int b = 0; b < batch_size; b++) {
        float ms = 0.0f;  // mean square
        
        // Calculate mean square
        for (int i = 0; i < last_dim; i++) {
            float val = x->data[b * last_dim + i];
            ms += val * val;
        }
        ms /= last_dim;
        
        // Calculate scaling factor
        float scale = 1.0f / sqrt(ms + eps);
        
        // Apply normalization
        for (int i = 0; i < last_dim; i++) {
            out->data[b * last_dim + i] = x->data[b * last_dim + i] * scale;
        }
    }
    
    if (out->requires_grad) {
        float* eps_ptr = malloc(sizeof(float));
        *eps_ptr = eps;
        tape[tape_len++] = (TapeEntry){RMSNORM, out, x, NULL, (int*)eps_ptr};
    }
    
    return out;
}

Tensor* tensor_softmax(Tensor* a) {
    if (!a || a->ndims < 1) return NULL;
    
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    int last_dim = a->dims[a->ndims - 1];
    int outer_size = a->size / last_dim;
    
    for (int i = 0; i < outer_size; i++) {
        // Find max for this batch
        float max_val = a->data[i * last_dim];
        for (int j = 1; j < last_dim; j++) {
            max_val = fmaxf(max_val, a->data[i * last_dim + j]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            float val = a->data[i * last_dim + j] - max_val;
            r->data[i * last_dim + j] = expf(val);
            sum += r->data[i * last_dim + j];
        }
        
        // Normalize
        for (int j = 0; j < last_dim; j++) {
            r->data[i * last_dim + j] /= sum;
        }
    }
    
    if (r->requires_grad) {
        tape[tape_len++] = (TapeEntry){SOFTMAX, r, a, NULL, NULL};
    }
    
    return r;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL, NULL};
    return r;
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
        r->data[i] = op == ADD ? av + bv : av - bv;
    }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){op, r, a, b, NULL};
    return r;
}

Tensor* tensor_add(Tensor* a, Tensor* b) { return tensor_op(a, b, ADD); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return tensor_op(a, b, SUB); }

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* e = &tape[t];
        Tensor *r = e->result, *a = e->input1, *b = e->input2;
        
        if (e->op == ADD || e->op == SUB) {
            for (int i = 0; i < r->size; i++) {
                if (a->requires_grad) 
                    a->grad[get_index(i, a->dims, a->ndims, r->dims, r->ndims)] += r->grad[i];
                if (b->requires_grad) 
                    b->grad[get_index(i, b->dims, b->ndims, r->dims, r->ndims)] += 
                        (e->op == ADD ? 1 : -1) * r->grad[i];
            }
        }
        else if (e->op == MATMUL) {
            int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
            int batch = r->size / (M * N);
            for (int n = 0; n < batch; n++)
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        float g = r->grad[n*M*N + i*N + j];
                        for (int k = 0; k < K; k++) {
                            if (a->requires_grad) 
                                a->grad[n*M*K + i*K + k] += g * b->data[n*K*N + k*N + j];
                            if (b->requires_grad) 
                                b->grad[n*K*N + k*N + j] += g * a->data[n*M*K + i*K + k];
                        }
                    }
        }else if (e->op == RESHAPE && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i];
        }
        else if (e->op == PERMUTE && a->requires_grad) {
            int* inv_perm = malloc(a->ndims * sizeof(int));
            for (int i = 0; i < a->ndims; i++) inv_perm[e->aux_data[i]] = i;
            
            int* a_strides = malloc(a->ndims * sizeof(int));
            int* r_strides = malloc(r->ndims * sizeof(int));
            
            a_strides[a->ndims - 1] = r_strides[r->ndims - 1] = 1;
            
            for (int i = a->ndims - 2; i >= 0; i--) {
                a_strides[i] = a_strides[i + 1] * a->dims[i + 1];
                r_strides[i] = r_strides[i + 1] * r->dims[i + 1];
            }
            
            for (int i = 0; i < r->size; i++) {
                int temp = i, old_idx = 0;
                for (int d = 0; d < r->ndims; d++) {
                    int coord = temp / r_strides[d];
                    temp %= r_strides[d];
                    old_idx += coord * a_strides[inv_perm[d]];
                }
                a->grad[old_idx] += r->grad[i];
            }
            
            free(a_strides); free(r_strides); free(inv_perm);
        }else if (e->op == RMSNORM && a->requires_grad) {
            float eps = *(float*)e->aux_data;
            int last_dim = a->dims[a->ndims - 1];
            int batch_size = a->size / last_dim;
            
            for (int b = 0; b < batch_size; b++) {
                float ms = 0.0f;
                for (int i = 0; i < last_dim; i++) {
                    float val = a->data[b * last_dim + i];
                    ms += val * val;
                }
                ms /= last_dim;
                float scale = 1.0f / sqrt(ms + eps);
                
                float sum_grad_times_val = 0.0f;
                for (int i = 0; i < last_dim; i++) {
                    sum_grad_times_val += r->grad[b * last_dim + i] * a->data[b * last_dim + i];
                }
                
                for (int i = 0; i < last_dim; i++) {
                    float val = a->data[b * last_dim + i];
                    a->grad[b * last_dim + i] += scale * r->grad[b * last_dim + i] -
                        (scale * scale * scale) * val * sum_grad_times_val / last_dim;
                }
            }
        }else if (e->op == SOFTMAX && a->requires_grad) {
            int last_dim = a->dims[a->ndims - 1];
            int outer_size = a->size / last_dim;
            
            for (int i = 0; i < outer_size; i++) {
                float sum = 0.0f;
                for (int j = 0; j < last_dim; j++) {
                    sum += r->grad[i * last_dim + j] * r->data[i * last_dim + j];
                }
                
                for (int j = 0; j < last_dim; j++) {
                    float softmax_j = r->data[i * last_dim + j];
                    a->grad[i * last_dim + j] += softmax_j * (r->grad[i * last_dim + j] - sum);
                }
            }
        }else if (e->op == HADAMARD) {
            for (int i = 0; i < r->size; i++) {
                if (a->requires_grad) {
                    int a_idx = get_index(i, a->dims, a->ndims, r->dims, r->ndims);
                    a->grad[a_idx] += r->grad[i] * b->data[get_index(i, b->dims, b->ndims, r->dims, r->ndims)];
                }
                if (b->requires_grad) {
                    int b_idx = get_index(i, b->dims, b->ndims, r->dims, r->ndims);
                    b->grad[b_idx] += r->grad[i] * a->data[get_index(i, a->dims, a->ndims, r->dims, r->ndims)];
                }
            }
        }
        
        if (e->aux_data) {
            free(e->aux_data);
            e->aux_data = NULL;
        }
    }
    tape_len = 0;
}

Tensor* tensor_multihead_attention(Tensor* Q, Tensor* K, Tensor* V, int num_heads) {
    // Initial validation
    if (!Q || !K || !V || Q->ndims != 3 || K->ndims != 3 || V->ndims != 3) return NULL;
    
    int batch_size = Q->dims[0];
    int seq_len_q = Q->dims[1];
    int seq_len_k = K->dims[1];
    int d_model = Q->dims[2];
    
    // Validate dimensions before proceeding
    if (d_model % num_heads != 0 || 
        K->dims[2] != d_model || 
        V->dims[2] != d_model || 
        seq_len_k != V->dims[1] ||
        batch_size != K->dims[0] || 
        batch_size != V->dims[0]) return NULL;
    
    int d_head = d_model / num_heads;
    
    // Clear existing tape entries to prevent accumulation
    tape_len = 0;

    // Reshape operations
    int reshape_dims[] = {batch_size, seq_len_q, num_heads, d_head};
    Tensor* Q_reshaped = tensor_reshape(Q, 4, reshape_dims);
    if (!Q_reshaped) return NULL;

    reshape_dims[1] = seq_len_k;
    Tensor* K_reshaped = tensor_reshape(K, 4, reshape_dims);
    if (!K_reshaped) return NULL;

    Tensor* V_reshaped = tensor_reshape(V, 4, reshape_dims);
    if (!V_reshaped) return NULL;

    // Permute operations
    int perm[] = {0, 2, 1, 3};
    Tensor* Q_perm = tensor_permute(Q_reshaped, perm, 4);
    if (!Q_perm) return NULL;

    Tensor* K_perm = tensor_permute(K_reshaped, perm, 4);
    if (!K_perm) return NULL;

    Tensor* V_perm = tensor_permute(V_reshaped, perm, 4);
    if (!V_perm) return NULL;

    // K transpose for attention
    int k_perm[] = {0, 1, 3, 2};
    Tensor* K_transpose = tensor_permute(K_perm, k_perm, 4);
    if (!K_transpose) return NULL;

    // Compute attention scores
    Tensor* scores = tensor_matmul(Q_perm, K_transpose);
    if (!scores) return NULL;

    // Scale scores
    float scale = 1.0f / sqrt(d_head);
    Tensor* scale_tensor = tensor_new(4, (int[]){1, 1, 1, 1}, (float[]){scale}, 0);
    if (!scale_tensor) return NULL;

    Tensor* scaled_scores = tensor_hadamard(scores, scale_tensor);
    if (!scaled_scores) return NULL;

    // Apply softmax
    Tensor* attention_weights = tensor_softmax(scaled_scores);
    if (!attention_weights) return NULL;

    // Apply attention to values
    Tensor* attention = tensor_matmul(attention_weights, V_perm);
    if (!attention) return NULL;

    // Final permute and reshape
    int inv_perm[] = {0, 2, 1, 3};
    Tensor* attention_perm = tensor_permute(attention, inv_perm, 4);
    if (!attention_perm) return NULL;

    int final_shape[] = {batch_size, seq_len_q, d_model};
    Tensor* output = tensor_reshape(attention_perm, 3, final_shape);
    if (!output) return NULL;

    printf("DEBUG: Final tape length: %d\n", tape_len);
    return output;
}
int main() {
    // Test 1: Basic functionality with clear attention patterns
{
    printf("\nTest 1: Basic Multi-Head Attention with Simple Values\n");
    int batch_size = 1;
    int seq_len = 2;  // Simplified sequence length
    int d_model = 4;
    int num_heads = 2;
    int qkv_dims[] = {batch_size, seq_len, d_model};
    
    // Simple values for easy tracking
    float q_data[] = {
        1.0f, 1.0f, 0.0f, 0.0f,  // First sequence position, split across 2 heads
        0.0f, 0.0f, 1.0f, 1.0f   // Second sequence position
    };
    float k_data[] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f
    };
    float v_data[] = {
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f
    };

    Tensor* Q = tensor_new(3, qkv_dims, q_data, 1);
    Tensor* K = tensor_new(3, qkv_dims, k_data, 1);
    Tensor* V = tensor_new(3, qkv_dims, v_data, 1);
    
    printf("\nInput values:");
    printf("\nQ:"); for(int i = 0; i < 8; i++) printf(" %f", q_data[i]);
    printf("\nK:"); for(int i = 0; i < 8; i++) printf(" %f", k_data[i]);
    printf("\nV:"); for(int i = 0; i < 8; i++) printf(" %f", v_data[i]);
    printf("\n");
    
    Tensor* output = tensor_multihead_attention(Q, K, V, num_heads);
    
    printf("\nFinal output values:\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Seq %d:", i);
        for (int j = 0; j < d_model; j++) {
            printf(" %6.3f", output->data[i * d_model + j]);
        }
        printf("\n");
    }
    
    // Test gradients
    printf("\nSetting output gradients to 1.0\n");
    for (int i = 0; i < output->size; i++) {
        output->grad[i] = 1.0f;
    }
    
    backward();
    
    printf("\nQ gradients:\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Seq %d:", i);
        for (int j = 0; j < d_model; j++) {
            printf(" %6.3f", Q->grad[i * d_model + j]);
        }
        printf("\n");
    }
    
    printf("\nK gradients:\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Seq %d:", i);
        for (int j = 0; j < d_model; j++) {
            printf(" %6.3f", K->grad[i * d_model + j]);
        }
        printf("\n");
    }
    printf("\n");
}

    // Test 2: Uniform attention pattern
    {
        printf("Test 2: Uniform Attention Pattern\n");
        int batch_size = 1;
        int seq_len = 3;
        int d_model = 4;
        int num_heads = 2;
        int qkv_dims[] = {batch_size, seq_len, d_model};
        
        // Create uniform attention by setting all Q and K values to the same value
        float q_data[12], k_data[12], v_data[12];
        for (int i = 0; i < seq_len * d_model; i++) {
            q_data[i] = 1.0f;
            k_data[i] = 1.0f;
            v_data[i] = (i / d_model) + 1.0f;  // Different values per position
        }

        Tensor* Q = tensor_new(3, qkv_dims, q_data, 1);
        Tensor* K = tensor_new(3, qkv_dims, k_data, 1);
        Tensor* V = tensor_new(3, qkv_dims, v_data, 1);
        
        Tensor* output = tensor_multihead_attention(Q, K, V, num_heads);
        printf("Output (should show uniform mixing):\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d: ", i);
            for (int j = 0; j < d_model; j++) {
                printf("%6.3f ", output->data[i * d_model + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Test 3: Gradient Flow
{
    printf("\nTest 3: Gradient Flow\n");
    int batch_size = 1;
    int seq_len = 2;
    int d_model = 4;
    int num_heads = 2;
    int dims[] = {batch_size, seq_len, d_model};
    
    // Use same pattern as Test 1 for consistency
    float q_data[] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f
    };
    float k_data[] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f
    };
    float v_data[] = {
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f
    };

    Tensor* Q = tensor_new(3, dims, q_data, 1);
    Tensor* K = tensor_new(3, dims, k_data, 1);
    Tensor* V = tensor_new(3, dims, v_data, 1);
    
    Tensor* output = tensor_multihead_attention(Q, K, V, num_heads);
    
    // Set gradients
    for (int i = 0; i < output->size; i++) {
        output->grad[i] = 1.0f;
    }
    
    backward();
    
    printf("Q gradients:\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Seq %d:", i);
        for (int j = 0; j < d_model; j++) {
            printf(" %6.3f", Q->grad[i * d_model + j]);
        }
        printf("\n");
    }
}


    // Test 4: Error cases
    {
        printf("Test 4: Error Cases\n");
        int batch_size = 2;
        int seq_len = 3;
        int d_model = 4;
        int qkv_dims[] = {batch_size, seq_len, d_model};
        float data[24] = {0};
        
        Tensor* Q = tensor_new(3, qkv_dims, data, 1);
        Tensor* K = tensor_new(3, qkv_dims, data, 1);
        Tensor* V = tensor_new(3, qkv_dims, data, 1);
        
        printf("Invalid num_heads (should be NULL): %s\n",
               tensor_multihead_attention(Q, K, V, 3) == NULL ? "PASS" : "FAIL");
        
        int wrong_dims[] = {batch_size, seq_len, d_model + 1};
        Tensor* Wrong = tensor_new(3, wrong_dims, data, 1);
        printf("Mismatched dimensions (should be NULL): %s\n",
               tensor_multihead_attention(Q, K, Wrong, 2) == NULL ? "PASS" : "FAIL");
    }

    clean_registry();
    return 0;
}