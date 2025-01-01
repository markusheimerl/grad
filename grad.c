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
        tape[tape_len++] = (TapeEntry){SOFTMAX, r, a, NULL};
    }
    
    return r;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL};
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

int main() {
    // Test 1: 3D tensors with exact shape match
    int dims1[] = {2, 3, 4};
    float data1[24];
    for (int i = 0; i < 24; i++) data1[i] = i + 1;
    Tensor* a = tensor_new(3, dims1, data1, 1);

    int dims2[] = {2, 3, 4};
    float data2[24];
    for (int i = 0; i < 24; i++) data2[i] = (i % 4) + 1;
    Tensor* b = tensor_new(3, dims2, data2, 1);

    Tensor* c = tensor_hadamard(a, b);
    
    printf("Test 1 - 3D * 3D:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                printf("%6.2f ", c->data[i*12 + j*4 + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Test 2: Broadcasting 2D to 3D
    int dims3[] = {1, 3, 4};
    float data3[12];
    for (int i = 0; i < 12; i++) data3[i] = i + 1;
    Tensor* d = tensor_new(3, dims3, data3, 1);

    Tensor* e = tensor_hadamard(a, d);
    
    printf("Test 2 - 3D * (1,3,4) broadcast:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                printf("%6.2f ", e->data[i*12 + j*4 + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Test 3: Broadcasting 1D to 3D
    int dims4[] = {4};
    float data4[] = {1, 2, 3, 4};
    Tensor* f = tensor_new(1, dims4, data4, 1);

    Tensor* g = tensor_hadamard(a, f);
    
    printf("Test 3 - 3D * (4) broadcast:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                printf("%6.2f ", g->data[i*12 + j*4 + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Test 4: Gradient checking
    // Set specific gradients in the output
    for (int i = 0; i < g->size; i++) {
        g->grad[i] = 1.0f;
    }
    
    backward();
    
    printf("Test 4 - Gradient check for first tensor (should match broadcast pattern):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                printf("%6.2f ", a->grad[i*12 + j*4 + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("Gradient check for broadcast tensor:\n");
    for (int i = 0; i < f->size; i++) {
        printf("%6.2f ", f->grad[i]);
    }
    printf("\n\n");

    // Test 5: Edge cases
    int dims5[] = {2, 1, 4};
    float data5[8];
    for (int i = 0; i < 8; i++) data5[i] = i + 1;
    Tensor* h = tensor_new(3, dims5, data5, 1);

    Tensor* i = tensor_hadamard(a, h);
    
    printf("Test 5 - 3D * (2,1,4) broadcast:\n");
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 3; y++) {
            for (int z = 0; z < 4; z++) {
                printf("%6.2f ", i->data[x*12 + y*4 + z]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Test 6: Error cases
    int wrong_dims[] = {2, 2, 3};
    float wrong_data[12];
    for (int i = 0; i < 12; i++) wrong_data[i] = i;
    Tensor* wrong = tensor_new(3, wrong_dims, wrong_data, 1);

    Tensor* should_be_null = tensor_hadamard(a, wrong);
    printf("Test 6 - Incompatible shapes should return NULL: %s\n", 
           should_be_null == NULL ? "PASS" : "FAIL");

    clean_registry();
    return 0;
}