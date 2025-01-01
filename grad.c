#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, ADD, SUB, RESHAPE, SOFTMAX, PERMUTE, RMSNORM } OpType;

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
        }
        
        if (e->aux_data) {
            free(e->aux_data);
            e->aux_data = NULL;
        }
    }
    tape_len = 0;
}

int main() {
    printf("=== Comprehensive RMSNorm Test Suite ===\n\n");

    // Test 1: Verify RMS of output is ~1.0
    printf("Test 1: Verify RMS of output equals 1.0\n");
    {
        int dims[] = {4};
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* normalized = tensor_rms_norm(x, 1e-6f);
        
        // Calculate RMS of output
        float rms = 0.0f;
        for(int i = 0; i < 4; i++) {
            rms += normalized->data[i] * normalized->data[i];
        }
        rms = sqrt(rms / 4);
        
        printf("Input: [%.2f %.2f %.2f %.2f]\n", data[0], data[1], data[2], data[3]);
        printf("Output: [%.6f %.6f %.6f %.6f]\n", 
               normalized->data[0], normalized->data[1], 
               normalized->data[2], normalized->data[3]);
        printf("RMS of output: %.9f (should be very close to 1.0)\n", rms);
    }
    printf("\n");

    // Test 2: Scale invariance
    printf("Test 2: Scale Invariance\n");
    {
        int dims[] = {4};
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float data2[] = {2.0f, 4.0f, 6.0f, 8.0f};  // scaled by 2
        
        Tensor* x1 = tensor_new(1, dims, data1, 1);
        Tensor* x2 = tensor_new(1, dims, data2, 1);
        Tensor* norm1 = tensor_rms_norm(x1, 1e-6f);
        Tensor* norm2 = tensor_rms_norm(x2, 1e-6f);
        
        printf("Original normalized: [%.6f %.6f %.6f %.6f]\n",
               norm1->data[0], norm1->data[1], norm1->data[2], norm1->data[3]);
        printf("Scaled normalized:   [%.6f %.6f %.6f %.6f]\n",
               norm2->data[0], norm2->data[1], norm2->data[2], norm2->data[3]);
    }
    printf("\n");

    // Test 3: Edge cases
    printf("Test 3: Edge Cases\n");
    {
        int dims[] = {4};
        // Test with very small values
        float small_data[] = {1e-6f, 2e-6f, 3e-6f, 4e-6f};
        // Test with very large values
        float large_data[] = {1e6f, 2e6f, 3e6f, 4e6f};
        // Test with zeros (with one non-zero to avoid division by zero)
        float zero_data[] = {0.0f, 0.0f, 0.0f, 1e-5f};
        
        Tensor* x_small = tensor_new(1, dims, small_data, 1);
        Tensor* x_large = tensor_new(1, dims, large_data, 1);
        Tensor* x_zero = tensor_new(1, dims, zero_data, 1);
        
        Tensor* norm_small = tensor_rms_norm(x_small, 1e-6f);
        Tensor* norm_large = tensor_rms_norm(x_large, 1e-6f);
        Tensor* norm_zero = tensor_rms_norm(x_zero, 1e-6f);
        
        printf("Small values normalized: [%.6f %.6f %.6f %.6f]\n",
               norm_small->data[0], norm_small->data[1], 
               norm_small->data[2], norm_small->data[3]);
        printf("Large values normalized: [%.6f %.6f %.6f %.6f]\n",
               norm_large->data[0], norm_large->data[1], 
               norm_large->data[2], norm_large->data[3]);
        printf("Near-zero values normalized: [%.6f %.6f %.6f %.6f]\n",
               norm_zero->data[0], norm_zero->data[1], 
               norm_zero->data[2], norm_zero->data[3]);
    }
    printf("\n");

    // Test 4: Gradient verification
    printf("Test 4: Gradient Verification\n");
    {
        int dims[] = {3};
        float data[] = {1.0f, 2.0f, 3.0f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* normalized = tensor_rms_norm(x, 1e-6f);
        
        // Set gradient to [1, 1, 1]
        for(int i = 0; i < 3; i++) normalized->grad[i] = 1.0f;
        
        backward();
        
        printf("Input: [%.2f %.2f %.2f]\n", data[0], data[1], data[2]);
        printf("Normalized: [%.6f %.6f %.6f]\n",
               normalized->data[0], normalized->data[1], normalized->data[2]);
        printf("Gradients: [%.6f %.6f %.6f]\n",
               x->grad[0], x->grad[1], x->grad[2]);
        
        // Verify gradient using finite differences
        float eps = 1e-4f;
        printf("\nNumerical gradient check (should be close to analytical gradients):\n");
        for(int i = 0; i < 3; i++) {
            data[i] += eps;
            Tensor* x_plus = tensor_new(1, dims, data, 0);
            Tensor* norm_plus = tensor_rms_norm(x_plus, 1e-6f);
            
            data[i] -= 2*eps;
            Tensor* x_minus = tensor_new(1, dims, data, 0);
            Tensor* norm_minus = tensor_rms_norm(x_minus, 1e-6f);
            
            float numerical_grad = 0;
            for(int j = 0; j < 3; j++) {
                numerical_grad += (norm_plus->data[j] - norm_minus->data[j]) / (2*eps);
            }
            
            printf("dim %d numerical grad: %.6f\n", i, numerical_grad);
            
            data[i] += eps;  // restore original value
        }
    }

    clean_registry();
    return 0;
}