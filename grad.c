#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, EXP, LOG, ADD, SUB, RESHAPE, REDUCE_SUM } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int reduce_dim;
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
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){op, r, a, b, -1};
    return r;
}

Tensor* tensor_add(Tensor* a, Tensor* b) { return tensor_op(a, b, ADD); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return tensor_op(a, b, SUB); }

Tensor* tensor_exp(Tensor* a) {
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) r->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){EXP, r, a, NULL, -1};
    return r;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) r->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){LOG, r, a, NULL, -1};
    return r;
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
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, r, a, b, -1};
    return r;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL, -1};
    return r;
}

Tensor* tensor_reduce_sum(Tensor* a, int dim) {
    if (!a || dim >= a->ndims) return NULL;
    
    int new_dims[32], new_ndims = a->ndims - 1;
    for (int i = 0, j = 0; i < a->ndims; i++)
        if (i != dim) new_dims[j++] = a->dims[i];
    
    Tensor* r = tensor_new(new_ndims, new_dims, NULL, a->requires_grad);
    
    int stride[32];
    stride[a->ndims - 1] = 1;
    for (int i = a->ndims - 2; i >= 0; i--)
        stride[i] = stride[i + 1] * a->dims[i + 1];
    
    memset(r->data, 0, r->size * sizeof(float));
    
    for (int i = 0; i < a->size; i++) {
        int coords[32], idx = i;
        for (int d = 0; d < a->ndims; d++) {
            coords[d] = idx / stride[d];
            idx %= stride[d];
        }
        
        int target_idx = 0;
        int stride_r = 1;
        for (int d = a->ndims - 1, r_d = new_ndims - 1; d >= 0; d--) {
            if (d != dim) {
                target_idx += coords[d] * stride_r;
                stride_r *= new_dims[r_d--];
            }
        }
        r->data[target_idx] += a->data[i];
    }
    
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){REDUCE_SUM, r, a, NULL, dim};
    return r;
}

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
        else if (e->op == EXP && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i] * r->data[i];
        }
        else if (e->op == LOG && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i] / fmaxf(a->data[i], MIN_LOG);
        }
        else if (e->op == RESHAPE && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i];
        }
        else if (e->op == REDUCE_SUM && a->requires_grad) {
            int stride[32];
            stride[a->ndims - 1] = 1;
            for (int i = a->ndims - 2; i >= 0; i--)
                stride[i] = stride[i + 1] * a->dims[i + 1];
            
            for (int i = 0; i < a->size; i++) {
                int coords[32], idx = i;
                for (int d = 0; d < a->ndims; d++) {
                    coords[d] = idx / stride[d];
                    idx %= stride[d];
                }
                
                int target_idx = 0;
                int stride_r = 1;
                for (int d = a->ndims - 1, r_d = r->ndims - 1; d >= 0; d--) {
                    if (d != e->reduce_dim) {
                        target_idx += coords[d] * stride_r;
                        stride_r *= r->dims[r_d--];
                    }
                }
                a->grad[i] += r->grad[target_idx];
            }
        }
    }
    tape_len = 0;
}

int main() {
    // Test 1: Simple 2D tensor reduction along dim 0
    {
        printf("\nTest 1: 2D tensor reduction along dim 0\n");
        float data[] = {1, 2, 3,
                       4, 5, 6};
        int dims[] = {2, 3};
        Tensor* a = tensor_new(2, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        
        printf("Input:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%.1f ", a->data[i*3 + j]);
            }
            printf("\n");
        }
        
        printf("Output (reduce dim 0):\n");
        for (int i = 0; i < 3; i++) {
            printf("%.1f ", b->data[i]);
        }
        printf("\n");
        
        // Test backward
        b->grad[0] = 1.0;
        b->grad[1] = 1.0;
        b->grad[2] = 1.0;
        backward();
        
        printf("Gradients:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%.1f ", a->grad[i*3 + j]);
            }
            printf("\n");
        }
    }

    // Test 2: 3D tensor reduction along dim 1
    {
        printf("\nTest 2: 3D tensor reduction along dim 1\n");
        float data[] = {1, 2,
                       3, 4,
                       5, 6,
                       
                       7, 8,
                       9, 10,
                       11, 12};
        int dims[] = {2, 3, 2};
        Tensor* a = tensor_new(3, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 1);
        
        printf("Input:\n");
        for (int i = 0; i < 2; i++) {
            printf("Batch %d:\n", i);
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 2; k++) {
                    printf("%.1f ", a->data[i*6 + j*2 + k]);
                }
                printf("\n");
            }
        }
        
        printf("Output (reduce dim 1):\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.1f ", b->data[i*2 + j]);
            }
            printf("\n");
        }
        
        // Test backward
        for (int i = 0; i < b->size; i++) {
            b->grad[i] = 1.0;
        }
        backward();
        
        printf("Gradients:\n");
        for (int i = 0; i < 2; i++) {
            printf("Batch %d:\n", i);
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 2; k++) {
                    printf("%.1f ", a->grad[i*6 + j*2 + k]);
                }
                printf("\n");
            }
        }
    }

    // Test 3: Edge case - reduction of 1D tensor
    {
        printf("\nTest 3: 1D tensor reduction\n");
        float data[] = {1, 2, 3, 4};
        int dims[] = {4};
        Tensor* a = tensor_new(1, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        
        printf("Input: ");
        for (int i = 0; i < 4; i++) {
            printf("%.1f ", a->data[i]);
        }
        printf("\nOutput: %.1f\n", b->data[0]);
        
        b->grad[0] = 1.0;
        backward();
        
        printf("Gradients: ");
        for (int i = 0; i < 4; i++) {
            printf("%.1f ", a->grad[i]);
        }
        printf("\n");
    }

    // Test 4: Verify numerical accuracy with larger numbers
    {
        printf("\nTest 4: Numerical accuracy test\n");
        float data[] = {100.5, 200.7, 300.3, 400.1};
        int dims[] = {4};
        Tensor* a = tensor_new(1, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        float expected_sum = 1001.6; // pre-calculated sum
        printf("Sum: %.6f (Expected: %.6f)\n", b->data[0], expected_sum);
        printf("Difference: %.6f\n", fabsf(b->data[0] - expected_sum));
    }

    // Test 5: Test reduction with zeros and negative numbers
    {
        printf("\nTest 5: Zeros and negative numbers\n");
        float data[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0};
        int dims[] = {2, 3};
        Tensor* a = tensor_new(2, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        printf("Input:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) printf("%.1f ", a->data[i*3 + j]);
            printf("\n");
        }
        printf("Output:\n");
        for (int i = 0; i < 3; i++) printf("%.1f ", b->data[i]);
        printf("\n");
    }

    // Test 6: Test reduction with dimension size 1
    {
        printf("\nTest 6: Dimension size 1\n");
        float data[] = {1.0, 2.0, 3.0};
        int dims[] = {1, 3};
        Tensor* a = tensor_new(2, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        printf("Input shape: [1, 3]\n");
        printf("Output shape: [3]\n");
        printf("Output: ");
        for (int i = 0; i < 3; i++) printf("%.1f ", b->data[i]);
        printf("\n");
    }

    // Test 7: Multiple reductions
    {
        printf("\nTest 7: Multiple reductions\n");
        float data[] = {1,2,3,4,5,6,7,8};
        int dims[] = {2,2,2};
        Tensor* a = tensor_new(3, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        Tensor* c = tensor_reduce_sum(b, 0);
        printf("Original sum: %.1f\n", c->data[0]);
        
        // Verify same result with different reduction order
        Tensor* d = tensor_reduce_sum(a, 1);
        Tensor* e = tensor_reduce_sum(d, 0);
        printf("Alternative reduction path sum: %.1f\n", e->data[0]);
    }

    // Test 8: Large tensor gradient accumulation
    {
        printf("\nTest 8: Large tensor gradient accumulation\n");
        float* data = malloc(1000 * sizeof(float));
        for (int i = 0; i < 1000; i++) data[i] = 1.0f;
        int dims[] = {10, 10, 10};
        Tensor* a = tensor_new(3, dims, data, 1);
        
        // First reduction and backward
        Tensor* b = tensor_reduce_sum(a, 0);
        for (int i = 0; i < b->size; i++) {
            b->grad[i] = 1.0f;
        }
        backward();
        
        // Second reduction and backward
        Tensor* c = tensor_reduce_sum(a, 0);
        for (int i = 0; i < c->size; i++) {
            c->grad[i] = 1.0f;
        }
        backward();
        
        // Check if gradients accumulated correctly
        float grad_sum = 0;
        for (int i = 0; i < a->size; i++) {
            grad_sum += a->grad[i];
        }
        printf("Input tensor size: %d\n", a->size);
        printf("Reduced tensor size: %d\n", b->size);
        printf("Gradient sum after two backwards: %.1f (Expected: 2000.0)\n", grad_sum);
        
        // Print first few gradients for debugging
        printf("First few gradients: ");
        for (int i = 0; i < 5; i++) {
            printf("%.1f ", a->grad[i]);
        }
        printf("\n");
        
        free(data);
    }

    // Test 9: Very small numbers
    {
        printf("\nTest 9: Very small numbers\n");
        float data[] = {1e-30f, 1e-31f, 1e-32f, 1e-33f};
        int dims[] = {4};
        Tensor* a = tensor_new(1, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 0);
        printf("Sum of very small numbers: %.2e\n", b->data[0]);
    }

    // Test 10: Invalid reduction dimension
    {
        printf("\nTest 10: Invalid reduction tests\n");
        float data[] = {1.0f, 2.0f};
        int dims[] = {2};
        Tensor* a = tensor_new(1, dims, data, 1);
        Tensor* b = tensor_reduce_sum(a, 1);  // Should return NULL
        printf("Reduction with invalid dim: %s\n", b == NULL ? "Correctly returned NULL" : "Failed");
    }

    clean_registry();
    return 0;
}