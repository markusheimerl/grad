#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, ADD, SUB, RESHAPE, SOFTMAX, PERMUTE } OpType;

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
        }
        
        if (e->aux_data) {
            free(e->aux_data);
            e->aux_data = NULL;
        }
    }
    tape_len = 0;
}

int main() {
    printf("=== Comprehensive Matrix Operations Test Suite ===\n\n");

    // Test 1: Basic matrix multiplication with 2D matrices
    printf("Test 1: Basic 2D Matrix Multiplication\n");
    {
        int dims1[] = {3, 4};
        int dims2[] = {4, 2};
        float data1[] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        float data2[] = {
            1, 2,
            3, 4,
            5, 6,
            7, 8
        };
        
        Tensor* a = tensor_new(2, dims1, data1, 1);
        Tensor* b = tensor_new(2, dims2, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        
        printf("Matrix A (3x4):\n");
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 4; j++) printf("%4.0f ", data1[i*4 + j]);
            printf("\n");
        }
        
        printf("\nMatrix B (4x2):\n");
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", data2[i*2 + j]);
            printf("\n");
        }
        
        printf("\nResult C (3x2):\n");
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", c->data[i*2 + j]);
            printf("\n");
        }
    }
    printf("\n");

    // Test 2: 3D batch matrix multiplication
    printf("Test 2: 3D Batch Matrix Multiplication\n");
    {
        int dims1[] = {2, 2, 3};  // 2 batches of 2x3 matrices
        int dims2[] = {2, 3, 2};  // 2 batches of 3x2 matrices
        float data1[] = {
            1, 2, 3,
            4, 5, 6,
            
            7, 8, 9,
            10, 11, 12
        };
        float data2[] = {
            1, 2,
            3, 4,
            5, 6,
            
            7, 8,
            9, 10,
            11, 12
        };
        
        Tensor* a = tensor_new(3, dims1, data1, 1);
        Tensor* b = tensor_new(3, dims2, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        
        printf("Batch 1 Result:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", c->data[i*2 + j]);
            printf("\n");
        }
        
        printf("\nBatch 2 Result:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", c->data[4 + i*2 + j]);
            printf("\n");
        }
    }
    printf("\n");

    // Test 3: Complex operation chain with gradients
    printf("Test 3: Complex Operation Chain with Gradients\n");
    {
        int dims1[] = {2, 2};
        int dims2[] = {2, 2};
        float data1[] = {1, 2,
                        3, 4};
        float data2[] = {5, 6,
                        7, 8};
        
        Tensor* a = tensor_new(2, dims1, data1, 1);
        Tensor* b = tensor_new(2, dims2, data2, 1);
        
        // Perform: C = (A @ B).permute(1,0)
        Tensor* c = tensor_matmul(a, b);
        int perm[] = {1, 0};
        Tensor* d = tensor_permute(c, perm, 2);
        
        printf("Initial A:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", a->data[i*2 + j]);
            printf("\n");
        }
        
        printf("\nInitial B:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", b->data[i*2 + j]);
            printf("\n");
        }
        
        printf("\nResult before permute:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", c->data[i*2 + j]);
            printf("\n");
        }
        
        printf("\nFinal result after permute:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.0f ", d->data[i*2 + j]);
            printf("\n");
        }
        
        // Set gradient and backpropagate
        d->grad[0] = 1.0f;
        d->grad[1] = 0.5f;
        backward();
        
        printf("\nGradients for A:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.1f ", a->grad[i*2 + j]);
            printf("\n");
        }
        
        printf("\nGradients for B:\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%4.1f ", b->grad[i*2 + j]);
            printf("\n");
        }
    }

    clean_registry();
    return 0;
}