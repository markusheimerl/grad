#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, EXP, LOG, ADD, RESHAPE } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static TapeEntry tape[MAX_TAPE];
static int tape_len = 0;
static Tensor* registry[MAX_TENSORS];
static int registry_len = 0;

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
    for (int i = 0; i < registry_len; i++) {
        free(registry[i]->data);
        free(registry[i]->grad);
        free(registry[i]->dims);
        free(registry[i]);
    }
    registry_len = 0;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = a->data[i] + b->data[i];
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){ADD, result, a, b};
    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
    
    int max_ndims = fmax(a->ndims, b->ndims);
    int* result_dims = malloc(max_ndims * sizeof(int));
    memcpy(result_dims, (a->ndims > b->ndims ? a : b)->dims, (max_ndims - 2) * sizeof(int));
    result_dims[max_ndims-2] = a->dims[a->ndims-2];
    result_dims[max_ndims-1] = b->dims[b->ndims-1];
    
    Tensor* result = tensor_new(max_ndims, result_dims, NULL, a->requires_grad || b->requires_grad);
    free(result_dims);
    
    int batch = result->size / (result->dims[max_ndims-2] * result->dims[max_ndims-1]);
    int M = a->dims[a->ndims-2], N = b->dims[b->ndims-1], K = a->dims[a->ndims-1];
    
    for (int n = 0; n < batch; n++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a->data[n*M*K + i*K + k] * b->data[n*K*N + k*N + j];
                result->data[n*M*N + i*N + j] = sum;
            }
    
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, result, a, b};
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){EXP, result, a, NULL};
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){LOG, result, a, NULL};
    return result;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* new_dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= new_dims[i];
    if (size != a->size) return NULL;

    Tensor* result = tensor_new(ndims, new_dims, a->data, a->requires_grad);
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, result, a, NULL};
    return result;
}

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
}

// Helper function to create a tensor filled with ones

Tensor* tensor_ones(int ndims, const int* dims) {
    Tensor* t = tensor_new(ndims, dims, NULL, 0);
    for (int i = 0; i < t->size; i++) t->data[i] = 1.0f;
    return t;
}

Tensor* tensor_reduce_sum(Tensor* a, int axis) {
    if (axis < 0 || axis >= a->ndims) return NULL;
    
    // Create shape for the ones tensor
    int ones_dims[2] = {1, a->dims[axis]};
    if (axis == a->ndims - 1) {
        ones_dims[0] = a->dims[axis];
        ones_dims[1] = 1;
    }
    
    // Calculate reshape dimensions
    int reshape_dims[2];
    if (axis == a->ndims - 1) {
        int prod = 1;
        for (int i = 0; i < a->ndims - 1; i++) prod *= a->dims[i];
        reshape_dims[0] = prod;
        reshape_dims[1] = a->dims[axis];
    } else {
        reshape_dims[0] = a->dims[axis];
        int prod = 1;
        for (int i = 0; i < a->ndims; i++) {
            if (i != axis) prod *= a->dims[i];
        }
        reshape_dims[1] = prod;
    }
    
    // Create ones tensor and perform reduction
    Tensor* ones = tensor_ones(2, ones_dims);
    Tensor* reshaped = tensor_reshape(a, 2, reshape_dims);
    Tensor* result = (axis == a->ndims - 1) ? 
                    tensor_matmul(reshaped, ones) : 
                    tensor_matmul(ones, reshaped);
    
    // If result should have more than 1 dimension, reshape it
    if (a->ndims > 2) {
        int new_ndims = a->ndims - 1;
        int* new_dims = malloc(new_ndims * sizeof(int));
        for (int i = 0, j = 0; i < a->ndims; i++) {
            if (i != axis) new_dims[j++] = a->dims[i];
        }
        Tensor* final = tensor_reshape(result, new_ndims, new_dims);
        free(new_dims);
        return final;
    }
    
    return result;
}

void backward() {
    if (!tape_len) return;
    Tensor* final = tape[tape_len-1].result;
    if (!final->requires_grad) return;
 
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* entry = &tape[t];
        Tensor *result = entry->result, *a = entry->input1, *b = entry->input2;
        
        switch (entry->op) {
            case MATMUL: {
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
                int batch = result->size / (M * N);
                
                if (a->requires_grad) {
                    for (int n = 0; n < batch; n++)
                        for (int i = 0; i < M; i++)
                            for (int k = 0; k < K; k++) {
                                float sum = 0;
                                for (int j = 0; j < N; j++)
                                    sum += result->grad[n*M*N + i*N + j] * b->data[n*K*N + k*N + j];
                                a->grad[n*M*K + i*K + k] += sum;
                            }
                }
                
                if (b->requires_grad) {
                    for (int n = 0; n < batch; n++)
                        for (int k = 0; k < K; k++)
                            for (int j = 0; j < N; j++) {
                                float sum = 0;
                                for (int i = 0; i < M; i++)
                                    sum += a->data[n*M*K + i*K + k] * result->grad[n*M*N + i*N + j];
                                b->grad[n*K*N + k*N + j] += sum;
                            }
                }
                break;
            }
            case ADD:
                if (a->requires_grad) {
                    for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                }
                if (b->requires_grad) {
                    for (int i = 0; i < b->size; i++) b->grad[i] += result->grad[i];
                }
                break;
            case EXP:
                if (a->requires_grad) {
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] * result->data[i];
                }
                break;
            case LOG:
                if (a->requires_grad) {
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] / fmaxf(a->data[i], MIN_LOG);
                }
                break;
            case RESHAPE:
                if (a->requires_grad) {
                    for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                }
                break;
        }
    }
    tape_len = 0;
}

void print_tensor(const Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) printf("%d%s", t->dims[i], i < t->ndims-1 ? "," : "");
    printf("]\nData: ");
    for (int i = 0; i < t->size; i++) printf("%.4f ", t->data[i]);
    if (t->grad) {
        printf("\nGrad: ");
        for (int i = 0; i < t->size; i++) printf("%.4f ", t->grad[i]);
    }
    printf("\n\n");
}

int main() {
    // Test 1: 3D Batch Matrix Multiplication
    {
        printf("\nTest 1: 3D Batch Matrix Multiplication\n");
        float data1[] = {
            0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8,  0.9, 1.0, 1.1, 1.2,
            0.2, 0.3, 0.4, 0.5,  0.6, 0.7, 0.8, 0.9,  1.0, 1.1, 1.2, 1.3
        };
        float data2[] = {
            0.05, 0.06, 0.07,  0.08, 0.09, 0.10,  0.11, 0.12, 0.13,  0.14, 0.15, 0.16,
            0.04, 0.05, 0.06,  0.07, 0.08, 0.09,  0.10, 0.11, 0.12,  0.13, 0.14, 0.15
        };
        int dims1[] = {2,3,4};
        int dims2[] = {2,4,3};
        
        Tensor *a = tensor_new(3, dims1, data1, 1);
        Tensor *b = tensor_new(3, dims2, data2, 1);
        Tensor *c = tensor_matmul(a, b);
        Tensor *d = tensor_exp(c);
        
        for (int i = 0; i < d->size; i++) d->grad[i] = 1.0f;
        
        print_tensor(a, "A");
        print_tensor(b, "B");
        print_tensor(c, "C = A @ B");
        print_tensor(d, "D = exp(C)");
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A");
        print_tensor(b, "B");
    }

    // Test 2: 4D Tensor Addition
    {
        printf("\nTest 2: 4D Tensor Addition\n");
        float data1[36], data2[36];
        for (int i = 0; i < 36; i++) {
            data1[i] = (float)(i + 1) * 0.1f;
            data2[i] = (float)(i + 1) * 0.05f;
        }
        int dims[] = {2,2,3,3};
        
        Tensor *a = tensor_new(4, dims, data1, 1);
        Tensor *b = tensor_new(4, dims, data2, 1);
        Tensor *c = tensor_add(a, b);
        Tensor *d = tensor_exp(c);
        
        for (int i = 0; i < d->size; i++) d->grad[i] = 1.0f;
        
        print_tensor(a, "A");
        print_tensor(b, "B");
        print_tensor(c, "C = A + B");
        print_tensor(d, "D = exp(C)");
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A");
        print_tensor(b, "B");
    }

    // Test 3: 3D Hadamard Product
    {
        printf("\nTest 3: 3D Hadamard Product\n");
        float data1[24], data2[24];
        for (int i = 0; i < 24; i++) {
            data1[i] = (float)(i + 1) * 0.2f;
            data2[i] = (float)(i + 1) * 0.15f;
        }
        int dims[] = {2,3,4};
        
        Tensor *a = tensor_new(3, dims, data1, 1);
        Tensor *b = tensor_new(3, dims, data2, 1);
        Tensor *c = tensor_hadamard(a, b);
        
        for (int i = 0; i < c->size; i++) c->grad[i] = 1.0f;
        
        print_tensor(a, "A");
        print_tensor(b, "B");
        print_tensor(c, "C = A ⊙ B");
        
        printf("Direct multiplication verification for first few elements: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", data1[i] * data2[i]);
        printf("\n\n");
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A");
        print_tensor(b, "B");
    }

    // Test 4: Reshape Operation
    {
        printf("\nTest 4: Reshape Operation\n");
        float data[24];
        for (int i = 0; i < 24; i++) {
            data[i] = (float)(i + 1) * 0.1f;
        }
        int orig_dims[] = {2,3,4};
        int new_dims[] = {4,6};
        
        Tensor *a = tensor_new(3, orig_dims, data, 1);
        Tensor *b = tensor_reshape(a, 2, new_dims);
        Tensor *c = tensor_exp(b);
        
        print_tensor(a, "A (original)");
        print_tensor(b, "B (reshaped)");
        print_tensor(c, "C = exp(B)");
        
        for (int i = 0; i < c->size; i++) c->grad[i] = 1.0f;
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A");
        print_tensor(b, "B");
    }

// Test 5: Reduce Sum
    {
        printf("\nTest 1: 3D Tensor Reduction\n");
        float data[24];
        for (int i = 0; i < 24; i++) data[i] = i + 1;
        int dims[] = {2, 3, 4};  // 2 batches, 3 rows, 4 columns
        
        Tensor* a = tensor_new(3, dims, data, 1);
        
        // Reduce along different axes
        Tensor* sum_axis0 = tensor_reduce_sum(a, 0);  // Should be (3x4)
        Tensor* sum_axis1 = tensor_reduce_sum(a, 1);  // Should be (2x4)
        Tensor* sum_axis2 = tensor_reduce_sum(a, 2);  // Should be (2x3)
        
        print_tensor(a, "A (2x3x4)");
        print_tensor(sum_axis0, "Sum along axis 0 (batch)");
        print_tensor(sum_axis1, "Sum along axis 1 (rows)");
        print_tensor(sum_axis2, "Sum along axis 2 (columns)");
        
        // Set gradients for backward
        for (int i = 0; i < sum_axis2->size; i++) sum_axis2->grad[i] = 1.0f;
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A gradients");
    }

    // Test 2: 4D tensor (2x2x3x3)
    {
        printf("\nTest 2: 4D Tensor Reduction\n");
        float data[36];
        for (int i = 0; i < 36; i++) data[i] = i + 1;
        int dims[] = {2, 2, 3, 3};  // 2 batches, 2 channels, 3 rows, 3 columns
        
        Tensor* a = tensor_new(4, dims, data, 1);
        
        // Reduce along different axes
        Tensor* sum_axis0 = tensor_reduce_sum(a, 0);  // Should be (2x3x3)
        Tensor* sum_axis1 = tensor_reduce_sum(a, 1);  // Should be (2x3x3)
        Tensor* sum_axis2 = tensor_reduce_sum(a, 2);  // Should be (2x2x3)
        Tensor* sum_axis3 = tensor_reduce_sum(a, 3);  // Should be (2x2x3)
        
        print_tensor(a, "A (2x2x3x3)");
        print_tensor(sum_axis0, "Sum along axis 0 (batch)");
        print_tensor(sum_axis1, "Sum along axis 1 (channels)");
        print_tensor(sum_axis2, "Sum along axis 2 (rows)");
        print_tensor(sum_axis3, "Sum along axis 3 (columns)");
        
        // Set gradients for backward
        for (int i = 0; i < sum_axis3->size; i++) sum_axis3->grad[i] = 1.0f;
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A gradients");
    }

    clean_registry();
    return 0;
}