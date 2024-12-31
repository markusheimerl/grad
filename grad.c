#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, EXP, LOG, ADD, RESHAPE, PERMUTE} OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int* perm;
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
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){ADD, result, a, b, NULL};
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
    
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, result, a, b, NULL};
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){EXP, result, a, NULL, NULL};
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){LOG, result, a, NULL, NULL};
    return result;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* new_dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= new_dims[i];
    if (size != a->size) return NULL;

    Tensor* result = tensor_new(ndims, new_dims, a->data, a->requires_grad);
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, result, a, NULL, NULL};
    return result;
}

Tensor* tensor_permute(Tensor* a, const int* perm) {
    if (!a || !perm || a->ndims <= 1) 
        return a ? tensor_new(a->ndims, a->dims, a->data, a->requires_grad) : NULL;
    
    // Validate permutation
    char used[32] = {0};
    for (int i = 0; i < a->ndims; i++) {
        if (perm[i] < 0 || perm[i] >= a->ndims || used[perm[i]]) 
            return NULL;
        used[perm[i]] = 1;
    }

    // Calculate new dimensions
    int new_dims[32];
    for (int i = 0; i < a->ndims; i++) 
        new_dims[i] = a->dims[perm[i]];
    
    Tensor* result = tensor_new(a->ndims, new_dims, NULL, a->requires_grad);
    
    // Calculate strides for both original and new dimensions
    int old_strides[32], new_strides[32];
    
    old_strides[a->ndims - 1] = 1;
    new_strides[a->ndims - 1] = 1;
    
    for (int i = a->ndims - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * a->dims[i + 1];
        new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
    }

    // Perform permutation
    for (int idx = 0; idx < a->size; idx++) {
        // Convert linear index to coordinates
        int coords[32];
        int temp = idx;
        for (int i = 0; i < a->ndims; i++) {
            coords[i] = temp / old_strides[i];
            temp %= old_strides[i];
        }
        
        // Calculate new index using permuted coordinates
        int new_idx = 0;
        for (int i = 0; i < a->ndims; i++) {
            new_idx += coords[perm[i]] * new_strides[i];
        }
        
        result->data[new_idx] = a->data[idx];
    }

    if (result->requires_grad) {
        int* inv_perm = malloc(a->ndims * sizeof(int));
        for (int i = 0; i < a->ndims; i++) 
            inv_perm[perm[i]] = i;
        tape[tape_len++] = (TapeEntry){PERMUTE, result, a, NULL, inv_perm};
    }
    
    return result;
}

void backward() {
    if (!tape_len) return;
    
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* entry = &tape[t];
        Tensor *result = entry->result, *a = entry->input1, *b = entry->input2;
        
        switch (entry->op) {
            case MATMUL: {
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
                int batch = result->size / (M * N);
                
                if (a->requires_grad || b->requires_grad) {
                    for (int n = 0; n < batch; n++)
                        for (int i = 0; i < M; i++)
                            for (int j = 0; j < N; j++)
                                for (int k = 0; k < K; k++) {
                                    float grad = result->grad[n*M*N + i*N + j];
                                    if (a->requires_grad)
                                        a->grad[n*M*K + i*K + k] += grad * b->data[n*K*N + k*N + j];
                                    if (b->requires_grad)
                                        b->grad[n*K*N + k*N + j] += grad * a->data[n*M*K + i*K + k];
                                }
                }
                break;
            }
            case ADD:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                if (b->requires_grad)
                    for (int i = 0; i < b->size; i++) b->grad[i] += result->grad[i];
                break;
            case EXP:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] * result->data[i];
                break;
            case LOG:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] / fmaxf(a->data[i], MIN_LOG);
                break;
            case RESHAPE:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                break;
            case PERMUTE: {
                if (a->requires_grad && entry->perm) {
                    // Create inverse permutation
                    int inv_perm[32];
                    for (int i = 0; i < a->ndims; i++) {
                        inv_perm[entry->perm[i]] = i;
                    }
                    
                    // Create tensor with permuted gradients
                    Tensor* grad_tensor = tensor_new(result->ndims, result->dims, result->grad, 0);
                    if (grad_tensor) {
                        // Permute the gradients using inverse permutation
                        Tensor* permuted_grad = tensor_permute(grad_tensor, inv_perm);
                        if (permuted_grad) {
                            // Add to input gradients
                            for (int i = 0; i < a->size; i++) {
                                a->grad[i] += permuted_grad->data[i];
                            }
                        }
                    }
                }
                free(entry->perm);
                break;
            }
        }
    }
    tape_len = 0;
}

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) 
        if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
}

int main() {
    // Define dimensions
    const int batch_size = 2;
    const int channels = 3;
    const int height = 4;
    const int width = 5;
    const int dims4d[] = {batch_size, channels, height, width};
    const int tensor_size = batch_size * channels * height * width;
    float* matrix_data = NULL;  // Initialize to NULL
    
    // Create test data
    float* data1 = malloc(tensor_size * sizeof(float));
    float* data2 = malloc(tensor_size * sizeof(float));
    
    if (!data1 || !data2) {
        printf("Failed to allocate input data\n");
        goto cleanup;
    }
    
    // Initialize with safe values
    for (int i = 0; i < tensor_size; i++) {
        data1[i] = (float)(i % 10) / 10.0f + 0.5f;
        data2[i] = (float)(i % 10) / 10.0f + 0.5f;
    }
    
    // Create input tensors
    Tensor* input1 = tensor_new(4, dims4d, data1, 1);
    if (!input1) {
        printf("Failed to create input1\n");
        goto cleanup;
    }

    // Create input2 with permuted dimensions
    int permuted_dims[] = {batch_size, width, channels, height};
    Tensor* input2 = tensor_new(4, permuted_dims, data2, 1);
    if (!input2) {
        printf("Failed to create input2\n");
        goto cleanup;
    }
    
    printf("Testing Permutation:\n");
    const int perm[] = {0, 3, 1, 2};
    Tensor* permuted = tensor_permute(input1, perm);
    if (!permuted) {
        printf("Permutation failed\n");
        goto cleanup;
    }
    
    printf("Original shape: %d,%d,%d,%d\n", 
           input1->dims[0], input1->dims[1], input1->dims[2], input1->dims[3]);
    printf("Permuted shape: %d,%d,%d,%d\n", 
           permuted->dims[0], permuted->dims[1], permuted->dims[2], permuted->dims[3]);
    printf("Input2 shape: %d,%d,%d,%d\n\n", 
           input2->dims[0], input2->dims[1], input2->dims[2], input2->dims[3]);
    
    // Add permuted and input2
    Tensor* sum = tensor_add(permuted, input2);
    if (!sum) {
        printf("Addition failed\n");
        goto cleanup;
    }
    printf("Addition successful\n");
    
    // Reshape for matrix multiplication
    const int reshape_dims[] = {batch_size, channels * height * width};
    Tensor* reshaped = tensor_reshape(sum, 2, reshape_dims);
    if (!reshaped) {
        printf("Reshape failed\n");
        goto cleanup;
    }
    printf("Reshape successful\n");
    
    // Create and initialize weight matrix
    const int matrix_rows = channels * height * width;
    const int matrix_cols = 10;
    const int matrix_dims[] = {matrix_rows, matrix_cols};
    
    matrix_data = malloc(matrix_rows * matrix_cols * sizeof(float));
    if (!matrix_data) {
        printf("Failed to allocate matrix data\n");
        goto cleanup;
    }
    
    for (int i = 0; i < matrix_rows * matrix_cols; i++) {
        matrix_data[i] = (float)(i % 10) / 20.0f;
    }
    
    Tensor* weight_matrix = tensor_new(2, matrix_dims, matrix_data, 1);
    if (!weight_matrix) {
        printf("Failed to create weight matrix\n");
        goto cleanup;
    }
    
    // Matrix multiplication
    Tensor* matmul_result = tensor_matmul(reshaped, weight_matrix);
    if (!matmul_result) {
        printf("Matrix multiplication failed\n");
        goto cleanup;
    }
    printf("Matrix multiplication successful\n");
    
    // Add activation functions
    Tensor* activated = tensor_exp(matmul_result);
    if (!activated) {
        printf("Activation failed\n");
        goto cleanup;
    }
    printf("Activation successful\n");
    
    Tensor* logged = tensor_log(activated);
    if (!logged) {
        printf("Log operation failed\n");
        goto cleanup;
    }
    printf("Log operation successful\n");
    
    // Hadamard product with itself
    Tensor* final_output = tensor_hadamard(logged, logged);
    if (!final_output) {
        printf("Hadamard product failed\n");
        goto cleanup;
    }
    printf("Hadamard product successful\n");
    
    // Set gradients for backward pass
    for (int i = 0; i < final_output->size; i++) {
        final_output->grad[i] = 1.0f;
    }
    
    // Perform backward pass
    backward();
    printf("Backward pass completed\n");
    
    // Print final output shape and gradient statistics
    printf("\nFinal output shape: %d,%d\n", 
           final_output->dims[0], final_output->dims[1]);
    
    // Calculate gradient statistics
    double input_grad_sum = 0.0;  // Use double for better precision
    float input_grad_max = -INFINITY;
    float input_grad_min = INFINITY;
    int valid_grads = 0;
    
    for (int i = 0; i < input1->size; i++) {
        if (isfinite(input1->grad[i])) {
            input_grad_sum += input1->grad[i];
            input_grad_max = fmaxf(input_grad_max, input1->grad[i]);
            input_grad_min = fminf(input_grad_min, input1->grad[i]);
            valid_grads++;
        }
    }
    
    printf("\nGradient statistics:\n");
    printf("Valid gradients: %d/%d\n", valid_grads, input1->size);
    printf("Sum: %.6f\n", input_grad_sum);
    printf("Max: %.6f\n", input_grad_max);
    printf("Min: %.6f\n", input_grad_min);
    if (valid_grads > 0) {
        printf("Mean: %.6f\n", input_grad_sum / valid_grads);
    }
    
cleanup:
    free(data1);
    free(data2);
    if (matrix_data) free(matrix_data);
    clean_registry();
    
    return 0;
}