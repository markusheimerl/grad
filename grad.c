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
    // 1. Define dimensions and sizes
    const int batch_size = 2;
    const int channels = 3;
    const int height = 4;
    const int width = 5;
    const int dims4d[] = {batch_size, channels, height, width};
    const int tensor_size = batch_size * channels * height * width;  // 120
    
    const int matrix_rows = 60;
    const int matrix_cols = 30;
    const int matrix_dims[] = {matrix_rows, matrix_cols};
    const int matrix_size = matrix_rows * matrix_cols;  // 1800
    
    // 2. Initialize input data
    float* data1 = malloc(tensor_size * sizeof(float));
    float* data2 = malloc(tensor_size * sizeof(float));
    float* matrix_data = malloc(matrix_size * sizeof(float));
    
    for (int i = 0; i < tensor_size; i++) {
        data1[i] = sinf(i * 0.1f) + 1.0f;  // Ensure positive for log
        data2[i] = cosf(i * 0.1f) + 1.0f;
    }
    
    for (int i = 0; i < matrix_size; i++) {
        matrix_data[i] = sinf(i * 0.05f) * 0.1f;
    }
    
    // 3. Create input tensors
    Tensor* input1 = tensor_new(4, dims4d, data1, 1);
    Tensor* input2 = tensor_new(4, dims4d, data2, 1);
    Tensor* weight_matrix = tensor_new(2, matrix_dims, matrix_data, 1);
    
    // 4. Build computation graph
    // First branch: input1 + input2
    Tensor* sum = tensor_add(input1, input2);
    
    // Reshape for matrix multiplication
    const int reshape_dims[] = {batch_size, matrix_rows};
    Tensor* reshaped = tensor_reshape(sum, 2, reshape_dims);
    
    // Matrix multiplication and activation
    Tensor* matmul_result = tensor_matmul(reshaped, weight_matrix);
    Tensor* activated = tensor_exp(matmul_result);
    Tensor* logged = tensor_log(activated);
    Tensor* final_output = tensor_hadamard(logged, logged);
    
    // 5. Set output gradients
    for (int i = 0; i < final_output->size; i++) {
        final_output->grad[i] = 1.0f;
    }
    
    // 6. Perform backward pass
    backward();
    
    // 7. Print results
    printf("Network Architecture:\n");
    printf("Input tensors: %dx%dx%dx%d\n", batch_size, channels, height, width);
    printf("Weight matrix: %dx%d\n", matrix_rows, matrix_cols);
    printf("Output tensor: %dx%d\n\n", final_output->dims[0], final_output->dims[1]);
    
    // Calculate gradient statistics
    float input_grad_sum = 0, weight_grad_sum = 0;
    for (int i = 0; i < tensor_size; i++) {
        input_grad_sum += input1->grad[i];
    }
    for (int i = 0; i < matrix_size; i++) {
        weight_grad_sum += weight_matrix->grad[i];
    }
    
    printf("Gradient Statistics:\n");
    printf("Input gradient sum: %f\n", input_grad_sum);
    printf("Weight gradient sum: %f\n", weight_grad_sum);
    printf("Output samples: [%f, %f, %f]\n", 
           final_output->data[0], 
           final_output->data[final_output->size/2], 
           final_output->data[final_output->size-1]);
    
    // 8. Clean up
    free(data1);
    free(data2);
    free(matrix_data);
    clean_registry();
    
    return 0;
}