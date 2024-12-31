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
    // Create 4D tensors (batch, channels, height, width)
    int dims4d[] = {2, 3, 4, 5};  // 2 batches, 3 channels, 4x5 spatial dims
    float data1[120] = {0};  // 2*3*4*5 = 120
    float data2[120] = {0};
    float matrix_data[1800] = {0};  // 60*30 = 1800
    
    // Initialize with interesting patterns
    for(int i = 0; i < 120; i++) {
        data1[i] = sinf(i * 0.1f) + 1.0f;  // ensure positive for log
        data2[i] = cosf(i * 0.1f) + 1.0f;
    }
    for(int i = 0; i < 1800; i++) {
        matrix_data[i] = sinf(i * 0.05f) * 0.1f;
    }

    Tensor* a = tensor_new(4, dims4d, data1, 1);
    Tensor* b = tensor_new(4, dims4d, data2, 1);

    // Test all operations in a complex computation graph
    Tensor* c = tensor_add(a, b);                    // Add
    
    int reshape_dims[] = {2, 60};                    // Reshape to 2x60
    Tensor* d = tensor_reshape(c, 2, reshape_dims);
    
    int matrix_dims[] = {60, 30};                    // Create matrix for matmul
    Tensor* matrix = tensor_new(2, matrix_dims, matrix_data, 1);
    
    Tensor* e = tensor_matmul(d, matrix);            // MatMul
    Tensor* f = tensor_exp(e);                       // Exp
    Tensor* g = tensor_log(f);                       // Log
    Tensor* h = tensor_hadamard(g, g);               // Hadamard

    // Set gradient at output
    for(int i = 0; i < h->size; i++) h->grad[i] = 1.0f;

    // Backward pass
    backward();

    // Print some statistics to verify results
    printf("Input tensor dims: %dx%dx%dx%d\n", dims4d[0], dims4d[1], dims4d[2], dims4d[3]);
    printf("Final tensor dims: %dx%d\n", h->dims[0], h->dims[1]);
    
    float sum_grad_a = 0, sum_grad_matrix = 0;
    for(int i = 0; i < a->size; i++) sum_grad_a += a->grad[i];
    for(int i = 0; i < matrix->size; i++) sum_grad_matrix += matrix->grad[i];
    
    printf("Sum of gradients in first tensor: %f\n", sum_grad_a);
    printf("Sum of gradients in matrix: %f\n", sum_grad_matrix);
    printf("Sample of final output: %f %f %f\n", 
           h->data[0], h->data[h->size/2], h->data[h->size-1]);

    clean_registry();
    return 0;
}