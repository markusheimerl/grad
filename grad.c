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

Tensor* tensor_ones(int ndims, const int* dims) {
    Tensor* t = tensor_new(ndims, dims, NULL, 0);
    for (int i = 0; i < t->size; i++) t->data[i] = 1.0f;
    return t;
}

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) 
        if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
}

Tensor* tensor_reduce_sum(Tensor* a, int axis) {
    if (axis < 0 || axis >= a->ndims) return NULL;
    
    int ones_dims[2] = {1, a->dims[axis]};
    int reshape_dims[2];
    
    if (axis == a->ndims - 1) {
        ones_dims[0] = a->dims[axis];
        ones_dims[1] = 1;
        reshape_dims[0] = a->size / a->dims[axis];
        reshape_dims[1] = a->dims[axis];
    } else {
        reshape_dims[0] = a->dims[axis];
        reshape_dims[1] = a->size / a->dims[axis];
    }
    
    Tensor* ones = tensor_ones(2, ones_dims);
    Tensor* reshaped = tensor_reshape(a, 2, reshape_dims);
    Tensor* result = (axis == a->ndims - 1) ? 
                    tensor_matmul(reshaped, ones) : 
                    tensor_matmul(ones, reshaped);
    
    if (a->ndims > 2) {
        int new_ndims = a->ndims - 1;
        int* new_dims = malloc(new_ndims * sizeof(int));
        for (int i = 0, j = 0; i < a->ndims; i++) 
            if (i != axis) new_dims[j++] = a->dims[i];
        Tensor* final = tensor_reshape(result, new_ndims, new_dims);
        free(new_dims);
        return final;
    }
    
    return result;
}

int main() {
    // Test data setup
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float data2[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    int dims[] = {2, 2, 2};

    // Create tensors
    Tensor *a = tensor_new(3, dims, data1, 1);
    Tensor *b = tensor_new(3, dims, data2, 1);

    printf("Initial tensors:\n");
    printf("A (2x2x2):\n");
    for (int i = 0; i < 8; i++) {
        printf("%.2f ", a->data[i]);
        if (i == 3) printf("\n");
    }
    printf("\n\nB (2x2x2):\n");
    for (int i = 0; i < 8; i++) {
        printf("%.2f ", b->data[i]);
        if (i == 3) printf("\n");
    }
    printf("\n");

    // Hadamard product
    Tensor *c = tensor_hadamard(a, b);
    printf("\nAfter Hadamard product (c = a ⊙ b):\n");
    for (int i = 0; i < 8; i++) {
        printf("%.2f ", c->data[i]);
        if (i == 3) printf("\n");
    }
    printf("\n");

    // Reduce sum
    Tensor *d = tensor_reduce_sum(c, 1);
    printf("\nAfter reduce_sum along axis 1:\n");
    for (int i = 0; i < d->size; i++) {
        printf("%.2f ", d->data[i]);
    }
    printf("\n");

    // Exponential
    Tensor *e = tensor_exp(d);
    printf("\nAfter exponential:\n");
    for (int i = 0; i < e->size; i++) {
        printf("%.2f ", e->data[i]);
    }
    printf("\n");

    // Reshape
    int reshape_dims[] = {2, 2, 1};
    Tensor *e_reshaped = tensor_reshape(e, 3, reshape_dims);
    printf("\nAfter reshape to (2,2,1):\n");
    for (int i = 0; i < e_reshaped->size; i++) {
        printf("%.2f ", e_reshaped->data[i]);
    }
    printf("\n");

    // Final matrix multiplication
    Tensor *f = tensor_matmul(a, e_reshaped);
    printf("\nFinal output (matrix multiplication):\n");
    for (int i = 0; i < f->size; i++) {
        printf("%.2f ", f->data[i]);
    }
    printf("\n");

    // Backward pass
    for (int i = 0; i < f->size; i++) {
        f->grad[i] = 1.0f;
    }
    printf("\nSetting gradients to 1.0\n");
    backward();

    printf("\nFinal gradients of A:\n");
    for (int i = 0; i < 8; i++) {
        printf("%.2f ", a->grad[i]);
        if (i == 3) printf("\n");
    }
    printf("\n");

    clean_registry();
    return 0;
}