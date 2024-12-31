#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, EXP, LOG } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static struct { TapeEntry entries[MAX_TAPE]; int len; } tape = {0};

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
    return t;
}

void tensor_free(Tensor* t) {
    free(t->data); free(t->grad); free(t->dims); free(t);
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
    
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){MATMUL, result, a, b};
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++)
        result->data[i] = expf(fmaxf(fminf(a->data[i], MAX_EXP), -MAX_EXP));
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){EXP, result, a, NULL};
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++)
        result->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){LOG, result, a, NULL};
    return result;
}

void backward() {
    if (!tape.len) return;
    
    Tensor* final = tape.entries[tape.len-1].result;
    if (!final->grad) {
        final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
    }
    
    for (int t = tape.len-1; t >= 0; t--) {
        TapeEntry* entry = &tape.entries[t];
        Tensor *result = entry->result, *a = entry->input1, *b = entry->input2;
        
        switch (entry->op) {
            case MATMUL: {
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
                int batch = result->size / (M * N);
                
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
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
                    if (!b->grad) b->grad = calloc(b->size, sizeof(float));
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
            case EXP:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] * result->data[i];
                }
                break;
            case LOG:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] / fmaxf(a->data[i], MIN_LOG);
                }
                break;
        }
    }
}

void print_tensor(const Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) printf("%d%s", t->dims[i], i < t->ndims-1 ? "," : "");
    printf("]\nData (first few elements):\n");
    for (int i = 0; i < fmin(t->size, 10); i++) printf("%8.4f ", t->data[i]);
    if (t->size > 10) printf("...");
    printf("\n");
    if (t->grad) {
        printf("Gradients (first few elements):\n");
        for (int i = 0; i < fmin(t->size, 10); i++) printf("%8.4f ", t->grad[i]);
        if (t->size > 10) printf("...");
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Test 1: 2D Matrix Multiplication
    {
        printf("Test 1: 2D Matrix Multiplication\n");
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f}, data2[] = {0.5f, 0.5f, 0.5f, 0.5f};
        int dims[] = {2, 2};
        
        Tensor *a = tensor_new(2, dims, data1, 1);
        Tensor *b = tensor_new(2, dims, data2, 1);
        Tensor *c = tensor_matmul(a, b);
        Tensor *d = tensor_exp(c);
        Tensor *e = tensor_log(d);
        
        e->grad = calloc(e->size, sizeof(float));
        e->grad[0] = 1.0f;
        
        print_tensor(a, "A (2x2)");
        print_tensor(b, "B (2x2)");
        print_tensor(c, "C = A @ B");
        print_tensor(d, "D = exp(C)");
        print_tensor(e, "E = log(D)");
        
        backward();
        printf("\nAfter backward pass:\n");
        print_tensor(a, "A (with gradients)");
        print_tensor(b, "B (with gradients)");
        
        tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(d); tensor_free(e);
        tape.len = 0;
    }

    // Test 2: 3D Tensor Multiplication
    {
        printf("\nTest 2: 3D Tensor Multiplication\n");
        int dims1[] = {2, 3, 4}, dims2[] = {2, 4, 2};
        float *data1 = malloc(24 * sizeof(float));
        float *data2 = malloc(16 * sizeof(float));
        
        for (int i = 0; i < 24; i++) data1[i] = i / 10.0f;
        for (int i = 0; i < 16; i++) data2[i] = i / 20.0f;
        
        Tensor *a = tensor_new(3, dims1, data1, 1);
        Tensor *b = tensor_new(3, dims2, data2, 1);
        Tensor *c = tensor_matmul(a, b);
        Tensor *d = tensor_exp(c);
        
        d->grad = calloc(d->size, sizeof(float));
        for (int i = 0; i < d->size; i++) d->grad[i] = 1.0f;
        
        print_tensor(a, "A (2x3x4)");
        print_tensor(b, "B (2x4x2)");
        print_tensor(c, "C = A @ B");
        print_tensor(d, "D = exp(C)");
        
        backward();
        printf("\nAfter backward pass:\n");
        print_tensor(a, "A (with gradients)");
        print_tensor(b, "B (with gradients)");
        
        free(data1); free(data2);
        tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(d);
        tape.len = 0;
    }

    // Test 3: 4D Tensor Multiplication
    {
        printf("\nTest 3: 4D Tensor Multiplication\n");
        int dims1[] = {2, 3, 4, 5}, dims2[] = {2, 3, 5, 3};
        float *data1 = malloc(120 * sizeof(float));
        float *data2 = malloc(90 * sizeof(float));
        
        for (int i = 0; i < 120; i++) data1[i] = i / 100.0f;
        for (int i = 0; i < 90; i++) data2[i] = i / 100.0f;
        
        Tensor *a = tensor_new(4, dims1, data1, 1);
        Tensor *b = tensor_new(4, dims2, data2, 1);
        Tensor *c = tensor_matmul(a, b);
        Tensor *d = tensor_log(c);
        
        d->grad = calloc(d->size, sizeof(float));
        for (int i = 0; i < d->size; i++) d->grad[i] = 1.0f;
        
        print_tensor(a, "A (2x3x4x5)");
        print_tensor(b, "B (2x3x5x3)");
        print_tensor(c, "C = A @ B");
        print_tensor(d, "D = log(C)");
        
        backward();
        printf("\nAfter backward pass:\n");
        print_tensor(a, "A (with gradients)");
        print_tensor(b, "B (with gradients)");
        
        free(data1); free(data2);
        tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(d);
        tape.len = 0;
    }
    
    return 0;
}