#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, EXP, LOG, ADD } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static struct {
    TapeEntry entries[MAX_TAPE];
    int len;
} tape = {0};

static struct {
    Tensor* tensors[MAX_TENSORS];
    int len;
} registry = {0};

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
    if (registry.len < MAX_TENSORS) registry.tensors[registry.len++] = t;
    return t;
}

void clean_registry() {
    for (int i = 0; i < registry.len; i++) {
        if (registry.tensors[i]) {
            free(registry.tensors[i]->data);
            free(registry.tensors[i]->grad);
            free(registry.tensors[i]->dims);
            free(registry.tensors[i]);
        }
    }
    registry.len = 0;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = a->data[i] + b->data[i];
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){ADD, result, a, b};
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
    
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){MATMUL, result, a, b};
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){EXP, result, a, NULL};
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){LOG, result, a, NULL};
    return result;
}

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
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
            case ADD:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                }
                if (b->requires_grad) {
                    if (!b->grad) b->grad = calloc(b->size, sizeof(float));
                    for (int i = 0; i < b->size; i++) b->grad[i] += result->grad[i];
                }
                break;
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
    tape.len = 0;
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
    // Test 1: Matrix Multiplication
    {
        printf("\nTest 1: 2D Matrix Multiplication\n");
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float data2[] = {0.5f, 0.5f, 0.5f, 0.5f};
        int dims[] = {2, 2};
        
        Tensor *a = tensor_new(2, dims, data1, 1);
        Tensor *b = tensor_new(2, dims, data2, 1);
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

    // Test 2: Addition
    {
        printf("\nTest 2: Element-wise Addition\n");
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float data2[] = {0.5f, 0.5f, 0.5f, 0.5f};
        int dims[] = {2, 2};
        
        Tensor *a = tensor_new(2, dims, data1, 1);
        Tensor *b = tensor_new(2, dims, data2, 1);
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

    // Test 3: Hadamard
    {
        printf("\nTest 3: Hadamard Product\n");
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float data2[] = {0.5f, 1.5f, 2.5f, 3.5f};
        int dims[] = {2, 2};
        
        Tensor *a = tensor_new(2, dims, data1, 1);
        Tensor *b = tensor_new(2, dims, data2, 1);
        Tensor *c = tensor_hadamard(a, b);
        
        for (int i = 0; i < c->size; i++) c->grad[i] = 1.0f;
        
        print_tensor(a, "A");
        print_tensor(b, "B");
        print_tensor(c, "C = A ⊙ B");
        
        printf("Direct multiplication verification: ");
        for (int i = 0; i < 4; i++) printf("%.4f ", data1[i] * data2[i]);
        printf("\n\n");
        
        backward();
        printf("After backward:\n");
        print_tensor(a, "A");
        print_tensor(b, "B");
    }

    clean_registry();
    return 0;
}