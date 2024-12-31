#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000

typedef enum { MATMUL, EXP, LOG } OpType;

typedef struct Tensor {
    float* data;
    float* grad;
    int* dims;
    int ndims;
    int size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor* result;
    Tensor* input1;
    Tensor* input2;
} TapeEntry;

static struct {
    TapeEntry entries[MAX_TAPE];
    int len;
} tape = {0};

int compute_size(int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = compute_size(ndims, dims);
    
    t->data = malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    
    t->requires_grad = requires_grad;
    if (requires_grad) t->grad = calloc(t->size, sizeof(float));
    return t;
}

void tensor_free(Tensor* t) {
    free(t->data);
    free(t->grad);
    free(t->dims);
    free(t);
}

int can_matmul(const Tensor* a, const Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1) return 0;
    
    int a_cols = a->dims[a->ndims - 1];
    int b_rows = b->dims[b->ndims - 2];
    
    return a_cols == b_rows;
}

int* get_broadcast_dims(const Tensor* a, const Tensor* b, int* out_ndims) {
    int max_ndims = (a->ndims > b->ndims) ? a->ndims : b->ndims;
    int* result_dims = malloc(max_ndims * sizeof(int));
    
    const Tensor* larger = (a->ndims > b->ndims) ? a : b;
    memcpy(result_dims, larger->dims, (max_ndims - 2) * sizeof(int));
    
    result_dims[max_ndims - 2] = a->dims[a->ndims - 2];
    result_dims[max_ndims - 1] = b->dims[b->ndims - 1];
    
    *out_ndims = max_ndims;
    return result_dims;
}

static void record_operation(OpType op, Tensor* result, Tensor* input1, Tensor* input2) {
    if (tape.len < MAX_TAPE && result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, input1, input2};
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!can_matmul(a, b)) {
        fprintf(stderr, "Invalid dimensions for matrix multiplication\n");
        return NULL;
    }
    
    int result_ndims;
    int* result_dims = get_broadcast_dims(a, b, &result_ndims);
    Tensor* result = tensor_new(result_ndims, result_dims, NULL, 
                               a->requires_grad || b->requires_grad);
    
    int batch_size = 1;
    for (int i = 0; i < result_ndims - 2; i++) {
        batch_size *= result_dims[i];
    }
    
    int M = a->dims[a->ndims - 2];
    int N = b->dims[b->ndims - 1];
    int K = a->dims[a->ndims - 1];
    
    for (int batch = 0; batch < batch_size; batch++) {
        int batch_offset = batch * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    int a_idx = batch * M * K + i * K + k;
                    int b_idx = batch * K * N + k * N + j;
                    sum += a->data[a_idx] * b->data[b_idx];
                }
                result->data[batch_offset + i * N + j] = sum;
            }
        }
    }
    
    record_operation(MATMUL, result, a, b);
    free(result_dims);
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = expf(fmaxf(fminf(a->data[i], 88.0f), -88.0f));
    }
    record_operation(EXP, result, a, NULL);
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = logf(fmaxf(a->data[i], 1e-7f));
    }
    record_operation(LOG, result, a, NULL);
    return result;
}

void backward() {
    if (!tape.len) return;
    
    Tensor* final = tape.entries[tape.len - 1].result;
    if (!final->grad) final->grad = calloc(final->size, sizeof(float));
    
    int all_zeros = 1;
    for (int i = 0; i < final->size && all_zeros; i++)
        if (final->grad[i] != 0.0f) all_zeros = 0;
    
    if (all_zeros)
        for (int i = 0; i < final->size; i++)
            final->grad[i] = 1.0f;
    
    for (int t = tape.len - 1; t >= 0; t--) {
        TapeEntry* entry = &tape.entries[t];
        Tensor *result = entry->result, *a = entry->input1, *b = entry->input2;
        
        switch (entry->op) {
            case MATMUL: {
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    
                    int M = a->dims[a->ndims - 2];
                    int K = a->dims[a->ndims - 1];
                    int N = b->dims[b->ndims - 1];
                    
                    int batch_size = result->size / (M * N);
                    
                    for (int batch = 0; batch < batch_size; batch++) {
                        for (int i = 0; i < M; i++) {
                            for (int k = 0; k < K; k++) {
                                float sum = 0.0f;
                                for (int j = 0; j < N; j++) {
                                    int grad_idx = batch * M * N + i * N + j;
                                    int b_idx = batch * K * N + k * N + j;
                                    sum += result->grad[grad_idx] * b->data[b_idx];
                                }
                                int a_idx = batch * M * K + i * K + k;
                                a->grad[a_idx] += sum;
                            }
                        }
                    }
                }
                
                if (b->requires_grad) {
                    if (!b->grad) b->grad = calloc(b->size, sizeof(float));
                    
                    int M = a->dims[a->ndims - 2];
                    int K = a->dims[a->ndims - 1];
                    int N = b->dims[b->ndims - 1];
                    
                    int batch_size = result->size / (M * N);
                    
                    for (int batch = 0; batch < batch_size; batch++) {
                        for (int k = 0; k < K; k++) {
                            for (int j = 0; j < N; j++) {
                                float sum = 0.0f;
                                for (int i = 0; i < M; i++) {
                                    int grad_idx = batch * M * N + i * N + j;
                                    int a_idx = batch * M * K + i * K + k;
                                    sum += a->data[a_idx] * result->grad[grad_idx];
                                }
                                int b_idx = batch * K * N + k * N + j;
                                b->grad[b_idx] += sum;
                            }
                        }
                    }
                }
                break;
            }
            
            case EXP:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++) {
                        a->grad[i] += result->grad[i] * result->data[i];
                    }
                }
                break;
                
            case LOG:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++) {
                        a->grad[i] += result->grad[i] / fmaxf(a->data[i], 1e-7f);
                    }
                }
                break;
        }
    }
}

void cleanup_tape() {
    tape.len = 0;
}

void print_tensor(const Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) {
        printf("%d%s", t->dims[i], i < t->ndims - 1 ? "," : "");
    }
    printf("]\n");
    
    printf("Data (first few elements):\n");
    for (int i = 0; i < t->size && i < 10; i++) {
        printf("%8.4f ", t->data[i]);
    }
    if (t->size > 10) printf("...");
    printf("\n");
    
    if (t->grad) {
        printf("Gradients (first few elements):\n");
        for (int i = 0; i < t->size && i < 10; i++) {
            printf("%8.4f ", t->grad[i]);
        }
        if (t->size > 10) printf("...");
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Test 1: Original 2D matrix multiplication with complete computation
    {
        printf("Test 1: 2D Matrix Multiplication\n");
        int dims[] = {2, 2};
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float data2[] = {0.5f, 0.5f, 0.5f, 0.5f};
        
        Tensor* a = tensor_new(2, dims, data1, 1);
        Tensor* b = tensor_new(2, dims, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        Tensor* d = tensor_exp(c);
        Tensor* e = tensor_log(d);
        
        // Set gradient for final tensor
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
        
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(d);
        tensor_free(e);
        cleanup_tape();
    }
    
    // Test 2: 3D tensor multiplication
    {
        printf("\nTest 2: 3D Tensor Multiplication\n");
        int dims1[] = {2, 3, 4};
        int dims2[] = {2, 4, 2};
        
        float* data1 = malloc(24 * sizeof(float));
        float* data2 = malloc(16 * sizeof(float));
        
        for (int i = 0; i < 24; i++) data1[i] = (float)i / 10.0f;
        for (int i = 0; i < 16; i++) data2[i] = (float)i / 20.0f;
        
        Tensor* a = tensor_new(3, dims1, data1, 1);
        Tensor* b = tensor_new(3, dims2, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        Tensor* d = tensor_exp(c);
        
        // Set gradient for final tensor
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
        
        free(data1);
        free(data2);
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(d);
        cleanup_tape();
    }
    
    // Test 3: 4D tensor multiplication
    {
        printf("\nTest 3: 4D Tensor Multiplication\n");
        int dims1[] = {2, 3, 4, 5};
        int dims2[] = {2, 3, 5, 3};
        
        float* data1 = malloc(120 * sizeof(float));
        float* data2 = malloc(90 * sizeof(float));
        
        for (int i = 0; i < 120; i++) data1[i] = (float)i / 100.0f;
        for (int i = 0; i < 90; i++) data2[i] = (float)i / 100.0f;
        
        Tensor* a = tensor_new(4, dims1, data1, 1);
        Tensor* b = tensor_new(4, dims2, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        Tensor* d = tensor_log(c);
        
        // Set gradient for final tensor
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
        
        free(data1);
        free(data2);
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(d);
        cleanup_tape();
    }
    
    return 0;
}