#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { ADD, MATMUL, NONE } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims;
    int ndims, size;
    int requires_grad;
    struct Tensor *children[2];
    OpType op;
} Tensor;

typedef struct {
    Tensor* ops[1000];
    int len;
} Tape;

static Tape tape = {0};

static inline int calc_size(const int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    if (!t->dims) { free(t); return NULL; }
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = calc_size(dims, ndims);
    t->data = malloc(t->size * sizeof(float));
    if (!t->data) { free(t->dims); free(t); return NULL; }
    
    if (data) {
        memcpy(t->data, data, t->size * sizeof(float));
    } else {
        memset(t->data, 0, t->size * sizeof(float));
    }
    
    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = calloc(t->size, sizeof(float));
        if (!t->grad) {
            free(t->data);
            free(t->dims);
            free(t);
            return NULL;
        }
    }
    
    if (requires_grad || data) tape.ops[tape.len++] = t;
    return t;
}

static void matmul_forward(const float* __restrict__ a, 
                          const float* __restrict__ b, 
                          float* __restrict__ out,
                          const int m, const int n, const int p) {
    memset(out, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            const float aik = a[i * n + k];
            for (int j = 0; j < p; j++) {
                out[i * p + j] += aik * b[k * p + j];
            }
        }
    }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!a || !b || a->ndims != b->ndims) return NULL;
    
    int out_dims[2];
    if (op == MATMUL) {
        if (a->dims[1] != b->dims[0]) return NULL;
        out_dims[0] = a->dims[0];
        out_dims[1] = b->dims[1];
    } else {
        if (a->dims[0] != b->dims[0] || a->dims[1] != b->dims[1]) return NULL;
        out_dims[0] = a->dims[0];
        out_dims[1] = a->dims[1];
    }
    
    Tensor* result = tensor_new(2, out_dims, NULL, 1);
    if (!result) return NULL;
    
    result->op = op;
    result->children[0] = a;
    result->children[1] = b;
    
    if (op == ADD) {
        for (int i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    } else {
        matmul_forward(a->data, b->data, result->data, 
                      a->dims[0], a->dims[1], b->dims[1]);
    }
    
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)

static void backward_op(Tensor* t) {
    if (!t || t->op == NONE) return;
    
    Tensor *a = t->children[0], *b = t->children[1];
    if (!a || !b) return;
    
    if (t->op == ADD) {
        if (a->requires_grad) {
            for (int i = 0; i < a->size; i++) {
                a->grad[i] += t->grad[i];
            }
        }
        if (b->requires_grad) {
            for (int i = 0; i < b->size; i++) {
                b->grad[i] += t->grad[i];
            }
        }
    } else if (t->op == MATMUL) {
        const int m = a->dims[0], n = a->dims[1], p = b->dims[1];
        
        if (a->requires_grad) {
            for (int i = 0; i < m; i++) {
                for (int k = 0; k < n; k++) {
                    float sum = 0;
                    for (int j = 0; j < p; j++) {
                        sum += t->grad[i * p + j] * b->data[k * p + j];
                    }
                    a->grad[i * n + k] += sum;
                }
            }
        }
        
        if (b->requires_grad) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0;
                    for (int i = 0; i < m; i++) {
                        sum += t->grad[i * p + j] * a->data[i * n + k];
                    }
                    b->grad[k * p + j] += sum;
                }
            }
        }
    }
}

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.ops[tape.len - 1];
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++) {
            final->grad[i] = 1.0f;
        }
        for (int i = tape.len - 1; i >= 0; i--) {
            backward_op(tape.ops[i]);
        }
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < t->dims[0]; i++) {
        for (int j = 0; j < t->dims[1]; j++) {
            printf("%f ", t->data[i * t->dims[1] + j]);
        }
        printf("\n");
    }
    if (t->grad) {
        printf("Gradients:\n");
        for (int i = 0; i < t->dims[0]; i++) {
            for (int j = 0; j < t->dims[1]; j++) {
                printf("%f ", t->grad[i * t->dims[1] + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    // Example: Neural network-like computation
    int dims[] = {2, 2};
    float w1_data[] = {1.0f, 0.5f, 0.5f, 1.0f};
    float w2_data[] = {0.5f, 1.0f, 1.0f, 0.5f};
    float x_data[] = {1.0f, 2.0f, 0.5f, 1.5f};
    
    // Create tensors
    Tensor* w1 = tensor_new(2, dims, w1_data, 1);  // First weight matrix
    Tensor* w2 = tensor_new(2, dims, w2_data, 1);  // Second weight matrix
    Tensor* x = tensor_new(2, dims, x_data, 1);    // Input matrix
    
    // Forward pass: x -> w1 -> w2
    Tensor* h = tensor_matmul(x, w1);    // First layer
    Tensor* y = tensor_matmul(h, w2);    // Second layer
    
    // Compute gradients
    backward();
    
    // Print results
    printf("Neural network computation example:\n\n");
    print_tensor(x, "Input (x)");
    print_tensor(w1, "First layer weights (w1)");
    print_tensor(w2, "Second layer weights (w2)");
    print_tensor(h, "Hidden layer (h)");
    print_tensor(y, "Output (y)");
    
    return 0;
}