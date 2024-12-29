#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

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

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->dims);
    free(t);
}

static int compatible_shapes(const Tensor* a, const Tensor* b, OpType op) {
    if (op == ADD) {
        if (a->ndims != b->ndims) return 0;
        for (int i = 0; i < a->ndims; i++) {
            if (a->dims[i] != b->dims[i]) return 0;
        }
        return 1;
    } else if (op == MATMUL) {
        if (a->ndims < 2 || b->ndims < 2) return 0;
        if (a->dims[a->ndims-1] != b->dims[b->ndims-2]) return 0;
        
        int batch_dims = MIN(a->ndims-2, b->ndims-2);
        for (int i = 0; i < batch_dims; i++) {
            if (a->dims[i] != b->dims[i] && 
                a->dims[i] != 1 && b->dims[i] != 1) return 0;
        }
        return 1;
    }
    return 0;
}

static void get_output_dims(const Tensor* a, const Tensor* b, 
                          OpType op, int* out_dims, int* out_ndims) {
    if (op == ADD) {
        *out_ndims = a->ndims;
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else if (op == MATMUL) {
        *out_ndims = MAX(a->ndims, b->ndims);
        
        int batch_dims = *out_ndims - 2;
        for (int i = 0; i < batch_dims; i++) {
            out_dims[i] = MAX(a->dims[i], b->dims[i]);
        }
        
        out_dims[*out_ndims-2] = a->dims[a->ndims-2];
        out_dims[*out_ndims-1] = b->dims[b->ndims-1];
    }
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

static void generalized_matmul(const Tensor* a, const Tensor* b, Tensor* out) {
    int batch_size = 1;
    int batch_dims = out->ndims - 2;
    for (int i = 0; i < batch_dims; i++) {
        batch_size *= out->dims[i];
    }
    
    int m = a->dims[a->ndims-2];
    int n = a->dims[a->ndims-1];
    int p = b->dims[b->ndims-1];
    
    for (int batch = 0; batch < batch_size; batch++) {
        int batch_offset = batch * m * p;
        const float* a_batch = &a->data[batch * m * n];
        const float* b_batch = &b->data[batch * n * p];
        float* out_batch = &out->data[batch_offset];
        
        matmul_forward(a_batch, b_batch, out_batch, m, n, p);
    }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!compatible_shapes(a, b, op)) return NULL;
    
    int out_dims[MAX_DIMS];
    int out_ndims;
    get_output_dims(a, b, op, out_dims, &out_ndims);
    
    Tensor* result = tensor_new(out_ndims, out_dims, NULL, 1);
    if (!result) return NULL;
    
    result->op = op;
    result->children[0] = a;
    result->children[1] = b;
    
    if (op == ADD) {
        for (int i = 0; i < result->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    } else if (op == MATMUL) {
        generalized_matmul(a, b, result);
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
        int batch_size = 1;
        int batch_dims = t->ndims - 2;
        for (int i = 0; i < batch_dims; i++) {
            batch_size *= t->dims[i];
        }
        
        int m = a->dims[a->ndims-2];
        int n = a->dims[a->ndims-1];
        int p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++) {
            int a_offset = batch * m * n;
            int b_offset = batch * n * p;
            int t_offset = batch * m * p;
            
            if (a->requires_grad) {
                for (int i = 0; i < m; i++) {
                    for (int k = 0; k < n; k++) {
                        float sum = 0;
                        for (int j = 0; j < p; j++) {
                            sum += t->grad[t_offset + i * p + j] * 
                                  b->data[b_offset + k * p + j];
                        }
                        a->grad[a_offset + i * n + k] += sum;
                    }
                }
            }
            
            if (b->requires_grad) {
                for (int k = 0; k < n; k++) {
                    for (int j = 0; j < p; j++) {
                        float sum = 0;
                        for (int i = 0; i < m; i++) {
                            sum += t->grad[t_offset + i * p + j] * 
                                  a->data[a_offset + i * n + k];
                        }
                        b->grad[b_offset + k * p + j] += sum;
                    }
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

void tape_clear() {
    tape.len = 0;
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (dims:", name);
    for (int i = 0; i < t->ndims; i++) {
        printf(" %d", t->dims[i]);
    }
    printf("):\n");
    
    // For simplicity, we'll just print the first two dimensions fully
    int rows = t->dims[0];
    int cols = t->dims[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", t->data[i * cols + j]);
        }
        printf("\n");
    }
    
    if (t->grad) {
        printf("Gradients:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%f ", t->grad[i * cols + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    // Example with 3D tensors (batch_size=2, rows=2, cols=2)
    int dims[] = {2, 2, 2};
    float w1_data[] = {1.0f, 0.5f, 0.5f, 1.0f, 0.8f, 0.2f, 0.3f, 0.7f};
    float w2_data[] = {0.5f, 1.0f, 1.0f, 0.5f, 0.4f, 0.6f, 0.9f, 0.1f};
    float x_data[] = {1.0f, 2.0f, 0.5f, 1.5f, 0.7f, 1.3f, 1.8f, 0.4f};
    
    Tensor* w1 = tensor_new(3, dims, w1_data, 1);
    Tensor* w2 = tensor_new(3, dims, w2_data, 1);
    Tensor* x = tensor_new(3, dims, x_data, 1);
    
    Tensor* h = tensor_matmul(x, w1);
    Tensor* y = tensor_matmul(h, w2);
    
    backward();
    
    printf("Neural network computation example with batched 3D tensors:\n\n");
    print_tensor(x, "Input (x)");
    print_tensor(w1, "First layer weights (w1)");
    print_tensor(w2, "Second layer weights (w2)");
    print_tensor(h, "Hidden layer (h)");
    print_tensor(y, "Output (y)");
    
    tensor_free(y);
    tensor_free(h);
    tensor_free(w2);
    tensor_free(w1);
    tensor_free(x);
    
    tape_clear();
    return 0;
}