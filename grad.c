#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { ADD, MATMUL, RELU, SIGMOID, RESHAPE } OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape;

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float sigmoid_derivative(float x) { float s = sigmoid(x); return s * (1 - s); }
static float relu(float x) { return x > 0 ? x : 0; }
static float relu_derivative(float x) { return x > 0 ? 1 : 0; }

static int calc_size(const int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndims = ndims;
    t->size = calc_size(dims, ndims);
    t->dims = malloc(ndims * sizeof(int));
    t->data = malloc(t->size * sizeof(float));
    
    if (!t->dims || !t->data) {
        free(t->dims); free(t->data); free(t);
        return NULL;
    }
    
    memcpy(t->dims, dims, ndims * sizeof(int));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    
    if ((t->requires_grad = requires_grad)) {
        if (!(t->grad = calloc(t->size, sizeof(float)))) {
            free(t->data); free(t->dims); free(t);
            return NULL;
        }
    }
    return t;
}

Tensor* tensor_reshape(Tensor* t, int new_ndims, int* new_dims) {
    int new_size = calc_size(new_dims, new_ndims);
    if (new_size != t->size) return NULL;
    
    Tensor* result = tensor_new(new_ndims, new_dims, NULL, t->requires_grad);
    if (!result) return NULL;
    
    // Copy data
    memcpy(result->data, t->data, t->size * sizeof(float));
    
    if (result->requires_grad) {
        if (!result->grad) {
            result->grad = calloc(result->size, sizeof(float));
        }
        tape.entries[tape.len++] = (TapeEntry){RESHAPE, result, t, NULL};
    }
    
    return result;
}

void tensor_free(Tensor* t) {
    if (t) { free(t->data); free(t->grad); free(t->dims); free(t); }
}

static int compatible_shapes(const Tensor* a, const Tensor* b, OpType op) {
    if (op == ADD) {
        if (a->ndims != b->ndims) return 0;
        for (int i = 0; i < a->ndims; i++)
            if (a->dims[i] != b->dims[i]) return 0;
        return 1;
    }
    if (op == MATMUL) {
        if (a->ndims < 2 || b->ndims < 2 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) 
            return 0;
        for (int i = 0; i < MIN(a->ndims-2, b->ndims-2); i++)
            if (a->dims[i] != b->dims[i] && a->dims[i] != 1 && b->dims[i] != 1) 
                return 0;
        return 1;
    }
    return 0;
}

static void get_output_dims(const Tensor* a, const Tensor* b, OpType op, int* out_dims, int* out_ndims) {
    *out_ndims = op == ADD ? a->ndims : MAX(a->ndims, b->ndims);
    if (op == ADD) {
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else {
        for (int i = 0; i < *out_ndims - 2; i++)
            out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[*out_ndims-2] = a->dims[a->ndims-2];
        out_dims[*out_ndims-1] = b->dims[b->ndims-1];
    }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (op == RELU || op == SIGMOID) {
        Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
        if (!result) return NULL;
        
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == RELU ? relu(a->data[i]) : sigmoid(a->data[i]);
        
        if (result->requires_grad)
            tape.entries[tape.len++] = (TapeEntry){op, result, a, NULL};
        return result;
    }
    
    if (!compatible_shapes(a, b, op)) return NULL;
    
    int out_dims[MAX_DIMS], out_ndims;
    get_output_dims(a, b, op, out_dims, &out_ndims);
    
    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    if (!result) return NULL;
    
    if (op == ADD) {
        for (int i = 0; i < result->size; i++)
            result->data[i] = a->data[i] + b->data[i];
    } else if (op == MATMUL) {
        int batch_size = calc_size(out_dims, out_ndims - 2);
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++) {
            float *out = &result->data[batch * m * p];
            const float *a_data = &a->data[batch * m * n];
            const float *b_data = &b->data[batch * n * p];
            
            memset(out, 0, m * p * sizeof(float));
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++) {
                    float aik = a_data[i * n + k];
                    for (int j = 0; j < p; j++)
                        out[i * p + j] += aik * b_data[k * p + j];
                }
        }
    }
    
    if (result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, a, b};
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)
#define tensor_relu(a) tensor_op(a, NULL, RELU)
#define tensor_sigmoid(a) tensor_op(a, NULL, SIGMOID)

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.entries[tape.len - 1].result;
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
        
        for (int i = tape.len - 1; i >= 0; i--) {
            TapeEntry* entry = &tape.entries[i];
            Tensor *t = entry->result, *a = entry->input1, *b = entry->input2;
            
            if (entry->op == RESHAPE) {
                if (a->requires_grad) {
                    for (int j = 0; j < a->size; j++) {
                        a->grad[j] += t->grad[j];
                    }
                }
            }
            else if (entry->op == RELU || entry->op == SIGMOID) {
                if (a->requires_grad)
                    for (int j = 0; j < a->size; j++)
                        a->grad[j] += t->grad[j] * 
                            (entry->op == RELU ? relu_derivative(a->data[j]) : 
                             sigmoid_derivative(a->data[j]));
            }
            else if (entry->op == ADD) {
                if (a->requires_grad)
                    for (int j = 0; j < a->size; j++) a->grad[j] += t->grad[j];
                if (b->requires_grad)
                    for (int j = 0; j < b->size; j++) b->grad[j] += t->grad[j];
            }
            else if (entry->op == MATMUL) {
                int batch_size = calc_size(t->dims, t->ndims - 2);
                int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
                
                for (int batch = 0; batch < batch_size; batch++) {
                    int a_off = batch * m * n, b_off = batch * n * p, t_off = batch * m * p;
                    
                    if (a->requires_grad)
                        for (int i = 0; i < m; i++)
                            for (int k = 0; k < n; k++) {
                                float sum = 0;
                                for (int j = 0; j < p; j++)
                                    sum += t->grad[t_off + i * p + j] * b->data[b_off + k * p + j];
                                a->grad[a_off + i * n + k] += sum;
                            }
                    
                    if (b->requires_grad)
                        for (int k = 0; k < n; k++)
                            for (int j = 0; j < p; j++) {
                                float sum = 0;
                                for (int i = 0; i < m; i++)
                                    sum += t->grad[t_off + i * p + j] * a->data[a_off + i * n + k];
                                b->grad[b_off + k * p + j] += sum;
                            }
                }
            }
        }
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (dims:", name);
    for (int i = 0; i < t->ndims; i++) printf(" %d", t->dims[i]);
    printf("):\n");
    
    if (t->ndims == 2) {
        int rows = t->dims[0], cols = t->dims[1];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) printf("%f ", t->data[i * cols + j]);
            printf("\n");
        }
        
        if (t->grad) {
            printf("Gradients:\n");
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) printf("%f ", t->grad[i * cols + j]);
                printf("\n");
            }
        }
    } else {
        printf("Data: ");
        for (int i = 0; i < t->size; i++) printf("%f ", t->data[i]);
        printf("\n");
        if (t->grad) {
            printf("Gradients: ");
            for (int i = 0; i < t->size; i++) printf("%f ", t->grad[i]);
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    // Define dimensions for weights and input
    int input_dims[] = {2, 2, 2};  // Original input shape
    int w1_dims[] = {2, 2};        // Weight matrices should be 2D
    int w2_dims[] = {2, 2};
    
    float w1_data[] = {1.0f, 0.5f, 0.5f, 1.0f};
    float w2_data[] = {0.5f, 1.0f, 1.0f, 0.5f};
    float x_data[] = {1.0f, 2.0f, 0.5f, 1.5f, 0.7f, 1.3f, 1.8f, 0.4f};
    
    // Create tensors with correct dimensions
    Tensor *x = tensor_new(3, input_dims, x_data, 1);
    Tensor *w1 = tensor_new(2, w1_dims, w1_data, 1);
    Tensor *w2 = tensor_new(2, w2_dims, w2_data, 1);
    
    // Reshape x from (2,2,2) to (4,2)
    int new_dims[] = {4, 2};
    Tensor *x_reshaped = tensor_reshape(x, 2, new_dims);
    
    if (!x_reshaped) {
        printf("Reshape operation failed\n");
        return 1;
    }
    
    // Neural network operations
    Tensor *h1 = tensor_matmul(x_reshaped, w1);
    if (!h1) {
        printf("First matrix multiplication failed\n");
        return 1;
    }
    
    Tensor *h1_act = tensor_relu(h1);
    if (!h1_act) {
        printf("ReLU activation failed\n");
        return 1;
    }
    
    Tensor *h2 = tensor_matmul(h1_act, w2);
    if (!h2) {
        printf("Second matrix multiplication failed\n");
        return 1;
    }
    
    Tensor *y = tensor_sigmoid(h2);
    if (!y) {
        printf("Sigmoid activation failed\n");
        return 1;
    }
    
    backward();
    
    printf("Neural network computation with reshape and activations:\n\n");
    print_tensor(x, "Original Input (x)");
    print_tensor(x_reshaped, "Reshaped Input (x_reshaped)");
    print_tensor(w1, "First layer weights (w1)");
    print_tensor(h1, "Pre-activation hidden layer (h1)");
    print_tensor(h1_act, "ReLU activation (h1_act)");
    print_tensor(w2, "Second layer weights (w2)");
    print_tensor(h2, "Pre-activation output (h2)");
    print_tensor(y, "Sigmoid output (y)");
    
    // Cleanup
    tensor_free(y);
    tensor_free(h2);
    tensor_free(h1_act);
    tensor_free(h1);
    tensor_free(x_reshaped);
    tensor_free(w2);
    tensor_free(w1);
    tensor_free(x);
    
    tape.len = 0;
    return 0;
}