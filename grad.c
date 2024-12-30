#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { ADD, MATMUL, RELU, SIGMOID } OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape = {0};

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

static float relu(float x) {
    return x > 0 ? x : 0;
}

static float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

static inline int calc_size(const int* dims, int ndims) {
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
        int batch_dims = MIN(a->ndims-2, b->ndims-2);
        for (int i = 0; i < batch_dims; i++)
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
        int batch_dims = *out_ndims - 2;
        for (int i = 0; i < batch_dims; i++)
            out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[*out_ndims-2] = a->dims[a->ndims-2];
        out_dims[*out_ndims-1] = b->dims[b->ndims-1];
    }
}

static void matmul_forward(const float* a, const float* b, float* out, int m, int n, int p) {
    memset(out, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            float aik = a[i * n + k];
            for (int j = 0; j < p; j++)
                out[i * p + j] += aik * b[k * p + j];
        }
}

static Tensor* tensor_unary_op(Tensor* a, OpType op) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    if (!result) return NULL;
    
    for (int i = 0; i < result->size; i++) {
        if (op == RELU)
            result->data[i] = relu(a->data[i]);
        else if (op == SIGMOID)
            result->data[i] = sigmoid(a->data[i]);
    }
    
    if (result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, a, NULL};
    
    return result;
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (op == RELU || op == SIGMOID)
        return tensor_unary_op(a, op);
        
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
        
        for (int batch = 0; batch < batch_size; batch++)
            matmul_forward(&a->data[batch * m * n], &b->data[batch * n * p], 
                          &result->data[batch * m * p], m, n, p);
    }

    if (result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, a, b};
    
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)
#define tensor_relu(a) tensor_op(a, NULL, RELU)
#define tensor_sigmoid(a) tensor_op(a, NULL, SIGMOID)

static void backward_op(TapeEntry* entry) {
    Tensor *t = entry->result, *a = entry->input1, *b = entry->input2;
    
    if (entry->op == RELU) {
        if (a->requires_grad)
            for (int i = 0; i < a->size; i++)
                a->grad[i] += t->grad[i] * relu_derivative(a->data[i]);
    }
    else if (entry->op == SIGMOID) {
        if (a->requires_grad)
            for (int i = 0; i < a->size; i++)
                a->grad[i] += t->grad[i] * sigmoid_derivative(a->data[i]);
    }
    else if (entry->op == ADD) {
        if (a->requires_grad)
            for (int i = 0; i < a->size; i++) 
                a->grad[i] += t->grad[i];
        if (b->requires_grad)
            for (int i = 0; i < b->size; i++) 
                b->grad[i] += t->grad[i];
    } else if (entry->op == MATMUL) {
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

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.entries[tape.len - 1].result;
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++) 
            final->grad[i] = 1.0f;
        
        for (int i = tape.len - 1; i >= 0; i--)
            backward_op(&tape.entries[i]);
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (dims:", name);
    for (int i = 0; i < t->ndims; i++) 
        printf(" %d", t->dims[i]);
    printf("):\n");
    
    int rows = t->dims[0], cols = t->dims[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
            printf("%f ", t->data[i * cols + j]);
        printf("\n");
    }
    
    if (t->grad) {
        printf("Gradients:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) 
                printf("%f ", t->grad[i * cols + j]);
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    int dims[] = {2, 2, 2};
    float w1_data[] = {1.0f, 0.5f, 0.5f, 1.0f, 0.8f, 0.2f, 0.3f, 0.7f};
    float w2_data[] = {0.5f, 1.0f, 1.0f, 0.5f, 0.4f, 0.6f, 0.9f, 0.1f};
    float x_data[] = {1.0f, 2.0f, 0.5f, 1.5f, 0.7f, 1.3f, 1.8f, 0.4f};
    
    Tensor *w1 = tensor_new(3, dims, w1_data, 1);
    Tensor *w2 = tensor_new(3, dims, w2_data, 1);
    Tensor *x = tensor_new(3, dims, x_data, 1);
    
    // Forward pass with activations
    Tensor *h1 = tensor_matmul(x, w1);
    Tensor *h1_act = tensor_relu(h1);
    Tensor *h2 = tensor_matmul(h1_act, w2);
    Tensor *y = tensor_sigmoid(h2);
    
    backward();
    
    printf("Neural network computation with ReLU and Sigmoid activations:\n\n");
    print_tensor(x, "Input (x)");
    print_tensor(w1, "First layer weights (w1)");
    print_tensor(h1, "Pre-activation hidden layer (h1)");
    print_tensor(h1_act, "ReLU activation (h1_act)");
    print_tensor(w2, "Second layer weights (w2)");
    print_tensor(h2, "Pre-activation output (h2)");
    print_tensor(y, "Sigmoid output (y)");
    
    tensor_free(y);
    tensor_free(h2);
    tensor_free(h1_act);
    tensor_free(h1);
    tensor_free(w2);
    tensor_free(w1);
    tensor_free(x);
    
    tape.len = 0;
    return 0;
}