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
    t->ndims = ndims;
    t->size = calc_size(dims, ndims);
    t->dims = malloc(ndims * sizeof(int));
    t->data = malloc(t->size * sizeof(float));
    memcpy(t->dims, dims, ndims * sizeof(int));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    if ((t->requires_grad = requires_grad)) {
        t->grad = calloc(t->size, sizeof(float));
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (t) { free(t->data); free(t->grad); free(t->dims); free(t); }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (op == RELU || op == SIGMOID) {
        Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == RELU ? relu(a->data[i]) : sigmoid(a->data[i]);
        if (result->requires_grad)
            tape.entries[tape.len++] = (TapeEntry){op, result, a, NULL};
        return result;
    }

    int out_dims[MAX_DIMS], out_ndims = op == ADD ? a->ndims : MAX(a->ndims, b->ndims);
    if (op == ADD) {
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else {
        for (int i = 0; i < out_ndims - 2; i++)
            out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[out_ndims-2] = a->dims[a->ndims-2];
        out_dims[out_ndims-1] = b->dims[b->ndims-1];
    }

    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    
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
    if (tape.len == 0) return;
    
    Tensor* final = tape.entries[tape.len - 1].result;
    if (!final->grad) final->grad = calloc(final->size, sizeof(float));
    for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
    
    for (int i = tape.len - 1; i >= 0; i--) {
        TapeEntry* entry = &tape.entries[i];
        Tensor *t = entry->result, *a = entry->input1, *b = entry->input2;
        
        if (entry->op == RELU || entry->op == SIGMOID) {
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
                        for (int k = 0; k < n; k++)
                            for (int j = 0; j < p; j++)
                                a->grad[a_off + i * n + k] += 
                                    t->grad[t_off + i * p + j] * b->data[b_off + k * p + j];
                
                if (b->requires_grad)
                    for (int k = 0; k < n; k++)
                        for (int j = 0; j < p; j++)
                            for (int i = 0; i < m; i++)
                                b->grad[b_off + k * p + j] += 
                                    t->grad[t_off + i * p + j] * a->data[a_off + i * n + k];
            }
        }
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (dims:", name);
    for (int i = 0; i < t->ndims; i++) printf(" %d", t->dims[i]);
    printf("):\n");
    
    for (int i = 0; i < t->dims[0]; i++) {
        for (int j = 0; j < t->dims[1]; j++)
            printf("%f ", t->data[i * t->dims[1] + j]);
        printf("\n");
    }
    
    if (t->grad) {
        printf("Gradients:\n");
        for (int i = 0; i < t->dims[0]; i++) {
            for (int j = 0; j < t->dims[1]; j++)
                printf("%f ", t->grad[i * t->dims[1] + j]);
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    int dims[] = {2, 3, 4, 4};  // [batch, channels, height, width]
    float *w1_data = malloc(96 * sizeof(float));
    float *w2_data = malloc(96 * sizeof(float));
    float *x_data = malloc(96 * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < 96; i++) {
        w1_data[i] = (float)rand() / RAND_MAX * 0.2f;
        w2_data[i] = (float)rand() / RAND_MAX * 0.2f;
        x_data[i] = (float)rand() / RAND_MAX;
    }
    
    // Create tensors and build network
    Tensor *w1 = tensor_new(4, dims, w1_data, 1);
    Tensor *w2 = tensor_new(4, dims, w2_data, 1);
    Tensor *x = tensor_new(4, dims, x_data, 1);
    
    Tensor *y = tensor_sigmoid(
        tensor_matmul(
            tensor_relu(
                tensor_matmul(x, w1)
            ), w2)
        );
    
    backward();
    
    // Print results
    printf("4D Network [%d, %d, %d, %d] - First 4 elements:\n", 
           dims[0], dims[1], dims[2], dims[3]);
    printf("Output: %.4f %.4f %.4f %.4f\n", 
           y->data[0], y->data[1], y->data[2], y->data[3]);
    printf("Gradients: %.4f %.4f %.4f %.4f\n", 
           w1->grad[0], w1->grad[1], w1->grad[2], w1->grad[3]);
    
    // Cleanup
    tensor_free(y);
    free(w1_data); free(w2_data); free(x_data);
    tensor_free(w1); tensor_free(w2); tensor_free(x);
    tape.len = 0;
    return 0;
}