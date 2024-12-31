#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 4
#define MAX_TAPE 1000

typedef enum { ADD, MATMUL, RELU, SIGMOID } OpType;

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

static float sigmoid(float x) {
    x = fmaxf(fminf(x, 88.0f), -88.0f);
    return 1.0f / (1.0f + expf(-x));
}

static float relu(float x) {
    return fmaxf(0.0f, x);
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = 1;
    for (int i = 0; i < ndims; i++) t->size *= dims[i];
    
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

static void record_operation(OpType op, Tensor* result, Tensor* input1, Tensor* input2) {
    if (tape.len < MAX_TAPE && result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, input1, input2};
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < result->size; i++)
        result->data[i] = a->data[i] + b->data[i];
    record_operation(ADD, result, a, b);
    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    int dims[] = {a->dims[0], b->dims[1]};
    Tensor* result = tensor_new(2, dims, NULL, a->requires_grad || b->requires_grad);
    
    for (int i = 0; i < a->dims[0]; i++)
        for (int j = 0; j < b->dims[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->dims[1]; k++)
                sum += a->data[i * a->dims[1] + k] * b->data[k * b->dims[1] + j];
            result->data[i * b->dims[1] + j] = sum;
        }
    
    record_operation(MATMUL, result, a, b);
    return result;
}

Tensor* tensor_relu(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++)
        result->data[i] = relu(a->data[i]);
    record_operation(RELU, result, a, NULL);
    return result;
}

Tensor* tensor_sigmoid(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++)
        result->data[i] = sigmoid(a->data[i]);
    record_operation(SIGMOID, result, a, NULL);
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
            case ADD:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i];
                }
                if (b->requires_grad) {
                    if (!b->grad) b->grad = calloc(b->size, sizeof(float));
                    for (int i = 0; i < b->size; i++)
                        b->grad[i] += result->grad[i];
                }
                break;
                
            case MATMUL:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->dims[0]; i++)
                        for (int j = 0; j < a->dims[1]; j++) {
                            float sum = 0.0f;
                            for (int k = 0; k < b->dims[1]; k++)
                                sum += result->grad[i * b->dims[1] + k] * b->data[j * b->dims[1] + k];
                            a->grad[i * a->dims[1] + j] += sum;
                        }
                }
                if (b->requires_grad) {
                    if (!b->grad) b->grad = calloc(b->size, sizeof(float));
                    for (int i = 0; i < b->dims[0]; i++)
                        for (int j = 0; j < b->dims[1]; j++) {
                            float sum = 0.0f;
                            for (int k = 0; k < a->dims[0]; k++)
                                sum += a->data[k * a->dims[1] + i] * result->grad[k * b->dims[1] + j];
                            b->grad[i * b->dims[1] + j] += sum;
                        }
                }
                break;
                
            case RELU:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] * (a->data[i] > 0.0f);
                }
                break;
                
            case SIGMOID:
                if (a->requires_grad) {
                    if (!a->grad) a->grad = calloc(a->size, sizeof(float));
                    for (int i = 0; i < a->size; i++) {
                        float s = sigmoid(a->data[i]);
                        a->grad[i] += result->grad[i] * s * (1.0f - s);
                    }
                }
                break;
        }
    }
}

void cleanup_tape() {
    tape.len = 0;
}

// Utility function to print tensor
void print_tensor(const Tensor* t, const char* name) {
    if (!t) {
        printf("%s: NULL tensor\n", name);
        return;
    }
    
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) {
        printf("%d%s", t->dims[i], i < t->ndims - 1 ? "," : "");
    }
    printf("]\n");
    
    if (t->ndims == 2) {
        for (int i = 0; i < t->dims[0]; i++) {
            for (int j = 0; j < t->dims[1]; j++) {
                printf("%8.4f ", t->data[i * t->dims[1] + j]);
            }
            printf("\n");
        }
    } else {
        printf("Data: ");
        for (int i = 0; i < t->size && i < 10; i++) {
            printf("%8.4f ", t->data[i]);
        }
        if (t->size > 10) printf("...");
        printf("\n");
    }
    
    if (t->grad) {
        printf("Gradients: ");
        for (int i = 0; i < t->size && i < 10; i++) {
            printf("%8.4f ", t->grad[i]);
        }
        if (t->size > 10) printf("...");
        printf("\n");
    }
    printf("\n");
}

void test_basic_operations() {
    // Create two 2x2 matrices
    int dims[] = {2, 2};
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[] = {0.5f, 0.5f, 0.5f, 0.5f};
    
    Tensor* a = tensor_new(2, dims, data1, 1);
    Tensor* b = tensor_new(2, dims, data2, 1);
    
    printf("Input tensors:\n");
    print_tensor(a, "A");
    print_tensor(b, "B");
    
    // Single matrix multiplication
    Tensor* c = tensor_matmul(a, b);
    
    // Set specific gradients for testing
    c->grad = calloc(c->size, sizeof(float));
    c->grad[0] = 1.0f;  // Only set gradient for first element
    c->grad[1] = 0.0f;
    c->grad[2] = 0.0f;
    c->grad[3] = 0.0f;
    
    printf("\nForward pass result:\n");
    print_tensor(c, "C = A @ B");
    printf("Setting dL/dC = [1, 0; 0, 0] for testing\n\n");
    
    // Test backward pass
    backward();
    
    printf("Analytical gradients:\n");
    printf("dL/dA should be:\n");
    printf("  [0.5, 0.5;   (gradient with respect to first row of A)\n");
    printf("   0.0, 0.0]   (gradient with respect to second row of A)\n\n");
    
    printf("dL/dB should be:\n");
    printf("  [1.0, 0.0;   (from first element of A)\n");
    printf("   2.0, 0.0]   (from second element of A)\n\n");
    
    printf("Computed gradients:\n");
    print_tensor(a, "dL/dA");
    print_tensor(b, "dL/dB");
    
    // Cleanup
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    cleanup_tape();
}

void test_chain_rule() {
    // Same setup as before
    int dims[] = {2, 2};
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};  // Matrix A
    float data2[] = {0.5f, 0.5f, 0.5f, 0.5f};  // Matrix B
    
    Tensor* a = tensor_new(2, dims, data1, 1);
    Tensor* b = tensor_new(2, dims, data2, 1);
    
    printf("Chain Rule Test: E = Sigmoid(ReLU(A @ B))\n");
    printf("Starting with dL/dE[0,0] = 1.0, others = 0\n\n");
    
    // Forward pass
    Tensor* c = tensor_matmul(a, b);        // C = A @ B
    Tensor* d = tensor_relu(c);             // D = ReLU(C)
    Tensor* e = tensor_sigmoid(d);          // E = Sigmoid(D)
    
    printf("Forward pass values:\n");
    printf("A = [1 2; 3 4]\n");
    printf("B = [0.5 0.5; 0.5 0.5]\n");
    
    // C = A @ B = [1.5 1.5; 3.5 3.5]
    printf("\nC = A @ B = [1.5 1.5; 3.5 3.5]\n");
    
    // D = ReLU(C) = [1.5 1.5; 3.5 3.5] (all positive)
    printf("D = ReLU(C) = [1.5 1.5; 3.5 3.5]\n");
    
    // E = Sigmoid(D)
    printf("E = Sigmoid(D)\n");
    
    // Set gradient at output
    e->grad = calloc(e->size, sizeof(float));
    e->grad[0] = 1.0f;  // Only first element
    
    printf("\nAnalytical gradient computation:\n");
    printf("1. dL/dE[0,0] = 1.0 (given)\n");
    
    printf("2. dL/dD = dL/dE * dSigmoid(D)\n");
    printf("   dSigmoid(x) = sigmoid(x) * (1 - sigmoid(x))\n");
    
    printf("3. dL/dC = dL/dD * dReLU(C)\n");
    printf("   dReLU(x) = 1 if x > 0, 0 otherwise\n");
    
    printf("4. dL/dA = dL/dC @ B^T\n");
    printf("   dL/dB = A^T @ dL/dC\n");
    
    backward();
    
    printf("\nComputed gradients:\n");
    print_tensor(a, "dL/dA");
    print_tensor(b, "dL/dB");
    print_tensor(c, "dL/dC");
    print_tensor(d, "dL/dD");
    print_tensor(e, "dL/dE");
    
    // Cleanup
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);
    cleanup_tape();
}

int main() {
    test_basic_operations();
    test_chain_rule();
    return 0;
}