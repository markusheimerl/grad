#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations
struct Tensor;
struct Operator;

typedef struct {
    int rows;
    int cols;
} Shape;

// Tensor structure
typedef struct Tensor {
    float* data;
    float* grad;
    Shape shape;
    int requires_grad;
    struct Operator* grad_fn;  // Points to the operator that created this tensor
} Tensor;

// Operator structure
typedef struct Operator {
    char* name;
    Tensor* (*forward)(Tensor**, int);  // Takes array of input tensors and count
    void (*backward)(Tensor*, Tensor**);  // Output grad and input tensors
    Tensor** inputs;
    int num_inputs;
} Operator;

// Utility functions
Tensor* create_tensor(int rows, int cols, int requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->shape.rows = rows;
    t->shape.cols = cols;
    t->data = (float*)calloc(rows * cols, sizeof(float));
    t->grad = requires_grad ? (float*)calloc(rows * cols, sizeof(float)) : NULL;
    t->requires_grad = requires_grad;
    t->grad_fn = NULL;
    return t;
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (%dx%d):\n", name, t->shape.rows, t->shape.cols);
    for (int i = 0; i < t->shape.rows; i++) {
        for (int j = 0; j < t->shape.cols; j++) {
            printf("%.2f ", t->data[i * t->shape.cols + j]);
        }
        printf("\n");
    }
    if (t->grad) {
        printf("Gradients:\n");
        for (int i = 0; i < t->shape.rows; i++) {
            for (int j = 0; j < t->shape.cols; j++) {
                printf("%.2f ", t->grad[i * t->shape.cols + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

// Matrix multiplication forward
Tensor* matmul_forward(Tensor** inputs, int num_inputs) {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];
    
    if (a->shape.cols != b->shape.rows) {
        printf("Invalid shapes for matrix multiplication!\n");
        return NULL;
    }

    Tensor* out = create_tensor(a->shape.rows, b->shape.cols, 1);
    
    for (int i = 0; i < a->shape.rows; i++) {
        for (int j = 0; j < b->shape.cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->shape.cols; k++) {
                sum += a->data[i * a->shape.cols + k] * b->data[k * b->shape.cols + j];
            }
            out->data[i * out->shape.cols + j] = sum;
        }
    }
    return out;
}

// Matrix multiplication backward
void matmul_backward(Tensor* grad_output, Tensor** inputs) {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];

    // Gradient for first input: grad_output @ b.T
    if (a->requires_grad) {
        for (int i = 0; i < a->shape.rows; i++) {
            for (int k = 0; k < a->shape.cols; k++) {
                float sum = 0;
                for (int j = 0; j < b->shape.cols; j++) {
                    sum += grad_output->data[i * grad_output->shape.cols + j] * 
                           b->data[k * b->shape.cols + j];
                }
                a->grad[i * a->shape.cols + k] += sum;
            }
        }
    }

    // Gradient for second input: a.T @ grad_output
    if (b->requires_grad) {
        for (int k = 0; k < b->shape.rows; k++) {
            for (int j = 0; j < b->shape.cols; j++) {
                float sum = 0;
                for (int i = 0; i < a->shape.rows; i++) {
                    sum += a->data[i * a->shape.cols + k] * 
                           grad_output->data[i * grad_output->shape.cols + j];
                }
                b->grad[k * b->shape.cols + j] += sum;
            }
        }
    }
}

// Addition forward
Tensor* add_forward(Tensor** inputs, int num_inputs) {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];
    
    if (a->shape.rows != b->shape.rows || a->shape.cols != b->shape.cols) {
        printf("Invalid shapes for addition!\n");
        return NULL;
    }

    Tensor* out = create_tensor(a->shape.rows, a->shape.cols, 1);
    
    for (int i = 0; i < a->shape.rows * a->shape.cols; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

// Addition backward
void add_backward(Tensor* grad_output, Tensor** inputs) {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];

    if (a->requires_grad) {
        for (int i = 0; i < a->shape.rows * a->shape.cols; i++) {
            a->grad[i] += grad_output->data[i];
        }
    }

    if (b->requires_grad) {
        for (int i = 0; i < b->shape.rows * b->shape.cols; i++) {
            b->grad[i] += grad_output->data[i];
        }
    }
}

// Create operator
Operator* create_operator(const char* name, 
                         Tensor* (*forward)(Tensor**, int),
                         void (*backward)(Tensor*, Tensor**),
                         Tensor** inputs,
                         int num_inputs) {
    Operator* op = (Operator*)malloc(sizeof(Operator));
    op->name = strdup(name);
    op->forward = forward;
    op->backward = backward;
    op->inputs = inputs;
    op->num_inputs = num_inputs;
    return op;
}

int main() {
    // Create input tensors
    Tensor* a = create_tensor(2, 3, 1);  // 2x3 matrix
    Tensor* b = create_tensor(3, 2, 1);  // 3x2 matrix
    Tensor* c = create_tensor(2, 2, 1);  // 2x2 matrix

    // Initialize some values
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 2, 3, 4, 5, 6};
    float c_data[] = {1, 1, 1, 1};
    
    memcpy(a->data, a_data, sizeof(float) * 6);
    memcpy(b->data, b_data, sizeof(float) * 6);
    memcpy(c->data, c_data, sizeof(float) * 4);

    // Create operators
    Tensor* inputs1[] = {a, b};
    Operator* matmul_op = create_operator("matmul", matmul_forward, matmul_backward, inputs1, 2);
    
    // Forward pass
    Tensor* d = matmul_op->forward(matmul_op->inputs, 2);
    d->grad_fn = matmul_op;
    
    Tensor* inputs2[] = {d, c};
    Operator* add_op = create_operator("add", add_forward, add_backward, inputs2, 2);
    
    Tensor* output = add_op->forward(add_op->inputs, 2);
    output->grad_fn = add_op;

    // Print initial tensors and result
    print_tensor(a, "A");
    print_tensor(b, "B");
    print_tensor(c, "C");
    print_tensor(d, "D (A @ B)");
    print_tensor(output, "Output (D + C)");

    // Backward pass
    // Initialize output gradient with ones
    for (int i = 0; i < output->shape.rows * output->shape.cols; i++) {
        output->grad[i] = 1.0;
    }

    // Backward pass through the graph
    add_op->backward(output, add_op->inputs);
    matmul_op->backward(d, matmul_op->inputs);

    // Print gradients
    printf("After backward pass:\n");
    print_tensor(a, "A");
    print_tensor(b, "B");
    print_tensor(c, "C");

    return 0;
}
