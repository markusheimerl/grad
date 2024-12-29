#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    int rows;
    int cols;
} Shape;

struct Operator;

typedef struct Tensor {
    float* data;
    float* grad;
    Shape shape;
    bool requires_grad;
    struct Operator* grad_fn;
} Tensor;

typedef struct Operator {
    char* name;
    Tensor* (*forward)(Tensor**);
    void (*backward)(Tensor*, Tensor**);
    Tensor** inputs;
    int num_inputs;
} Operator;

Tensor* create_tensor(int rows, int cols, bool requires_grad) {
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

Tensor* matmul_forward(Tensor** inputs) {
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

void matmul_backward(Tensor* grad_output, Tensor** inputs) {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];

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

Tensor* add_forward(Tensor** inputs) {
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

Operator* create_operator(const char* name, 
                         Tensor* (*forward)(Tensor**),
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
    Tensor* a = create_tensor(2, 3, true);
    Tensor* b = create_tensor(3, 2, true);
    Tensor* c = create_tensor(2, 2, true);

    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 2, 3, 4, 5, 6};
    float c_data[] = {1, 1, 1, 1};
    
    memcpy(a->data, a_data, sizeof(float) * 6);
    memcpy(b->data, b_data, sizeof(float) * 6);
    memcpy(c->data, c_data, sizeof(float) * 4);

    Tensor* inputs1[] = {a, b};
    Operator* matmul_op = create_operator("matmul", matmul_forward, matmul_backward, inputs1, 2);
    
    Tensor* d = matmul_op->forward(matmul_op->inputs);
    d->grad_fn = matmul_op;
    
    Tensor* inputs2[] = {d, c};
    Operator* add_op = create_operator("add", add_forward, add_backward, inputs2, 2);
    
    Tensor* output = add_op->forward(add_op->inputs);
    output->grad_fn = add_op;

    print_tensor(a, "A");
    print_tensor(b, "B");
    print_tensor(c, "C");
    print_tensor(d, "D (A @ B)");
    print_tensor(output, "Output (D + C)");

    for (int i = 0; i < output->shape.rows * output->shape.cols; i++) {
        output->grad[i] = 1.0;
    }

    add_op->backward(output, add_op->inputs);
    matmul_op->backward(d, matmul_op->inputs);

    printf("After backward pass:\n");
    print_tensor(a, "A");
    print_tensor(b, "B");
    print_tensor(c, "C");

    // Free memory
    free(a->data);
    free(a->grad);
    free(a);
    free(b->data);
    free(b->grad);
    free(b);
    free(c->data);
    free(c->grad);
    free(c);
    free(d->data);
    free(d->grad);
    free(d);
    free(output->data);
    free(output->grad);
    free(output);
    free(matmul_op->name);
    free(matmul_op);
    free(add_op->name);
    free(add_op);

    return 0;
}