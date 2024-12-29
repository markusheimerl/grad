#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef enum { SCALAR, MATRIX } TensorType;
typedef struct { int rows, cols; } Shape;
typedef struct Node Node;

typedef struct {
    float* data;
    Shape shape;
    bool requires_grad;
    Node* grad_node;
    TensorType type;
} Tensor;

typedef struct {
    char* name;
    Tensor* (*forward)(Node*);
    void (*backward)(Node*);
} Operator;

typedef struct Tape {
    Node** nodes;
    int num_nodes, capacity;
    struct Tape* parent_tape;
} Tape;

struct Node {
    Tensor* tensor;
    Operator* op;
    Node** inputs;
    int num_inputs;
    bool is_leaf;
    Tape* tape;
};

static Tape* current_tape = NULL;

// Core tensor operations
Tensor* create_tensor(int rows, int cols, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));
    t->shape = (Shape){rows, cols};
    t->data = calloc(rows * cols, sizeof(float));
    t->requires_grad = requires_grad;
    t->grad_node = NULL;
    t->type = (rows == 1 && cols == 1) ? SCALAR : MATRIX;
    return t;
}

Tensor* create_scalar(float value, bool requires_grad) {
    Tensor* t = create_tensor(1, 1, requires_grad);
    t->data[0] = value;
    return t;
}

// Graph operations
Node* create_node(Tensor* tensor, Operator* op, Node** inputs, int num_inputs) {
    Node* node = malloc(sizeof(Node));
    *node = (Node){tensor, op, inputs, num_inputs, (op == NULL), current_tape};
    
    if (current_tape) {
        if (current_tape->num_nodes >= current_tape->capacity) {
            current_tape->capacity *= 2;
            current_tape->nodes = realloc(current_tape->nodes, 
                                        current_tape->capacity * sizeof(Node*));
        }
        current_tape->nodes[current_tape->num_nodes++] = node;
    }
    return node;
}

Tape* create_tape(void) {
    Tape* tape = malloc(sizeof(Tape));
    *tape = (Tape){malloc(sizeof(Node*) * 16), 0, 16, current_tape};
    return tape;
}

void start_recording(void) { 
    current_tape = create_tape(); 
}

Tape* stop_recording(void) {
    Tape* tape = current_tape;
    current_tape = tape->parent_tape;
    return tape;
}

void accumulate_grad(Tensor* tensor, Tensor* grad) {
    if (!tensor->grad_node) {
        tensor->grad_node = create_node(grad, NULL, NULL, 0);
    } else {
        int size = grad->shape.rows * grad->shape.cols;
        for (int i = 0; i < size; i++) {
            tensor->grad_node->tensor->data[i] += grad->data[i];
        }
    }
}

// Element-wise operations
Tensor* add_forward(Node* node) {
    Tensor *a = node->inputs[0]->tensor, *b = node->inputs[1]->tensor;
    Tensor* out = create_tensor(a->shape.rows, a->shape.cols, true);
    
    int size = a->shape.rows * a->shape.cols;
    for (int i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

void add_backward(Node* node) {
    Tensor* grad_output = node->tensor->grad_node->tensor;
    
    for (int i = 0; i < 2; i++) {
        if (node->inputs[i]->tensor->requires_grad) {
            Tensor* grad = create_tensor(node->inputs[i]->tensor->shape.rows,
                                       node->inputs[i]->tensor->shape.cols, false);
            int size = grad->shape.rows * grad->shape.cols;
            memcpy(grad->data, grad_output->data, size * sizeof(float));
            accumulate_grad(node->inputs[i]->tensor, grad);
        }
    }
}

// Matrix and scalar operations
Tensor* matmul_forward(Node* node) {
    Tensor *a = node->inputs[0]->tensor, *b = node->inputs[1]->tensor;
    if (a->shape.cols != b->shape.rows) return NULL;

    Tensor* out = create_tensor(a->shape.rows, b->shape.cols, true);
    for (int i = 0; i < a->shape.rows; i++)
        for (int j = 0; j < b->shape.cols; j++)
            for (int k = 0; k < a->shape.cols; k++)
                out->data[i * out->shape.cols + j] += 
                    a->data[i * a->shape.cols + k] * b->data[k * b->shape.cols + j];
    return out;
}

void matmul_backward(Node* node) {
    Tensor *grad_output = node->tensor->grad_node->tensor,
           *a = node->inputs[0]->tensor,
           *b = node->inputs[1]->tensor;

    if (a->requires_grad) {
        Tensor* a_grad = create_tensor(a->shape.rows, a->shape.cols, false);
        for (int i = 0; i < a->shape.rows; i++)
            for (int k = 0; k < a->shape.cols; k++)
                for (int j = 0; j < b->shape.cols; j++)
                    a_grad->data[i * a->shape.cols + k] += 
                        grad_output->data[i * grad_output->shape.cols + j] * 
                        b->data[k * b->shape.cols + j];
        accumulate_grad(a, a_grad);
    }

    if (b->requires_grad) {
        Tensor* b_grad = create_tensor(b->shape.rows, b->shape.cols, false);
        for (int k = 0; k < b->shape.rows; k++)
            for (int j = 0; j < b->shape.cols; j++)
                for (int i = 0; i < a->shape.rows; i++)
                    b_grad->data[k * b->shape.cols + j] += 
                        a->data[i * a->shape.cols + k] * 
                        grad_output->data[i * grad_output->shape.cols + j];
        accumulate_grad(b, b_grad);
    }
}

Tensor* mul_forward(Node* node) {
    Tensor *a = node->inputs[0]->tensor, *b = node->inputs[1]->tensor;
    
    if (a->type == SCALAR) {
        Tensor* out = create_tensor(b->shape.rows, b->shape.cols, true);
        float scalar = a->data[0];
        int size = b->shape.rows * b->shape.cols;
        for (int i = 0; i < size; i++) {
            out->data[i] = scalar * b->data[i];
        }
        return out;
    } else if (b->type == SCALAR) {
        Tensor* out = create_tensor(a->shape.rows, a->shape.cols, true);
        float scalar = b->data[0];
        int size = a->shape.rows * a->shape.cols;
        for (int i = 0; i < size; i++) {
            out->data[i] = scalar * a->data[i];
        }
        return out;
    } else {
        Tensor* out = create_tensor(a->shape.rows, a->shape.cols, true);
        int size = a->shape.rows * a->shape.cols;
        for (int i = 0; i < size; i++) {
            out->data[i] = a->data[i] * b->data[i];
        }
        return out;
    }
}

void mul_backward(Node* node) {
    Tensor *grad_output = node->tensor->grad_node->tensor,
           *a = node->inputs[0]->tensor,
           *b = node->inputs[1]->tensor;
    
    if (a->requires_grad) {
        Tensor* grad_a;
        if (a->type == SCALAR) {
            grad_a = create_scalar(0.0, false);
            int size = b->shape.rows * b->shape.cols;
            for (int i = 0; i < size; i++) {
                grad_a->data[0] += grad_output->data[i] * b->data[i];
            }
        } else {
            grad_a = create_tensor(a->shape.rows, a->shape.cols, false);
            int size = a->shape.rows * a->shape.cols;
            for (int i = 0; i < size; i++) {
                grad_a->data[i] = grad_output->data[i] * 
                    (b->type == SCALAR ? b->data[0] : b->data[i]);
            }
        }
        accumulate_grad(a, grad_a);
    }
    
    if (b->requires_grad) {
        Tensor* grad_b;
        if (b->type == SCALAR) {
            grad_b = create_scalar(0.0, false);
            int size = a->shape.rows * a->shape.cols;
            for (int i = 0; i < size; i++) {
                grad_b->data[0] += grad_output->data[i] * a->data[i];
            }
        } else {
            grad_b = create_tensor(b->shape.rows, b->shape.cols, false);
            int size = b->shape.rows * b->shape.cols;
            for (int i = 0; i < size; i++) {
                grad_b->data[i] = grad_output->data[i] * 
                    (a->type == SCALAR ? a->data[0] : a->data[i]);
            }
        }
        accumulate_grad(b, grad_b);
    }
}

Operator* create_operator(const char* name, 
                         Tensor* (*forward)(Node*), 
                         void (*backward)(Node*)) {
    Operator* op = malloc(sizeof(Operator));
    *op = (Operator){strdup(name), forward, backward};
    return op;
}

Node* apply_operator(Operator* op, Node** inputs, int num_inputs) {
    Node* node = create_node(NULL, op, inputs, num_inputs);
    node->tensor = op->forward(node);
    return node;
}

void backward_pass(Node* node) {
    if (!node->tensor->requires_grad) return;
    
    if (!node->tensor->grad_node) {
        Tensor* grad = create_tensor(node->tensor->shape.rows, 
                                   node->tensor->shape.cols, false);
        int size = grad->shape.rows * grad->shape.cols;
        for (int i = 0; i < size; i++) grad->data[i] = 1.0;
        node->tensor->grad_node = create_node(grad, NULL, NULL, 0);
    }
    
    for (int i = node->tape->num_nodes - 1; i >= 0; i--)
        if (node->tape->nodes[i]->op && node->tape->nodes[i]->op->backward)
            node->tape->nodes[i]->op->backward(node->tape->nodes[i]);
}

void print_tensor(Tensor* t, const char* name) {
    if (t->type == SCALAR) {
        printf("%s (scalar): %.2f\n\n", name, t->data[0]);
        return;
    }
    
    printf("%s (%dx%d):\n", name, t->shape.rows, t->shape.cols);
    for (int i = 0; i < t->shape.rows; i++) {
        for (int j = 0; j < t->shape.cols; j++)
            printf("%.2f ", t->data[i * t->shape.cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    Operator *matmul = create_operator("matmul", matmul_forward, matmul_backward);
    Operator *add = create_operator("add", add_forward, add_backward);
    Operator *mul = create_operator("mul", mul_forward, mul_backward);
    
    // Create tensors
    Tensor *a_tensor = create_tensor(2, 2, true);
    float a_data[] = {1, 2, 3, 4};
    memcpy(a_tensor->data, a_data, sizeof(a_data));
    
    Tensor *b_tensor = create_tensor(2, 2, true);
    float b_data[] = {1, 0, 0, 1};
    memcpy(b_tensor->data, b_data, sizeof(b_data));
    
    Tensor *scalar = create_scalar(2.0, true);
    
    Node *a = create_node(a_tensor, NULL, NULL, 0);
    Node *b = create_node(b_tensor, NULL, NULL, 0);
    Node *s = create_node(scalar, NULL, NULL, 0);
    
    start_recording();
    
    // c = a @ b
    Node* inputs1[] = {a, b};
    Node* c = apply_operator(matmul, inputs1, 2);
    
    // d = c * s
    Node* inputs2[] = {c, s};
    Node* d = apply_operator(mul, inputs2, 2);
    
    // e = d + a
    Node* inputs3[] = {d, a};
    Node* e = apply_operator(add, inputs3, 2);
    
    print_tensor(a->tensor, "A");
    print_tensor(b->tensor, "B");
    print_tensor(s->tensor, "Scalar");
    print_tensor(c->tensor, "C (A @ B)");
    print_tensor(d->tensor, "D (C * scalar)");
    print_tensor(e->tensor, "E (D + A)");
    
    backward_pass(e);
    
    printf("Gradients (note accumulation for A):\n");
    print_tensor(a->tensor->grad_node->tensor, "dE/dA");
    print_tensor(b->tensor->grad_node->tensor, "dE/dB");
    print_tensor(s->tensor->grad_node->tensor, "dE/dscalar");
    
    // Basic cleanup
    free(matmul->name);
    free(matmul);
    free(add->name);
    free(add);
    free(mul->name);
    free(mul);
    
    return 0;
}