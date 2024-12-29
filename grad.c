#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    int rows, cols;
} Shape;

typedef struct Node Node;
typedef struct {
    float* data;
    Shape shape;
    bool requires_grad;
    Node* grad_node;
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

Tensor* create_tensor(int rows, int cols, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));
    t->shape = (Shape){rows, cols};
    t->data = calloc(rows * cols, sizeof(float));
    t->requires_grad = requires_grad;
    t->grad_node = NULL;
    return t;
}

Node* create_node(Tensor* tensor, Operator* op, Node** inputs, int num_inputs) {
    Node* node = malloc(sizeof(Node));
    *node = (Node){tensor, op, inputs, num_inputs, (op == NULL), current_tape};
    
    if (current_tape) {
        if (current_tape->num_nodes >= current_tape->capacity) {
            current_tape->capacity *= 2;
            current_tape->nodes = realloc(current_tape->nodes, current_tape->capacity * sizeof(Node*));
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

void start_recording(void) { current_tape = create_tape(); }
Tape* stop_recording(void) {
    Tape* tape = current_tape;
    current_tape = tape->parent_tape;
    return tape;
}

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
                        grad_output->data[i * grad_output->shape.cols + j] * b->data[k * b->shape.cols + j];
        node->inputs[0]->tensor->grad_node = create_node(a_grad, NULL, NULL, 0);
    }

    if (b->requires_grad) {
        Tensor* b_grad = create_tensor(b->shape.rows, b->shape.cols, false);
        for (int k = 0; k < b->shape.rows; k++)
            for (int j = 0; j < b->shape.cols; j++)
                for (int i = 0; i < a->shape.rows; i++)
                    b_grad->data[k * b->shape.cols + j] += 
                        a->data[i * a->shape.cols + k] * grad_output->data[i * grad_output->shape.cols + j];
        node->inputs[1]->tensor->grad_node = create_node(b_grad, NULL, NULL, 0);
    }
}

Operator* create_operator(const char* name, Tensor* (*forward)(Node*), void (*backward)(Node*)) {
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
        Tensor* grad = create_tensor(node->tensor->shape.rows, node->tensor->shape.cols, false);
        for (int i = 0; i < grad->shape.rows * grad->shape.cols; i++) grad->data[i] = 1.0;
        node->tensor->grad_node = create_node(grad, NULL, NULL, 0);
    }
    
    for (int i = node->tape->num_nodes - 1; i >= 0; i--)
        if (node->tape->nodes[i]->op && node->tape->nodes[i]->op->backward)
            node->tape->nodes[i]->op->backward(node->tape->nodes[i]);
}

Node* grad(Node* output, Node* input, int order) {
    Node* result = output;
    for (int i = 0; i < order; i++) {
        start_recording();
        backward_pass(result);
        result = input->tensor->grad_node;
        stop_recording();
    }
    return result;
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (%dx%d):\n", name, t->shape.rows, t->shape.cols);
    for (int i = 0; i < t->shape.rows; i++) {
        for (int j = 0; j < t->shape.cols; j++)
            printf("%.2f ", t->data[i * t->shape.cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    Operator* matmul = create_operator("matmul", matmul_forward, matmul_backward);
    
    Tensor *a_tensor = create_tensor(2, 3, true),
           *b_tensor = create_tensor(3, 2, true);
    
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 2, 3, 4, 5, 6};
    memcpy(a_tensor->data, a_data, sizeof(float) * 6);
    memcpy(b_tensor->data, b_data, sizeof(float) * 6);
    
    Node *a = create_node(a_tensor, NULL, NULL, 0),
         *b = create_node(b_tensor, NULL, NULL, 0);
    
    start_recording();
    Node* inputs[] = {a, b};
    Node* c = apply_operator(matmul, inputs, 2);
    
    print_tensor(a->tensor, "A");
    print_tensor(b->tensor, "B");
    print_tensor(c->tensor, "C (A @ B)");
    
    backward_pass(c);
    
    printf("First-order gradients:\n");
    print_tensor(a->tensor->grad_node->tensor, "dC/dA");
    print_tensor(b->tensor->grad_node->tensor, "dC/dB");
    
    Node* grad_a = grad(c, a, 2);
    if (grad_a) {
        printf("Second-order gradient:\n");
        print_tensor(grad_a->tensor, "d²C/dA²");
    }
    
    free(matmul->name);
    free(matmul);
    
    return 0;
}
