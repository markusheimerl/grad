#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct { int rows, cols; } Shape;

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

void free_tensor(Tensor* t) {
    if (t) { free(t->data); free(t); }
}

void free_node(Node* node) {
    if (node) { free(node->inputs); free(node); }
}

void free_tape(Tape* tape) {
    if (!tape) return;
    for (int i = tape->num_nodes - 1; i >= 0; i--) {
        Node* node = tape->nodes[i];
        if (!node->is_leaf) {
            if (node->tensor) {
                if (node->tensor->grad_node) {
                    free_tensor(node->tensor->grad_node->tensor);
                    free_node(node->tensor->grad_node);
                }
                free_tensor(node->tensor);
            }
            free_node(node);
        }
    }
    free(tape->nodes);
    free(tape);
}

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
    node->tensor = tensor;
    node->op = op;
    node->inputs = num_inputs ? malloc(num_inputs * sizeof(Node*)) : NULL;
    if (num_inputs) memcpy(node->inputs, inputs, num_inputs * sizeof(Node*));
    node->num_inputs = num_inputs;
    node->is_leaf = !op;
    node->tape = current_tape;
    
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
    tape->nodes = malloc(sizeof(Node*) * 16);
    tape->num_nodes = 0;
    tape->capacity = 16;
    tape->parent_tape = current_tape;
    return tape;
}

void start_recording(void) { current_tape = create_tape(); }
Tape* stop_recording(void) {
    Tape* tape = current_tape;
    current_tape = tape->parent_tape;
    return tape;
}

void zero_grad(Node* node) {
    if (node->tensor->grad_node) {
        memset(node->tensor->grad_node->tensor->data, 0,
               node->tensor->shape.rows * node->tensor->shape.cols * sizeof(float));
    }
}

Tensor* matmul_forward(Node* node) {
    Tensor *a = node->inputs[0]->tensor, *b = node->inputs[1]->tensor;
    if (a->shape.cols != b->shape.rows) return NULL;
    
    Tensor* out = create_tensor(a->shape.rows, b->shape.cols, true);
    for (int i = 0; i < a->shape.rows; i++)
        for (int j = 0; j < b->shape.cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->shape.cols; k++)
                sum += a->data[i * a->shape.cols + k] * b->data[k * b->shape.cols + j];
            out->data[i * out->shape.cols + j] = sum;
        }
    return out;
}

void matmul_backward(Node* node) {
    Tensor *grad_output = node->tensor->grad_node->tensor,
           *a = node->inputs[0]->tensor,
           *b = node->inputs[1]->tensor;

    if (a->requires_grad) {
        if (!node->inputs[0]->tensor->grad_node) {
            node->inputs[0]->tensor->grad_node = create_node(
                create_tensor(a->shape.rows, a->shape.cols, false), NULL, NULL, 0);
        }
        
        Tensor* a_grad = node->inputs[0]->tensor->grad_node->tensor;
        for (int i = 0; i < a->shape.rows; i++)
            for (int k = 0; k < a->shape.cols; k++) {
                float sum = 0;
                for (int j = 0; j < b->shape.cols; j++)
                    sum += grad_output->data[i * grad_output->shape.cols + j] * 
                          b->data[k * b->shape.cols + j];
                a_grad->data[i * a->shape.cols + k] += sum;
            }
    }

    if (b->requires_grad) {
        if (!node->inputs[1]->tensor->grad_node) {
            node->inputs[1]->tensor->grad_node = create_node(
                create_tensor(b->shape.rows, b->shape.cols, false), NULL, NULL, 0);
        }
        
        Tensor* b_grad = node->inputs[1]->tensor->grad_node->tensor;
        for (int k = 0; k < b->shape.rows; k++)
            for (int j = 0; j < b->shape.cols; j++) {
                float sum = 0;
                for (int i = 0; i < a->shape.rows; i++)
                    sum += a->data[i * a->shape.cols + k] * 
                          grad_output->data[i * grad_output->shape.cols + j];
                b_grad->data[k * b->shape.cols + j] += sum;
            }
    }
}

Operator* create_operator(const char* name, Tensor* (*forward)(Node*), void (*backward)(Node*)) {
    Operator* op = malloc(sizeof(Operator));
    op->name = strdup(name);
    op->forward = forward;
    op->backward = backward;
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
    
    for (int i = node->tape->num_nodes - 1; i >= 0; i--) {
        Node* current = node->tape->nodes[i];
        if (current->op && current->op->backward) current->op->backward(current);
    }
}

Node* grad(Node* output, Node* input, int order) {
    Node* result = output;
    Tape* last_tape = NULL;
    Tape* final_tape = NULL;
    
    for (int i = 0; i < order; i++) {
        start_recording();
        backward_pass(result);
        result = input->tensor->grad_node;
        if (last_tape) free_tape(last_tape);
        last_tape = stop_recording();
        if (i == order - 1) final_tape = last_tape;
    }
    
    if (result) {
        Tensor* persistent_tensor = create_tensor(result->tensor->shape.rows, 
                                                result->tensor->shape.cols, false);
        memcpy(persistent_tensor->data, result->tensor->data, 
               result->tensor->shape.rows * result->tensor->shape.cols * sizeof(float));
        Node* persistent_node = create_node(persistent_tensor, NULL, NULL, 0);
        free_tape(final_tape);
        return persistent_node;
    }
    
    if (final_tape) free_tape(final_tape);
    return NULL;
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (%dx%d):\n", name, t->shape.rows, t->shape.cols);
    for (int i = 0; i < t->shape.rows; i++) {
        for (int j = 0; j < t->shape.cols; j++) printf("%.2f ", t->data[i * t->shape.cols + j]);
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
    memcpy(a_tensor->data, a_data, sizeof(a_data));
    memcpy(b_tensor->data, b_data, sizeof(b_data));
    
    Node *a = create_node(a_tensor, NULL, NULL, 0),
         *b = create_node(b_tensor, NULL, NULL, 0);
    
    start_recording();
    Node* c = apply_operator(matmul, (Node*[]){a, b}, 2);
    
    print_tensor(a->tensor, "A");
    print_tensor(b->tensor, "B");
    print_tensor(c->tensor, "C (A @ B)");
    
    backward_pass(c);
    printf("First-order gradients:\n");
    print_tensor(a->tensor->grad_node->tensor, "dC/dA");
    print_tensor(b->tensor->grad_node->tensor, "dC/dB");
    
    Tape* first_tape = stop_recording();
    
    zero_grad(a);
    zero_grad(b);
    
    start_recording();
    Node* c2 = apply_operator(matmul, (Node*[]){a, b}, 2);
    backward_pass(c2);
    
    printf("Gradients after second backward pass:\n");
    print_tensor(a->tensor->grad_node->tensor, "dC/dA");
    print_tensor(b->tensor->grad_node->tensor, "dC/dB");
    
    Tape* second_tape = stop_recording();
    
    Node* grad_a = grad(c, a, 2);
    if (grad_a) {
        printf("Second-order gradient:\n");
        print_tensor(grad_a->tensor, "d²C/dA²");
        free_tensor(grad_a->tensor);
        free_node(grad_a);
    }
    
    free_tape(first_tape);
    free_tape(second_tape);
    
    if (a->tensor->grad_node) {
        free_tensor(a->tensor->grad_node->tensor);
        free_node(a->tensor->grad_node);
    }
    if (b->tensor->grad_node) {
        free_tensor(b->tensor->grad_node->tensor);
        free_node(b->tensor->grad_node);
    }
    
    free(matmul->name);
    free(matmul);
    free_tensor(a_tensor);
    free_tensor(b_tensor);
    free_node(a);
    free_node(b);
    
    return 0;
}