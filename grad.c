#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Graph tracking structure
typedef struct Tape {
    struct Node** nodes;
    int num_nodes, capacity;
    struct Tape* parent;
} Tape;

static Tape* current_tape = NULL;

// Core structures
typedef struct {
    float* data;
    int rows, cols;
    bool requires_grad;
    struct Node* grad;
} Tensor;

typedef struct Node {
    Tensor* tensor;
    struct Node** inputs;
    int num_inputs;
    void (*backward)(struct Node*);
    Tape* tape;
} Node;

// Tape management
Tape* create_tape(void) {
    Tape* tape = malloc(sizeof(Tape));
    *tape = (Tape){
        .nodes = malloc(sizeof(Node*) * 16),
        .num_nodes = 0,
        .capacity = 16,
        .parent = current_tape
    };
    return tape;
}

void start_recording(void) {
    current_tape = create_tape();
}

Tape* stop_recording(void) {
    Tape* tape = current_tape;
    current_tape = tape->parent;
    return tape;
}

// Core functions
Tensor* tensor(int rows, int cols, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));
    *t = (Tensor){
        .data = calloc(rows * cols, sizeof(float)),
        .rows = rows,
        .cols = cols,
        .requires_grad = requires_grad,
        .grad = NULL
    };
    return t;
}

Node* node(Tensor* t, Node** inputs, int num_inputs, void (*backward)(Node*)) {
    Node* n = malloc(sizeof(Node));
    *n = (Node){t, inputs, num_inputs, backward, current_tape};
    
    if (current_tape) {
        if (current_tape->num_nodes >= current_tape->capacity) {
            current_tape->capacity *= 2;
            current_tape->nodes = realloc(current_tape->nodes, 
                                        current_tape->capacity * sizeof(Node*));
        }
        current_tape->nodes[current_tape->num_nodes++] = n;
    }
    return n;
}

// Matrix multiplication
void matmul_backward(Node* n) {
    Tensor *grad = n->tensor->grad->tensor,
           *a = n->inputs[0]->tensor,
           *b = n->inputs[1]->tensor;

    if (a->requires_grad) {
        Tensor* a_grad = tensor(a->rows, a->cols, false);
        for (int i = 0; i < a->rows; i++)
            for (int k = 0; k < a->cols; k++)
                for (int j = 0; j < b->cols; j++)
                    a_grad->data[i * a->cols + k] += 
                        grad->data[i * grad->rows + j] * b->data[k * b->cols + j];
        n->inputs[0]->tensor->grad = node(a_grad, NULL, 0, NULL);
    }

    if (b->requires_grad) {
        Tensor* b_grad = tensor(b->rows, b->cols, false);
        for (int k = 0; k < b->rows; k++)
            for (int j = 0; j < b->cols; j++)
                for (int i = 0; i < a->rows; i++)
                    b_grad->data[k * b->cols + j] += 
                        a->data[i * a->cols + k] * grad->data[i * grad->rows + j];
        n->inputs[1]->tensor->grad = node(b_grad, NULL, 0, NULL);
    }
}

Node* matmul(Node* a, Node* b) {
    Tensor* out = tensor(a->tensor->rows, b->tensor->cols, true);
    for (int i = 0; i < a->tensor->rows; i++)
        for (int j = 0; j < b->tensor->cols; j++)
            for (int k = 0; k < a->tensor->cols; k++)
                out->data[i * out->cols + j] += 
                    a->tensor->data[i * a->tensor->cols + k] * 
                    b->tensor->data[k * b->tensor->cols + j];

    Node** inputs = malloc(2 * sizeof(Node*));
    inputs[0] = a; inputs[1] = b;
    return node(out, inputs, 2, matmul_backward);
}

// Backward pass
void backward(Node* n) {
    if (!n->tensor->requires_grad) return;
    
    if (!n->tensor->grad) {
        Tensor* grad = tensor(n->tensor->rows, n->tensor->cols, false);
        for (int i = 0; i < grad->rows * grad->cols; i++) 
            grad->data[i] = 1.0;
        n->tensor->grad = node(grad, NULL, 0, NULL);
    }
    
    for (int i = n->tape->num_nodes - 1; i >= 0; i--) {
        Node* current = n->tape->nodes[i];
        if (current->backward) current->backward(current);
    }
}

// Higher-order gradients
Node* grad(Node* output, Node* input, int order) {
    Node* result = output;
    for (int i = 0; i < order; i++) {
        start_recording();
        backward(result);
        result = input->tensor->grad;
        stop_recording();
    }
    return result;
}

// Utility function
void print_tensor(Tensor* t, const char* name) {
    printf("%s (%dx%d):\n", name, t->rows, t->cols);
    for (int i = 0; i < t->rows; i++) {
        for (int j = 0; j < t->cols; j++)
            printf("%.2f ", t->data[i * t->cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Create tensors
    Tensor *a_tensor = tensor(2, 3, true),
           *b_tensor = tensor(3, 2, true);
    
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 2, 3, 4, 5, 6};
    memcpy(a_tensor->data, a_data, sizeof(a_data));
    memcpy(b_tensor->data, b_data, sizeof(b_data));
    
    Node *a = node(a_tensor, NULL, 0, NULL),
         *b = node(b_tensor, NULL, 0, NULL);
    
    // Forward and backward passes
    start_recording();
    Node* c = matmul(a, b);
    
    print_tensor(a->tensor, "A");
    print_tensor(b->tensor, "B");
    print_tensor(c->tensor, "C (A @ B)");
    
    backward(c);
    
    print_tensor(a->tensor->grad->tensor, "dC/dA");
    print_tensor(b->tensor->grad->tensor, "dC/dB");
    
    // Second-order gradients
    Node* grad_a = grad(c, a, 2);
    if (grad_a) {
        printf("Second-order gradient:\n");
        print_tensor(grad_a->tensor, "d²C/dA²");
    }
    
    stop_recording();
    return 0;
}