#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Forward declarations
struct Node;
struct Operator;
struct Tape;

typedef struct {
    int rows;
    int cols;
} Shape;

typedef struct {
    float* data;
    Shape shape;
    bool requires_grad;
    struct Node* grad_node;  // Points to gradient computation node
} Tensor;

typedef struct Node {
    Tensor* tensor;
    struct Operator* op;
    struct Node** inputs;
    int num_inputs;
    bool is_leaf;
    struct Tape* tape;  // For higher-order derivatives
} Node;

typedef struct Operator {
    char* name;
    Tensor* (*forward)(Node* node);
    void (*backward)(Node* node);
} Operator;

typedef struct Tape {
    Node** nodes;
    int num_nodes;
    int capacity;
    struct Tape* parent_tape;  // For nested derivatives
} Tape;

// Global tape for automatic differentiation
Tape* current_tape = NULL;

// Utility functions
Tensor* create_tensor(int rows, int cols, bool requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->shape.rows = rows;
    t->shape.cols = cols;
    t->data = (float*)calloc(rows * cols, sizeof(float));
    t->requires_grad = requires_grad;
    t->grad_node = NULL;
    return t;
}

Node* create_node(Tensor* tensor, Operator* op, Node** inputs, int num_inputs) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->tensor = tensor;
    node->op = op;
    node->inputs = inputs;
    node->num_inputs = num_inputs;
    node->is_leaf = (op == NULL);
    node->tape = current_tape;
    
    // Add to tape if we're recording
    if (current_tape != NULL) {
        if (current_tape->num_nodes >= current_tape->capacity) {
            current_tape->capacity *= 2;
            current_tape->nodes = (Node**)realloc(current_tape->nodes, 
                                                current_tape->capacity * sizeof(Node*));
        }
        current_tape->nodes[current_tape->num_nodes++] = node;
    }
    
    return node;
}

Tape* create_tape() {
    Tape* tape = (Tape*)malloc(sizeof(Tape));
    tape->nodes = (Node**)malloc(sizeof(Node*) * 16);  // Initial capacity
    tape->num_nodes = 0;
    tape->capacity = 16;
    tape->parent_tape = current_tape;  // Link to parent tape for nested gradients
    return tape;
}

// Start recording operations for automatic differentiation
void start_recording() {
    current_tape = create_tape();
}

// Stop recording and return the tape
Tape* stop_recording() {
    Tape* tape = current_tape;
    current_tape = tape->parent_tape;  // Restore parent tape
    return tape;
}

// Forward operations
Tensor* matmul_forward(Node* node) {
    Tensor* a = node->inputs[0]->tensor;
    Tensor* b = node->inputs[1]->tensor;
    
    if (a->shape.cols != b->shape.rows) {
        printf("Invalid shapes for matrix multiplication!\n");
        return NULL;
    }

    Tensor* out = create_tensor(a->shape.rows, b->shape.cols, true);
    
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

// Backward operations
void matmul_backward(Node* node) {
    Tensor* grad_output = node->tensor->grad_node->tensor;
    Tensor* a = node->inputs[0]->tensor;
    Tensor* b = node->inputs[1]->tensor;

    if (a->requires_grad) {
        Tensor* a_grad = create_tensor(a->shape.rows, a->shape.cols, false);
        for (int i = 0; i < a->shape.rows; i++) {
            for (int k = 0; k < a->shape.cols; k++) {
                float sum = 0;
                for (int j = 0; j < b->shape.cols; j++) {
                    sum += grad_output->data[i * grad_output->shape.cols + j] * 
                           b->data[k * b->shape.cols + j];
                }
                a_grad->data[i * a->shape.cols + k] = sum;
            }
        }
        node->inputs[0]->tensor->grad_node = create_node(a_grad, NULL, NULL, 0);
    }

    if (b->requires_grad) {
        Tensor* b_grad = create_tensor(b->shape.rows, b->shape.cols, false);
        for (int k = 0; k < b->shape.rows; k++) {
            for (int j = 0; j < b->shape.cols; j++) {
                float sum = 0;
                for (int i = 0; i < a->shape.rows; i++) {
                    sum += a->data[i * a->shape.cols + k] * 
                           grad_output->data[i * grad_output->shape.cols + j];
                }
                b_grad->data[k * b->shape.cols + j] = sum;
            }
        }
        node->inputs[1]->tensor->grad_node = create_node(b_grad, NULL, NULL, 0);
    }
}

// Operator creation and application
Operator* create_operator(const char* name, 
                         Tensor* (*forward)(Node*),
                         void (*backward)(Node*)) {
    Operator* op = (Operator*)malloc(sizeof(Operator));
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

// Backward pass through computation graph
void backward_pass(Node* node) {
    if (!node->tensor->requires_grad) return;
    
    // Create gradient tensor filled with ones if this is the output node
    if (!node->tensor->grad_node) {
        Tensor* grad = create_tensor(node->tensor->shape.rows, 
                                   node->tensor->shape.cols, 
                                   false);
        for (int i = 0; i < grad->shape.rows * grad->shape.cols; i++) {
            grad->data[i] = 1.0;
        }
        node->tensor->grad_node = create_node(grad, NULL, NULL, 0);
    }
    
    // Traverse tape in reverse order
    for (int i = node->tape->num_nodes - 1; i >= 0; i--) {
        Node* current = node->tape->nodes[i];
        if (current->op && current->op->backward) {
            current->op->backward(current);
        }
    }
}

// Higher-order gradient computation
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
        for (int j = 0; j < t->shape.cols; j++) {
            printf("%.2f ", t->data[i * t->shape.cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Create operators
    Operator* matmul = create_operator("matmul", matmul_forward, matmul_backward);
    
    // Create input tensors
    Tensor* a_tensor = create_tensor(2, 3, true);
    Tensor* b_tensor = create_tensor(3, 2, true);
    
    // Initialize tensors
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 2, 3, 4, 5, 6};
    memcpy(a_tensor->data, a_data, sizeof(float) * 6);
    memcpy(b_tensor->data, b_data, sizeof(float) * 6);
    
    // Create leaf nodes
    Node* a = create_node(a_tensor, NULL, NULL, 0);
    Node* b = create_node(b_tensor, NULL, NULL, 0);
    
    // Start recording operations
    start_recording();
    
    // Forward pass
    Node* inputs[] = {a, b};
    Node* c = apply_operator(matmul, inputs, 2);
    
    // Print initial tensors
    print_tensor(a->tensor, "A");
    print_tensor(b->tensor, "B");
    print_tensor(c->tensor, "C (A @ B)");
    
    // Compute first-order gradients
    backward_pass(c);
    
    printf("First-order gradients:\n");
    print_tensor(a->tensor->grad_node->tensor, "dC/dA");
    print_tensor(b->tensor->grad_node->tensor, "dC/dB");
    
    // Compute second-order gradients (if needed)
    Node* grad_a = grad(c, a, 2);
    if (grad_a) {
        printf("Second-order gradient:\n");
        print_tensor(grad_a->tensor, "d²C/dA²");
    }
    
    // Clean up (basic cleanup, in practice you'd need more comprehensive memory management)
    free(matmul->name);
    free(matmul);
    
    return 0;
}