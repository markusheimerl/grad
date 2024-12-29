#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAX_DIMS 8
#define MAX_CHILDREN 2
#define MAX_TAPE_LENGTH 1000

typedef enum {
    ADD,
    MATMUL,
    NONE
} OpType;

typedef struct Tensor {
    float* data;
    float* grad;
    int* dims;
    int ndims;
    int requires_grad;
    int num_children;
    struct Tensor* children[MAX_CHILDREN];
    OpType op;
    int size;
    int tape_idx;  // Position in tape
} Tensor;

// Global tape for automatic differentiation
typedef struct {
    Tensor* operations[MAX_TAPE_LENGTH];
    int length;
} Tape;

Tape tape = {.length = 0};

// Helper functions
void tape_push(Tensor* t) {
    if (t->requires_grad || (t->children[0] && t->children[0]->requires_grad) || 
        (t->children[1] && t->children[1]->requires_grad)) {
        t->tape_idx = tape.length;
        tape.operations[tape.length++] = t;
    }
}

void tape_clear() {
    tape.length = 0;
}

int compute_size(int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_create(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndims = ndims;
    t->dims = (int*)malloc(ndims * sizeof(int));
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = compute_size(dims, ndims);
    t->data = (float*)malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    
    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = (float*)calloc(t->size, sizeof(float));
    } else {
        t->grad = NULL;
    }
    
    t->num_children = 0;
    t->op = NONE;
    t->tape_idx = -1;
    
    if (requires_grad) {
        tape_push(t);
    }
    
    return t;
}

void tensor_print(Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) {
        printf("%d%s", t->dims[i], i < t->ndims - 1 ? "," : "");
    }
    printf("], data=[");
    for (int i = 0; i < t->size; i++) {
        printf("%.2f%s", t->data[i], i < t->size - 1 ? "," : "");
    }
    printf("]");
    if (t->grad) {
        printf(", grad=[");
        for (int i = 0; i < t->size; i++) {
            printf("%.2f%s", t->grad[i], i < t->size - 1 ? "," : "");
        }
        printf("]");
    }
    printf("\n");
}

void zero_grad(Tensor* t) {
    if (t->grad) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    assert(a->ndims == b->ndims);
    for (int i = 0; i < a->ndims; i++) {
        assert(a->dims[i] == b->dims[i]);
    }
    
    Tensor* result = tensor_create(a->ndims, a->dims, NULL, 
                                 a->requires_grad || b->requires_grad);
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    result->op = ADD;
    result->num_children = 2;
    result->children[0] = a;
    result->children[1] = b;
    
    tape_push(result);
    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    assert(a->ndims == 2 && b->ndims == 2);
    assert(a->dims[1] == b->dims[0]);
    
    int out_dims[] = {a->dims[0], b->dims[1]};
    Tensor* result = tensor_create(2, out_dims, NULL, 
                                 a->requires_grad || b->requires_grad);
    
    for (int i = 0; i < a->dims[0]; i++) {
        for (int j = 0; j < b->dims[1]; j++) {
            float sum = 0;
            for (int k = 0; k < a->dims[1]; k++) {
                sum += a->data[i * a->dims[1] + k] * b->data[k * b->dims[1] + j];
            }
            result->data[i * b->dims[1] + j] = sum;
        }
    }
    
    result->op = MATMUL;
    result->num_children = 2;
    result->children[0] = a;
    result->children[1] = b;
    
    tape_push(result);
    return result;
}

void backward_add(Tensor* t) {
    Tensor *a = t->children[0], *b = t->children[1];
    
    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++) {
            a->grad[i] += t->grad[i];
        }
    }
    
    if (b->requires_grad) {
        for (int i = 0; i < b->size; i++) {
            b->grad[i] += t->grad[i];
        }
    }
}

void backward_matmul(Tensor* t) {
    Tensor *a = t->children[0], *b = t->children[1];
    
    if (a->requires_grad) {
        for (int i = 0; i < a->dims[0]; i++) {
            for (int j = 0; j < a->dims[1]; j++) {
                float sum = 0;
                for (int k = 0; k < b->dims[1]; k++) {
                    sum += t->grad[i * b->dims[1] + k] * b->data[j * b->dims[1] + k];
                }
                a->grad[i * a->dims[1] + j] += sum;
            }
        }
    }
    
    if (b->requires_grad) {
        for (int i = 0; i < b->dims[0]; i++) {
            for (int j = 0; j < b->dims[1]; j++) {
                float sum = 0;
                for (int k = 0; k < a->dims[0]; k++) {
                    sum += t->grad[k * b->dims[1] + j] * a->data[k * a->dims[1] + i];
                }
                b->grad[i * b->dims[1] + j] += sum;
            }
        }
    }
}

void backward() {
    // Initialize gradient of final tensor to 1
    if (tape.length > 0) {
        Tensor* final_tensor = tape.operations[tape.length - 1];
        if (final_tensor->grad == NULL) {
            final_tensor->grad = (float*)calloc(final_tensor->size, sizeof(float));
        }
        final_tensor->grad[0] = 1.0;
        
        // Backward pass through the tape
        for (int i = tape.length - 1; i >= 0; i--) {
            Tensor* t = tape.operations[i];
            switch (t->op) {
                case ADD:
                    backward_add(t);
                    break;
                case MATMUL:
                    backward_matmul(t);
                    break;
                case NONE:
                    break;
            }
        }
    }
}

int main() {
    // Example of dynamic computation graph
    float data1[] = {1, 2, 3, 4};
    int dims1[] = {2, 2};
    Tensor* a = tensor_create(2, dims1, data1, 1);
    
    float data2[] = {5, 6, 7, 8};
    Tensor* b = tensor_create(2, dims1, data2, 1);
    
    // Forward pass
    Tensor* c = tensor_add(a, b);
    Tensor* d = tensor_matmul(c, b);
    
    printf("Forward pass results:\n");
    tensor_print(a, "a");
    tensor_print(b, "b");
    tensor_print(c, "c = a + b");
    tensor_print(d, "d = c @ b");
    
    // Backward pass
    printf("\nBackward pass results:\n");
    backward();
    
    tensor_print(a, "a");
    tensor_print(b, "b");
    tensor_print(c, "c");
    tensor_print(d, "d");
    
    // Clear tape and gradients for new computation
    tape_clear();
    zero_grad(a);
    zero_grad(b);
    
    // New computation
    printf("\nNew computation after tape clear:\n");
    Tensor* e = tensor_matmul(a, b);
    backward();
    
    tensor_print(a, "a");
    tensor_print(b, "b");
    tensor_print(e, "e = a @ b");
    
    return 0;
}
