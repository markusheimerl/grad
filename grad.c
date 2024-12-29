#ifndef __GRAD_H__
#define __GRAD_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef enum { ADD, MATMUL, NONE } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad, num_children;
    struct Tensor *children[2];
    OpType op;
} Tensor;

typedef struct {
    Tensor* ops[1000];
    int len;
} Tape;

static Tape tape = {0};

static int calc_size(int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = memcpy(malloc(ndims * sizeof(int)), dims, ndims * sizeof(int));
    t->size = calc_size(dims, ndims);
    t->data = malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    t->requires_grad = requires_grad;
    if (requires_grad) t->grad = calloc(t->size, sizeof(float));
    
    if (requires_grad || data) {  // Track leaf nodes and computed tensors
        tape.ops[tape.len++] = t;
    }
    return t;
}

void tensor_print(Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) 
        printf("%d%s", t->dims[i], i < t->ndims-1 ? "," : "");
    
    printf("], data=[");
    for (int i = 0; i < t->size; i++)
        printf("%.2f%s", t->data[i], i < t->size-1 ? "," : "");
    
    if (t->grad) {
        printf("], grad=[");
        for (int i = 0; i < t->size; i++)
            printf("%.2f%s", t->grad[i], i < t->size-1 ? "," : "");
    }
    printf("]\n");
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    assert(a->ndims == b->ndims);
    if (op == MATMUL) {
        assert(a->dims[1] == b->dims[0]);
    } else {
        for (int i = 0; i < a->ndims; i++) 
            assert(a->dims[i] == b->dims[i]);
    }
    
    int out_dims[2] = {op == MATMUL ? a->dims[0] : a->dims[0], 
                       op == MATMUL ? b->dims[1] : b->dims[1]};
    
    Tensor* result = tensor_new(2, out_dims, NULL, 1);  // Always track computed tensors
    result->op = op;
    result->num_children = 2;
    result->children[0] = a;
    result->children[1] = b;
    
    if (op == ADD) {
        for (int i = 0; i < a->size; i++)
            result->data[i] = a->data[i] + b->data[i];
    } else {  // MATMUL
        for (int i = 0; i < a->dims[0]; i++)
            for (int j = 0; j < b->dims[1]; j++) {
                float sum = 0;
                for (int k = 0; k < a->dims[1]; k++)
                    sum += a->data[i * a->dims[1] + k] * b->data[k * b->dims[1] + j];
                result->data[i * b->dims[1] + j] = sum;
            }
    }
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)

static void backward_op(Tensor* t) {
    if (!t || t->op == NONE) return;
    
    Tensor *a = t->children[0], *b = t->children[1];
    if (!a || !b) return;
    
    if (t->op == ADD) {
        if (a->requires_grad)
            for (int i = 0; i < a->size; i++)
                a->grad[i] += t->grad[i];
        if (b->requires_grad)
            for (int i = 0; i < b->size; i++)
                b->grad[i] += t->grad[i];
    } else if (t->op == MATMUL) {
        if (a->requires_grad)
            for (int i = 0; i < a->dims[0]; i++)
                for (int j = 0; j < a->dims[1]; j++) {
                    float sum = 0;
                    for (int k = 0; k < b->dims[1]; k++)
                        sum += t->grad[i * b->dims[1] + k] * b->data[j * b->dims[1] + k];
                    a->grad[i * a->dims[1] + j] += sum;
                }
        if (b->requires_grad)
            for (int i = 0; i < b->dims[0]; i++)
                for (int j = 0; j < b->dims[1]; j++) {
                    float sum = 0;
                    for (int k = 0; k < a->dims[0]; k++)
                        sum += t->grad[k * b->dims[1] + j] * a->data[k * a->dims[1] + i];
                    b->grad[i * b->dims[1] + j] += sum;
                }
    }
}

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.ops[tape.len - 1];
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        final->grad[0] = 1.0;
        
        for (int i = tape.len - 1; i >= 0; i--)
            backward_op(tape.ops[i]);
    }
}

void zero_grad(Tensor* t) {
    if (t->grad) memset(t->grad, 0, t->size * sizeof(float));
}

void tape_clear() { tape.len = 0; }


#endif // __GRAD_H__

int main() {
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    
    Tensor *a = tensor_new(2, dims, data1, 1);
    Tensor *b = tensor_new(2, dims, data2, 1);
    
    Tensor *c = tensor_add(a, b);
    Tensor *d = tensor_matmul(c, b);
    
    printf("Forward pass:\n");
    tensor_print(a, "a"); tensor_print(b, "b");
    tensor_print(c, "c = a + b"); tensor_print(d, "d = c @ b");
    
    backward();
    printf("\nBackward pass:\n");
    tensor_print(a, "a"); tensor_print(b, "b");
    tensor_print(c, "c"); tensor_print(d, "d");
    
    tape_clear();
    zero_grad(a); zero_grad(b);
    
    printf("\nNew computation:\n");
    Tensor *e = tensor_matmul(a, b);
    backward();
    tensor_print(a, "a"); tensor_print(b, "b");
    tensor_print(e, "e = a @ b");
    
    return 0;
}