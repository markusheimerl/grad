#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

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
    if (requires_grad || data) tape.ops[tape.len++] = t;
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
    
    Tensor* result = tensor_new(2, out_dims, NULL, 1);
    result->op = op;
    result->num_children = 2;
    result->children[0] = a;
    result->children[1] = b;
    
    if (op == ADD) {
        for (int i = 0; i < a->size; i++)
            result->data[i] = a->data[i] + b->data[i];
    } else {
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
        if (a->requires_grad) {
            for (int i = 0; i < a->dims[0]; i++)
                for (int k = 0; k < a->dims[1]; k++) {
                    float sum = 0;
                    for (int j = 0; j < b->dims[1]; j++)
                        sum += t->grad[i * b->dims[1] + j] * b->data[k * b->dims[1] + j];
                    a->grad[i * a->dims[1] + k] += sum;
                }
        }
        if (b->requires_grad) {
            for (int k = 0; k < b->dims[0]; k++)
                for (int j = 0; j < b->dims[1]; j++) {
                    float sum = 0;
                    for (int i = 0; i < a->dims[0]; i++)
                        sum += t->grad[i * b->dims[1] + j] * a->data[i * a->dims[1] + k];
                    b->grad[k * b->dims[1] + j] += sum;
                }
        }
    }
}

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.ops[tape.len - 1];
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++)
            final->grad[i] = 1.0;
        for (int i = tape.len - 1; i >= 0; i--)
            backward_op(tape.ops[i]);
    }
}

void zero_grad(Tensor* t) {
    if (t->grad) memset(t->grad, 0, t->size * sizeof(float));
}

void tape_clear() { tape.len = 0; }

void compare_with_pytorch(float* values, const char* pytorch_str, int expected_size, const char* name, float tol) {
    char* str_copy = strdup(pytorch_str);
    char* saveptr;
    char* token = strtok_r(str_copy, " ", &saveptr);
    
    for (int i = 0; i < expected_size; i++) {
        if (!token) {
            printf("Error: Not enough values in PyTorch results for %s\n", name);
            free(str_copy);
            exit(1);
        }
        float pytorch_val = strtof(token, NULL);
        float diff = fabs(values[i] - pytorch_val);
        if (diff > tol) {
            printf("Error: Mismatch in %s at index %d: got %f, expected %f (diff: %f)\n",
                   name, i, values[i], pytorch_val, diff);
            free(str_copy);
            exit(1);
        }
        token = strtok_r(NULL, " ", &saveptr);
    }
    
    if (token) {
        printf("Error: Extra values in PyTorch results for %s\n", name);
        free(str_copy);
        exit(1);
    }
    
    free(str_copy);
    printf("✓ %s matches PyTorch\n", name);
}

int main() {
    // Generate Python comparison script
    FILE* f = fopen("compare.py", "w");
    fprintf(f, "import torch\n\n");
    
    // Test 1
    fprintf(f, "a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)\n");
    fprintf(f, "b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)\n");
    fprintf(f, "c = a + b\n");
    fprintf(f, "d = c @ b\n");
    fprintf(f, "d.backward(torch.ones_like(d))\n");
    fprintf(f, "print('TEST1_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, a.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, b.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, c.detach().numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, d.detach().numpy().flatten())))\n");

    // Test 2
    fprintf(f, "m1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)\n");
    fprintf(f, "m2 = torch.tensor([[7., 8.], [9., 10.], [11., 12.]], requires_grad=True)\n");
    fprintf(f, "m3 = m1 @ m2\n");
    fprintf(f, "m3.backward(torch.ones_like(m3))\n");
    fprintf(f, "print('TEST2_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, m1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, m2.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, m3.detach().numpy().flatten())))\n");
    
    fclose(f);

    // Run Python script and capture output
    FILE* pipe = popen("python3 compare.py", "r");
    if (!pipe) {
        printf("Failed to run Python comparison\n");
        return 1;
    }

    char buffer[1024];
    char pytorch_results[8][1024];
    int result_idx = -1;
    
    while (fgets(buffer, sizeof(buffer), pipe)) {
        buffer[strcspn(buffer, "\n")] = 0;
        if (strcmp(buffer, "TEST1_RESULTS") == 0) {
            result_idx = 0;
            continue;
        }
        if (strcmp(buffer, "TEST2_RESULTS") == 0) {
            result_idx = 4;
            continue;
        }
        if (result_idx >= 0 && result_idx < 8) {
            strcpy(pytorch_results[result_idx++], buffer);
        }
    }
    
    if (pclose(pipe) != 0) {
        printf("Error: Python script execution failed\n");
        return 1;
    }
    
    if (system("rm compare.py") != 0) {
        printf("Warning: Failed to remove temporary Python file\n");
    }

    printf("Running tests and comparing with PyTorch...\n\n");
    float tol = 1e-5;
    
    // Test 1
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    
    Tensor *a = tensor_new(2, dims, data1, 1);
    Tensor *b = tensor_new(2, dims, data2, 1);
    Tensor *c = tensor_add(a, b);
    Tensor *d = tensor_matmul(c, b);
    
    printf("Test 1 forward pass:\n");
    tensor_print(a, "a");
    tensor_print(b, "b");
    tensor_print(c, "c");
    tensor_print(d, "d");
    
    backward();
    
    printf("\nComparing Test 1 results with PyTorch:\n");
    compare_with_pytorch(a->grad, pytorch_results[0], a->size, "a.grad", tol);
    compare_with_pytorch(b->grad, pytorch_results[1], b->size, "b.grad", tol);
    compare_with_pytorch(c->data, pytorch_results[2], c->size, "c values", tol);
    compare_with_pytorch(d->data, pytorch_results[3], d->size, "d values", tol);

    tape_clear();
    zero_grad(a); zero_grad(b);
    
    // Test 2
    printf("\nTest 2:\n");
    int dims2[] = {2, 3};
    int dims3[] = {3, 2};
    float data3[] = {1, 2, 3, 4, 5, 6};
    float data4[] = {7, 8, 9, 10, 11, 12};
    
    Tensor *m1 = tensor_new(2, dims2, data3, 1);
    Tensor *m2 = tensor_new(2, dims3, data4, 1);
    Tensor *m3 = tensor_matmul(m1, m2);
    
    printf("Test 2 forward pass:\n");
    tensor_print(m1, "m1");
    tensor_print(m2, "m2");
    tensor_print(m3, "m3");
    
    backward();
    
    printf("\nComparing Test 2 results with PyTorch:\n");
    compare_with_pytorch(m1->grad, pytorch_results[4], m1->size, "m1.grad", tol);
    compare_with_pytorch(m2->grad, pytorch_results[5], m2->size, "m2.grad", tol);
    compare_with_pytorch(m3->data, pytorch_results[6], m3->size, "m3 values", tol);

    printf("\nAll tests passed successfully!\n");
    return 0;
}