#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

typedef enum { ADD, MATMUL, NONE } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims;
    int ndims, size;
    int requires_grad;
    struct Tensor *children[2];
    OpType op;
} Tensor;

typedef struct {
    Tensor* ops[1000];
    int len;
} Tape;

static Tape tape = {0};

static inline int calc_size(const int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    if (!t->dims) { free(t); return NULL; }
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = calc_size(dims, ndims);
    t->data = malloc(t->size * sizeof(float));
    if (!t->data) { free(t->dims); free(t); return NULL; }
    
    if (data) {
        memcpy(t->data, data, t->size * sizeof(float));
    } else {
        memset(t->data, 0, t->size * sizeof(float));
    }
    
    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = calloc(t->size, sizeof(float));
        if (!t->grad) {
            free(t->data);
            free(t->dims);
            free(t);
            return NULL;
        }
    }
    
    if (requires_grad || data) tape.ops[tape.len++] = t;
    return t;
}

static void matmul_forward(const float* __restrict__ a, 
                          const float* __restrict__ b, 
                          float* __restrict__ out,
                          const int m, const int n, const int p) {
    memset(out, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            const float aik = a[i * n + k];
            for (int j = 0; j < p; j++) {
                out[i * p + j] += aik * b[k * p + j];
            }
        }
    }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!a || !b || a->ndims != b->ndims) return NULL;
    
    int out_dims[2];
    if (op == MATMUL) {
        if (a->dims[1] != b->dims[0]) return NULL;
        out_dims[0] = a->dims[0];
        out_dims[1] = b->dims[1];
    } else {
        if (a->dims[0] != b->dims[0] || a->dims[1] != b->dims[1]) return NULL;
        out_dims[0] = a->dims[0];
        out_dims[1] = a->dims[1];
    }
    
    Tensor* result = tensor_new(2, out_dims, NULL, 1);
    if (!result) return NULL;
    
    result->op = op;
    result->children[0] = a;
    result->children[1] = b;
    
    if (op == ADD) {
        for (int i = 0; i < a->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    } else {
        matmul_forward(a->data, b->data, result->data, 
                      a->dims[0], a->dims[1], b->dims[1]);
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
    } else if (t->op == MATMUL) {
        const int m = a->dims[0], n = a->dims[1], p = b->dims[1];
        
        if (a->requires_grad) {
            for (int i = 0; i < m; i++) {
                for (int k = 0; k < n; k++) {
                    float sum = 0;
                    for (int j = 0; j < p; j++) {
                        sum += t->grad[i * p + j] * b->data[k * p + j];
                    }
                    a->grad[i * n + k] += sum;
                }
            }
        }
        
        if (b->requires_grad) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0;
                    for (int i = 0; i < m; i++) {
                        sum += t->grad[i * p + j] * a->data[i * n + k];
                    }
                    b->grad[k * p + j] += sum;
                }
            }
        }
    }
}

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.ops[tape.len - 1];
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++) {
            final->grad[i] = 1.0f;
        }
        for (int i = tape.len - 1; i >= 0; i--) {
            backward_op(tape.ops[i]);
        }
    }
}

void zero_grad(Tensor* t) {
    if (t->grad) memset(t->grad, 0, t->size * sizeof(float));
}

void tape_clear() { tape.len = 0; }

static void compare_with_pytorch(float* values, const char* pytorch_str, 
                               int expected_size, const char* name, float tol) {
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
    FILE* f = fopen("compare.py", "w");
    fprintf(f, "import torch\n\n");
    
    // Test 1: Basic Addition and MatMul
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

    // Test 2: Different Shaped MatMul
    fprintf(f, "\nm1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)\n");
    fprintf(f, "m2 = torch.tensor([[7., 8.], [9., 10.], [11., 12.]], requires_grad=True)\n");
    fprintf(f, "m3 = m1 @ m2\n");
    fprintf(f, "m3.backward(torch.ones_like(m3))\n");
    fprintf(f, "print('TEST2_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, m1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, m2.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, m3.detach().numpy().flatten())))\n");

    // Test 3: Complex Graph
    fprintf(f, "\nx1 = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)\n");
    fprintf(f, "x2 = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)\n");
    fprintf(f, "x3 = torch.tensor([[9., 10.], [11., 12.]], requires_grad=True)\n");
    fprintf(f, "y = (x1 + x2) @ x3\n");
    fprintf(f, "y.backward(torch.ones_like(y))\n");
    fprintf(f, "print('TEST3_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, y.detach().numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x2.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x3.grad.numpy().flatten())))\n");

    // Test 4: Repeated Use of Same Tensor
    fprintf(f, "\nx4 = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)\n");
    fprintf(f, "intermediate1 = x4 @ x4\n");
    fprintf(f, "intermediate2 = x4 @ intermediate1\n");
    fprintf(f, "intermediate2.backward(torch.ones_like(intermediate2))\n");
    fprintf(f, "print('TEST4_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x4.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, intermediate2.detach().numpy().flatten())))\n");

    // Test 5: Deep Chain
    fprintf(f, "\nx5 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)\n");
    fprintf(f, "current = x5\n");
    fprintf(f, "for _ in range(5):\n");
    fprintf(f, "    current = current @ x5\n");
    fprintf(f, "current.backward(torch.ones_like(current))\n");
    fprintf(f, "print('TEST5_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x5.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, current.detach().numpy().flatten())))\n");

    // Test 6: Mixed Operations Tree
    fprintf(f, "\nx6 = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)\n");
    fprintf(f, "x7 = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)\n");
    fprintf(f, "branch1 = x6 @ x7\n");
    fprintf(f, "branch2 = x6 + x7\n");
    fprintf(f, "branch3 = branch2 @ branch1\n");
    fprintf(f, "branch3.backward(torch.ones_like(branch3))\n");
    fprintf(f, "print('TEST6_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x6.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x7.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, branch3.detach().numpy().flatten())))\n");

    // Test 7: Zero Gradients and Mixed Requires Grad
    fprintf(f, "\nx8 = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)\n");
    fprintf(f, "x9 = torch.tensor([[1., 1.], [1., 1.]], requires_grad=False)\n");
    fprintf(f, "x10 = torch.tensor([[2., 2.], [2., 2.]], requires_grad=True)\n");
    fprintf(f, "result = (x8 + x9) @ x10\n");
    fprintf(f, "result.backward(torch.ones_like(result))\n");
    fprintf(f, "print('TEST7_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x8.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x10.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, result.detach().numpy().flatten())))\n");

    // Test 8: Larger Matrices (3x3)
    fprintf(f, "\nx11 = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], requires_grad=True)\n");
    fprintf(f, "x12 = torch.tensor([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]], requires_grad=True)\n");
    fprintf(f, "result2 = (x11 @ x12 + x11) @ x12\n");
    fprintf(f, "result2.backward(torch.ones_like(result2))\n");
    fprintf(f, "print('TEST8_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x11.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x12.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, result2.detach().numpy().flatten())))\n");

    // Test 9: Complex Multi-Branch with Reuse
    fprintf(f, "\nx13 = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)\n");
    fprintf(f, "x14 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)\n");
    fprintf(f, "branch4 = x13 @ x14\n");
    fprintf(f, "branch5 = x14 @ x13\n");
    fprintf(f, "branch6 = branch4 + branch5\n");
    fprintf(f, "final = branch6 @ (x13 + x14)\n");
    fprintf(f, "final.backward(torch.ones_like(final))\n");
    fprintf(f, "print('TEST9_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x13.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x14.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, final.detach().numpy().flatten())))\n");

    fclose(f);

    FILE* pipe = popen("python3 compare.py", "r");
    if (!pipe) {
        printf("Failed to run Python comparison\n");
        return 1;
    }

    char buffer[1024];
    char pytorch_results[30][1024];
    int result_idx = -1;
    int current_section = 0;
    
    while (fgets(buffer, sizeof(buffer), pipe)) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (strstr(buffer, "_RESULTS")) {
            if (strstr(buffer, "TEST1")) current_section = 0;
            else if (strstr(buffer, "TEST2")) current_section = 4;
            else if (strstr(buffer, "TEST3")) current_section = 7;
            else if (strstr(buffer, "TEST4")) current_section = 11;
            else if (strstr(buffer, "TEST5")) current_section = 13;
            else if (strstr(buffer, "TEST6")) current_section = 15;
            else if (strstr(buffer, "TEST7")) current_section = 18;
            else if (strstr(buffer, "TEST8")) current_section = 21;
            else if (strstr(buffer, "TEST9")) current_section = 24;
            result_idx = 0;
            continue;
        }
        
        if (result_idx >= 0) {
            strcpy(pytorch_results[current_section + result_idx], buffer);
            result_idx++;
        }
    }
    
    pclose(pipe);
    remove("compare.py");

    printf("Running comprehensive tests...\n\n");
    float tol = 1e-5;
    
    // Test 1: Basic Addition and MatMul
    printf("=== Test 1: Basic Addition and MatMul ===\n");
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    
    Tensor *a = tensor_new(2, dims, data1, 1);
    Tensor *b = tensor_new(2, dims, data2, 1);
    Tensor *c = tensor_add(a, b);
    Tensor *d = tensor_matmul(c, b);

    backward();
    compare_with_pytorch(a->grad, pytorch_results[0], a->size, "a.grad", tol);
    compare_with_pytorch(b->grad, pytorch_results[1], b->size, "b.grad", tol);
    compare_with_pytorch(c->data, pytorch_results[2], c->size, "c values", tol);
    compare_with_pytorch(d->data, pytorch_results[3], d->size, "d values", tol);

    tape_clear();
    
    // Test 2: Different Shaped MatMul
    printf("\n=== Test 2: Different Shaped MatMul ===\n");
    int dims2[] = {2, 3};
    int dims3[] = {3, 2};
    float data3[] = {1, 2, 3, 4, 5, 6};
    float data4[] = {7, 8, 9, 10, 11, 12};
    
    Tensor *m1 = tensor_new(2, dims2, data3, 1);
    Tensor *m2 = tensor_new(2, dims3, data4, 1);
    Tensor *m3 = tensor_matmul(m1, m2);
    
    backward();
    compare_with_pytorch(m1->grad, pytorch_results[4], m1->size, "m1.grad", tol);
    compare_with_pytorch(m2->grad, pytorch_results[5], m2->size, "m2.grad", tol);
    compare_with_pytorch(m3->data, pytorch_results[6], m3->size, "m3 values", tol);

    tape_clear();

   // Test 3: Complex Graph
    printf("\n=== Test 3: Complex Graph ===\n");
    float data5[] = {9, 10, 11, 12};
    Tensor *x1 = tensor_new(2, dims, data1, 1);
    Tensor *x2 = tensor_new(2, dims, data2, 1);
    Tensor *x3 = tensor_new(2, dims, data5, 1);
    Tensor *sum = tensor_add(x1, x2);
    Tensor *y = tensor_matmul(sum, x3);
    
    backward();
    compare_with_pytorch(y->data, pytorch_results[7], y->size, "y values", tol);
    compare_with_pytorch(x1->grad, pytorch_results[8], x1->size, "x1.grad", tol);
    compare_with_pytorch(x2->grad, pytorch_results[9], x2->size, "x2.grad", tol);
    compare_with_pytorch(x3->grad, pytorch_results[10], x3->size, "x3.grad", tol);

    tape_clear();

    // Test 4: Repeated Use of Same Tensor
    printf("\n=== Test 4: Repeated Use of Same Tensor ===\n");
    float data6[] = {1, 2, 3, 4};
    Tensor *x4 = tensor_new(2, dims, data6, 1);
    Tensor *intermediate1 = tensor_matmul(x4, x4);
    Tensor *intermediate2 = tensor_matmul(x4, intermediate1);
    
    backward();
    compare_with_pytorch(x4->grad, pytorch_results[11], x4->size, "x4.grad", tol);
    compare_with_pytorch(intermediate2->data, pytorch_results[12], intermediate2->size, "intermediate2 values", tol);

    tape_clear();

    // Test 5: Deep Chain
    printf("\n=== Test 5: Deep Chain ===\n");
    float data7[] = {0.5, 0.5, 0.5, 0.5};
    Tensor *x5 = tensor_new(2, dims, data7, 1);
    Tensor *current = x5;
    for(int i = 0; i < 5; i++) {
        current = tensor_matmul(current, x5);
    }
    
    backward();
    compare_with_pytorch(x5->grad, pytorch_results[13], x5->size, "x5.grad", tol);
    compare_with_pytorch(current->data, pytorch_results[14], current->size, "final values", tol);

    tape_clear();

    // Test 6: Mixed Operations Tree
    printf("\n=== Test 6: Mixed Operations Tree ===\n");
    float data8[] = {1, 0, 0, 1};
    float data9[] = {0, 1, 1, 0};
    Tensor *x6 = tensor_new(2, dims, data8, 1);
    Tensor *x7 = tensor_new(2, dims, data9, 1);
    Tensor *branch1 = tensor_matmul(x6, x7);
    Tensor *branch2 = tensor_add(x6, x7);
    Tensor *branch3 = tensor_matmul(branch2, branch1);
    
    backward();
    compare_with_pytorch(x6->grad, pytorch_results[15], x6->size, "x6.grad", tol);
    compare_with_pytorch(x7->grad, pytorch_results[16], x7->size, "x7.grad", tol);
    compare_with_pytorch(branch3->data, pytorch_results[17], branch3->size, "branch3 values", tol);

    tape_clear();

    // Test 7: Zero Gradients and Mixed Requires Grad
    printf("\n=== Test 7: Zero Gradients and Mixed Requires Grad ===\n");
    float data10[] = {0, 0, 0, 0};
    float data11[] = {1, 1, 1, 1};
    float data12[] = {2, 2, 2, 2};
    Tensor *x8 = tensor_new(2, dims, data10, 1);
    Tensor *x9 = tensor_new(2, dims, data11, 0);  // requires_grad = false
    Tensor *x10 = tensor_new(2, dims, data12, 1);
    Tensor *result = tensor_matmul(tensor_add(x8, x9), x10);
    
    backward();
    compare_with_pytorch(x8->grad, pytorch_results[18], x8->size, "x8.grad", tol);
    compare_with_pytorch(x10->grad, pytorch_results[19], x10->size, "x10.grad", tol);
    compare_with_pytorch(result->data, pytorch_results[20], result->size, "result values", tol);

    tape_clear();

    // Test 8: Larger Matrices
    printf("\n=== Test 8: Larger Matrices ===\n");
    int dims_3x3[] = {3, 3};
    float data13[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float data14[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    Tensor *x11 = tensor_new(2, dims_3x3, data13, 1);
    Tensor *x12 = tensor_new(2, dims_3x3, data14, 1);
    Tensor *temp = tensor_add(tensor_matmul(x11, x12), x11);
    Tensor *result2 = tensor_matmul(temp, x12);
    
    backward();
    compare_with_pytorch(x11->grad, pytorch_results[21], x11->size, "x11.grad", tol);
    compare_with_pytorch(x12->grad, pytorch_results[22], x12->size, "x12.grad", tol);
    compare_with_pytorch(result2->data, pytorch_results[23], result2->size, "result2 values", tol);

    tape_clear();

    // Test 9: Complex Multi-Branch with Reuse
    printf("\n=== Test 9: Complex Multi-Branch with Reuse ===\n");
    float data15[] = {1, 2, 3, 4};
    float data16[] = {0.1, 0.2, 0.3, 0.4};
    Tensor *x13 = tensor_new(2, dims, data15, 1);
    Tensor *x14 = tensor_new(2, dims, data16, 1);
    Tensor *branch4 = tensor_matmul(x13, x14);
    Tensor *branch5 = tensor_matmul(x14, x13);
    Tensor *branch6 = tensor_add(branch4, branch5);
    Tensor *final = tensor_matmul(branch6, tensor_add(x13, x14));
    
    backward();
    compare_with_pytorch(x13->grad, pytorch_results[24], x13->size, "x13.grad", tol);
    compare_with_pytorch(x14->grad, pytorch_results[25], x14->size, "x14.grad", tol);
    compare_with_pytorch(final->data, pytorch_results[26], final->size, "final values", tol);

    printf("\nAll tests completed successfully!\n");
    return 0;
}