#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { ADD, MATMUL } OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape = {0};

static inline int calc_size(const int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndims = ndims;
    t->size = calc_size(dims, ndims);
    t->dims = malloc(ndims * sizeof(int));
    t->data = malloc(t->size * sizeof(float));
    
    if (!t->dims || !t->data) {
        free(t->dims); free(t->data); free(t);
        return NULL;
    }
    
    memcpy(t->dims, dims, ndims * sizeof(int));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    
    if ((t->requires_grad = requires_grad)) {
        if (!(t->grad = calloc(t->size, sizeof(float)))) {
            free(t->data); free(t->dims); free(t);
            return NULL;
        }
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (t) { free(t->data); free(t->grad); free(t->dims); free(t); }
}

static int compatible_shapes(const Tensor* a, const Tensor* b, OpType op) {
    if (op == ADD) {
        if (a->ndims != b->ndims) return 0;
        for (int i = 0; i < a->ndims; i++)
            if (a->dims[i] != b->dims[i]) return 0;
        return 1;
    }
    if (op == MATMUL) {
        if (a->ndims < 2 || b->ndims < 2 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) 
            return 0;
        int batch_dims = MIN(a->ndims-2, b->ndims-2);
        for (int i = 0; i < batch_dims; i++)
            if (a->dims[i] != b->dims[i] && a->dims[i] != 1 && b->dims[i] != 1) 
                return 0;
        return 1;
    }
    return 0;
}

static void get_output_dims(const Tensor* a, const Tensor* b, OpType op, int* out_dims, int* out_ndims) {
    *out_ndims = op == ADD ? a->ndims : MAX(a->ndims, b->ndims);
    if (op == ADD) {
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else {
        int batch_dims = *out_ndims - 2;
        for (int i = 0; i < batch_dims; i++)
            out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[*out_ndims-2] = a->dims[a->ndims-2];
        out_dims[*out_ndims-1] = b->dims[b->ndims-1];
    }
}

static void matmul_forward(const float* a, const float* b, float* out, int m, int n, int p) {
    memset(out, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++) {
            float aik = a[i * n + k];
            for (int j = 0; j < p; j++)
                out[i * p + j] += aik * b[k * p + j];
        }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!compatible_shapes(a, b, op)) return NULL;
    
    int out_dims[MAX_DIMS], out_ndims;
    get_output_dims(a, b, op, out_dims, &out_ndims);
    
    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    if (!result) return NULL;
    
    if (op == ADD) {
        for (int i = 0; i < result->size; i++)
            result->data[i] = a->data[i] + b->data[i];
    } else if (op == MATMUL) {
        int batch_size = calc_size(out_dims, out_ndims - 2);
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++)
            matmul_forward(&a->data[batch * m * n], &b->data[batch * n * p], 
                          &result->data[batch * m * p], m, n, p);
    }

    if (result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, a, b};
    
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)

static void backward_op(TapeEntry* entry) {
    Tensor *t = entry->result, *a = entry->input1, *b = entry->input2;
    
    if (entry->op == ADD) {
        if (a->requires_grad)
            for (int i = 0; i < a->size; i++) 
                a->grad[i] += t->grad[i];
        if (b->requires_grad)
            for (int i = 0; i < b->size; i++) 
                b->grad[i] += t->grad[i];
    } else if (entry->op == MATMUL) {
        int batch_size = calc_size(t->dims, t->ndims - 2);
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++) {
            int a_off = batch * m * n, b_off = batch * n * p, t_off = batch * m * p;
            
            if (a->requires_grad)
                for (int i = 0; i < m; i++)
                    for (int k = 0; k < n; k++) {
                        float sum = 0;
                        for (int j = 0; j < p; j++)
                            sum += t->grad[t_off + i * p + j] * b->data[b_off + k * p + j];
                        a->grad[a_off + i * n + k] += sum;
                    }
            
            if (b->requires_grad)
                for (int k = 0; k < n; k++)
                    for (int j = 0; j < p; j++) {
                        float sum = 0;
                        for (int i = 0; i < m; i++)
                            sum += t->grad[t_off + i * p + j] * a->data[a_off + i * n + k];
                        b->grad[b_off + k * p + j] += sum;
                    }
        }
    }
}

void backward() {
    if (tape.len > 0) {
        Tensor* final = tape.entries[tape.len - 1].result;
        if (!final->grad) final->grad = calloc(final->size, sizeof(float));
        for (int i = 0; i < final->size; i++) 
            final->grad[i] = 1.0f;
        
        for (int i = tape.len - 1; i >= 0; i--)
            backward_op(&tape.entries[i]);
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s (dims:", name);
    for (int i = 0; i < t->ndims; i++) 
        printf(" %d", t->dims[i]);
    printf("):\n");
    
    int rows = t->dims[0], cols = t->dims[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
            printf("%f ", t->data[i * cols + j]);
        printf("\n");
    }
    
    if (t->grad) {
        printf("Gradients:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) 
                printf("%f ", t->grad[i * cols + j]);
            printf("\n");
        }
    }
    printf("\n");
}

#include <math.h>

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

void tape_clear() { tape.len = 0; }

int main() {
    FILE* f = fopen("compare.py", "w");
    fprintf(f, "import torch\n\n");
    
    // Test 1: Edge Case - Very Small Numbers
    fprintf(f, "# Test 1: Edge Case - Very Small Numbers\n");
    fprintf(f, "a = torch.tensor([[[1e-7, 2e-7], [3e-7, 4e-7]], [[5e-7, 6e-7], [7e-7, 8e-7]]], requires_grad=True)\n");
    fprintf(f, "b = torch.tensor([[[1e-7, 2e-7], [3e-7, 4e-7]], [[5e-7, 6e-7], [7e-7, 8e-7]]], requires_grad=True)\n");
    fprintf(f, "c = torch.matmul(a, b)\n");
    fprintf(f, "c.backward(torch.ones_like(c))\n");
    fprintf(f, "print('TEST1_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, a.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, b.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, c.detach().numpy().flatten())))\n");

    // Test 2: Edge Case - Very Large Numbers
    fprintf(f, "\n# Test 2: Edge Case - Very Large Numbers\n");
    fprintf(f, "x1 = torch.tensor([[[1e5, 2e5], [3e5, 4e5]], [[5e5, 6e5], [7e5, 8e5]]], requires_grad=True)\n");
    fprintf(f, "x2 = torch.tensor([[[1e-5, 2e-5], [3e-5, 4e-5]], [[5e-5, 6e-5], [7e-5, 8e-5]]], requires_grad=True)\n");
    fprintf(f, "y = x1 + x2\n");
    fprintf(f, "z = torch.matmul(y, x1)\n");
    fprintf(f, "z.backward(torch.ones_like(z))\n");
    fprintf(f, "print('TEST2_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, x1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, x2.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, z.detach().numpy().flatten())))\n");

    // Test 3: Complex Graph with Mixed Operations
    fprintf(f, "\n# Test 3: Complex Graph with Mixed Operations\n");
    fprintf(f, "m1 = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], requires_grad=True)\n");
    fprintf(f, "m2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)\n");
    fprintf(f, "m3 = torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]], requires_grad=True)\n");
    fprintf(f, "temp1 = torch.matmul(m1, m2)\n");
    fprintf(f, "temp2 = torch.matmul(m2, m3)\n");
    fprintf(f, "temp3 = temp1 + temp2\n");
    fprintf(f, "result = torch.matmul(temp3, torch.matmul(m1, m3))\n");
    fprintf(f, "result.backward(torch.ones_like(result))\n");
    fprintf(f, "print('TEST3_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, m1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, m2.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, m3.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, result.detach().numpy().flatten())))\n");

    // Test 4: Deep Chain with Alternating Operations
    fprintf(f, "\n# Test 4: Deep Chain with Alternating Operations\n");
    fprintf(f, "d1 = torch.tensor([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]], requires_grad=True)\n");
    fprintf(f, "d2 = torch.tensor([[[0.2, 0.2], [0.2, 0.2]], [[0.2, 0.2], [0.2, 0.2]]], requires_grad=True)\n");
    fprintf(f, "current = d1\n");
    fprintf(f, "for i in range(5):\n");
    fprintf(f, "    if i % 2 == 0:\n");
    fprintf(f, "        current = torch.matmul(current, d2)\n");
    fprintf(f, "    else:\n");
    fprintf(f, "        current = current + d1\n");
    fprintf(f, "current.backward(torch.ones_like(current))\n");
    fprintf(f, "print('TEST4_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, d1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, d2.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, current.detach().numpy().flatten())))\n");

    // Test 5: Identity and Zero Matrices
    fprintf(f, "\n# Test 5: Identity and Zero Matrices\n");
    fprintf(f, "i1 = torch.eye(2).reshape(1, 2, 2).repeat(2, 1, 1).requires_grad_(True)\n");
    fprintf(f, "z1 = torch.zeros(2, 2, 2).requires_grad_(True)\n");
    fprintf(f, "temp = torch.matmul(i1, z1)\n");
    fprintf(f, "final = temp + i1\n");
    fprintf(f, "final.backward(torch.ones_like(final))\n");
    fprintf(f, "print('TEST5_RESULTS')\n");
    fprintf(f, "print(' '.join(map(str, i1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, z1.grad.numpy().flatten())))\n");
    fprintf(f, "print(' '.join(map(str, final.detach().numpy().flatten())))\n");

    fclose(f);

    // Run Python and collect results
    FILE* pipe = popen("python3 compare.py", "r");
    if (!pipe) {
        printf("Failed to run Python comparison\n");
        return 1;
    }

    char buffer[1024];
    char pytorch_results[25][1024];  // Increased size for more results
    int result_idx = -1;
    int current_section = 0;
    
    while (fgets(buffer, sizeof(buffer), pipe)) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (strstr(buffer, "_RESULTS")) {
            if (strstr(buffer, "TEST1")) current_section = 0;
            else if (strstr(buffer, "TEST2")) current_section = 3;
            else if (strstr(buffer, "TEST3")) current_section = 6;
            else if (strstr(buffer, "TEST4")) current_section = 10;
            else if (strstr(buffer, "TEST5")) current_section = 13;
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

    printf("Running comprehensive tensor tests...\n\n");
    float tol = 1e-5;

    // Test 1: Very Small Numbers
    printf("=== Test 1: Edge Case - Very Small Numbers ===\n");
    int dims[] = {2, 2, 2};
    float data1[8], data2[8];
    for(int i = 0; i < 8; i++) {
        data1[i] = (i + 1) * 1e-7f;
        data2[i] = (i + 1) * 1e-7f;
    }
    
    Tensor *a = tensor_new(3, dims, data1, 1);
    Tensor *b = tensor_new(3, dims, data2, 1);
    Tensor *c = tensor_matmul(a, b);
    
    backward();
    compare_with_pytorch(a->grad, pytorch_results[0], a->size, "a.grad (small numbers)", tol);
    compare_with_pytorch(b->grad, pytorch_results[1], b->size, "b.grad (small numbers)", tol);
    compare_with_pytorch(c->data, pytorch_results[2], c->size, "c values (small numbers)", tol);

    tape_clear();
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);

        // Test 2: Very Large Numbers
    printf("\n=== Test 2: Edge Case - Very Large Numbers ===\n");
    float data3[8], data4[8];
    for(int i = 0; i < 8; i++) {
        data3[i] = (i + 1) * 1e5f;
        data4[i] = (i + 1) * 1e-5f;
    }
    
    Tensor *x1 = tensor_new(3, dims, data3, 1);
    Tensor *x2 = tensor_new(3, dims, data4, 1);
    Tensor *y = tensor_add(x1, x2);
    Tensor *z = tensor_matmul(y, x1);
    
    backward();
    compare_with_pytorch(x1->grad, pytorch_results[3], x1->size, "x1.grad (large numbers)", 1e5);
    compare_with_pytorch(x2->grad, pytorch_results[4], x2->size, "x2.grad (large numbers)", 1e5);
    compare_with_pytorch(z->data, pytorch_results[5], z->size, "z values (large numbers)", 1e5);

    tape_clear();
    tensor_free(x1);
    tensor_free(x2);
    tensor_free(y);
    tensor_free(z);

    // Test 3: Complex Graph with Mixed Operations
    printf("\n=== Test 3: Complex Graph with Mixed Operations ===\n");
    float data5[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float data6[] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
    float data7[] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    
    Tensor *m1 = tensor_new(3, dims, data5, 1);
    Tensor *m2 = tensor_new(3, dims, data6, 1);
    Tensor *m3 = tensor_new(3, dims, data7, 1);
    
    Tensor *temp1 = tensor_matmul(m1, m2);
    Tensor *temp2 = tensor_matmul(m2, m3);
    Tensor *temp3 = tensor_add(temp1, temp2);
    Tensor *inner_prod = tensor_matmul(m1, m3);
    Tensor *result = tensor_matmul(temp3, inner_prod);
    
    backward();
    compare_with_pytorch(m1->grad, pytorch_results[6], m1->size, "m1.grad (complex)", tol);
    compare_with_pytorch(m2->grad, pytorch_results[7], m2->size, "m2.grad (complex)", tol);
    compare_with_pytorch(m3->grad, pytorch_results[8], m3->size, "m3.grad (complex)", tol);
    compare_with_pytorch(result->data, pytorch_results[9], result->size, "result values (complex)", tol);

    tape_clear();
    tensor_free(m1);
    tensor_free(m2);
    tensor_free(m3);
    tensor_free(temp1);
    tensor_free(temp2);
    tensor_free(temp3);
    tensor_free(inner_prod);
    tensor_free(result);

// Test 4: Deep Chain with Alternating Operations
printf("\n=== Test 4: Deep Chain with Alternating Operations ===\n");
float data8[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
float data9[] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

printf("Creating initial tensors...\n");
Tensor *d1 = tensor_new(3, dims, data8, 1);
Tensor *d2 = tensor_new(3, dims, data9, 1);
Tensor *current = d1;

// Store intermediate results to free them after backward
Tensor* intermediates[5] = {NULL};
int intermediate_count = 0;

printf("Starting loop...\n");
for(int i = 0; i < 5; i++) {
    printf("Iteration %d\n", i);
    Tensor* new_tensor;
    if(i % 2 == 0) {
        printf("Performing matmul...\n");
        new_tensor = tensor_matmul(current, d2);
    } else {
        printf("Performing add...\n");
        new_tensor = tensor_add(current, d1);
    }
    if(!new_tensor) {
        printf("Operation failed!\n");
        exit(1);
    }
    if(i > 0) {
        // Store the previous intermediate result
        intermediates[intermediate_count++] = current;
    }
    current = new_tensor;
    printf("Iteration %d complete\n", i);
}

printf("Starting backward pass...\n");
backward();
printf("Backward pass complete\n");

printf("Comparing results...\n");
compare_with_pytorch(d1->grad, pytorch_results[10], d1->size, "d1.grad (chain)", tol);
compare_with_pytorch(d2->grad, pytorch_results[11], d2->size, "d2.grad (chain)", tol);
compare_with_pytorch(current->data, pytorch_results[12], current->size, "current values (chain)", tol);

printf("Cleaning up...\n");
// Free intermediate tensors (excluding the final result)
for(int i = 0; i < intermediate_count; i++) {
    tensor_free(intermediates[i]);
}
tape_clear();
tensor_free(d1);
tensor_free(d2);
tensor_free(current);  // Free the final result

    // Test 5: Identity and Zero Matrices
    printf("\n=== Test 5: Identity and Zero Matrices ===\n");
    float identity[] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
    float zeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    Tensor *i1 = tensor_new(3, dims, identity, 1);
    Tensor *z1 = tensor_new(3, dims, zeros, 1);
    Tensor *temp = tensor_matmul(i1, z1);
    Tensor *final = tensor_add(temp, i1);
    
    backward();
    compare_with_pytorch(i1->grad, pytorch_results[13], i1->size, "i1.grad (identity)", tol);
    compare_with_pytorch(z1->grad, pytorch_results[14], z1->size, "z1.grad (zeros)", tol);
    compare_with_pytorch(final->data, pytorch_results[15], final->size, "final values (identity)", tol);

    tape_clear();
    tensor_free(i1);
    tensor_free(z1);
    tensor_free(temp);
    tensor_free(final);

    printf("\nAll comprehensive tests completed successfully!\n");
    return 0;
}