#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f
#define MAX_DIMS 32

typedef enum { MATMUL, EXP, LOG, ADD, RESHAPE } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static TapeEntry tape[MAX_TAPE];
static int tape_len = 0;
static Tensor* registry[MAX_TENSORS];
static int registry_len = 0;

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    t->size = 1;
    for (int i = 0; i < ndims; i++) t->size *= dims[i];
    memcpy(t->dims, dims, ndims * sizeof(int));
    t->data = malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    if ((t->requires_grad = requires_grad)) t->grad = calloc(t->size, sizeof(float));
    registry[registry_len++] = t;
    return t;
}

void clean_registry() {
    for (int i = 0; i < registry_len; i++) free(registry[i]->data), free(registry[i]->grad), free(registry[i]->dims), free(registry[i]);
    registry_len = 0;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
    int max_ndims = fmax(a->ndims, b->ndims);
    int* result_dims = malloc(max_ndims * sizeof(int));
    memcpy(result_dims, (a->ndims > b->ndims ? a : b)->dims, (max_ndims - 2) * sizeof(int));
    result_dims[max_ndims-2] = a->dims[a->ndims-2];
    result_dims[max_ndims-1] = b->dims[b->ndims-1];
    Tensor* result = tensor_new(max_ndims, result_dims, NULL, a->requires_grad || b->requires_grad);
    free(result_dims);
    int batch = result->size / (result->dims[max_ndims-2] * result->dims[max_ndims-1]);
    int M = a->dims[a->ndims-2], N = b->dims[b->ndims-1], K = a->dims[a->ndims-1];
    for (int n = 0; n < batch; n++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += a->data[n*M*K + i*K + k] * b->data[n*K*N + k*N + j];
                result->data[n*M*N + i*N + j] = sum;
            }
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, result, a, b};
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){EXP, result, a, NULL};
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){LOG, result, a, NULL};
    return result;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = a->data[i] + b->data[i];
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){ADD, result, a, b};
    return result;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* new_dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= new_dims[i];
    if (size != a->size) return NULL;
    Tensor* result = tensor_new(ndims, new_dims, a->data, a->requires_grad);
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, result, a, NULL};
    return result;
}

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* entry = &tape[t];
        Tensor *result = entry->result, *a = entry->input1, *b = entry->input2;
        switch (entry->op) {
            case MATMUL: {
                if (!a->requires_grad && !b->requires_grad) break;
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
                int batch = result->size / (M * N);
                for (int n = 0; n < batch; n++)
                    for (int i = 0; i < M; i++)
                        for (int j = 0; j < N; j++) {
                            float grad = result->grad[n*M*N + i*N + j];
                            for (int k = 0; k < K; k++) {
                                if (a->requires_grad) a->grad[n*M*K + i*K + k] += grad * b->data[n*K*N + k*N + j];
                                if (b->requires_grad) b->grad[n*K*N + k*N + j] += grad * a->data[n*M*K + i*K + k];
                            }
                        }
                break;
            }
            case ADD:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                if (b->requires_grad) for (int i = 0; i < b->size; i++) b->grad[i] += result->grad[i];
                break;
            case EXP:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i] * result->data[i];
                break;
            case LOG:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i] / fmaxf(a->data[i], MIN_LOG);
                break;
            case RESHAPE:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += result->grad[i];
                break;
        }
    }
    tape_len = 0;
}

// Only works for positive numbers
Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
}

Tensor* tensor_permute(Tensor* a, const int* p) {
    if (!a || !p || a->ndims <= 1) return a ? tensor_new(a->ndims, a->dims, a->data, a->requires_grad) : NULL;
    char u[MAX_DIMS] = {0}; int n[MAX_DIMS], s = a->size, t[2][MAX_DIMS];
    for (int i = 0; i < a->ndims; i++) {
        if (p[i] < 0 || p[i] >= a->ndims || u[p[i]]) return NULL;
        u[p[i]] = 1; n[i] = a->dims[p[i]];
    }
    float* m = calloc(s * s, sizeof(float));
    t[0][a->ndims-1] = t[1][a->ndims-1] = 1;
    for (int i = a->ndims-2; i >= 0; i--) t[0][i] = t[0][i+1] * a->dims[i+1], t[1][i] = t[1][i+1] * n[i+1];
    for (int i = 0; i < s; i++) {
        int x = i, y = 0, c[MAX_DIMS];
        for (int d = 0; d < a->ndims; d++) c[d] = (x / t[0][d]) % a->dims[d];
        for (int d = 0; d < a->ndims; d++) y += c[p[d]] * t[1][d];
        m[y * s + i] = 1;
    }
    Tensor* r = tensor_reshape(tensor_matmul(tensor_new(2, (int[]){s,s}, m, 0), tensor_reshape(a, 2, (int[]){s,1})), a->ndims, n);
    free(m);
    return r;
}

float relative_error(float a, float b) {
    return fabsf(a - b) / (fabsf(a) + fabsf(b) + 1e-8);
}

void assert_close(float a, float b, float tolerance, const char* message) {
    if (relative_error(a, b) > tolerance) {
        printf("ASSERTION FAILED: %s\n", message);
        printf("Expected: %.7f, Got: %.7f, Relative Error: %.7f\n", a, b, relative_error(a, b));
        exit(1);
    }
}

void test_matmul() {
    printf("\n=== Advanced Matrix Multiplication Tests ===\n");
    
    // Test 1: 3D batch matrix multiplication
    float a_data[] = {
        1,2,3,  // First batch
        4,5,6,
        7,8,9,
        
        9,8,7,  // Second batch
        6,5,4,
        3,2,1
    };
    float b_data[] = {
        1,2,    // First batch
        3,4,
        5,6,
        
        6,5,    // Second batch
        4,3,
        2,1
    };
    
    Tensor* a = tensor_new(3, (int[]){2,3,3}, a_data, 1);
    Tensor* b = tensor_new(3, (int[]){2,3,2}, b_data, 1);
    Tensor* c = tensor_matmul(a, b);

    // Manual verification of calculations:
    // First batch:
    // [1,2,3] · [[1,2], [3,4], [5,6]] = [1×1 + 2×3 + 3×5, 1×2 + 2×4 + 3×6] = [22,28]
    // [4,5,6] · [[1,2], [3,4], [5,6]] = [4×1 + 5×3 + 6×5, 4×2 + 5×4 + 6×6] = [49,64]
    // [7,8,9] · [[1,2], [3,4], [5,6]] = [7×1 + 8×3 + 9×5, 7×2 + 8×4 + 9×6] = [76,100]

    // Second batch:
    // [9,8,7] · [[6,5], [4,3], [2,1]] = [9×6 + 8×4 + 7×2, 9×5 + 8×3 + 7×1] = [100,76]
    // [6,5,4] · [[6,5], [4,3], [2,1]] = [6×6 + 5×4 + 4×2, 6×5 + 5×3 + 4×1] = [64,49]
    // [3,2,1] · [[6,5], [4,3], [2,1]] = [3×6 + 2×4 + 1×2, 3×5 + 2×3 + 1×1] = [28,22]
    
    printf("First batch results:\n");
    printf("Expected: [22,28] [49,64] [76,100]\n");
    printf("Got: [%.1f,%.1f] [%.1f,%.1f] [%.1f,%.1f]\n", 
           c->data[0], c->data[1], c->data[2], c->data[3], c->data[4], c->data[5]);
    
    printf("\nSecond batch results:\n");
    printf("Expected: [100,76] [64,49] [28,22]\n");
    printf("Got: [%.1f,%.1f] [%.1f,%.1f] [%.1f,%.1f]\n", 
           c->data[6], c->data[7], c->data[8], c->data[9], c->data[10], c->data[11]);

    // First batch verifications
    assert_close(c->data[0], 22, 1e-5, "Batch 1, position (0,0)");
    assert_close(c->data[1], 28, 1e-5, "Batch 1, position (0,1)");
    assert_close(c->data[2], 49, 1e-5, "Batch 1, position (1,0)");
    assert_close(c->data[3], 64, 1e-5, "Batch 1, position (1,1)");
    assert_close(c->data[4], 76, 1e-5, "Batch 1, position (2,0)");
    assert_close(c->data[5], 100, 1e-5, "Batch 1, position (2,1)");

    // Second batch verifications
    assert_close(c->data[6], 100, 1e-5, "Batch 2, position (0,0)");
    assert_close(c->data[7], 76, 1e-5, "Batch 2, position (0,1)");
    assert_close(c->data[8], 64, 1e-5, "Batch 2, position (1,0)");
    assert_close(c->data[9], 49, 1e-5, "Batch 2, position (1,1)");
    assert_close(c->data[10], 28, 1e-5, "Batch 2, position (2,0)");
    assert_close(c->data[11], 22, 1e-5, "Batch 2, position (2,1)");
    
    // Test gradient propagation
    for(int i = 0; i < c->size; i++) c->grad[i] = 1.0f;
    backward();
    
    printf("\nGradient verification:\n");
    printf("A gradients:\n");
    for(int i = 0; i < a->size; i++) {
        printf("%.1f ", a->grad[i]);
        if((i+1) % 3 == 0) printf("\n");
    }
    printf("\nB gradients:\n");
    for(int i = 0; i < b->size; i++) {
        printf("%.1f ", b->grad[i]);
        if((i+1) % 2 == 0) printf("\n");
    }
}

void test_exp_log() {
    printf("\n=== Advanced Exp and Log Tests ===\n");
    
    // Test values within stable numerical range
    float data[] = {-10.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 10.0f};
    Tensor* x = tensor_new(1, (int[]){7}, data, 1);
    
    printf("Testing exp with various values:\n");
    Tensor* exp_x = tensor_exp(x);
    for(int i = 0; i < x->size; i++) {
        float expected = expf(fminf(data[i], MAX_EXP));
        printf("exp(%.1f) = %.6f (expected: %.6f)\n", 
               data[i], exp_x->data[i], expected);
        assert_close(exp_x->data[i], expected, 1e-5, "Exp computation");
    }
    
    printf("\nTesting log with positive values:\n");
    float log_data[] = {0.1f, 1.0f, 2.0f, 10.0f, 100.0f};
    Tensor* y = tensor_new(1, (int[]){5}, log_data, 1);
    Tensor* log_y = tensor_log(y);
    for(int i = 0; i < y->size; i++) {
        float expected = logf(fmaxf(log_data[i], MIN_LOG));
        printf("log(%.1f) = %.6f (expected: %.6f)\n", 
               log_data[i], log_y->data[i], expected);
        assert_close(log_y->data[i], expected, 1e-5, "Log computation");
    }
    
    printf("\nTesting log(exp(x)) composition for stable range:\n");
    // Test composition for a single value to verify gradient
    float comp_data[] = {0.5f};
    Tensor* z = tensor_new(1, (int[]){1}, comp_data, 1);
    Tensor* exp_z = tensor_exp(z);
    Tensor* log_exp_z = tensor_log(exp_z);
    
    printf("Forward pass:\n");
    printf("x = %.6f\n", comp_data[0]);
    printf("exp(x) = %.6f\n", exp_z->data[0]);
    printf("log(exp(x)) = %.6f\n", log_exp_z->data[0]);
    assert_close(log_exp_z->data[0], comp_data[0], 1e-5, "Log-Exp composition");
    
    // Test gradient propagation
    printf("\nTesting gradient propagation:\n");
    printf("Setting output gradient to 1.0\n");
    log_exp_z->grad[0] = 1.0f;
    backward();
    
    // For log(exp(x)), the gradient should be 1.0
    printf("Input gradient = %.6f (expected: 1.0)\n", z->grad[0]);
    assert_close(z->grad[0], 1.0f, 1e-5, "Gradient through log-exp");
    
    // Additional gradient test with exp only
    printf("\nTesting exp gradient separately:\n");
    Tensor* a = tensor_new(1, (int[]){1}, (float[]){1.0f}, 1);
    Tensor* exp_a = tensor_exp(a);
    exp_a->grad[0] = 1.0f;
    backward();
    printf("exp'(1.0) = %.6f (expected: %.6f)\n", a->grad[0], expf(1.0f));
    assert_close(a->grad[0], expf(1.0f), 1e-5, "Exp gradient");
    
    // Test log gradient separately
    printf("\nTesting log gradient separately:\n");
    Tensor* b = tensor_new(1, (int[]){1}, (float[]){2.0f}, 1);
    Tensor* log_b = tensor_log(b);
    log_b->grad[0] = 1.0f;
    backward();
    printf("log'(2.0) = %.6f (expected: %.6f)\n", b->grad[0], 1.0f/2.0f);
    assert_close(b->grad[0], 1.0f/2.0f, 1e-5, "Log gradient");
}

void test_add() {
    printf("\n=== Advanced Addition Tests ===\n");
    
    // Test 3D tensor addition with various patterns
    int dims[] = {2,2,3};
    float a_data[] = {
        1.0f, -2.0f, 3.0f,
        -4.0f, 5.0f, -6.0f,
        7.0f, -8.0f, 9.0f,
        -10.0f, 11.0f, -12.0f
    };
    float b_data[] = {
        -1.0f, 2.0f, -3.0f,
        4.0f, -5.0f, 6.0f,
        -7.0f, 8.0f, -9.0f,
        10.0f, -11.0f, 12.0f
    };
    
    Tensor* a = tensor_new(3, dims, a_data, 1);
    Tensor* b = tensor_new(3, dims, b_data, 1);
    Tensor* c = tensor_add(a, b);
    
    printf("Testing addition with positive/negative patterns:\n");
    for(int i = 0; i < 12; i++) {
        printf("%.1f + %.1f = %.1f\n", a_data[i], b_data[i], c->data[i]);
        assert_close(c->data[i], a_data[i] + b_data[i], 1e-5, "Addition verification");
    }
}

void test_reshape() {
    printf("\n=== Advanced Reshape Tests ===\n");
    
    // Test complex reshape chain
    int dims[] = {2,3,2};
    float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    Tensor* x = tensor_new(3, dims, data, 1);
    
    printf("Original shape: (2,3,2)\n");
    Tensor* r1 = tensor_reshape(x, 2, (int[]){3,4});
    Tensor* r2 = tensor_reshape(r1, 1, (int[]){12});
    Tensor* r3 = tensor_reshape(r2, 4, (int[]){2,2,1,3});
    
    printf("Reshape chain: (2,3,2) -> (3,4) -> (12) -> (2,2,1,3)\n");
    printf("Verifying data continuity through reshapes:\n");
    for(int i = 0; i < 12; i++) {
        printf("Position %d: %.1f\n", i, r3->data[i]);
        assert_close(r3->data[i], data[i], 1e-5, "Reshape data continuity");
    }
}

void test_numerical_gradient() {
    printf("\n=== Numerical Gradient Verification ===\n");
    
    float eps = 1e-4;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* x = tensor_new(2, (int[]){2,2}, data, 1);
    
    // Test exp gradient
    Tensor* exp_x = tensor_exp(x);
    exp_x->grad[0] = 1.0f;
    backward();
    
    // Compute numerical gradient
    float numerical_grad = (expf(data[0] + eps) - expf(data[0])) / eps;
    assert_close(x->grad[0], numerical_grad, 1e-3, "Exp numerical gradient");
}

void test_permute() {
    printf("\n=== Advanced Permute Tests ===\n");
    
    // Test 1: 4D tensor permute with complex pattern
    printf("Test 1: 4D tensor permute (2,3,4,2) -> (4,2,2,3)\n");
    float* data4d = malloc(48 * sizeof(float));
    for(int i = 0; i < 48; i++) data4d[i] = i + 1;
    
    Tensor* t4d = tensor_new(4, (int[]){2,3,4,2}, data4d, 1);
    Tensor* p4d = tensor_permute(t4d, (int[]){2,3,0,1});
    
    printf("Original shape: (2,3,4,2), Permuted shape: (4,2,2,3)\n");
    printf("Verifying corner elements:\n");
    // Print and verify some strategic elements
    printf("Original[0,0,0,0] = %.1f -> Permuted[0,0,0,0] = %.1f\n", t4d->data[0], p4d->data[0]);
    printf("Original[1,2,3,1] = %.1f -> Permuted[3,1,1,2] = %.1f\n", 
           t4d->data[1*24 + 2*8 + 3*2 + 1], p4d->data[3*12 + 1*6 + 1*3 + 2]);
    
    // Test 2: Permute with singleton dimensions
    printf("\nTest 2: Permute with singleton dimensions (1,4,1,3) -> (4,1,3,1)\n");
    float* data_single = malloc(12 * sizeof(float));
    for(int i = 0; i < 12; i++) data_single[i] = i + 1;
    Tensor* t_single = tensor_new(4, (int[]){1,4,1,3}, data_single, 1);
    Tensor* p_single = tensor_permute(t_single, (int[]){1,0,3,2});
    printf("Verifying singleton dimension permute:\n");
    for(int i = 0; i < 12; i++) {
        printf("%.1f ", p_single->data[i]);
        if((i+1) % 3 == 0) printf("\n");
    }
    
    // Test 3: Permute with repeated application
    printf("\nTest 3: Double permute should return to original\n");
    int dims[] = {2,3,2};
    float* data3d = malloc(12 * sizeof(float));
    for(int i = 0; i < 12; i++) data3d[i] = i + 1;
    
    Tensor* t3d = tensor_new(3, dims, data3d, 1);
    Tensor* p1 = tensor_permute(t3d, (int[]){2,0,1});
    Tensor* p2 = tensor_permute(p1, (int[]){1,2,0});
    
    printf("Testing if double permute preserves values:\n");
    for(int i = 0; i < 12; i++) {
        assert_close(t3d->data[i], p2->data[i], 1e-5, "Double permute");
        if(i < 5) printf("%.1f -> %.1f -> %.1f\n", t3d->data[i], p1->data[i], p2->data[i]);
    }
    
    // Test 4: Gradient propagation through permute
    printf("\nTest 4: Gradient propagation through permute\n");
    Tensor* x = tensor_new(2, (int[]){2,3}, (float[]){1,2,3,4,5,6}, 1);
    Tensor* px = tensor_permute(x, (int[]){1,0});
    for(int i = 0; i < px->size; i++) px->grad[i] = 1.0f;
    backward();
    
    printf("Input gradients after permute:\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) printf("%.1f ", x->grad[i*3 + j]);
        printf("\n");
    }
    
    free(data4d);
    free(data_single);
    free(data3d);
}

void test_hadamard() {
    printf("\n=== Advanced Hadamard Product Tests ===\n");
    
    // Test 1: Basic Hadamard product with positive numbers
    printf("Test 1: Basic Hadamard product with positive numbers\n");
    float data1[] = {0.5f, 1.0f, 1.5f, 2.0f};  // Using positive numbers
    float data2[] = {1.0f, 1.5f, 2.0f, 2.5f};  // Using positive numbers
    Tensor* h1 = tensor_new(2, (int[]){2,2}, data1, 1);
    Tensor* h2 = tensor_new(2, (int[]){2,2}, data2, 1);
    Tensor* h_result = tensor_hadamard(h1, h2);
    
    printf("Results:\n");
    for(int i = 0; i < 4; i++) {
        float expected = data1[i] * data2[i];
        printf("%.2f * %.2f = %.2f (expected: %.2f)\n", 
               data1[i], data2[i], h_result->data[i], expected);
        assert_close(h_result->data[i], expected, 1e-4, "Basic hadamard");
    }
    
    // Test 2: Chain of Hadamard products
    printf("\nTest 2: Chain of Hadamard products\n");
    float data3[] = {1.0f, 1.2f, 1.4f, 1.6f};
    Tensor* h3 = tensor_new(2, (int[]){2,2}, data3, 1);
    Tensor* chain1 = tensor_hadamard(h3, h3);  // square
    Tensor* chain2 = tensor_hadamard(chain1, h3);  // cube
    
    printf("Original -> Square -> Cube:\n");
    for(int i = 0; i < 4; i++) {
        float expected = data3[i] * data3[i] * data3[i];
        printf("%.2f -> %.2f -> %.2f (expected: %.2f)\n", 
               data3[i], chain1->data[i], chain2->data[i], expected);
        assert_close(chain2->data[i], expected, 1e-4, "Chained hadamard");
    }
    
    // Test 3: Hadamard with ones (identity)
    printf("\nTest 3: Hadamard with ones (identity)\n");
    float ones_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float test_data[] = {1.5f, 2.0f, 2.5f, 3.0f};
    Tensor* ones = tensor_new(2, (int[]){2,2}, ones_data, 1);
    Tensor* test = tensor_new(2, (int[]){2,2}, test_data, 1);
    Tensor* id_result = tensor_hadamard(test, ones);
    
    printf("Testing identity property (x * 1 = x):\n");
    for(int i = 0; i < 4; i++) {
        printf("%.2f * 1.00 = %.2f\n", test_data[i], id_result->data[i]);
        assert_close(id_result->data[i], test_data[i], 1e-4, "Identity hadamard");
    }
    
    // Test 4: Gradient verification
    printf("\nTest 4: Gradient verification\n");
    Tensor* a = tensor_new(2, (int[]){2,2}, (float[]){1.5f, 2.0f, 2.5f, 3.0f}, 1);
    Tensor* b = tensor_new(2, (int[]){2,2}, (float[]){2.0f, 2.5f, 3.0f, 3.5f}, 1);
    Tensor* c = tensor_hadamard(a, b);
    
    // Set all gradients to 1.0
    for(int i = 0; i < 4; i++) c->grad[i] = 1.0f;
    backward();
    
    printf("Gradients:\n");
    printf("dC/dA (should be equal to B):\n");
    for(int i = 0; i < 4; i++) {
        printf("%.2f ", a->grad[i]);
        assert_close(a->grad[i], b->data[i], 1e-4, "Gradient wrt A");
    }
    printf("\ndC/dB (should be equal to A):\n");
    for(int i = 0; i < 4; i++) {
        printf("%.2f ", b->grad[i]);
        assert_close(b->grad[i], a->data[i], 1e-4, "Gradient wrt B");
    }
    printf("\n");
    
    // Test 5: Commutativity
    printf("\nTest 5: Testing commutativity (a * b = b * a)\n");
    Tensor* h_ab = tensor_hadamard(a, b);
    Tensor* h_ba = tensor_hadamard(b, a);
    
    for(int i = 0; i < 4; i++) {
        printf("%.2f = %.2f\n", h_ab->data[i], h_ba->data[i]);
        assert_close(h_ab->data[i], h_ba->data[i], 1e-4, "Commutativity");
    }
}

Tensor* tensor_reduce_sum(Tensor* a, int axis) {
    if (!a || axis < 0 || axis >= a->ndims) return NULL;
    
    int pre_size = 1;
    int axis_size = a->dims[axis];
    int post_size = 1;
    
    for (int i = 0; i < axis; i++) pre_size *= a->dims[i];
    for (int i = axis + 1; i < a->ndims; i++) post_size *= a->dims[i];
    
    // Create ones vector
    float* ones_data = malloc(axis_size * sizeof(float));
    for (int i = 0; i < axis_size; i++) ones_data[i] = 1.0f;
    
    // Shape the ones vector based on the reduction axis
    Tensor* ones;
    Tensor* reshaped;
    if (axis == 0) {
        // For axis 0, we want (1, axis_size) × (axis_size, pre_size * post_size)
        ones = tensor_new(2, (int[]){1, axis_size}, ones_data, 0);
        reshaped = tensor_reshape(a, 2, (int[]){axis_size, pre_size * post_size});
        
    } else {
        // For other axes, we want (pre_size * post_size, axis_size) × (axis_size, 1)
        ones = tensor_new(2, (int[]){axis_size, 1}, ones_data, 0);
        reshaped = tensor_reshape(a, 2, (int[]){pre_size * post_size, axis_size});
    }
    free(ones_data);
    
    // Perform matrix multiplication in the correct order
    Tensor* result = axis == 0 ? tensor_matmul(ones, reshaped) : tensor_matmul(reshaped, ones);
    
    // Calculate final shape
    int final_ndims = a->ndims - 1;
    if (final_ndims == 0) final_ndims = 1;
    
    int* final_dims = malloc(final_ndims * sizeof(int));
    int j = 0;
    for (int i = 0; i < a->ndims; i++) {
        if (i != axis) final_dims[j++] = a->dims[i];
    }
    if (j == 0) final_dims[0] = 1;  // Handle scalar result
    
    // Reshape to final shape
    Tensor* final_result = tensor_reshape(result, final_ndims, final_dims);
    free(final_dims);
    
    return final_result;
}

void test_reduce_sum() {
    printf("\n=== Reduce Sum Tests ===\n");
    
    // Test 1: Basic verification with all axes
    printf("Test 1: Basic verification with all axes\n");
    float data3d[] = {
        1.0f, 2.0f,  // [0,0,:]
        3.0f, 4.0f,  // [0,1,:]
        5.0f, 6.0f,  // [1,0,:]
        7.0f, 8.0f   // [1,1,:]
    };
    Tensor* t3d = tensor_new(3, (int[]){2,2,2}, data3d, 1);
    
    Tensor* sum0 = tensor_reduce_sum(t3d, 0);  // (2,2)
    Tensor* sum1 = tensor_reduce_sum(t3d, 1);  // (2,2)
    Tensor* sum2 = tensor_reduce_sum(t3d, 2);  // (2,2)
    
    printf("Original 3D tensor (2,2,2):\n");
    for(int i = 0; i < 2; i++) {
        printf("Layer %d:\n", i);
        for(int j = 0; j < 2; j++) {
            printf("[%.1f %.1f] ", data3d[i*4 + j*2], data3d[i*4 + j*2 + 1]);
        }
        printf("\n");
    }
    
    printf("\nReductions:\n");
    printf("Axis 0 (should be [[6,8],[10,12]]):\n");
    for(int i = 0; i < 4; i++) {
        printf("%.1f ", sum0->data[i]);
        if(i % 2 == 1) printf("\n");
        float expected = data3d[i] + data3d[i+4];
        assert_close(sum0->data[i], expected, 1e-5, "Axis 0 reduction");
    }
    
    // Test 2: Chain of reductions
    printf("\nTest 2: Chain of reductions\n");
    Tensor* chain1 = tensor_reduce_sum(t3d, 2);  // First reduce last dim
    Tensor* chain2 = tensor_reduce_sum(chain1, 1); // Then middle dim
    Tensor* chain3 = tensor_reduce_sum(chain2, 0); // Finally first dim
    
    float total = 0;
    for(int i = 0; i < 8; i++) total += data3d[i];
    
    printf("Sequential reduction result (should be %.1f): %.1f\n", 
           total, chain3->data[0]);
    assert_close(chain3->data[0], total, 1e-5, "Chain reduction");
    
    // Test 3: Gradient through multiple paths
    printf("\nTest 3: Gradient through multiple paths\n");
    Tensor* x = tensor_new(2, (int[]){2,2}, (float[]){1,2,3,4}, 1);
    Tensor* y1 = tensor_reduce_sum(x, 0);  // Sum rows
    Tensor* y2 = tensor_reduce_sum(x, 1);  // Sum columns
    Tensor* z = tensor_add(y1, y2);        // Add both reductions
    
    z->grad[0] = 1.0f;
    z->grad[1] = 1.0f;
    backward();
    
    printf("Input gradients (should be [2,2,2,2]):\n");
    for(int i = 0; i < 4; i++) {
        printf("%.1f ", x->grad[i]);
        assert_close(x->grad[i], 2.0f, 1e-5, "Multiple path gradients");
    }
    printf("\n");
    
    // Test 4: Reduction with singleton dimensions
    printf("\nTest 4: Reduction with singleton dimensions\n");
    Tensor* singleton = tensor_new(4, (int[]){2,1,2,1}, (float[]){1,2,3,4}, 1);
    Tensor* sum_singleton = tensor_reduce_sum(singleton, 1);
    
    printf("Original shape: (2,1,2,1), After reduction: (2,2,1)\n");
    printf("Values (should be unchanged): ");
    for(int i = 0; i < 4; i++) {
        printf("%.1f ", sum_singleton->data[i]);
        assert_close(sum_singleton->data[i], singleton->data[i], 1e-5, "Singleton reduction");
    }
    printf("\n");
    
    // Test 5: Reduction to scalar
    printf("\nTest 5: Reduction to scalar\n");
    Tensor* scalar = tensor_new(1, (int[]){4}, (float[]){1,2,3,4}, 1);
    Tensor* sum_scalar = tensor_reduce_sum(scalar, 0);
    
    printf("Sum of [1,2,3,4] = %.1f (should be 10)\n", sum_scalar->data[0]);
    assert_close(sum_scalar->data[0], 10.0f, 1e-5, "Scalar reduction");
    
    // Test 6: Gradient accumulation with single reduction
    printf("\nTest 6: Gradient accumulation with single reduction\n");
    Tensor* a = tensor_new(2, (int[]){2,3}, (float[]){1,2,3,4,5,6}, 1);
    Tensor* b = tensor_reduce_sum(a, 1);  // Reduce along axis 1
    b->grad[0] = 1.0f;  // Set gradient for first row sum
    b->grad[1] = 2.0f;  // Set gradient for second row sum
    backward();
    
    printf("Gradients after reduction (should be [1,1,1,2,2,2]):\n");
    for(int i = 0; i < 6; i++) {
        printf("%.1f ", a->grad[i]);
        float expected = i < 3 ? 1.0f : 2.0f;  // First three gradients should be 1.0, last three 2.0
        assert_close(a->grad[i], expected, 1e-5, "Gradient accumulation");
    }
    printf("\n");
}

int main() {
    test_matmul();
    test_exp_log();
    test_add();
    test_reshape();
    test_numerical_gradient();
    test_permute();
    test_hadamard();
    test_reduce_sum();
    
    printf("\nAll tests passed successfully!\n");
    clean_registry();
    return 0;
}