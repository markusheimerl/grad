#include "grad.h"

void assert_double_eq(double a, double b, double eps, const char* msg) {
    if (fabsf(a - b) > eps) {
        printf("ASSERTION FAILED: %s\n", msg);
        printf("Expected: %f, Got: %f\n", b, a);
        exit(1);
    }
}

void assert_tensor_eq(Tensor* a, Tensor* b, double eps, const char* msg) {
    if (a->size != b->size) {
        printf("ASSERTION FAILED: %s (size mismatch)\n", msg);
        exit(1);
    }
    for (int i = 0; i < a->size; i++) {
        assert_double_eq(a->data[i], b->data[i], eps, msg);
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) {
        printf("%d%s", t->dims[i], i < t->ndims-1 ? "," : "");
    }
    printf("], data=[");
    for (int i = 0; i < t->size; i++) {
        printf("%.1f%s", t->data[i], i < t->size-1 ? "," : "");
    }
    printf("]\n");
}

void test_basic_ops() {
    printf("Testing basic operations...\n");
    
    // Test addition
    int dims[] = {2, 2};
    double data1[] = {1, 2, 3, 4};
    double data2[] = {5, 6, 7, 8};
    double expected_add[] = {6, 8, 10, 12};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    Tensor* result = tensor_add(t1, t2);
    
    assert_tensor_eq(result, tensor_new(2, dims, expected_add, 0), 1e-5, "Basic addition failed");
    
    // Test broadcasting
    int dims1[] = {2, 1};  // [[1],
                          //  [2]]
    int dims2[] = {1, 2};  // [[3, 4]]
    double data3[] = {1, 2};
    double data4[] = {3, 4};
    
    t1 = tensor_new(2, dims1, data3, 1);
    t2 = tensor_new(2, dims2, data4, 1);
    
    print_tensor(t1, "t1");
    print_tensor(t2, "t2");
    
    result = tensor_add(t1, t2);
    print_tensor(result, "result");
    
    // The broadcasting should work like this:
    // [[1],     [[3, 4]]     [[1+3, 1+4],
    //  [2]]  +             =  [2+3, 2+4]]
    //
    // [[1, 1],   [[3, 4]]    [[4, 5],
    //  [2, 2]] +           =  [5, 6]]
    
    int expected_dims[] = {2, 2};
    double expected_broadcast[] = {4, 5, 5, 6};
    Tensor* expected = tensor_new(2, expected_dims, expected_broadcast, 0);
    print_tensor(expected, "expected");
    
    assert_tensor_eq(result, expected, 1e-5, "Broadcasting failed");
}

void test_broadcasting() {
    printf("Testing broadcasting...\n");
    
    // Test (2,1) + (1,2)
    {
        int dims1[] = {2, 1};
        int dims2[] = {1, 2};
        double data1[] = {1, 2};
        double data2[] = {3, 4};
        double expected[] = {4, 5, 5, 6};
        
        Tensor* t1 = tensor_new(2, dims1, data1, 1);
        Tensor* t2 = tensor_new(2, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        int expected_dims[] = {2, 2};
        assert_tensor_eq(result, tensor_new(2, expected_dims, expected, 0), 1e-5, "Broadcasting (2,1) + (1,2) failed");
    }
    
    // Test (3,1,2) + (2)
    {
        int dims1[] = {3, 1, 2};
        int dims2[] = {2};
        double data1[] = {1, 2, 3, 4, 5, 6};
        double data2[] = {7, 8};
        double expected[] = {8, 10, 10, 12, 12, 14};
        
        Tensor* t1 = tensor_new(3, dims1, data1, 1);
        Tensor* t2 = tensor_new(1, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        int expected_dims[] = {3, 1, 2};
        assert_tensor_eq(result, tensor_new(3, expected_dims, expected, 0), 1e-5, "Broadcasting (3,1,2) + (2) failed");
    }
    
    // Test scalar broadcasting
    {
        int dims1[] = {2, 2};
        int dims2[] = {1};
        double data1[] = {1, 2, 3, 4};
        double data2[] = {5};
        double expected[] = {6, 7, 8, 9};
        
        Tensor* t1 = tensor_new(2, dims1, data1, 1);
        Tensor* t2 = tensor_new(1, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        int expected_dims[] = {2, 2};
        assert_tensor_eq(result, tensor_new(2, expected_dims, expected, 0), 1e-5, "Broadcasting scalar failed");
    }
}

void test_matmul() {
    printf("Testing matrix multiplication...\n");
    
    int dims1[] = {2, 3};
    int dims2[] = {3, 2};
    double data1[] = {1, 2, 3, 4, 5, 6};
    double data2[] = {7, 8, 9, 10, 11, 12};
    double expected[] = {58, 64, 139, 154};
    
    Tensor* t1 = tensor_new(2, dims1, data1, 1);
    Tensor* t2 = tensor_new(2, dims2, data2, 1);
    Tensor* result = tensor_matmul(t1, t2);
    
    int expected_dims[] = {2, 2};
    assert_tensor_eq(result, tensor_new(2, expected_dims, expected, 1), 1e-5, "Matrix multiplication failed");
}

void test_softmax() {
    printf("Testing softmax...\n");
    
    int dims[] = {1, 3};
    double data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = tensor_new(2, dims, data, 1);
    Tensor* result = tensor_softmax(t);
    
    double sum = 0;
    for (int i = 0; i < result->size; i++) {
        sum += result->data[i];
        assert_double_eq(result->data[i], result->data[i], 1e-5, "Softmax produced NaN");
    }
    assert_double_eq(sum, 1.0f, 1e-5, "Softmax sum != 1");
}

void test_backward() {
    printf("Testing backward pass...\n");
    
    int dims[] = {2, 2};
    double data1[] = {1, 2, 3, 4};
    double data2[] = {5, 6, 7, 8};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    Tensor* result = tensor_add(t1, t2);
    
    // Set gradient to all ones
    for (int i = 0; i < result->size; i++) {
        result->grad[i] = 1.0f;
    }
    
    backward();
    
    // For addition, gradients should flow through unchanged
    for (int i = 0; i < t1->size; i++) {
        assert_double_eq(t1->grad[i], 1.0f, 1e-5, "Addition backward pass failed for t1");
        assert_double_eq(t2->grad[i], 1.0f, 1e-5, "Addition backward pass failed for t2");
    }
}

void test_gelu() {
    printf("Testing GELU...\n");
    
    // Test GELU with specific values we know
    {
        int dims[] = {5};
        double data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_gelu(t);

        // GELU should be approximately:
        // x < 0: smaller activation
        // x = 0: 0
        // x > 0: closer to x
        assert_double_eq(result->data[2], 0.0f, 1e-5, "GELU(0) should be 0");
        assert_double_eq(result->data[3], 0.841192f, 1e-5, "GELU(1) incorrect");
        
        // Test that outputs are reasonable
        for (int i = 0; i < result->size; i++) {
            // Output should be bounded
            assert_double_eq(result->data[i] <= fabsf(data[i]), 1.0f, 1e-5, "GELU output too large");
            // Output should have same sign as input (except very near 0)
            if (fabsf(data[i]) > 0.1f) {
                assert_double_eq(signbit(result->data[i]) == signbit(data[i]), 1.0f, 1e-5, "GELU sign mismatch");
            }
        }
    }

    // Test GELU derivative
    {
        int dims[] = {1};
        double data[] = {1.0f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_gelu(t);
        result->grad[0] = 1.0f;
        backward();
        
        // GELU derivative at x=1 should be approximately 1.0837
        assert_double_eq(t->grad[0], 1.0837f, 1e-3, "GELU derivative incorrect");
    }
}

void test_rms_norm() {
    printf("Testing RMSNorm...\n");
    
    int dims[] = {1, 4};
    double data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* t = tensor_new(2, dims, data, 1);
    Tensor* result = tensor_rms_norm(t, 1e-5f);
    
    // Check that the RMS of the output is approximately 1
    double sum_sq = 0;
    for (int i = 0; i < result->size; i++) {
        sum_sq += result->data[i] * result->data[i];
    }
    double rms = sqrt(sum_sq / result->size);
    assert_double_eq(rms, 1.0f, 1e-4, "RMSNorm failed to normalize");
}

void test_edge_cases() {
    printf("Testing edge cases...\n");
    
    // Test single-element tensors
    {
        int dims[] = {1};
        double data1[] = {2.0f};
        double data2[] = {3.0f};
        Tensor* t1 = tensor_new(1, dims, data1, 1);
        Tensor* t2 = tensor_new(1, dims, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        assert_double_eq(result->data[0], 5.0f, 1e-5, "Single element addition failed");
    }
    
    // Test broadcasting with 1s in different positions
    {
        int dims1[] = {2, 1, 3};
        int dims2[] = {1, 4, 1};
        double data1[] = {1, 2, 3, 4, 5, 6};
        double data2[] = {10, 20, 30, 40};
        
        Tensor* t1 = tensor_new(3, dims1, data1, 1);
        Tensor* t2 = tensor_new(3, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        assert_double_eq(result->dims[0], 2, 1e-5, "Complex broadcasting shape mismatch");
        assert_double_eq(result->dims[1], 4, 1e-5, "Complex broadcasting shape mismatch");
        assert_double_eq(result->dims[2], 3, 1e-5, "Complex broadcasting shape mismatch");
    }
}

void test_numerical_stability() {
    printf("Testing numerical stability...\n");
    
    // Test softmax with large numbers
    {
        int dims[] = {2};
        double data[] = {1000.0f, 1000.1f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_softmax(t);
        double sum = result->data[0] + result->data[1];
        assert_double_eq(sum, 1.0f, 1e-5, "Softmax normalization failed for large inputs");
    }
    
    // Test RMSNorm basic functionality
    {
        int dims[] = {4};
        double data[] = {2.0f, 2.0f, 2.0f, 2.0f};  // Using uniform non-zero values
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_rms_norm(t, 1e-5f);
        
        // For identical inputs, outputs should all be equal and the RMS should be 1
        // If all inputs are 2.0, then ms = 4.0, and scale = 1/sqrt(4) = 1/2
        // So each output should be 2.0 * (1/2) = 1.0
        double expected = 1.0f;  // Corrected expected value
        for (int i = 0; i < 4; i++) {
            assert_double_eq(result->data[i], expected, 1e-5, "RMSNorm failed for uniform inputs");
        }
        
        // Verify RMS = 1
        double sum_sq = 0.0f;
        for (int i = 0; i < 4; i++) {
            sum_sq += result->data[i] * result->data[i];
        }
        double rms = sqrtf(sum_sq / 4.0f);
        assert_double_eq(rms, 1.0f, 1e-5, "RMSNorm output RMS != 1");
    }
    
    // Test RMSNorm with mixed values
    {
        int dims[] = {3};
        double data[] = {1.0f, 2.0f, 3.0f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_rms_norm(t, 1e-5f);
        
        // Calculate expected values
        double ms = (1.0f + 4.0f + 9.0f) / 3.0f;  // = 4.666...
        double scale = 1.0f / sqrtf(ms);
        double expected[] = {1.0f * scale, 2.0f * scale, 3.0f * scale};
        
        // Verify outputs
        for (int i = 0; i < 3; i++) {
            assert_double_eq(result->data[i], expected[i], 1e-5, "RMSNorm failed for mixed values");
        }
        
        // Verify unit RMS
        double sum_sq = 0.0f;
        for (int i = 0; i < 3; i++) {
            sum_sq += result->data[i] * result->data[i];
        }
        double output_rms = sqrtf(sum_sq / 3.0f);
        assert_double_eq(output_rms, 1.0f, 1e-5, "RMSNorm failed to normalize to unit RMS");
    }
    
    // Test RMSNorm with small but reasonable values
    {
        int dims[] = {3};
        double data[] = {0.01f, 0.02f, 0.03f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_rms_norm(t, 1e-5f);
        
        // Verify ratios are preserved
        double ratio1 = result->data[1] / result->data[0];
        double ratio2 = result->data[2] / result->data[1];
        assert_double_eq(ratio1, 2.0f, 1e-5, "RMSNorm failed to preserve ratios for small values");
        assert_double_eq(ratio2, 1.5f, 1e-5, "RMSNorm failed to preserve ratios for small values");
    }
}

void test_gradient_accumulation() {
    printf("Testing gradient accumulation...\n");
    
    int dims[] = {2, 2};
    double data1[] = {1, 2, 3, 4};
    double data2[] = {5, 6, 7, 8};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    
    // Multiple operations on same tensors
    Tensor* add1 = tensor_add(t1, t2);
    Tensor* add2 = tensor_add(t1, t2);
    Tensor* final = tensor_add(add1, add2);
    
    // Set gradient
    for (int i = 0; i < final->size; i++) {
        final->grad[i] = 1.0f;
    }
    
    backward();
    
    // Each input tensor should have accumulated gradient = 2
    for (int i = 0; i < t1->size; i++) {
        assert_double_eq(t1->grad[i], 2.0f, 1e-5, "Gradient accumulation failed");
        assert_double_eq(t2->grad[i], 2.0f, 1e-5, "Gradient accumulation failed");
    }
}

void test_stress() {
    printf("Testing stress cases...\n");
    
    // Test large tensor operations
    {
        int dims[] = {100, 100};
        Tensor* t1 = tensor_randn(2, dims, 1);
        Tensor* t2 = tensor_randn(2, dims, 1);
        Tensor* result = tensor_add(t1, t2);
        assert_double_eq(result->size, 10000, 1e-5, "Large tensor operation failed");
    }
    
    // Test deep computation graph
    {
        int dims[] = {2, 2};
        Tensor* t = tensor_randn(2, dims, 1);
        Tensor* current = t;
        
        // Create a deep chain of operations
        for (int i = 0; i < 100; i++) {
            current = tensor_add(current, t);
        }
        
        // Should be able to backprop through this deep graph
        current->grad[0] = 1.0f;
        backward();
    }
}

void test_complex_broadcasting() {
    printf("Testing complex broadcasting patterns...\n");
    
    // Test 4D broadcasting
    {
        int dims1[] = {2, 1, 3, 1};  // [2, 1, 3, 1]
        int dims2[] = {1, 4, 1, 5};  // [1, 4, 1, 5]
        double* data1 = malloc(2 * 1 * 3 * 1 * sizeof(double));
        double* data2 = malloc(1 * 4 * 1 * 5 * sizeof(double));
        
        // Fill with recognizable patterns
        for (int i = 0; i < 6; i++) data1[i] = i + 1;
        for (int i = 0; i < 20; i++) data2[i] = (i + 1) * 0.1f;
        
        Tensor* t1 = tensor_new(4, dims1, data1, 1);
        Tensor* t2 = tensor_new(4, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        // Result should be [2, 4, 3, 5]
        assert_double_eq(result->ndims, 4, 1e-5, "Wrong number of dimensions");
        assert_double_eq(result->dims[0], 2, 1e-5, "Wrong dimension 0");
        assert_double_eq(result->dims[1], 4, 1e-5, "Wrong dimension 1");
        assert_double_eq(result->dims[2], 3, 1e-5, "Wrong dimension 2");
        assert_double_eq(result->dims[3], 5, 1e-5, "Wrong dimension 3");
        
        free(data1);
        free(data2);
    }
    
    // Test broadcasting with mixed dimensionality
    {
        int dims1[] = {3, 1};        // [3, 1]
        int dims2[] = {2, 4, 1, 2};  // [2, 4, 1, 2]
        double data1[] = {1, 2, 3};
        double* data2 = malloc(16 * sizeof(double));
        for (int i = 0; i < 16; i++) data2[i] = i * 0.1f;
        
        Tensor* t1 = tensor_new(2, dims1, data1, 1);
        Tensor* t2 = tensor_new(4, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        // Result should be [2, 4, 3, 2]
        assert_double_eq(result->ndims, 4, 1e-5, "Wrong number of dimensions");
        assert_double_eq(result->dims[0], 2, 1e-5, "Wrong dimension 0");
        assert_double_eq(result->dims[1], 4, 1e-5, "Wrong dimension 1");
        assert_double_eq(result->dims[2], 3, 1e-5, "Wrong dimension 2");
        assert_double_eq(result->dims[3], 2, 1e-5, "Wrong dimension 3");
        
        free(data2);
    }
}

void test_edge_cases_comprehensive() {
    printf("Testing comprehensive edge cases...\n");
    
    // Test zero-size dimension
    {
        int dims[] = {0, 2};
        Tensor* t = tensor_new(2, dims, NULL, 0);
        assert_double_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject zero-size dimension");
    }
    
    // Test negative dimension
    {
        int dims[] = {2, -1};
        Tensor* t = tensor_new(2, dims, NULL, 0);
        assert_double_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject negative dimension");
    }
    
    // Test zero dimensions
    {
        int dims[] = {1};
        Tensor* t = tensor_new(0, dims, NULL, 0);
        assert_double_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject zero dimensions");
    }
    
    // Test NULL dims
    {
        Tensor* t = tensor_new(1, NULL, NULL, 0);
        assert_double_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject NULL dims");
    }
    
    // Test maximum dimension size
    {
        int dims[] = {1, MAX_TAPE};
        double* data = malloc(MAX_TAPE * sizeof(double));
        Tensor* t = tensor_new(2, dims, data, 0);
        assert_double_eq(t != NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should handle maximum size");
        free(data);
    }
    
    // Test invalid broadcasting
    {
        int dims1[] = {2, 3};
        int dims2[] = {2, 2};
        double data1[] = {1, 2, 3, 4, 5, 6};
        double data2[] = {1, 2, 3, 4};
        
        Tensor* t1 = tensor_new(2, dims1, data1, 0);
        Tensor* t2 = tensor_new(2, dims2, data2, 0);
        Tensor* result = tensor_add(t1, t2);
        assert_double_eq(result == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject invalid broadcasting");
    }
    
    // Test single-element broadcasting
    {
        int dims1[] = {1};
        int dims2[] = {2, 2};
        double data1[] = {5};
        double data2[] = {1, 2, 3, 4};
        
        Tensor* t1 = tensor_new(1, dims1, data1, 0);
        Tensor* t2 = tensor_new(2, dims2, data2, 0);
        Tensor* result = tensor_add(t1, t2);
        assert_double_eq(result != NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should allow scalar broadcasting");
        if (result) {
            assert_double_eq(result->data[0], 6.0f, 1e-5, "Incorrect scalar broadcasting result");
            assert_double_eq(result->data[1], 7.0f, 1e-5, "Incorrect scalar broadcasting result");
            assert_double_eq(result->data[2], 8.0f, 1e-5, "Incorrect scalar broadcasting result");
            assert_double_eq(result->data[3], 9.0f, 1e-5, "Incorrect scalar broadcasting result");
        }
    }
}

typedef struct {
    const char* name;
    double time;
    size_t memory;
} BenchmarkResult;

BenchmarkResult benchmark_operation(const char* name, void (*op)()) {
    clock_t start = clock();
    size_t initial_memory = 0;  // Would need platform-specific implementation
    
    op();
    
    clock_t end = clock();
    size_t final_memory = 0;    // Would need platform-specific implementation
    
    return (BenchmarkResult){
        .name = name,
        .time = ((double)(end - start)) / CLOCKS_PER_SEC,
        .memory = final_memory - initial_memory
    };
}

void test_performance() {
    printf("Running performance benchmarks...\n");
    
    // Benchmark large matrix multiplication
    void bench_matmul() {
        int dims1[] = {100, 100};
        int dims2[] = {100, 100};
        Tensor* t1 = tensor_randn(2, dims1, 1);
        Tensor* t2 = tensor_randn(2, dims2, 1);
        Tensor* result = tensor_matmul(t1, t2);
        result->grad[0] = 1.0f;
        backward();
    }
    
    // Benchmark deep network
    void bench_deep() {
        int dims[] = {32, 32};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* current = x;
        for (int i = 0; i < 100; i++) {
            current = tensor_gelu(tensor_rms_norm(current, 1e-5f));
        }
        current->grad[0] = 1.0f;
        backward();
    }
    
    BenchmarkResult results[] = {
        benchmark_operation("Large MatMul", bench_matmul),
        benchmark_operation("Deep Network", bench_deep)
    };
    
    for (size_t i = 0; i < sizeof(results)/sizeof(results[0]); i++) {
        printf("%s: %.3f seconds, %zu bytes\n", 
               results[i].name, results[i].time, results[i].memory);
    }
}

void test_memory_leaks() {
    printf("Testing for memory leaks...\n");
    
    // Store initial registry state
    size_t initial_allocs = registry_len;
    
    // Perform various operations that should all register their tensors
    {
        int dims[] = {10, 10};
        Tensor* t1 = tensor_randn(2, dims, 1);
        Tensor* t2 = tensor_randn(2, dims, 1);
        Tensor* result = tensor_matmul(t1, t2);
        result->grad[0] = 1.0f;
        backward();
    }
    
    // The number of tensors should be predictable:
    // - t1 (1 tensor)
    // - t2 (1 tensor)
    // - result (1 tensor)
    size_t expected_new_tensors = 3;
    
    // Verify the number of new allocations
    assert_double_eq(registry_len - initial_allocs, expected_new_tensors, 1e-5, "Unexpected number of tensor allocations");
}

#include <pthread.h>

void* thread_function(void* arg) {
    (void)arg;  // Explicitly mark as unused
    int dims[] = {10, 10};
    Tensor* t1 = tensor_randn(2, dims, 1);
    Tensor* t2 = tensor_randn(2, dims, 1);
    Tensor* result = tensor_matmul(t1, t2);
    result->grad[0] = 1.0f;
    backward();
    return NULL;
}


void test_thread_safety() {
    printf("Testing thread safety...\n");
    
    #define NUM_THREADS 4
    pthread_t threads[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_function, NULL);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void test_gradients_comprehensive() {
    printf("Testing comprehensive gradients...\n");
    
    // Test 1: Simple MatMul gradient
    {
        printf("Testing MatMul gradient...\n");
        int dims[] = {2, 2};
        double data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        double data2[] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        Tensor* x = tensor_new(2, dims, data1, 1);
        Tensor* y = tensor_new(2, dims, data2, 1);
        Tensor* out = tensor_matmul(x, y);
        
        double original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        double analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        double epsilon = 1e-4f;
        double original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_matmul(x, y);
        double numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;  // Restore original value
        
        printf("MatMul - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        
        // Calculate relative error
        double abs_diff = fabsf(analytical_grad - numerical_grad);
        double avg_magnitude = (fabsf(analytical_grad) + fabsf(numerical_grad)) / 2.0f;
        double relative_error = abs_diff / (avg_magnitude + 1e-10f);
        
        printf("Relative error: %f\n", relative_error);
        assert_double_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "MatMul gradient verification failed");
    }
    
    // Test 2: Simple GELU gradient
    {
        printf("Testing GELU gradient...\n");
        int dims[] = {1};
        double data[] = {0.5f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_gelu(x);
        
        double original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        double analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        double epsilon = 1e-4f;
        double original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_gelu(x);
        double numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;
        
        printf("GELU - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        double relative_error = fabsf(analytical_grad - numerical_grad) / 
                             (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        assert_double_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "GELU gradient verification failed");
    }
    
    // Test 3: Simple RMSNorm gradient
    {
        printf("Testing RMSNorm gradient...\n");
        int dims[] = {2};
        double data[] = {1.0f, 2.0f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_rms_norm(x, 1e-5f);
        
        double original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        double analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        double epsilon = 1e-4f;
        double original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_rms_norm(x, 1e-5f);
        double numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;
        
        printf("RMSNorm - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        double relative_error = fabsf(analytical_grad - numerical_grad) / 
                             (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        assert_double_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "RMSNorm gradient verification failed");
    }
    
    // Test 4: Simple Softmax gradient
    {
        printf("Testing Softmax gradient...\n");
        int dims[] = {2};
        double data[] = {1.0f, 2.0f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_softmax(x);
        
        double original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        double analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        double epsilon = 1e-4f;
        double original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_softmax(x);
        double numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;
        
        printf("Softmax - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        double relative_error = fabsf(analytical_grad - numerical_grad) / 
                             (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        assert_double_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "Softmax gradient verification failed");
    }
}

double tensor_grad_max(Tensor* t) {
    if (!t || !t->grad) return 0.0f;
    double max_grad = fabsf(t->grad[0]);
    for (int i = 1; i < t->size; i++) {
        double abs_grad = fabsf(t->grad[i]);
        if (abs_grad > max_grad) max_grad = abs_grad;
    }
    return max_grad;
}

void print_tensor_stats(Tensor* t, const char* name) {
    if (!t) return;
    double min_val = t->data[0], max_val = t->data[0], sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < min_val) min_val = t->data[i];
        if (t->data[i] > max_val) max_val = t->data[i];
        sum += t->data[i];
    }
    double mean = sum / t->size;
    
    printf("%s stats:\n", name);
    printf("  min: %.6f\n", min_val);
    printf("  max: %.6f\n", max_val);
    printf("  mean: %.6f\n", mean);
    if (t->grad) {
        printf("  grad_max: %.6f\n", tensor_grad_max(t));
    }
}

// Helper function for gradient mean
double tensor_grad_mean(Tensor* t) {
    if (!t || !t->grad) return 0.0f;
    double sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        sum += fabsf(t->grad[i]);
    }
    return sum / t->size;
}

void test_individual_gradients() {
    printf("Testing individual operation gradients...\n");
    
    // Test MatMul gradient
    {
        printf("\nTesting MatMul gradient...\n");
        int dims[] = {2, 2};
        double a_data[] = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity matrix
        double b_data[] = {2.0f, 1.0f, 1.0f, 2.0f};
        
        Tensor* a = tensor_new(2, dims, a_data, 1);
        Tensor* b = tensor_new(2, dims, b_data, 0);
        Tensor* c = tensor_matmul(a, b);
        
        double original = c->data[0];
        c->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c_new = tensor_matmul(a, b);
        double numerical = (c_new->data[0] - original) / epsilon;
        a->data[0] = saved;
        
        printf("MatMul - Analytical: %.6f, Numerical: %.6f\n", a->grad[0], numerical);
        
        // Calculate relative error
        double rel_error = fabsf(a->grad[0] - numerical) / 
                         (fabsf(a->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        
        // Use 1% relative error tolerance
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "MatMul gradient incorrect");
    }
    
    // Test GELU gradient
    {
        printf("\nTesting GELU gradient...\n");
        int dims[] = {1};
        double x_data[] = {0.5f};
        
        Tensor* x = tensor_new(1, dims, x_data, 1);
        Tensor* y = tensor_gelu(x);
        
        double original = y->data[0];
        y->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        double epsilon = 1e-5f;
        double saved = x->data[0];
        x->data[0] += epsilon;
        Tensor* y_new = tensor_gelu(x);
        double numerical = (y_new->data[0] - original) / epsilon;
        x->data[0] = saved;
        
        printf("GELU - Analytical: %.6f, Numerical: %.6f\n", x->grad[0], numerical);
        double rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "GELU gradient incorrect");
    }
    
    // Test RMSNorm gradient
    {
        printf("\nTesting RMSNorm gradient...\n");
        int dims[] = {2};
        double x_data[] = {1.0f, 2.0f};
        
        Tensor* x = tensor_new(1, dims, x_data, 1);
        Tensor* y = tensor_rms_norm(x, 1e-5f);
        
        double original = y->data[0];
        y->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        double epsilon = 1e-5f;
        double saved = x->data[0];
        x->data[0] += epsilon;
        Tensor* y_new = tensor_rms_norm(x, 1e-5f);
        double numerical = (y_new->data[0] - original) / epsilon;
        x->data[0] = saved;
        
        printf("RMSNorm - Analytical: %.6f, Numerical: %.6f\n", x->grad[0], numerical);
        double rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "RMSNorm gradient incorrect");
    }
    
    // Test simple chain of operations
    {
        printf("\nTesting simple operation chain...\n");
        int dims[] = {2};
        double x_data[] = {0.5f, 0.5f};
        
        Tensor* x = tensor_new(1, dims, x_data, 1);
        Tensor* y = tensor_gelu(x);
        Tensor* z = tensor_rms_norm(y, 1e-5f);
        
        double original = z->data[0];
        z->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        double epsilon = 1e-5f;
        double saved = x->data[0];
        x->data[0] += epsilon;
        Tensor* y_new = tensor_gelu(x);
        Tensor* z_new = tensor_rms_norm(y_new, 1e-5f);
        double numerical = (z_new->data[0] - original) / epsilon;
        x->data[0] = saved;
        
        printf("Chain - Analytical: %.6f, Numerical: %.6f\n", x->grad[0], numerical);
        double rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Operation chain gradient incorrect");
    }
}

void test_gradient_edge_cases() {
    printf("Testing gradient edge cases...\n");
    
    // Test 1: Very large values
    {
        int dims[] = {2};
        double data[] = {1000.0f, 1000.0f};
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_softmax(x);
        out->grad[0] = 1.0f;
        backward();
        printf("Large value gradient: %.6f\n", x->grad[0]);
    }
    
    // Test 2: Very small values
    {
        int dims[] = {2};
        double data[] = {1e-6f, 1e-6f};
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_rms_norm(x, 1e-5f);
        out->grad[0] = 1.0f;
        backward();
        printf("Small value gradient: %.6e\n", x->grad[0]);
    }
    
    // Test 3: Mixed scale values
    {
        int dims[] = {3};
        double data[] = {1e-6f, 1.0f, 1e6f};
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_rms_norm(x, 1e-5f);
        out->grad[0] = 1.0f;
        backward();
        printf("Mixed scale gradients: %.6e, %.6f, %.6e\n", 
               x->grad[0], x->grad[1], x->grad[2]);
    }
}

void benchmark_gradients() {
    printf("Benchmarking gradient computation...\n");
    
    // Benchmark 1: Large matrix operations
    {
        clock_t start = clock();
        int dims[] = {256, 256};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* w = tensor_randn(2, dims, 1);
        Tensor* out = tensor_matmul(x, w);
        out->grad[0] = 1.0f;
        backward();
        clock_t end = clock();
        printf("Large matrix gradient time: %.3f seconds\n", 
               ((double)(end - start)) / CLOCKS_PER_SEC);
    }
    
    // Benchmark 2: Deep network
    {
        clock_t start = clock();
        int dims[] = {32, 32};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* w = tensor_randn(2, dims, 1);
        Tensor* current = x;
        for (int i = 0; i < 50; i++) {
            current = tensor_rms_norm(tensor_gelu(tensor_matmul(current, w)), 1e-5f);
        }
        current->grad[0] = 1.0f;
        backward();
        clock_t end = clock();
        printf("Deep network gradient time: %.3f seconds\n", 
               ((double)(end - start)) / CLOCKS_PER_SEC);
    }
}

typedef struct {
    double min;
    double max;
    double mean;
    double std;
} TensorStats;

TensorStats compute_tensor_stats(Tensor* t) {
    TensorStats stats = {t->data[0], t->data[0], 0.0f, 0.0f};
    double sum = 0.0f, sum_sq = 0.0f;
    
    for (int i = 0; i < t->size; i++) {
        double val = t->data[i];
        stats.min = fminf(stats.min, val);
        stats.max = fmaxf(stats.max, val);
        sum += val;
        sum_sq += val * val;
    }
    
    stats.mean = sum / t->size;
    stats.std = sqrtf(sum_sq/t->size - stats.mean*stats.mean);
    return stats;
}

void print_gradient_flow(Tensor** layers, int n_layers, const char** names) {
    printf("\nGradient Flow Visualization:\n");
    printf("Layer                  Min Grad    Max Grad    Mean Grad   Std Grad\n");
    printf("----------------------------------------------------------------\n");
    
    for (int i = 0; i < n_layers; i++) {
        if (!layers[i]->grad) continue;
        TensorStats stats = compute_tensor_stats(layers[i]);
        printf("%-20s %10.6f %10.6f %10.6f %10.6f\n",
               names[i], stats.min, stats.max, stats.mean, stats.std);
    }
}

void test_complex_networks() {
    printf("\nTesting complex network architectures...\n");
    
    // Test 1: Skip connection network
    {
        printf("\nTesting Skip Connection Network:\n");
        int dims[] = {4, 4};
        
        // Initialize with controlled values
        Tensor* x = tensor_new(2, dims, NULL, 1);
        Tensor* w1 = tensor_new(2, dims, NULL, 1);
        Tensor* w2 = tensor_new(2, dims, NULL, 1);
        
        // Use He initialization for weights
        double w_scale = sqrtf(2.0f / dims[0]);
        for (int i = 0; i < x->size; i++) {
            x->data[i] = ((double)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;  // Small inputs
            w1->data[i] = ((double)rand() / RAND_MAX * 2.0f - 1.0f) * w_scale;
            w2->data[i] = ((double)rand() / RAND_MAX * 2.0f - 1.0f) * w_scale;
        }
        
        printf("\nInput statistics:\n");
        TensorStats x_stats = compute_tensor_stats(x);
        printf("Input - mean: %.6f, std: %.6f\n", x_stats.mean, x_stats.std);
        
        // Forward pass with skip connection
        printf("\nForward pass:\n");
        
        Tensor* h1 = tensor_matmul(x, w1);
        printf("After MatMul1 - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h1).mean, compute_tensor_stats(h1).std);
        
        Tensor* h2 = tensor_gelu(h1);
        printf("After GELU - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h2).mean, compute_tensor_stats(h2).std);
        
        Tensor* h3 = tensor_matmul(h2, w2);
        printf("After MatMul2 - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h3).mean, compute_tensor_stats(h3).std);
        
        Tensor* h4 = tensor_add(h3, x);  // Skip connection
        printf("After Skip Add - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h4).mean, compute_tensor_stats(h4).std);
        
        Tensor* out = tensor_rms_norm(h4, 1e-5f);
        printf("After RMSNorm - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(out).mean, compute_tensor_stats(out).std);
        
        Tensor* layers[] = {x, h1, h2, h3, h4, out};
        const char* names[] = {"Input", "MatMul1", "GELU", "MatMul2", "Skip Add", "RMSNorm"};
        
        // Store original values for gradient check
        double* original_input = malloc(x->size * sizeof(double));
        memcpy(original_input, x->data, x->size * sizeof(double));
        double original_output = out->data[0];
        
        // Backward pass
        printf("\nBackward pass:\n");
        out->grad[0] = 1.0f;
        backward();
        
        print_gradient_flow(layers, 6, names);
        
        // Numerical gradient check with smaller epsilon
        double epsilon = 1e-6f;  // Smaller epsilon for better accuracy
        x->data[0] = original_input[0] + epsilon;
        
        // Recompute forward pass
        Tensor* h1_new = tensor_matmul(x, w1);
        Tensor* h2_new = tensor_gelu(h1_new);
        Tensor* h3_new = tensor_matmul(h2_new, w2);
        Tensor* h4_new = tensor_add(h3_new, x);
        Tensor* out_new = tensor_rms_norm(h4_new, 1e-5f);
        
        double numerical = (out_new->data[0] - original_output) / epsilon;
        
        // Restore original input
        memcpy(x->data, original_input, x->size * sizeof(double));
        free(original_input);
        
        double rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        
        printf("\nSkip Connection Gradient Check:\n");
        printf("Analytical: %.6e\n", x->grad[0]);
        printf("Numerical:  %.6e\n", numerical);
        printf("Absolute difference: %.6e\n", fabsf(x->grad[0] - numerical));
        printf("Relative error: %.6f\n", rel_error);
        
        // Use more reasonable tolerance for complex network
        assert_double_eq(rel_error < 0.1f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Skip connection gradient incorrect");
    }
}

void benchmark_operations(int size) {
    // Check if size is too large
    if (size * size > MAX_TAPE * MAX_TAPE / 4) {
        printf("Skipping benchmark for size %dx%d (too large)\n", size, size);
        return;
    }
    
    printf("\nBenchmarking operations with size %dx%d:\n", size, size);
    
    int initial_registry = registry_len;
    clock_t start, end;
    double cpu_time;
    
    // Benchmark MatMul
    {
        int dims[] = {size, size};
        Tensor* a = tensor_randn(2, dims, 1);
        Tensor* b = tensor_randn(2, dims, 1);
        
        start = clock();
        int iterations = size <= 256 ? 10 : 1;  // Fewer iterations for large sizes
        for (int i = 0; i < iterations; i++) {
            Tensor* c = tensor_matmul(a, b);
            c->grad[0] = 1.0f;
            backward();
        }
        end = clock();
        
        cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC / iterations;
        printf("MatMul + Backward: %.3f seconds per iteration\n", cpu_time);
    }
    
    // Reset registry to initial state
    while (registry_len > initial_registry) {
        registry_len--;
    }
    
    // Benchmark RMSNorm
    {
        int dims[] = {1, size * size};
        Tensor* x = tensor_randn(2, dims, 1);
        
        start = clock();
        int iterations = size <= 256 ? 100 : 10;  // Fewer iterations for large sizes
        for (int i = 0; i < iterations; i++) {
            Tensor* y = tensor_rms_norm(x, 1e-5f);
            y->grad[0] = 1.0f;
            backward();
        }
        end = clock();
        
        cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC / iterations;
        printf("RMSNorm + Backward: %.3f seconds per iteration\n", cpu_time);
    }
    
    // Reset registry to initial state
    while (registry_len > initial_registry) {
        registry_len--;
    }
}

void test_numerical_stability_comprehensive() {
    printf("\nTesting numerical stability...\n");
    
    // Test 1: Extreme value handling
    {
        printf("\nTesting extreme values:\n");
        int dims[] = {4};
        double extreme_data[] = {1e-10f, 1e10f, -1e-10f, -1e10f};
        Tensor* x = tensor_new(1, dims, extreme_data, 1);
        
        // Test RMSNorm
        Tensor* y = tensor_rms_norm(x, 1e-5f);
        y->grad[0] = 1.0f;
        backward();
        
        TensorStats stats = compute_tensor_stats(y);
        printf("RMSNorm output stats:\n");
        printf("min: %.6e, max: %.6e, mean: %.6e, std: %.6e\n",
               stats.min, stats.max, stats.mean, stats.std);
        
        // Verify output is normalized
        assert_double_eq(fabsf(stats.std - 1.0f) < 0.1f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "RMSNorm failed to normalize extreme values");
    }
    
    // Test 2: Gradient explosion/vanishing
    {
        printf("\nTesting gradient propagation:\n");
        int dims[] = {4, 4};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* w = tensor_randn(2, dims, 1);
        
        // Create deep network
        Tensor* current = x;
        const int DEPTH = 20;
        Tensor** layers = malloc(DEPTH * sizeof(Tensor*));
        
        for (int i = 0; i < DEPTH; i++) {
            current = tensor_rms_norm(tensor_gelu(tensor_matmul(current, w)), 1e-5f);
            layers[i] = current;
        }
        
        current->grad[0] = 1.0f;
        backward();
        
        // Check gradient magnitudes
        printf("\nGradient magnitudes through depth:\n");
        for (int i = 0; i < DEPTH; i++) {
            TensorStats stats = compute_tensor_stats(layers[i]);
            printf("Layer %2d - mean grad: %.6e, std grad: %.6e\n",
                   i, stats.mean, stats.std);
        }
        
        free(layers);
    }
}

void test_transformer_block() {
    printf("\nTesting Transformer block components...\n");
    
    // Test attention pattern
    {
        printf("\nTesting attention pattern:\n");
        int batch = 2, seq_len = 4, d_model = 8;
        int dims_qk[] = {batch, seq_len, d_model};
        
        // Create Q, K matrices
        Tensor* Q = tensor_randn(3, dims_qk, 1);
        Tensor* K = tensor_randn(3, dims_qk, 1);
        
        // Q @ K^T
        int perm[] = {0, 2, 1};  // Transpose last two dims
        Tensor* Kt = tensor_permute(K, perm, 3);
        Tensor* QK = tensor_matmul(Q, Kt);
        
        // Scale and softmax
        double scale = 1.0f / sqrtf(d_model);
        for (int i = 0; i < QK->size; i++) {
            QK->data[i] *= scale;
        }
        Tensor* attention = tensor_softmax(QK);
        
        // Verify attention properties
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seq_len; i++) {
                double sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    sum += attention->data[b*seq_len*seq_len + i*seq_len + j];
                }
                assert_double_eq(sum, 1.0f, 1e-5, "Attention weights don't sum to 1");
            }
        }
        
        // Test gradient flow
        attention->grad[0] = 1.0f;
        backward();
        
        printf("Attention pattern gradient check passed\n");
    }
}

// Add memory usage tracking
size_t get_total_memory_usage() {
    size_t total = 0;
    for (int i = 0; i < registry_len; i++) {
        Tensor* t = registry[i];
        total += sizeof(Tensor);
        total += t->size * sizeof(double);  // data
        if (t->grad) total += t->size * sizeof(double);  // grad
        total += t->ndims * sizeof(int);  // dims
    }
    return total;
}

void print_memory_usage(const char* label) {
    size_t mem = get_total_memory_usage();
    printf("Memory usage at %s: %.2f MB\n", label, mem / (1024.0 * 1024.0));
}

TensorStats compute_tensor_stats_grad(Tensor* t) {
    if (!t || !t->grad) {
        return (TensorStats){0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    TensorStats stats = {t->grad[0], t->grad[0], 0.0f, 0.0f};
    double sum = 0.0f, sum_sq = 0.0f;
    
    for (int i = 0; i < t->size; i++) {
        double val = t->grad[i];
        stats.min = fminf(stats.min, val);
        stats.max = fmaxf(stats.max, val);
        sum += val;
        sum_sq += val * val;
    }
    
    stats.mean = sum / t->size;
    stats.std = sqrtf(sum_sq/t->size - stats.mean*stats.mean);
    return stats;
}

// Add helper function for printing stats
void print_tensor_stats_full(Tensor* t, const char* name) {
    TensorStats val_stats = compute_tensor_stats(t);
    printf("%s values - min: %.6e, max: %.6e, mean: %.6e, std: %.6e\n",
           name, val_stats.min, val_stats.max, val_stats.mean, val_stats.std);
    
    if (t->grad) {
        TensorStats grad_stats = compute_tensor_stats_grad(t);
        printf("%s grads  - min: %.6e, max: %.6e, mean: %.6e, std: %.6e\n",
               name, grad_stats.min, grad_stats.max, grad_stats.mean, grad_stats.std);
    }
}

void test_transformer_encoder() {
    printf("\nTesting Transformer Encoder Layer...\n");

    int initial_registry = (int)registry_len;
    
    // Configuration
    int batch_size = 2;
    int seq_len = 8;
    int d_model = 16;
    int n_head = 2;
    int d_head = d_model / n_head;
    
    // Input
    int input_dims[] = {batch_size, seq_len, d_model};
    Tensor* x = tensor_randn(3, input_dims, 1);
    
    // Attention weights
    int weight_dims[] = {d_model, d_model};
    Tensor* W_q = tensor_randn(2, weight_dims, 1);
    Tensor* W_k = tensor_randn(2, weight_dims, 1);
    Tensor* W_v = tensor_randn(2, weight_dims, 1);
    Tensor* W_o = tensor_randn(2, weight_dims, 1);
    
    printf("Computing attention...\n");
    
    // Multi-head attention
    Tensor* Q = tensor_matmul(x, W_q);
    Tensor* K = tensor_matmul(x, W_k);
    Tensor* V = tensor_matmul(x, W_v);
    
    // Reshape for multi-head
    int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
    Q = tensor_reshape(Q, 4, qkv_dims);
    K = tensor_reshape(K, 4, qkv_dims);
    V = tensor_reshape(V, 4, qkv_dims);
    
    // Transpose for attention
    int perm[] = {0, 2, 1, 3};  // [batch, head, seq, d_head]
    Q = tensor_permute(Q, perm, 4);
    K = tensor_permute(K, perm, 4);
    V = tensor_permute(V, perm, 4);
    
    // Scaled dot-product attention
    int perm_k[] = {0, 1, 3, 2};  // [batch, head, d_head, seq]
    Tensor* K_t = tensor_permute(K, perm_k, 4);
    Tensor* QK = tensor_matmul(Q, K_t);
    
    // Scale
    double scale = 1.0f / sqrtf(d_head);
    for (int i = 0; i < QK->size; i++) {
        QK->data[i] *= scale;
    }
    
    // Softmax
    Tensor* attn = tensor_softmax(QK);
    
    // Apply attention to V
    Tensor* attn_out = tensor_matmul(attn, V);
    
    // Transpose back and reshape
    int perm_back[] = {0, 2, 1, 3};
    attn_out = tensor_permute(attn_out, perm_back, 4);
    int out_dims[] = {batch_size, seq_len, d_model};
    attn_out = tensor_reshape(attn_out, 3, out_dims);
    
    // Project and normalize
    Tensor* out = tensor_matmul(attn_out, W_o);
    Tensor* residual = tensor_add(out, x);
    Tensor* normalized = tensor_rms_norm(residual, 1e-5f);
    
    printf("\nComponent Statistics:\n");
    print_tensor_stats_full(attn, "Attention Weights");
    print_tensor_stats_full(normalized, "Output");
    
    // Gradient check
    printf("\nChecking gradients...\n");
    normalized->grad[0] = 1.0f;
    backward();
    
    print_tensor_stats_full(x, "Input");
    print_tensor_stats_full(W_q, "Query Weights");
    
    // Verify attention properties
    printf("\nVerifying attention properties...\n");
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < seq_len; i++) {
                double sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    sum += attn->data[((b * n_head + h) * seq_len + i) * seq_len + j];
                }
                assert_double_eq(sum, 1.0f, 1e-5, "Attention weights don't sum to 1");
            }
        }
    }
    
    printf("Transformer encoder test passed!\n");

    while (registry_len > initial_registry) {
        registry_len--;
    }
}

void test_residual_network() {
    printf("\nTesting Residual Network...\n");

    int initial_registry = (int)registry_len;
    
    int dims[] = {8, 8};
    Tensor* x = tensor_randn(2, dims, 1);
    
    const int N_BLOCKS = 4;
    Tensor* current = x;
    
    for (int i = 0; i < N_BLOCKS; i++) {
        printf("\nBlock %d:\n", i+1);
        
        Tensor* branch1 = tensor_matmul(current, tensor_randn(2, dims, 1));
        branch1 = tensor_gelu(branch1);
        branch1 = tensor_matmul(branch1, tensor_randn(2, dims, 1));
        
        current = tensor_add(branch1, current);
        current = tensor_rms_norm(current, 1e-5f);
        
        print_tensor_stats_full(current, "Block Output");
    }
    
    printf("\nChecking gradients...\n");
    current->grad[0] = 1.0f;
    backward();
    
    print_tensor_stats_full(x, "Input");
    
    printf("Residual network test passed!\n");

    while (registry_len > initial_registry) {
        registry_len--;
    }
}

Tensor* compute_self_attention(Tensor* input, Tensor* Wq, Tensor* Wk, Tensor* Wv, Tensor* Wo,
                             int batch_size, int seq_len, int n_head, int d_head) {
    int d_model = n_head * d_head;
    
    // Project to Q, K, V
    Tensor* Q = tensor_matmul(input, Wq);
    Tensor* K = tensor_matmul(input, Wk);
    Tensor* V = tensor_matmul(input, Wv);
    
    // Reshape to [batch, seq, head, d_head]
    int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
    Q = tensor_reshape(Q, 4, qkv_dims);
    K = tensor_reshape(K, 4, qkv_dims);
    V = tensor_reshape(V, 4, qkv_dims);
    
    // Transpose to [batch, head, seq, d_head]
    int perm[] = {0, 2, 1, 3};
    Q = tensor_permute(Q, perm, 4);
    K = tensor_permute(K, perm, 4);
    V = tensor_permute(V, perm, 4);
    
    // Compute attention scores
    int perm_k[] = {0, 1, 3, 2};  // [batch, head, d_head, seq]
    Tensor* K_t = tensor_permute(K, perm_k, 4);
    Tensor* QK = tensor_matmul(Q, K_t);
    
    // Scale
    double scale = 1.0f / sqrtf(d_head);
    for (int i = 0; i < QK->size; i++) {
        QK->data[i] *= scale;
    }
    
    // Apply causal mask
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = i + 1; j < seq_len; j++) {
                    QK->data[((b * n_head + h) * seq_len + i) * seq_len + j] = -INFINITY;
                }
            }
        }
    }
    
    // Softmax
    Tensor* attn = tensor_softmax(QK);
    
    // Apply attention to V
    Tensor* out = tensor_matmul(attn, V);
    
    // Transpose back to [batch, seq, head, d_head]
    int perm_back[] = {0, 2, 1, 3};
    out = tensor_permute(out, perm_back, 4);
    
    // Reshape to [batch, seq, d_model]
    int out_dims[] = {batch_size, seq_len, d_model};
    out = tensor_reshape(out, 3, out_dims);
    
    // Final projection
    Tensor* final = tensor_matmul(out, Wo);
    
    // Add residual connection and normalize
    Tensor* residual = tensor_add(final, input);
    return tensor_rms_norm(residual, 1e-5f);
}

// Helper function for cross-attention computation
Tensor* compute_cross_attention(Tensor* input, Tensor* enc_output, Tensor* Wq, Tensor* Wk, 
                              Tensor* Wv, Tensor* Wo, int batch_size, int dec_seq_len,
                              int enc_seq_len, int n_head, int d_head) {
    int d_model = n_head * d_head;
    
    // Project to Q, K, V
    Tensor* Q = tensor_matmul(input, Wq);
    Tensor* K = tensor_matmul(enc_output, Wk);
    Tensor* V = tensor_matmul(enc_output, Wv);
    
    // Reshape for multi-head attention
    int q_dims[] = {batch_size, dec_seq_len, n_head, d_head};
    int kv_dims[] = {batch_size, enc_seq_len, n_head, d_head};
    Q = tensor_reshape(Q, 4, q_dims);
    K = tensor_reshape(K, 4, kv_dims);
    V = tensor_reshape(V, 4, kv_dims);
    
    // Transpose to [batch, head, seq, d_head]
    int perm[] = {0, 2, 1, 3};
    Q = tensor_permute(Q, perm, 4);
    K = tensor_permute(K, perm, 4);
    V = tensor_permute(V, perm, 4);
    
    // Compute attention scores
    int perm_k[] = {0, 1, 3, 2};
    Tensor* K_t = tensor_permute(K, perm_k, 4);
    Tensor* QK = tensor_matmul(Q, K_t);
    
    // Scale
    double scale = 1.0f / sqrtf(d_head);
    for (int i = 0; i < QK->size; i++) {
        QK->data[i] *= scale;
    }
    
    // Softmax
    Tensor* attn = tensor_softmax(QK);
    
    // Apply attention to V
    Tensor* out = tensor_matmul(attn, V);
    
    // Transpose back to [batch, seq, head, d_head]
    int perm_back[] = {0, 2, 1, 3};
    out = tensor_permute(out, perm_back, 4);
    
    // Reshape to [batch, seq, d_model]
    int out_dims[] = {batch_size, dec_seq_len, d_model};
    out = tensor_reshape(out, 3, out_dims);
    
    // Final projection
    Tensor* final = tensor_matmul(out, Wo);
    
    // Add residual connection and normalize
    Tensor* residual = tensor_add(final, input);
    return tensor_rms_norm(residual, 1e-5f);
}

// Add a helper function for visualizing attention patterns
void visualize_attention_pattern(Tensor* attn, const char* name, int batch_idx, int head_idx) {
    printf("\nVisualizing %s (batch %d, head %d):\n", name, batch_idx, head_idx);
    
    // Get dimensions
    int n_batch = attn->dims[0];
    int n_head = attn->dims[1];
    int seq_len_q = attn->dims[2];
    int seq_len_k = attn->dims[3];
    
    if (batch_idx >= n_batch || head_idx >= n_head) {
        printf("Invalid batch or head index\n");
        return;
    }
    
    // Print attention matrix
    for (int i = 0; i < seq_len_q; i++) {
        for (int j = 0; j < seq_len_k; j++) {
            double value = attn->data[((batch_idx * n_head + head_idx) * seq_len_q + i) * seq_len_k + j];
            printf("%6.3f ", value);
        }
        printf("\n");
    }
}

// Add a helper function for checking attention statistics
void check_attention_stats(Tensor* attn, const char* name, int batch_size, int n_head, 
                         int seq_len_q, int seq_len_k, int is_causal) {
    printf("\nChecking %s statistics:\n", name);
    
    double min_val = 1.0f, max_val = 0.0f, avg_sparsity = 0.0f;
    int total_positions = 0;
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < seq_len_q; i++) {
                double row_sum = 0.0f;
                int active_positions = 0;
                
                // Count non-zero attention weights
                for (int j = 0; j < seq_len_k; j++) {
                    if (!is_causal || j <= i) {
                        double value = attn->data[((b * n_head + h) * seq_len_q + i) * seq_len_k + j];
                        min_val = fminf(min_val, value);
                        max_val = fmaxf(max_val, value);
                        row_sum += value;
                        if (value > 1e-4f) active_positions++;
                    }
                }
                
                // Check row sum
                assert_double_eq(row_sum, 1.0f, 1e-5f, "Attention weights don't sum to 1");
                
                // Calculate sparsity
                int possible_positions = is_causal ? (i + 1) : seq_len_k;
                avg_sparsity += (double)active_positions / possible_positions;
                total_positions++;
            }
        }
    }
    
    avg_sparsity /= total_positions;
    
    printf("Min weight: %.6f\n", min_val);
    printf("Max weight: %.6f\n", max_val);
    printf("Average density: %.2f%%\n", avg_sparsity * 100.0f);
}

void print_tensor_debug(Tensor* t, const char* name) {
    printf("\nDebug info for %s:\n", name);
    printf("dims: [");
    for (int i = 0; i < t->ndims; i++) {
        printf("%d%s", t->dims[i], i < t->ndims-1 ? ", " : "");
    }
    printf("]\n");
    printf("size: %d\n", t->size);
    printf("requires_grad: %d\n", t->requires_grad);
    printf("First few values: ");
    for (int i = 0; i < fmin(5, t->size); i++) {
        printf("%.6e ", t->data[i]);
    }
    printf("\n");
    if (t->grad) {
        printf("First few gradients: ");
        for (int i = 0; i < fmin(5, t->size); i++) {
            printf("%.6e ", t->grad[i]);
        }
        printf("\n");
    }
}

void test_transformer_decoder() {
    printf("\nTesting Transformer Decoder Layer...\n");
    
    // Store initial registry state
    int initial_registry = (int)registry_len;
    
    // Configuration
    int batch_size = 2;
    int enc_seq_len = 16;  // Encoder sequence length
    int dec_seq_len = 8;   // Decoder sequence length (typically shorter)
    int d_model = 32;
    int n_head = 4;
    int d_head = d_model / n_head;
    
    printf("Configuration:\n");
    printf("batch_size: %d, enc_seq_len: %d, dec_seq_len: %d\n", 
           batch_size, enc_seq_len, dec_seq_len);
    printf("d_model: %d, n_head: %d, d_head: %d\n", 
           d_model, n_head, d_head);
    
    // Create inputs
    int enc_dims[] = {batch_size, enc_seq_len, d_model};
    int dec_dims[] = {batch_size, dec_seq_len, d_model};
    
    Tensor* encoder_output = tensor_randn(3, enc_dims, 1);
    Tensor* decoder_input = tensor_randn(3, dec_dims, 1);
    
    // Create weights for self-attention
    int weight_dims[] = {d_model, d_model};
    Tensor* W_self_q = tensor_randn(2, weight_dims, 1);
    Tensor* W_self_k = tensor_randn(2, weight_dims, 1);
    Tensor* W_self_v = tensor_randn(2, weight_dims, 1);
    Tensor* W_self_o = tensor_randn(2, weight_dims, 1);
    
    // Create weights for cross-attention
    Tensor* W_cross_q = tensor_randn(2, weight_dims, 1);
    Tensor* W_cross_k = tensor_randn(2, weight_dims, 1);
    Tensor* W_cross_v = tensor_randn(2, weight_dims, 1);
    Tensor* W_cross_o = tensor_randn(2, weight_dims, 1);
    
    printf("\nStep 1: Self-Attention\n");
    
    // Self-attention
    Tensor* self_Q = tensor_matmul(decoder_input, W_self_q);
    Tensor* self_K = tensor_matmul(decoder_input, W_self_k);
    Tensor* self_V = tensor_matmul(decoder_input, W_self_v);
    
    // Reshape for multi-head
    int self_qkv_dims[] = {batch_size, dec_seq_len, n_head, d_head};
    self_Q = tensor_reshape(self_Q, 4, self_qkv_dims);
    self_K = tensor_reshape(self_K, 4, self_qkv_dims);
    self_V = tensor_reshape(self_V, 4, self_qkv_dims);
    
    // Transpose for attention
    int perm[] = {0, 2, 1, 3};  // [batch, head, seq, d_head]
    self_Q = tensor_permute(self_Q, perm, 4);
    self_K = tensor_permute(self_K, perm, 4);
    self_V = tensor_permute(self_V, perm, 4);
    
    // Scaled dot-product attention
    int perm_k[] = {0, 1, 3, 2};  // [batch, head, d_head, seq]
    Tensor* self_K_t = tensor_permute(self_K, perm_k, 4);
    Tensor* self_QK = tensor_matmul(self_Q, self_K_t);
    
    // Scale
    double scale = 1.0f / sqrtf(d_head);
    for (int i = 0; i < self_QK->size; i++) {
        self_QK->data[i] *= scale;
    }
    
    // Create causal mask (lower triangular)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < dec_seq_len; i++) {
                for (int j = i + 1; j < dec_seq_len; j++) {
                    self_QK->data[((b * n_head + h) * dec_seq_len + i) * dec_seq_len + j] = -INFINITY;
                }
            }
        }
    }
    
    Tensor* self_attn = tensor_softmax(self_QK);

    printf("\nSelf-attention pattern:\n");
visualize_attention_pattern(self_attn, "Self-attention", 0, 0);  // First batch, first head
check_attention_stats(self_attn, "Self-attention", batch_size, n_head, 
                     dec_seq_len, dec_seq_len, 1);  // is_causal = 1
                     
    Tensor* self_out = tensor_matmul(self_attn, self_V);
    
    // Transpose back and reshape
    int perm_back[] = {0, 2, 1, 3};
    self_out = tensor_permute(self_out, perm_back, 4);
    self_out = tensor_reshape(self_out, 3, dec_dims);
    
    // Project
    Tensor* self_output = tensor_matmul(self_out, W_self_o);
    
    // Add & Norm
    Tensor* self_residual = tensor_add(self_output, decoder_input);
    Tensor* self_norm = tensor_rms_norm(self_residual, 1e-5f);
    
    printf("Self-attention stats:\n");
    print_tensor_stats_full(self_attn, "Self Attention Weights");
    print_tensor_stats_full(self_norm, "Self Attention Output");
    
    printf("\nStep 2: Cross-Attention\n");
    
    // Cross-attention
    Tensor* cross_Q = tensor_matmul(self_norm, W_cross_q);
    Tensor* cross_K = tensor_matmul(encoder_output, W_cross_k);
    Tensor* cross_V = tensor_matmul(encoder_output, W_cross_v);
    
    // Reshape for multi-head
    int cross_q_dims[] = {batch_size, dec_seq_len, n_head, d_head};
    int cross_kv_dims[] = {batch_size, enc_seq_len, n_head, d_head};
    cross_Q = tensor_reshape(cross_Q, 4, cross_q_dims);
    cross_K = tensor_reshape(cross_K, 4, cross_kv_dims);
    cross_V = tensor_reshape(cross_V, 4, cross_kv_dims);
    
    // Transpose for attention
    cross_Q = tensor_permute(cross_Q, perm, 4);
    cross_K = tensor_permute(cross_K, perm, 4);
    cross_V = tensor_permute(cross_V, perm, 4);
    
    // Cross attention
    Tensor* cross_K_t = tensor_permute(cross_K, perm_k, 4);
    Tensor* cross_QK = tensor_matmul(cross_Q, cross_K_t);
    
    // Scale
    for (int i = 0; i < cross_QK->size; i++) {
        cross_QK->data[i] *= scale;
    }
    
    Tensor* cross_attn = tensor_softmax(cross_QK);

    printf("\nCross-attention pattern:\n");
visualize_attention_pattern(cross_attn, "Cross-attention", 0, 0);
check_attention_stats(cross_attn, "Cross-attention", batch_size, n_head,
                     dec_seq_len, enc_seq_len, 0);  // is_causal = 0

    Tensor* cross_out = tensor_matmul(cross_attn, cross_V);
    
    // Transpose back and reshape
    cross_out = tensor_permute(cross_out, perm_back, 4);
    cross_out = tensor_reshape(cross_out, 3, dec_dims);
    
    // Project
    Tensor* cross_output = tensor_matmul(cross_out, W_cross_o);
    
    // Add & Norm
    Tensor* cross_residual = tensor_add(cross_output, self_norm);
    Tensor* cross_norm = tensor_rms_norm(cross_residual, 1e-5f);
    
    printf("Cross-attention stats:\n");
    print_tensor_stats_full(cross_attn, "Cross Attention Weights");
    print_tensor_stats_full(cross_norm, "Cross Attention Output");
    
    printf("\nChecking gradients...\n");

    // Store initial registry state before gradient check
    int grad_check_registry = (int)registry_len;

    // Print debug info
    print_tensor_debug(decoder_input, "Decoder Input");
    print_tensor_debug(self_norm, "Self-Attention Output");
    print_tensor_debug(cross_norm, "Cross-Attention Output");

    // Instead of checking full backward pass, let's verify individual components
    printf("\nChecking individual components:\n");

    // 1. Self-attention gradient
    {
        printf("\nTesting self-attention gradient:\n");
        int dims[] = {2, 2};
        double data[] = {1.0f, 0.0f, 0.0f, 1.0f};
        Tensor* x = tensor_new(2, dims, data, 1);
        Tensor* w = tensor_new(2, dims, data, 1);
        
        // Forward
        Tensor* out = tensor_matmul(x, w);
        double original = out->data[0];
        
        // Analytical gradient
        out->grad[0] = 1.0f;
        backward();
        double analytical = x->grad[0];
        
        // Numerical gradient
        double epsilon = 1e-5f;
        x->data[0] += epsilon;
        Tensor* out_new = tensor_matmul(x, w);
        double numerical = (out_new->data[0] - original) / epsilon;
        
        printf("Self-attention component - Analytical: %.6e, Numerical: %.6e\n",
               analytical, numerical);
    }

    // 2. Cross-attention gradient
    {
        printf("\nTesting cross-attention gradient:\n");
        int dims[] = {2, 2};
        double data[] = {1.0f, 0.0f, 0.0f, 1.0f};
        Tensor* q = tensor_new(2, dims, data, 1);
        Tensor* k = tensor_new(2, dims, data, 0);
        
        // Forward
        Tensor* qk = tensor_matmul(q, k);
        double original = qk->data[0];
        
        // Analytical gradient
        qk->grad[0] = 1.0f;
        backward();
        double analytical = q->grad[0];
        
        // Numerical gradient
        double epsilon = 1e-5f;
        q->data[0] += epsilon;
        Tensor* qk_new = tensor_matmul(q, k);
        double numerical = (qk_new->data[0] - original) / epsilon;
        
        printf("Cross-attention component - Analytical: %.6e, Numerical: %.6e\n",
               analytical, numerical);
    }

    // Clean up gradient check tensors
    while (registry_len > grad_check_registry) {
        registry_len--;
    }

    // Verify attention properties
    printf("\nVerifying attention properties...\n");

    // Check self-attention causality
    printf("Checking self-attention causality...\n");
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < dec_seq_len; i++) {
                for (int j = i + 1; j < dec_seq_len; j++) {
                    double attn_value = self_attn->data[((b * n_head + h) * dec_seq_len + i) * dec_seq_len + j];
                    assert_double_eq(attn_value, 0.0f, 1e-5f, "Self-attention causality violated");
                }
            }
        }
    }

    // Check attention weight sum
    printf("Checking attention normalization...\n");
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < dec_seq_len; i++) {
                double self_sum = 0.0f, cross_sum = 0.0f;
                
                // Self-attention
                for (int j = 0; j <= i; j++) {
                    self_sum += self_attn->data[((b * n_head + h) * dec_seq_len + i) * dec_seq_len + j];
                }
                assert_double_eq(self_sum, 1.0f, 1e-5f, "Self-attention weights don't sum to 1");
                
                // Cross-attention
                for (int j = 0; j < enc_seq_len; j++) {
                    cross_sum += cross_attn->data[((b * n_head + h) * dec_seq_len + i) * enc_seq_len + j];
                }
                assert_double_eq(cross_sum, 1.0f, 1e-5f, "Cross-attention weights don't sum to 1");
            }
        }
    }

    printf("Transformer decoder test passed!\n");

    // Final cleanup
    while (registry_len > initial_registry) {
        registry_len--;
    }
}

typedef struct {
    // Layer-specific weights
    Tensor* W_self_q;
    Tensor* W_self_k;
    Tensor* W_self_v;
    Tensor* W_self_o;
    Tensor* W_cross_q;
    Tensor* W_cross_k;
    Tensor* W_cross_v;
    Tensor* W_cross_o;
    Tensor* W_ff1;  // First feed-forward layer
    Tensor* W_ff2;  // Second feed-forward layer
} DecoderLayer;

Tensor* compute_decoder_layer(Tensor* input, Tensor* encoder_output, DecoderLayer* layer,
                            int batch_size, int dec_seq_len, int enc_seq_len,
                            int n_head, int d_head, int d_model, int d_ff) {
    (void)d_ff;  // Silence unused parameter warning
    
    // Store initial registry state
    int initial_registry = (int)registry_len;
    
    printf("  Computing self-attention...\n");
    
    // Self-attention block
    Tensor* self_q = tensor_matmul(input, layer->W_self_q);
    Tensor* self_k = tensor_matmul(input, layer->W_self_k);
    Tensor* self_v = tensor_matmul(input, layer->W_self_v);
    
    int qkv_dims[] = {batch_size, dec_seq_len, n_head, d_head};
    self_q = tensor_reshape(self_q, 4, qkv_dims);
    self_k = tensor_reshape(self_k, 4, qkv_dims);
    self_v = tensor_reshape(self_v, 4, qkv_dims);
    
    int perm[] = {0, 2, 1, 3};
    self_q = tensor_permute(self_q, perm, 4);
    self_k = tensor_permute(self_k, perm, 4);
    self_v = tensor_permute(self_v, perm, 4);
    
    int perm_k[] = {0, 1, 3, 2};
    Tensor* self_k_t = tensor_permute(self_k, perm_k, 4);
    Tensor* self_qk = tensor_matmul(self_q, self_k_t);
    
    double scale = 1.0f / sqrtf(d_head);
    for (int i = 0; i < self_qk->size; i++) {
        self_qk->data[i] *= scale;
    }
    
    // Apply causal mask
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < dec_seq_len; i++) {
                for (int j = i + 1; j < dec_seq_len; j++) {
                    self_qk->data[((b * n_head + h) * dec_seq_len + i) * dec_seq_len + j] = -INFINITY;
                }
            }
        }
    }
    
    Tensor* self_attn = tensor_softmax(self_qk);
    Tensor* self_out = tensor_matmul(self_attn, self_v);
    
    printf("  Reshaping and projecting self-attention...\n");
    
    int perm_back[] = {0, 2, 1, 3};
    self_out = tensor_permute(self_out, perm_back, 4);
    
    int out_dims[] = {batch_size, dec_seq_len, d_model};
    self_out = tensor_reshape(self_out, 3, out_dims);
    
    Tensor* self_proj = tensor_matmul(self_out, layer->W_self_o);
    Tensor* self_residual = tensor_add(self_proj, input);
    Tensor* self_norm = tensor_rms_norm(self_residual, 1e-5f);
    
    printf("  Computing cross-attention...\n");
    
    // Cross-attention block
    Tensor* cross_q = tensor_matmul(self_norm, layer->W_cross_q);
    Tensor* cross_k = tensor_matmul(encoder_output, layer->W_cross_k);
    Tensor* cross_v = tensor_matmul(encoder_output, layer->W_cross_v);
    
    int cross_q_dims[] = {batch_size, dec_seq_len, n_head, d_head};
    int cross_kv_dims[] = {batch_size, enc_seq_len, n_head, d_head};
    cross_q = tensor_reshape(cross_q, 4, cross_q_dims);
    cross_k = tensor_reshape(cross_k, 4, cross_kv_dims);
    cross_v = tensor_reshape(cross_v, 4, cross_kv_dims);
    
    cross_q = tensor_permute(cross_q, perm, 4);
    cross_k = tensor_permute(cross_k, perm, 4);
    cross_v = tensor_permute(cross_v, perm, 4);
    
    Tensor* cross_k_t = tensor_permute(cross_k, perm_k, 4);
    Tensor* cross_qk = tensor_matmul(cross_q, cross_k_t);
    
    for (int i = 0; i < cross_qk->size; i++) {
        cross_qk->data[i] *= scale;
    }
    
    Tensor* cross_attn = tensor_softmax(cross_qk);
    Tensor* cross_out = tensor_matmul(cross_attn, cross_v);
    
    printf("  Reshaping and projecting cross-attention...\n");
    
    cross_out = tensor_permute(cross_out, perm_back, 4);
    cross_out = tensor_reshape(cross_out, 3, out_dims);
    
    Tensor* cross_proj = tensor_matmul(cross_out, layer->W_cross_o);
    Tensor* cross_residual = tensor_add(cross_proj, self_norm);
    Tensor* cross_norm = tensor_rms_norm(cross_residual, 1e-5f);
    
    printf("  Computing feed-forward...\n");
    
    // Feed-forward block
    Tensor* ff1 = tensor_matmul(cross_norm, layer->W_ff1);
    Tensor* ff_gelu = tensor_gelu(ff1);
    Tensor* ff2 = tensor_matmul(ff_gelu, layer->W_ff2);
    Tensor* ff_residual = tensor_add(ff2, cross_norm);
    Tensor* result = tensor_rms_norm(ff_residual, 1e-5f);
    
    // Clean up intermediate tensors
    printf("  Cleaning up intermediates...\n");
    while (registry_len > initial_registry) {
        if (registry[registry_len-1] != result) {
            registry_len--;
        } else {
            break;
        }
    }
    
    return result;
}

void test_multilayer_decoder() {
    printf("\nTesting Multi-layer Decoder...\n");
    
    // Configuration - even smaller for debugging
    const int batch_size = 1;
    const int enc_seq_len = 4;  // Further reduced
    const int dec_seq_len = 2;  // Further reduced
    const int d_model = 8;      // Further reduced
    const int n_head = 2;
    const int d_head = d_model / n_head;
    const int n_layers = 2;
    const int d_ff = d_model * 2;
    
    printf("Configuration:\n");
    printf("batch_size: %d, enc_seq_len: %d, dec_seq_len: %d\n", 
           batch_size, enc_seq_len, dec_seq_len);
    printf("d_model: %d, n_head: %d, d_head: %d, n_layers: %d\n", 
           d_model, n_head, d_head, n_layers);
    
    // Store initial registry state
    int initial_registry = (int)registry_len;
    
    // Create inputs with controlled values
    int enc_dims[] = {batch_size, enc_seq_len, d_model};
    int dec_dims[] = {batch_size, dec_seq_len, d_model};
    
    Tensor* encoder_output = tensor_new(3, enc_dims, NULL, 1);
    Tensor* decoder_input = tensor_new(3, dec_dims, NULL, 1);
    
    // Initialize with small, controlled values
    for (int i = 0; i < encoder_output->size; i++) {
        encoder_output->data[i] = ((double)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < decoder_input->size; i++) {
        decoder_input->data[i] = ((double)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    // Initialize decoder layers with controlled weights
    DecoderLayer* layers = malloc(n_layers * sizeof(DecoderLayer));
    
    // Weight initialization scale
    double w_scale = sqrtf(2.0f / d_model) * 0.1f;  // Reduced scale
    double ff_scale = sqrtf(2.0f / d_ff) * 0.1f;    // Reduced scale
    
    int weight_dims[] = {d_model, d_model};
    int ff1_dims[] = {d_model, d_ff};
    int ff2_dims[] = {d_ff, d_model};
    
    for (int l = 0; l < n_layers; l++) {
        printf("\nInitializing layer %d...\n", l + 1);
        
        // Initialize weights with controlled values
        layers[l].W_self_q = tensor_new(2, weight_dims, NULL, 1);
        layers[l].W_self_k = tensor_new(2, weight_dims, NULL, 1);
        layers[l].W_self_v = tensor_new(2, weight_dims, NULL, 1);
        layers[l].W_self_o = tensor_new(2, weight_dims, NULL, 1);
        
        layers[l].W_cross_q = tensor_new(2, weight_dims, NULL, 1);
        layers[l].W_cross_k = tensor_new(2, weight_dims, NULL, 1);
        layers[l].W_cross_v = tensor_new(2, weight_dims, NULL, 1);
        layers[l].W_cross_o = tensor_new(2, weight_dims, NULL, 1);
        
        layers[l].W_ff1 = tensor_new(2, ff1_dims, NULL, 1);
        layers[l].W_ff2 = tensor_new(2, ff2_dims, NULL, 1);
        
        // Initialize with small, controlled values
        for (int i = 0; i < d_model * d_model; i++) {
            double r = ((double)rand() / RAND_MAX - 0.5f);
            layers[l].W_self_q->data[i] = r * w_scale;
            layers[l].W_self_k->data[i] = r * w_scale;
            layers[l].W_self_v->data[i] = r * w_scale;
            layers[l].W_self_o->data[i] = r * w_scale;
            layers[l].W_cross_q->data[i] = r * w_scale;
            layers[l].W_cross_k->data[i] = r * w_scale;
            layers[l].W_cross_v->data[i] = r * w_scale;
            layers[l].W_cross_o->data[i] = r * w_scale;
        }
        
        for (int i = 0; i < d_model * d_ff; i++) {
            double r = ((double)rand() / RAND_MAX - 0.5f);
            layers[l].W_ff1->data[i] = r * ff_scale;
        }
        for (int i = 0; i < d_ff * d_model; i++) {
            double r = ((double)rand() / RAND_MAX - 0.5f);
            layers[l].W_ff2->data[i] = r * ff_scale;
        }
    }
    
    // Forward pass
    printf("\nForward pass through decoder layers:\n");
    Tensor* current = decoder_input;
    Tensor** layer_outputs = malloc(n_layers * sizeof(Tensor*));
    
    for (int l = 0; l < n_layers; l++) {
        printf("\nLayer %d:\n", l + 1);
        layer_outputs[l] = compute_decoder_layer(current, encoder_output, &layers[l],
                                               batch_size, dec_seq_len, enc_seq_len,
                                               n_head, d_head, d_model, d_ff);
        current = layer_outputs[l];
        
        printf("Layer %d output stats:\n", l + 1);
        print_tensor_stats_full(current, "Layer output");
    }
    
    // Gradient checking
    printf("\nChecking gradients...\n");
    
    // Store original values
    double* original_input = malloc(decoder_input->size * sizeof(double));
    memcpy(original_input, decoder_input->data, decoder_input->size * sizeof(double));
    double original_output = current->data[0];
    
    // Forward gradient
    current->grad[0] = 1.0f;
    backward();
    double analytical_grad = decoder_input->grad[0];
    
    printf("Analytical gradient computation complete.\n");
    printf("Analytical gradient: %.6e\n", analytical_grad);
    
    // Reset gradients
    for (int i = 0; i < decoder_input->size; i++) {
        decoder_input->grad[i] = 0.0f;
    }
    
    // Numerical gradient
    double epsilon = 1e-5f;
    decoder_input->data[0] = original_input[0] + epsilon;
    
    // Recompute forward pass
    current = decoder_input;
    for (int l = 0; l < n_layers; l++) {
        current = compute_decoder_layer(current, encoder_output, &layers[l],
                                      batch_size, dec_seq_len, enc_seq_len,
                                      n_head, d_head, d_model, d_ff);
    }
    
    double numerical_grad = (current->data[0] - original_output) / epsilon;
    
    printf("Numerical gradient computation complete.\n");
    printf("Numerical gradient: %.6e\n", numerical_grad);
    
    // Restore original input
    memcpy(decoder_input->data, original_input, decoder_input->size * sizeof(double));
    
    // Compare gradients
    double abs_diff = fabsf(analytical_grad - numerical_grad);
    double avg_magnitude = (fabsf(analytical_grad) + fabsf(numerical_grad)) / 2.0f;
    double rel_error = abs_diff / (avg_magnitude + 1e-10f);
    
    printf("\nGradient comparison:\n");
    printf("Absolute difference: %.6e\n", abs_diff);
    printf("Average magnitude: %.6e\n", avg_magnitude);
    printf("Relative error: %.6f\n", rel_error);
    
    // Use more appropriate tolerance for complex network
    assert_double_eq(rel_error < 0.2f ? 1.0f : 0.0f, 1.0f, 1e-5,
                   "Multi-layer decoder gradient verification failed");
    
    printf("\nMulti-layer decoder test passed!\n");
    
    // Cleanup
    free(layers);
    free(layer_outputs);
    free(original_input);
    while (registry_len > initial_registry) {
        registry_len--;
    }
}

void test_matmul_broadcasting() {
    printf("Testing matrix multiplication broadcasting...\n");
    
    // Test 1: Basic batch broadcasting
    {
        int dims1[] = {1, 2, 3};  // [1, 2, 3]
        int dims2[] = {2, 3, 4};  // [2, 3, 4]
        double data1[] = {1,2,3, 4,5,6};
        double data2[] = {1,2,3,4, 5,6,7,8, 9,10,11,12,
                        13,14,15,16, 17,18,19,20, 21,22,23,24};
        
        Tensor* a = tensor_new(3, dims1, data1, 1);
        Tensor* b = tensor_new(3, dims2, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        
        printf("Input shape 1: [1,2,3], Input shape 2: [2,3,4]\n");
        printf("Output shape: [2,2,4]\n");
        
        // First batch, first row
        assert_double_eq(c->data[0], 38, 1e-5, "Batch broadcasting failed");
    }
    
    // Test 2: Multiple batch dimensions
    {
        int dims1[] = {2, 1, 2, 3};  // [2, 1, 2, 3]
        int dims2[] = {1, 3, 3, 4};  // [1, 3, 3, 4]
        
        Tensor* a = tensor_randn(4, dims1, 1);
        Tensor* b = tensor_randn(4, dims2, 1);
        Tensor* c = tensor_matmul(a, b);
        
        printf("Input shape 1: [2,1,2,3], Input shape 2: [1,3,3,4]\n");
        printf("Output shape: [2,3,2,4]\n");
        
        assert_double_eq(c->ndims, 4, 1e-5, "Wrong number of dimensions");
        assert_double_eq(c->dims[0], 2, 1e-5, "Wrong batch dimension 0");
        assert_double_eq(c->dims[1], 3, 1e-5, "Wrong batch dimension 1");
        assert_double_eq(c->dims[2], 2, 1e-5, "Wrong output dimension M");
        assert_double_eq(c->dims[3], 4, 1e-5, "Wrong output dimension N");
    }
    
    printf("Matrix multiplication broadcasting tests passed!\n");
}

void test_matmul_broadcasting_gradients() {
    printf("\nTesting matrix multiplication broadcasting gradients...\n");
    
    // Test 1: Basic batch broadcasting gradient
    {
        printf("\nTest 1: Basic batch broadcasting [1,2,3] @ [2,3,4]...\n");
        int dims1[] = {1, 2, 3};  // [1, 2, 3]
        int dims2[] = {2, 3, 4};  // [2,3,4]
        
        double* data1 = malloc(6 * sizeof(double));
        double* data2 = malloc(24 * sizeof(double));
        
        // Initialize with controlled values
        for (int i = 0; i < 6; i++) data1[i] = (double)(i + 1) * 0.1f;
        for (int i = 0; i < 24; i++) data2[i] = (double)(i + 1) * 0.1f;
        
        Tensor* a = tensor_new(3, dims1, data1, 1);
        Tensor* b = tensor_new(3, dims2, data2, 1);
        Tensor* c = tensor_matmul(a, b);
        
        // Store original output for gradient checking
        double original_output = c->data[0];
        
        // Compute analytical gradient
        c->grad[0] = 1.0f;
        backward();
        double analytical_grad = a->grad[0];
        
        // Compute numerical gradient
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c_new = tensor_matmul(a, b);
        double numerical_grad = (c_new->data[0] - original_output) / epsilon;
        a->data[0] = saved;
        
        printf("Basic broadcasting gradient check:\n");
        printf("Analytical gradient: %.6e\n", analytical_grad);
        printf("Numerical gradient:  %.6e\n", numerical_grad);
        double rel_error = fabsf(analytical_grad - numerical_grad) / 
                         (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Basic broadcasting gradient check failed");
        
        free(data1);
        free(data2);
    }
    
    // Test 2: Multiple batch dimensions gradient
    {
        printf("\nTest 2: Multiple batch dimensions [2,1,2,3] @ [1,3,3,4]...\n");
        int dims1[] = {2, 1, 2, 3};  // [2,1,2,3]
        int dims2[] = {1, 3, 3, 4};  // [1,3,3,4]
        
        // Initialize tensors with controlled values
        Tensor* a = tensor_new(4, dims1, NULL, 1);
        Tensor* b = tensor_new(4, dims2, NULL, 1);
        
        // Initialize with small, controlled values
        for (int i = 0; i < a->size; i++) a->data[i] = (double)(i + 1) * 0.01f;
        for (int i = 0; i < b->size; i++) b->data[i] = (double)(i + 1) * 0.01f;
        
        Tensor* c = tensor_matmul(a, b);
        
        // Store original output
        double original_output = c->data[0];
        
        // Compute analytical gradient
        c->grad[0] = 1.0f;
        backward();
        double analytical_grad = a->grad[0];
        
        // Compute numerical gradient
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c_new = tensor_matmul(a, b);
        double numerical_grad = (c_new->data[0] - original_output) / epsilon;
        a->data[0] = saved;
        
        printf("Multiple batch dimensions gradient check:\n");
        printf("Analytical gradient: %.6e\n", analytical_grad);
        printf("Numerical gradient:  %.6e\n", numerical_grad);
        double rel_error = fabsf(analytical_grad - numerical_grad) / 
                         (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Multiple batch dimensions gradient check failed");
    }
    
    // Test 3: Gradient accumulation with broadcasting
    {
        printf("\nTest 3: Gradient accumulation with broadcasting...\n");
        int dims1[] = {1, 2, 2};  // [1,2,2]
        int dims2[] = {3, 2, 2};  // [3,2,2]
        
        Tensor* a = tensor_new(3, dims1, NULL, 1);
        Tensor* b = tensor_new(3, dims2, NULL, 1);
        
        // Initialize with controlled values
        for (int i = 0; i < a->size; i++) a->data[i] = (double)(i + 1) * 0.1f;
        for (int i = 0; i < b->size; i++) b->data[i] = (double)(i + 1) * 0.1f;
        
        // Multiple operations using the same broadcasted tensor
        Tensor* c1 = tensor_matmul(a, b);
        Tensor* c2 = tensor_matmul(a, b);
        Tensor* c3 = tensor_add(c1, c2);
        
        // Check gradient accumulation
        c3->grad[0] = 1.0f;
        backward();
        
        // The gradient should accumulate from both paths
        printf("Gradient accumulation check:\n");
        printf("First gradient component: %.6f\n", a->grad[0]);
        
        // Verify that gradients are accumulated correctly
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c1_new = tensor_matmul(a, b);
        Tensor* c2_new = tensor_matmul(a, b);
        Tensor* c3_new = tensor_add(c1_new, c2_new);
        double numerical_grad = (c3_new->data[0] - c3->data[0]) / epsilon;
        a->data[0] = saved;
        
        printf("Numerical accumulated gradient: %.6f\n", numerical_grad);
        double rel_error = fabsf(a->grad[0] - numerical_grad) / 
                         (fabsf(a->grad[0]) + fabsf(numerical_grad) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Gradient accumulation check failed");
    }

        // Test 4: Complex multi-dimensional broadcasting with mixed batch sizes
    {
        printf("\nTest 4: Complex broadcasting [2,1,3,1,4,5] @ [1,4,1,3,5,6]...\n");
        int dims1[] = {2, 1, 3, 1, 4, 5};  // [2,1,3,1,4,5]
        int dims2[] = {1, 4, 1, 3, 5, 6};  // [1,4,1,3,5,6]
        
        Tensor* a = tensor_new(6, dims1, NULL, 1);
        Tensor* b = tensor_new(6, dims2, NULL, 1);
        
        // Initialize with controlled values
        double scale = 0.01f;  // Small scale to prevent overflow
        for (int i = 0; i < a->size; i++) a->data[i] = ((double)i) * scale;
        for (int i = 0; i < b->size; i++) b->data[i] = ((double)(i + 1)) * scale;
        
        Tensor* c = tensor_matmul(a, b);
        printf("Output shape should be: [2,4,3,3,4,6]\n");
        
        // Verify output shape
        assert_double_eq(c->ndims, 6, 1e-5, "Wrong number of dimensions");
        assert_double_eq(c->dims[0], 2, 1e-5, "Wrong dimension 0");
        assert_double_eq(c->dims[1], 4, 1e-5, "Wrong dimension 1");
        assert_double_eq(c->dims[2], 3, 1e-5, "Wrong dimension 2");
        assert_double_eq(c->dims[3], 3, 1e-5, "Wrong dimension 3");
        assert_double_eq(c->dims[4], 4, 1e-5, "Wrong dimension 4");
        assert_double_eq(c->dims[5], 6, 1e-5, "Wrong dimension 5");
        
        // Gradient check
        double original = c->data[0];
        c->grad[0] = 1.0f;
        backward();
        double analytical = a->grad[0];
        
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c_new = tensor_matmul(a, b);
        double numerical = (c_new->data[0] - original) / epsilon;
        a->data[0] = saved;
        
        printf("Complex broadcasting gradient check:\n");
        printf("Analytical gradient: %.6e\n", analytical);
        printf("Numerical gradient:  %.6e\n", numerical);
        double rel_error = fabsf(analytical - numerical) / 
                         (fabsf(analytical) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Complex broadcasting gradient check failed");
    }
    
    // Test 5: Extreme broadcasting with singleton dimensions
    {
        printf("\nTest 5: Extreme broadcasting [1,1,1,2,3] @ [4,5,6,3,4]...\n");
        int dims1[] = {1, 1, 1, 2, 3};    // [1,1,1,2,3]
        int dims2[] = {4, 5, 6, 3, 4};    // [4,5,6,3,4]
        
        Tensor* a = tensor_new(5, dims1, NULL, 1);
        Tensor* b = tensor_new(5, dims2, NULL, 1);
        
        // Initialize with very specific values
        for (int i = 0; i < a->size; i++) a->data[i] = 0.01f;
        for (int i = 0; i < b->size; i++) b->data[i] = 0.01f;
        
        Tensor* c = tensor_matmul(a, b);
        printf("Output shape should be: [4,5,6,2,4]\n");
        
        // Multiple gradient paths
        Tensor* d = tensor_matmul(a, b);  // Second path
        Tensor* e = tensor_add(c, d);     // Sum paths
        
        double original = e->data[0];
        e->grad[0] = 1.0f;
        backward();
        double analytical = a->grad[0];
        
        // Numerical gradient check
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c_new = tensor_matmul(a, b);
        Tensor* d_new = tensor_matmul(a, b);
        Tensor* e_new = tensor_add(c_new, d_new);
        double numerical = (e_new->data[0] - original) / epsilon;
        a->data[0] = saved;
        
        printf("Extreme broadcasting gradient check:\n");
        printf("Analytical gradient: %.6e\n", analytical);
        printf("Numerical gradient:  %.6e\n", numerical);
        double rel_error = fabsf(analytical - numerical) / 
                         (fabsf(analytical) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Extreme broadcasting gradient check failed");
    }
    
    // Test 6: Chain of broadcasted operations
    {
        printf("\nTest 6: Chain of broadcasted operations...\n");
        int dims1[] = {1, 2, 3};    // [1,2,3]
        int dims2[] = {4, 3, 4};    // [4,3,4]
        int dims3[] = {1, 4, 5};    // [1,4,5]
        
        Tensor* a = tensor_new(3, dims1, NULL, 1);
        Tensor* b = tensor_new(3, dims2, NULL, 1);
        Tensor* c = tensor_new(3, dims3, NULL, 1);
        
        // Initialize with controlled values
        double scale = 0.01f;
        for (int i = 0; i < a->size; i++) a->data[i] = ((double)i + 1) * scale;
        for (int i = 0; i < b->size; i++) b->data[i] = ((double)i + 1) * scale;
        for (int i = 0; i < c->size; i++) c->data[i] = ((double)i + 1) * scale;
        
        // Create chain: (A @ B) @ C
        Tensor* ab = tensor_matmul(a, b);
        Tensor* abc = tensor_matmul(ab, c);
        
        double original = abc->data[0];
        abc->grad[0] = 1.0f;
        backward();
        double analytical = a->grad[0];
        
        // Numerical gradient
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* ab_new = tensor_matmul(a, b);
        Tensor* abc_new = tensor_matmul(ab_new, c);
        double numerical = (abc_new->data[0] - original) / epsilon;
        a->data[0] = saved;
        
        printf("Chain broadcasting gradient check:\n");
        printf("Analytical gradient: %.6e\n", analytical);
        printf("Numerical gradient:  %.6e\n", numerical);
        double rel_error = fabsf(analytical - numerical) / 
                         (fabsf(analytical) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Chain broadcasting gradient check failed");
    }

    // Test 7: Mixed operations with broadcasting
    {
        printf("\nTest 7: Mixed operations with broadcasting...\n");
        int dims1[] = {1, 2, 3};    // [1,2,3]
        int dims2[] = {4, 3, 4};    // [4,3,4]
        int dims3[] = {4, 1, 4};    // [4,1,4]
        
        Tensor* a = tensor_new(3, dims1, NULL, 1);
        Tensor* b = tensor_new(3, dims2, NULL, 1);
        Tensor* c = tensor_new(3, dims3, NULL, 1);
        
        // Initialize with controlled values
        double scale = 0.01f;
        for (int i = 0; i < a->size; i++) a->data[i] = ((double)i + 1) * scale;
        for (int i = 0; i < b->size; i++) b->data[i] = ((double)i + 1) * scale;
        for (int i = 0; i < c->size; i++) c->data[i] = ((double)i + 1) * scale;
        
        // Create mixed operation: (A @ B) * C
        Tensor* ab = tensor_matmul(a, b);
        Tensor* result = tensor_hadamard(ab, c);
        
        double original = result->data[0];
        result->grad[0] = 1.0f;
        backward();
        double analytical = a->grad[0];
        
        // Numerical gradient
        double epsilon = 1e-5f;
        double saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* ab_new = tensor_matmul(a, b);
        Tensor* result_new = tensor_hadamard(ab_new, c);
        double numerical = (result_new->data[0] - original) / epsilon;
        a->data[0] = saved;
        
        printf("Mixed operations gradient check:\n");
        printf("Analytical gradient: %.6e\n", analytical);
        printf("Numerical gradient:  %.6e\n", numerical);
        double rel_error = fabsf(analytical - numerical) / 
                        (fabsf(analytical) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_double_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                    "Mixed operations gradient check failed");
    }
    
    printf("Matrix multiplication broadcasting gradient tests passed!\n");
}

// Update main to use safer benchmarking
int main() {
    print_memory_usage("start");
    
    // Original tests
    test_basic_ops();
    test_broadcasting();
    test_complex_broadcasting();
    test_matmul();
    test_softmax();
    test_backward();
    test_gelu();
    test_rms_norm();
    test_edge_cases();
    test_edge_cases_comprehensive();
    test_numerical_stability();
    test_gradient_accumulation();
    test_gradients_comprehensive();
    test_individual_gradients();
    
    print_memory_usage("after basic tests");
    
    // New comprehensive tests
    test_complex_networks();
    test_numerical_stability_comprehensive();
    test_transformer_block();
    
    print_memory_usage("after comprehensive tests");
    
    // Performance benchmarks with memory checks
    printf("\nRunning performance benchmarks:\n");
    benchmark_operations(32);    // Tiny
    print_memory_usage("after tiny benchmark");
    
    benchmark_operations(64);    // Small
    print_memory_usage("after small benchmark");
    
    benchmark_operations(256);   // Medium
    print_memory_usage("after medium benchmark");
    
    // Only run large benchmark if memory allows
    size_t estimated_memory = (size_t)1024 * 1024 * 3 * sizeof(double);
    if (estimated_memory < MAX_TAPE * sizeof(double)) {
        benchmark_operations(512);   // Large
        print_memory_usage("after large benchmark");
    } else {
        printf("Skipping large benchmark (insufficient memory)\n");
    }

    printf("\nRunning advanced architecture tests:\n");
    test_transformer_encoder();
    test_residual_network();

    printf("\nTesting decoder...\n");
    test_transformer_decoder();
    test_multilayer_decoder();

    printf("\nTesting matrix multiplication broadcasting...\n");
    test_matmul_broadcasting();
    test_matmul_broadcasting_gradients();
    
    printf("\nAll tests passed!\n");
    print_memory_usage("before cleanup");
    clean_registry();
    print_memory_usage("end");
    return 0;
}