#include "grad.h"

void print_tensor(Tensor* t, const char* name) {
    if (t->ndims == 2) {
        printf("%s (%dx%d):\n", name, t->dims[0], t->dims[1]);
        for (int i = 0; i < t->dims[0]; i++) {
            for (int j = 0; j < t->dims[1]; j++) {
                printf("%6.2f ", t->data[i * t->dims[1] + j]);
            }
            printf("\n");
        }
    } else if (t->ndims == 3) {
        printf("%s (%dx%dx%d):\n", name, t->dims[0], t->dims[1], t->dims[2]);
        for (int i = 0; i < t->dims[0]; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < t->dims[1]; j++) {
                for (int k = 0; k < t->dims[2]; k++) {
                    printf("%6.2f ", t->data[i * t->dims[1] * t->dims[2] + j * t->dims[2] + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

void run_test(const char* test_name, void (*test_func)()) {
    printf("\n%s\n", test_name);
    printf("----------------------------------------\n");
    test_func();
    cleanup_tape();
}

void test_basic_slicing() {
    int dims[] = {2, 3, 4};
    float* data = malloc(24 * sizeof(float));
    for (int i = 0; i < 24; i++) data[i] = i;
    Tensor* t = tensor_new(3, dims, data, 1);

    int start[] = {0, 1, 1};
    int end[] = {1, 2, 3};
    Tensor* sliced = tensor_slice(t, start, end);

    print_tensor(t, "Original tensor");
    print_tensor(sliced, "Sliced tensor");

    backward();
    tensor_free(sliced);
    tensor_free(t);
    free(data);
}

void test_slice_gradients() {
    int dims[] = {2, 2};
    float data[] = {1.0, 2.0, 3.0, 4.0};
    Tensor* t = tensor_new(2, dims, data, 1);

    int start[] = {0, 0};
    int end[] = {1, 2};
    Tensor* sliced = tensor_slice(t, start, end);

    print_tensor(t, "Original tensor");
    print_tensor(sliced, "Sliced tensor");

    backward();
    printf("\nGradients in original tensor:\n");
    print_tensor(t, "Gradients");

    tensor_free(sliced);
    tensor_free(t);
}

void test_slice_operations() {
    int dims[] = {2, 3};
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor* t = tensor_new(2, dims, data, 1);

    int start[] = {0, 1};
    int end[] = {2, 2};
    Tensor* sliced = tensor_slice(t, start, end);
    Tensor* activated = tensor_relu(sliced);

    print_tensor(t, "Original tensor");
    print_tensor(sliced, "Sliced tensor");
    print_tensor(activated, "Activated tensor");

    backward();
    tensor_free(activated);
    tensor_free(sliced);
    tensor_free(t);
}

void test_combined_operations() {
    int dims[] = {2, 4, 3};
    float* data = malloc(24 * sizeof(float));
    for (int i = 0; i < 24; i++) data[i] = (float)(i) / 4.0f;
    Tensor* input = tensor_new(3, dims, data, 1);

    int w_dims[] = {2, 2};
    float w_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
    Tensor* weights = tensor_new(2, w_dims, w_data, 1);

    int start[] = {0, 1, 0};
    int end[] = {1, 3, 2};
    Tensor* sliced = tensor_slice(input, start, end);
    int reshape_dims[] = {2, 2};
    Tensor* reshaped = tensor_reshape(sliced, 2, reshape_dims);
    Tensor* matmul_result = tensor_matmul(reshaped, weights);
    Tensor* relu_result = tensor_relu(matmul_result);
    Tensor* final_result = tensor_sigmoid(relu_result);

    print_tensor(input, "Input tensor");
    print_tensor(sliced, "Sliced tensor");
    print_tensor(reshaped, "Reshaped tensor");
    print_tensor(weights, "Weight matrix");
    print_tensor(matmul_result, "Matrix multiplication result");
    print_tensor(final_result, "Final result");

    backward();
    tensor_free(final_result);
    tensor_free(relu_result);
    tensor_free(matmul_result);
    tensor_free(reshaped);
    tensor_free(sliced);
    tensor_free(weights);
    tensor_free(input);
    free(data);
}

void test_permute() {
    int dims[] = {2, 3, 4};
    float* data = malloc(24 * sizeof(float));
    for (int i = 0; i < 24; i++) data[i] = i;
    Tensor* t = tensor_new(3, dims, data, 1);

    int permutation[] = {2, 0, 1};
    Tensor* permuted = tensor_permute(t, permutation);
    Tensor* activated = tensor_relu(permuted);

    print_tensor(t, "Original tensor");
    print_tensor(permuted, "Permuted tensor");

    backward();
    printf("\nGradients in original tensor:\n");
    print_tensor(t, "Gradients");

    tensor_free(activated);
    tensor_free(permuted);
    tensor_free(t);
    free(data);

    // Simple 2D permute test
    int dims2d[] = {2, 3};
    float data2d[] = {1, 2, 3, 4, 5, 6};
    Tensor* t2d = tensor_new(2, dims2d, data2d, 1);
    
    int perm2d[] = {1, 0};
    Tensor* permuted2d = tensor_permute(t2d, perm2d);

    print_tensor(t2d, "Original 2D tensor");
    print_tensor(permuted2d, "Permuted 2D tensor");

    tensor_free(permuted2d);
    tensor_free(t2d);
}

void test_gather() {
    int dims[] = {3, 4};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor* t = tensor_new(2, dims, data, 1);
    
    int indices[] = {2, 1, 0};
    Tensor* gathered = tensor_gather(t, 0, indices, 3);
    Tensor* activated = tensor_relu(gathered);

    print_tensor(t, "Original tensor");
    print_tensor(gathered, "Gathered tensor");

    backward();
    printf("\nGradients in original tensor:\n");
    print_tensor(t, "Gradients");

    tensor_free(activated);
    tensor_free(gathered);
    tensor_free(t);
}

void test_hadamard() {
    int dims[] = {2, 3};
    float data1[] = {1, 2, 3, 4, 5, 6};
    float data2[] = {2, 3, 4, 5, 6, 7};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    Tensor* result = tensor_hadamard(t1, t2);

    print_tensor(t1, "Matrix 1");
    print_tensor(t2, "Matrix 2");
    print_tensor(result, "Hadamard product");

    backward();
    printf("\nGradients:\n");
    print_tensor(t1, "Matrix 1 gradients");
    print_tensor(t2, "Matrix 2 gradients");

    tensor_free(result);
    tensor_free(t1);
    tensor_free(t2);
}

void test_power() {
    int dims[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    float exponent = 2.0;
    
    Tensor* t = tensor_new(2, dims, data, 1);
    Tensor* result = tensor_pow(t, exponent);

    print_tensor(t, "Original tensor");
    print_tensor(result, "Power result");

    backward();
    printf("\nGradients:\n");
    print_tensor(t, "Gradients");

    tensor_free(result);
    tensor_free(t);
}

void test_exponential() {
    int dims[] = {2, 3};
    float data[] = {0, 0.5, 1, -1, -0.5, 0.1};
    
    Tensor* t = tensor_new(2, dims, data, 1);
    Tensor* result = tensor_exp(t);

    print_tensor(t, "Original tensor");
    print_tensor(result, "Exponential result");

    backward();
    printf("\nGradients:\n");
    print_tensor(t, "Gradients");

    tensor_free(result);
    tensor_free(t);
}

void test_reduce_sum() {
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = i + 1;
    Tensor* t = tensor_new(3, dims, data, 1);

    int axes[] = {1};
    Tensor* sum = tensor_reduce_sum(t, axes, 1);

    print_tensor(t, "Original tensor");
    print_tensor(sum, "Sum result");

    backward();
    tensor_free(sum);
    tensor_free(t);
}

void test_reduce_max() {
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = i + 1;
    Tensor* t = tensor_new(3, dims, data, 1);

    int axes[] = {1};
    Tensor* max_result = tensor_reduce_max(t, axes, 1);

    print_tensor(t, "Original tensor");
    print_tensor(max_result, "Max result");

    backward();
    printf("\nGradients in original tensor:\n");
    print_tensor(t, "Gradients");

    tensor_free(max_result);
    tensor_free(t);
}

void test_basic_operations1() {
    // Test matrix multiplication with requires_grad
    int dims1[] = {3, 4};
    int dims2[] = {4, 2};
    float data1[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float data2[] = {1,2, 3,4, 5,6, 7,8};
    
    Tensor* a = tensor_new(2, dims1, data1, 1);
    Tensor* b = tensor_new(2, dims2, data2, 1);
    Tensor* c = tensor_matmul(a, b);
    
    print_tensor(c, "Matrix multiplication result");
    backward();
    print_tensor(a, "Gradient of first matrix");
    print_tensor(b, "Gradient of second matrix");
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_complex_graph() {
    // Create a more complex computational graph
    int dims[] = {3, 3};
    float data1[] = {1,2,3, 4,5,6, 7,8,9};
    float data2[] = {0.1,0.2,0.3, 0.4,0.5,0.6, 0.7,0.8,0.9};
    
    Tensor* x = tensor_new(2, dims, data1, 1);
    Tensor* w = tensor_new(2, dims, data2, 1);
    
    // (relu(x @ w) + sigmoid(w)) * exp(x)
    Tensor* t1 = tensor_matmul(x, w);
    Tensor* t2 = tensor_relu(t1);
    Tensor* t3 = tensor_sigmoid(w);
    Tensor* t4 = tensor_add(t2, t3);
    Tensor* t5 = tensor_exp(x);
    Tensor* result = tensor_hadamard(t4, t5);
    
    print_tensor(result, "Complex graph result");
    backward();
    print_tensor(x, "Gradient of x");
    print_tensor(w, "Gradient of w");
    
    tensor_free(x);
    tensor_free(w);
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    tensor_free(t4);
    tensor_free(t5);
    tensor_free(result);
}

void test_reduction_operations1() {
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = i + 1;
    
    Tensor* x = tensor_new(3, dims, data, 1);
    
    // Test reduce_max
    int axes1[] = {1};
    Tensor* max_result = tensor_reduce_max(x, axes1, 1);
    print_tensor(max_result, "Reduce max result");
    
    // Test reduce_sum
    int axes2[] = {0, 2};
    Tensor* sum_result = tensor_reduce_sum(x, axes2, 2);
    print_tensor(sum_result, "Reduce sum result");
    
    backward();
    print_tensor(x, "Original tensor gradients");
    
    tensor_free(x);
    tensor_free(max_result);
    tensor_free(sum_result);
}

void test_reshape_and_permute() {
    int dims[] = {2, 3, 4};
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = i + 1;
    
    Tensor* x = tensor_new(3, dims, data, 1);
    
    // Test reshape
    int new_dims[] = {4, 6};
    Tensor* reshaped = tensor_reshape(x, 2, new_dims);
    print_tensor(reshaped, "Reshaped tensor");
    
    // Test permute
    int perm[] = {2, 0, 1};
    Tensor* permuted = tensor_permute(x, perm);
    print_tensor(permuted, "Permuted tensor");
    
    backward();
    print_tensor(x, "Original tensor gradients");
    
    tensor_free(x);
    tensor_free(reshaped);
    tensor_free(permuted);
}

void test_slice_and_gather() {
    int dims[] = {4, 5};
    float data[20];
    for (int i = 0; i < 20; i++) data[i] = i + 1;
    
    Tensor* x = tensor_new(2, dims, data, 1);
    
    // Test slice
    int start[] = {1, 2};
    int end[] = {3, 4};
    Tensor* sliced = tensor_slice(x, start, end);
    print_tensor(sliced, "Sliced tensor");
    
    // Test gather
    int indices[] = {0, 2, 1};
    Tensor* gathered = tensor_gather(x, 0, indices, 3);
    print_tensor(gathered, "Gathered tensor");
    
    backward();
    print_tensor(x, "Original tensor gradients");
    
    tensor_free(x);
    tensor_free(sliced);
    tensor_free(gathered);
}

void test_large_scale() {
    // Test with larger dimensions
    int dims1[] = {100, 100};
    int dims2[] = {100, 50};
    float* data1 = malloc(10000 * sizeof(float));
    float* data2 = malloc(5000 * sizeof(float));
    
    for (int i = 0; i < 10000; i++) data1[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < 5000; i++) data2[i] = (float)rand() / RAND_MAX;
    
    Tensor* a = tensor_new(2, dims1, data1, 1);
    Tensor* b = tensor_new(2, dims2, data2, 1);
    
    Tensor* c = tensor_matmul(a, b);
    Tensor* d = tensor_relu(c);
    Tensor* e = tensor_sigmoid(d);
    
    printf("Large scale test completed successfully\n");
    backward();
    printf("Large scale backward pass completed successfully\n");
    
    free(data1);
    free(data2);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);
}

void test_basic_operations2() {
    // Test matrix multiplication
    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {2, 0, 1, 3};
    int dims[] = {2, 2};
    
    Tensor* a = tensor_new(2, dims, a_data, 1);
    Tensor* b = tensor_new(2, dims, b_data, 1);
    
    Tensor* c = tensor_matmul(a, b);
    printf("Matrix Multiplication Result:\n");
    print_tensor(c, "C");
    
    backward();
    printf("Gradients for A:\n");
    for(int i = 0; i < 4; i++) printf("%.2f ", a->grad[i]);
    printf("\n");
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_complex_operations() {
    // Test chain of operations
    float a_data[] = {1, -2, 3, -4, 5, -6};
    int dims[] = {2, 3};
    
    Tensor* a = tensor_new(2, dims, a_data, 1);
    Tensor* b = tensor_relu(a);
    Tensor* c = tensor_sigmoid(b);
    
    printf("Original:\n");
    print_tensor(a, "A");
    printf("After ReLU:\n");
    print_tensor(b, "B");
    printf("After Sigmoid:\n");
    print_tensor(c, "C");
    
    backward();
    printf("Gradients through chain:\n");
    print_tensor(a, "A grad");
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_reduction_operations2() {
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int dims[] = {2, 2, 2};
    Tensor* a = tensor_new(3, dims, data, 1);
    
    int axes[] = {1}; // Reduce along middle dimension
    Tensor* b = tensor_reduce_max(a, axes, 1);
    Tensor* c = tensor_reduce_sum(a, axes, 1);
    
    printf("Original 3D tensor:\n");
    print_tensor(a, "A");
    printf("After max reduction:\n");
    print_tensor(b, "B");
    printf("After sum reduction:\n");
    print_tensor(c, "C");
    
    backward();
    printf("Gradients:\n");
    print_tensor(a, "A grad");
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_reshape_and_slice() {
    float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int dims[] = {2,2,3};
    Tensor* a = tensor_new(3, dims, data, 1);
    
    int new_dims[] = {3,4};
    Tensor* b = tensor_reshape(a, 2, new_dims);
    
    int start[] = {0,1,1};
    int end[] = {2,2,3};
    Tensor* c = tensor_slice(a, start, end);
    
    printf("Original:\n");
    print_tensor(a, "A");
    printf("Reshaped:\n");
    print_tensor(b, "B");
    printf("Sliced:\n");
    print_tensor(c, "C");
    
    backward();
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_gather_and_permute() {
    float data[] = {1,2,3,4,5,6};
    int dims[] = {2,3};
    Tensor* a = tensor_new(2, dims, data, 1);
    
    int indices[] = {2,1,0};
    Tensor* b = tensor_gather(a, 1, indices, 3);
    
    int perm[] = {1,0};
    Tensor* c = tensor_permute(a, perm);
    
    printf("Original:\n");
    print_tensor(a, "A");
    printf("After gather:\n");
    print_tensor(b, "B");
    printf("After permute:\n");
    print_tensor(c, "C");
    
    backward();
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_batch_operations() {
    // Test batch matrix multiplication
    float a_data[] = {1,2,3,4, 5,6,7,8};
    float b_data[] = {1,0, 0,1, 2,1, 1,2};
    int a_dims[] = {2,2,2}; // 2 batches of 2x2 matrices
    int b_dims[] = {2,2,2};
    
    Tensor* a = tensor_new(3, a_dims, a_data, 1);
    Tensor* b = tensor_new(3, b_dims, b_data, 1);
    
    Tensor* c = tensor_matmul(a, b);
    printf("Batch Matrix Multiplication Result:\n");
    print_tensor(c, "C");
    
    backward();
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_gradient_flow() {
    // Test complex computation graph
    float a_data[] = {1,2,3,4};
    int dims[] = {2,2};
    
    Tensor* a = tensor_new(2, dims, a_data, 1);
    Tensor* b = tensor_pow(a, 2.0);
    Tensor* c = tensor_relu(b);
    Tensor* d = tensor_sigmoid(c);
    
    printf("Forward pass results:\n");
    print_tensor(d, "Final output");
    
    backward();
    printf("Gradient flow:\n");
    print_tensor(a, "Initial gradients");
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
}

int main() {
    run_test("Test 1: Basic slicing", test_basic_slicing);
    run_test("Test 2: Slice and compute gradients", test_slice_gradients);
    run_test("Test 3: Slice and perform operations", test_slice_operations);
    run_test("Test 4: Combined operations", test_combined_operations);
    run_test("Test 5: Permute operation", test_permute);
    run_test("Test 6: Gather operation", test_gather);
    run_test("Test 7: Hadamard multiplication", test_hadamard);
    run_test("Test 8: Power operation", test_power);
    run_test("Test 9: Exponential operation", test_exponential);
    run_test("Test 10: Reduce sum operation", test_reduce_sum);
    run_test("Test 11: Reduce max operation", test_reduce_max);
    run_test("Test 12: Basic Operations Test", test_basic_operations1);
    run_test("Test 13: Complex Graph Test", test_complex_graph);
    run_test("Test 14: Reduction Operations Test", test_reduction_operations1);
    run_test("Test 15: Reshape and Permute Test", test_reshape_and_permute);
    run_test("Test 16: Slice and Gather Test", test_slice_and_gather);
    run_test("Test 17: Large Scale Test", test_large_scale);
    run_test("Test 18: Basic Operations Test", test_basic_operations2);
    run_test("Test 19: Complex Operations Test", test_complex_operations);
    run_test("Test 20: Reduction Operations Test", test_reduction_operations2);
    run_test("Test 21: Reshape and Slice Test", test_reshape_and_slice);
    run_test("Test 22: Gather and Permute Test", test_gather_and_permute);
    run_test("Test 23: Batch Operations Test", test_batch_operations);
    run_test("Test 24: Gradient Flow Test", test_gradient_flow);
    return 0;
}