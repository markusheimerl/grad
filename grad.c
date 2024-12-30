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
    return 0;
}