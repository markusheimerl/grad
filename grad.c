#include "grad.h"

int main() {
    // Test 1: Basic slicing
    printf("Test 1: Basic slicing\n");
    {
        int dims[] = {2, 3, 4};
        float* data = malloc(24 * sizeof(float));
        for (int i = 0; i < 24; i++) data[i] = i;
        Tensor* t = tensor_new(3, dims, data, 1);

        int start[] = {0, 1, 1};
        int end[] = {1, 2, 3};
        Tensor* sliced = tensor_slice(t, start, end);

        printf("Original tensor shape: %dx%dx%d\n", t->dims[0], t->dims[1], t->dims[2]);
        printf("Sliced tensor shape: %dx%dx%d\n", sliced->dims[0], sliced->dims[1], sliced->dims[2]);

        printf("\nOriginal tensor:\n");
        for (int i = 0; i < t->dims[0]; i++) {
            for (int j = 0; j < t->dims[1]; j++) {
                for (int k = 0; k < t->dims[2]; k++) {
                    printf("%.1f ", t->data[i * t->dims[1] * t->dims[2] + j * t->dims[2] + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        printf("Sliced tensor:\n");
        for (int i = 0; i < sliced->dims[0]; i++) {
            for (int j = 0; j < sliced->dims[1]; j++) {
                for (int k = 0; k < sliced->dims[2]; k++) {
                    printf("%.1f ", sliced->data[i * sliced->dims[1] * sliced->dims[2] + j * sliced->dims[2] + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        backward();
        cleanup_tape();
        tensor_free(sliced);
        tensor_free(t);
        free(data);
    }

    // Test 2: Slice and compute gradients
    printf("\nTest 2: Slice and compute gradients\n");
    {
        int dims[] = {2, 2};
        float data[] = {1.0, 2.0, 3.0, 4.0};
        Tensor* t = tensor_new(2, dims, data, 1);

        int start[] = {0, 0};
        int end[] = {1, 2};
        Tensor* sliced = tensor_slice(t, start, end);
        
        printf("Original tensor:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.1f ", t->data[i * 2 + j]);
            }
            printf("\n");
        }

        printf("\nSliced tensor:\n");
        for (int j = 0; j < 2; j++) {
            printf("%.1f ", sliced->data[j]);
        }
        printf("\n");

        backward();
        
        printf("\nGradients in original tensor:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.1f ", t->grad[i * 2 + j]);
            }
            printf("\n");
        }

        cleanup_tape();
        tensor_free(sliced);
        tensor_free(t);
    }

    // Test 3: Slice and perform operations
    printf("\nTest 3: Slice and perform operations\n");
    {
        int dims[] = {2, 3};
        float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        Tensor* t = tensor_new(2, dims, data, 1);

        int start[] = {0, 1};
        int end[] = {2, 2};
        Tensor* sliced = tensor_slice(t, start, end);
        Tensor* activated = tensor_relu(sliced);

        printf("Original tensor:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%.1f ", t->data[i * 3 + j]);
            }
            printf("\n");
        }

        printf("\nSliced tensor:\n");
        for (int i = 0; i < 2; i++) {
            printf("%.1f\n", sliced->data[i]);
        }

        printf("\nActivated tensor:\n");
        for (int i = 0; i < 2; i++) {
            printf("%.1f\n", activated->data[i]);
        }

        backward();
        cleanup_tape();
        tensor_free(activated);
        tensor_free(sliced);
        tensor_free(t);
    }

    // Test 4: Combined operations with slicing
    printf("\nTest 4: Combined operations with slicing\n");
    {
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

        printf("Input tensor (first slice):\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%.2f ", input->data[i * 3 + j]);
            }
            printf("\n");
        }

        printf("\nSliced tensor:\n");
        for (int i = 0; i < sliced->dims[0]; i++) {
            for (int j = 0; j < sliced->dims[1]; j++) {
                for (int k = 0; k < sliced->dims[2]; k++) {
                    printf("%.2f ", sliced->data[i * sliced->dims[1] * sliced->dims[2] + j * sliced->dims[2] + k]);
                }
                printf("\n");
            }
        }

        printf("\nReshaped tensor (2x2):\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.2f ", reshaped->data[i * 2 + j]);
            }
            printf("\n");
        }

        printf("\nWeight matrix:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.2f ", weights->data[i * 2 + j]);
            }
            printf("\n");
        }

        printf("\nMatrix multiplication result:\n");
        for (int i = 0; i < matmul_result->dims[0]; i++) {
            for (int j = 0; j < matmul_result->dims[1]; j++) {
                printf("%.2f ", matmul_result->data[i * matmul_result->dims[1] + j]);
            }
            printf("\n");
        }

        printf("\nFinal result (after ReLU and sigmoid):\n");
        for (int i = 0; i < final_result->dims[0]; i++) {
            for (int j = 0; j < final_result->dims[1]; j++) {
                printf("%.4f ", final_result->data[i * final_result->dims[1] + j]);
            }
            printf("\n");
        }

        backward();
        cleanup_tape();
        tensor_free(final_result);
        tensor_free(relu_result);
        tensor_free(matmul_result);
        tensor_free(reshaped);
        tensor_free(sliced);
        tensor_free(weights);
        tensor_free(input);
        free(data);
    }

    // Test 5: Permute operation
    printf("\nTest 5: Permute operation\n");
    {
        int dims[] = {2, 3, 4};
        float* data = malloc(24 * sizeof(float));
        for (int i = 0; i < 24; i++) data[i] = i;
        Tensor* t = tensor_new(3, dims, data, 1);

        int permutation[] = {2, 0, 1};
        Tensor* permuted = tensor_permute(t, permutation);

        printf("Original tensor shape: %dx%dx%d\n", t->dims[0], t->dims[1], t->dims[2]);
        printf("Permuted tensor shape: %dx%dx%d\n", permuted->dims[0], permuted->dims[1], permuted->dims[2]);

        printf("\nOriginal tensor:\n");
        for (int i = 0; i < t->dims[0]; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < t->dims[1]; j++) {
                for (int k = 0; k < t->dims[2]; k++) {
                    printf("%2.0f ", t->data[i * t->dims[1] * t->dims[2] + j * t->dims[2] + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        printf("Permuted tensor:\n");
        for (int i = 0; i < permuted->dims[0]; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < permuted->dims[1]; j++) {
                for (int k = 0; k < permuted->dims[2]; k++) {
                    printf("%2.0f ", permuted->data[i * permuted->dims[1] * permuted->dims[2] + j * permuted->dims[2] + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        Tensor* activated = tensor_relu(permuted);
        backward();

        printf("Gradients in original tensor:\n");
        for (int i = 0; i < t->dims[0]; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < t->dims[1]; j++) {
                for (int k = 0; k < t->dims[2]; k++) {
                    printf("%.1f ", t->grad[i * t->dims[1] * t->dims[2] + j * t->dims[2] + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        cleanup_tape();
        tensor_free(activated);
        tensor_free(permuted);
        tensor_free(t);
        free(data);

        // Simple 2D permute test
        printf("\nSimple 2D permute test:\n");
        int dims2d[] = {2, 3};
        float data2d[] = {1, 2, 3, 4, 5, 6};
        Tensor* t2d = tensor_new(2, dims2d, data2d, 1);
        
        int perm2d[] = {1, 0};
        Tensor* permuted2d = tensor_permute(t2d, perm2d);

        printf("Original 2D tensor (%dx%d):\n", t2d->dims[0], t2d->dims[1]);
        for (int i = 0; i < t2d->dims[0]; i++) {
            for (int j = 0; j < t2d->dims[1]; j++) {
                printf("%2.0f ", t2d->data[i * t2d->dims[1] + j]);
            }
            printf("\n");
        }

        printf("\nPermuted 2D tensor (%dx%d):\n", permuted2d->dims[0], permuted2d->dims[1]);
        for (int i = 0; i < permuted2d->dims[0]; i++) {
            for (int j = 0; j < permuted2d->dims[1]; j++) {
                printf("%2.0f ", permuted2d->data[i * permuted2d->dims[1] + j]);
            }
            printf("\n");
        }

        cleanup_tape();
        tensor_free(permuted2d);
        tensor_free(t2d);
    }

    // Test 6: Gather operation
    printf("\nTest 6: Gather operation\n");
    {
        int dims[] = {3, 4};
        float data[] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        Tensor* t = tensor_new(2, dims, data, 1);
        
        int indices[] = {2, 1, 0};  // Reverse the rows
        Tensor* gathered = tensor_gather(t, 0, indices, 3);
        
        printf("Original tensor:\n");
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                printf("%2.0f ", t->data[i * dims[1] + j]);
            }
            printf("\n");
        }
        
        printf("\nGathered tensor (reversed rows):\n");
        for (int i = 0; i < gathered->dims[0]; i++) {
            for (int j = 0; j < gathered->dims[1]; j++) {
                printf("%2.0f ", gathered->data[i * gathered->dims[1] + j]);
            }
            printf("\n");
        }
        
        Tensor* activated = tensor_relu(gathered);
        backward();
        
        printf("\nGradients in original tensor:\n");
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                printf("%.1f ", t->grad[i * dims[1] + j]);
            }
            printf("\n");
        }
        
        cleanup_tape();
        tensor_free(activated);
        tensor_free(gathered);
        tensor_free(t);
    }

    return 0;
}