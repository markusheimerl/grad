#include "grad.h"

int main() {
    // Create a test tensor
    int dims[] = {2, 3};
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    Tensor* x = tensor_new(2, dims, data, 1);
    
    // Apply GELU
    Tensor* y = tensor_gelu(x);
    
    // Print results
    printf("Input:\n");
    for (int i = 0; i < x->size; i++) {
        printf("%f ", x->data[i]);
    }
    printf("\n\nGELU output:\n");
    for (int i = 0; i < y->size; i++) {
        printf("%f ", y->data[i]);
    }
    printf("\n");
    
    // Test backward pass
    // Set gradient of output to 1.0
    for (int i = 0; i < y->size; i++) {
        y->grad[i] = 1.0f;
    }
    
    backward();
    
    printf("\nGradients:\n");
    for (int i = 0; i < x->size; i++) {
        printf("%f ", x->grad[i]);
    }
    printf("\n");
    
    clean_registry();
    return 0;
}