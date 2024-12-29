#include "grad.h"

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