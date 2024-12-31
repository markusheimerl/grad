#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f
#define MAX_DIMS 32

typedef enum { MATMUL, EXP, LOG, ADD, RESHAPE, PERMUTE } OpType;

typedef struct Tensor {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int* perm;
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
    for (int i = 0; i < registry_len; i++) {
        free(registry[i]->data);
        free(registry[i]->grad);
        free(registry[i]->dims);
        free(registry[i]);
    }
    registry_len = 0;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) 
        if (a->dims[i] != b->dims[i]) return NULL;
    
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) 
        result->data[i] = a->data[i] + b->data[i];
    if (result->requires_grad) 
        tape[tape_len++] = (TapeEntry){ADD, result, a, b, NULL};
    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) 
        return NULL;
    
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
                for (int k = 0; k < K; k++)
                    sum += a->data[n*M*K + i*K + k] * b->data[n*K*N + k*N + j];
                result->data[n*M*N + i*N + j] = sum;
            }
    
    if (result->requires_grad) 
        tape[tape_len++] = (TapeEntry){MATMUL, result, a, b, NULL};
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) 
        result->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (result->requires_grad) 
        tape[tape_len++] = (TapeEntry){EXP, result, a, NULL, NULL};
    return result;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) 
        result->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (result->requires_grad) 
        tape[tape_len++] = (TapeEntry){LOG, result, a, NULL, NULL};
    return result;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* new_dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= new_dims[i];
    if (size != a->size) return NULL;
    
    Tensor* result = tensor_new(ndims, new_dims, a->data, a->requires_grad);
    if (result->requires_grad) 
        tape[tape_len++] = (TapeEntry){RESHAPE, result, a, NULL, NULL};
    return result;
}

Tensor* tensor_permute(Tensor* a, const int* perm) {
    if (!a || !perm || a->ndims <= 1) return a ? tensor_new(a->ndims, a->dims, a->data, a->requires_grad) : NULL;
    char u[MAX_DIMS] = {0}; int d[MAX_DIMS], o[MAX_DIMS], n[MAX_DIMS], j, t;
    for (int i = 0; i < a->ndims; i++) if (perm[i] < 0 || perm[i] >= a->ndims || u[perm[i]]) return NULL; else u[perm[i]] = 1;
    for (int i = 0; i < a->ndims; i++) d[i] = a->dims[perm[i]];
    Tensor* r = tensor_new(a->ndims, d, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) {
        for (t = i, j = a->ndims-1; j >= 0; j--) { o[j] = t % a->dims[j]; t /= a->dims[j]; }
        for (j = 0; j < a->ndims; j++) n[j] = o[perm[j]];
        for (t = 0, j = 0; j < a->ndims; j++) t = t * d[j] + n[j];
        r->data[t] = a->data[i];
    }
    if (r->requires_grad) {
        int* p = malloc(a->ndims * sizeof(int));
        for (int i = 0; i < a->ndims; i++) p[perm[i]] = i;
        tape[tape_len++] = (TapeEntry){PERMUTE, r, a, NULL, p};
    }
    return r;
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
                                if (a->requires_grad)
                                    a->grad[n*M*K + i*K + k] += grad * b->data[n*K*N + k*N + j];
                                if (b->requires_grad)
                                    b->grad[n*K*N + k*N + j] += grad * a->data[n*M*K + i*K + k];
                            }
                        }
                break;
            }
            case ADD:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++) 
                        a->grad[i] += result->grad[i];
                if (b->requires_grad)
                    for (int i = 0; i < b->size; i++) 
                        b->grad[i] += result->grad[i];
                break;
            case EXP:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] * result->data[i];
                break;
            case LOG:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += result->grad[i] / fmaxf(a->data[i], MIN_LOG);
                break;
            case RESHAPE:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++) 
                        a->grad[i] += result->grad[i];
                break;
            case PERMUTE:
                if (a->requires_grad) {
                    Tensor* permuted_grad = tensor_permute(tensor_new(result->ndims, result->dims, result->grad, 0), entry->perm);
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += permuted_grad->data[i];
                }
                free(entry->perm);
                break;
        }
    }
    tape_len = 0;
}

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) 
        if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
}

Tensor* tensor_ones(int ndims, const int* dims) {
    Tensor* t = tensor_new(ndims, dims, NULL, 0);  // no grad needed for ones
    for (int i = 0; i < t->size; i++) 
        t->data[i] = 1.0f;
    return t;
}

Tensor* tensor_reduce_sum(Tensor* a, int axis) {
    if (axis < 0 || axis >= a->ndims) return NULL;
    
    // Create permutation to move the reduction axis to the end
    int* perm = malloc(a->ndims * sizeof(int));
    int j = 0;
    for (int i = 0; i < a->ndims; i++)
        if (i != axis) perm[j++] = i;
    perm[a->ndims-1] = axis;
    
    // Permute the tensor to move reduction axis to the end
    Tensor* permuted = tensor_permute(a, perm);
    free(perm);
    
    // Reshape to 2D tensor where the last dimension is the reduction axis
    int new_rows = 1;
    for (int i = 0; i < a->ndims-1; i++)
        new_rows *= permuted->dims[i];
    int new_cols = permuted->dims[a->ndims-1];
    
    int reshape_dims[] = {new_rows, new_cols};
    Tensor* reshaped = tensor_reshape(permuted, 2, reshape_dims);
    
    // Create a column vector of ones with shape (new_cols, 1)
    int ones_dims[] = {new_cols, 1};
    Tensor* ones = tensor_ones(2, ones_dims);
    
    // Matmul will sum along the reduction axis
    Tensor* result = tensor_matmul(reshaped, ones);
    
    // Reshape back to original dimensions minus the reduced axis
    int* final_dims = malloc((a->ndims-1) * sizeof(int));
    j = 0;
    for (int i = 0; i < a->ndims; i++)
        if (i != axis) final_dims[j++] = a->dims[i];
    
    Tensor* final_result = tensor_reshape(result, a->ndims-1, final_dims);
    free(final_dims);
    
    return final_result;
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s: shape(", name);
    for (int i = 0; i < t->ndims; i++) 
        printf("%d%s", t->dims[i], i < t->ndims - 1 ? "," : ")");
    printf(" first[%.4f,%.4f] grad[%.4f,%.4f]\n", 
           t->data[0], t->data[1],
           t->requires_grad ? t->grad[0] : 0.0f,
           t->requires_grad ? t->grad[1] : 0.0f);
}

int main() {
    // Create a 3D tensor with shape (2, 3, 4)
    const int dims[] = {2, 3, 4};
    const int size = 24;  // 2 * 3 * 4
    
    // Initialize data with increasing values
    float* data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        data[i] = i + 1;
    }
    
    // Create input tensor
    Tensor* input = tensor_new(3, dims, data, 1);
    printf("Original tensor:\n");
    print_tensor(input, "input");
    
    // Print the actual values for better visualization
    printf("\nOriginal tensor values:\n");
    for (int i = 0; i < dims[0]; i++) {
        printf("i=%d:\n", i);
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                printf("%.1f ", input->data[i*dims[1]*dims[2] + j*dims[2] + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // Reduce sum along different axes
    printf("\nReducing along different axes:\n");
    
    // Reduce along axis 0 (first dimension)
    Tensor* reduced0 = tensor_reduce_sum(input, 0);
    print_tensor(reduced0, "reduced_axis0");
    
    // Reduce along axis 1 (middle dimension)
    Tensor* reduced1 = tensor_reduce_sum(input, 1);
    print_tensor(reduced1, "reduced_axis1");
    
    // Reduce along axis 2 (last dimension)
    Tensor* reduced2 = tensor_reduce_sum(input, 2);
    print_tensor(reduced2, "reduced_axis2");
    
    // Test backpropagation
    printf("\nTesting backpropagation:\n");
    
    // Set gradients of reduced tensor to 1.0
    for (int i = 0; i < reduced1->size; i++) {
        reduced1->grad[i] = 1.0f;
    }
    
    // Perform backward pass
    backward();
    
    // Print input gradients
    print_tensor(input, "input_grad");
    
    // Clean up
    free(data);
    clean_registry();
    return 0;
}