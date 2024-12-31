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
    if (!a || !perm || a->ndims <= 1) 
        return a ? tensor_new(a->ndims, a->dims, a->data, a->requires_grad) : NULL;
    
    char used[MAX_DIMS] = {0};
    for (int i = 0; i < a->ndims; i++) {
        if (perm[i] < 0 || perm[i] >= a->ndims || used[perm[i]]) return NULL;
        used[perm[i]] = 1;
    }

    int new_dims[MAX_DIMS], old_strides[MAX_DIMS], new_strides[MAX_DIMS];
    for (int i = 0; i < a->ndims; i++) new_dims[i] = a->dims[perm[i]];
    
    Tensor* result = tensor_new(a->ndims, new_dims, NULL, a->requires_grad);
    
    old_strides[a->ndims - 1] = new_strides[a->ndims - 1] = 1;
    for (int i = a->ndims - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * a->dims[i + 1];
        new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
    }

    for (int idx = 0; idx < a->size; idx++) {
        int coords[MAX_DIMS], temp = idx, new_idx = 0;
        for (int i = 0; i < a->ndims; i++) {
            coords[i] = temp / old_strides[i];
            temp %= old_strides[i];
        }
        for (int i = 0; i < a->ndims; i++) 
            new_idx += coords[perm[i]] * new_strides[i];
        result->data[new_idx] = a->data[idx];
    }

    if (result->requires_grad) {
        int* inv_perm = malloc(a->ndims * sizeof(int));
        for (int i = 0; i < a->ndims; i++) inv_perm[perm[i]] = i;
        tape[tape_len++] = (TapeEntry){PERMUTE, result, a, NULL, inv_perm};
    }
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
                if (a->requires_grad && entry->perm) {
                    Tensor* grad_tensor = tensor_new(result->ndims, result->dims, result->grad, 0);
                    if (grad_tensor) {
                        Tensor* permuted_grad = tensor_permute(grad_tensor, entry->perm);
                        if (permuted_grad)
                            for (int i = 0; i < a->size; i++)
                                a->grad[i] += permuted_grad->data[i];
                    }
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
    const int b = 2, c = 3, h = 4, w = 5;
    const int dims4d[] = {b, c, h, w};
    const int size = b * c * h * w;
    
    float *data1 = malloc(size * sizeof(float));
    float *data2 = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) 
        data1[i] = data2[i] = (float)(i % 10) / 10.0f + 0.5f;
    
    Tensor* input1 = tensor_new(4, dims4d, data1, 1);
    print_tensor(input1, "input1");
    
    const int perm[] = {0, 3, 1, 2};
    Tensor* permuted = tensor_permute(input1, perm);
    print_tensor(permuted, "permuted");
    
    int permuted_dims[] = {b, w, c, h};
    Tensor* input2 = tensor_new(4, permuted_dims, data2, 1);
    
    Tensor* sum = tensor_add(permuted, input2);
    print_tensor(sum, "sum");
    
    const int reshape_dims[] = {b, c * h * w};
    Tensor* reshaped = tensor_reshape(sum, 2, reshape_dims);
    print_tensor(reshaped, "reshaped");
    
    const int matrix_dims[] = {c * h * w, 10};
    float* matrix_data = malloc(matrix_dims[0] * matrix_dims[1] * sizeof(float));
    for (int i = 0; i < matrix_dims[0] * matrix_dims[1]; i++) 
        matrix_data[i] = (float)(i % 10) / 20.0f;
    
    Tensor* weight_matrix = tensor_new(2, matrix_dims, matrix_data, 1);
    Tensor* matmul_result = tensor_matmul(reshaped, weight_matrix);
    print_tensor(matmul_result, "matmul");
    
    Tensor* final = tensor_hadamard(
        tensor_log(tensor_exp(matmul_result)),
        tensor_log(tensor_exp(matmul_result))
    );
    print_tensor(final, "final");
    
    for (int i = 0; i < final->size; i++) 
        final->grad[i] = 1.0f;
    
    backward();
    
    printf("\nGradients:\n");
    print_tensor(input1, "input1");
    print_tensor(weight_matrix, "weight");
    
    free(data1);
    free(data2);
    free(matrix_data);
    clean_registry();
    return 0;
}