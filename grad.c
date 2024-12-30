#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { ADD, MATMUL, RELU, SIGMOID, RESHAPE, SLICE, PERMUTE } OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int *slice_start, *slice_end, *permutation;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape;

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-fmaxf(fminf(x, 88.0f), -88.0f))); }
static float d_sigmoid(float x) { float s = sigmoid(x); return s * (1 - s); }
static float relu(float x) { return x > 0 ? x : 0; }
static float d_relu(float x) { return x > 0 ? 1 : 0; }

static int coords_to_index(const int* coords, const int* dims, int ndims) {
    int index = 0, stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        index += coords[i] * stride;
        stride *= dims[i];
    }
    return index;
}

static void index_to_coords(int index, int* coords, const int* dims, int ndims) {
    for (int i = ndims - 1; i >= 0; i--) {
        coords[i] = index % dims[i];
        index /= dims[i];
    }
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = 1;
    for (int i = 0; i < ndims; i++) t->size *= dims[i];
    
    t->data = malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    
    if ((t->requires_grad = requires_grad)) t->grad = calloc(t->size, sizeof(float));
    return t;
}

Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (op == RELU || op == SIGMOID) {
        Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == RELU ? relu(a->data[i]) : sigmoid(a->data[i]);
        if (result->requires_grad)
            tape.entries[tape.len++] = (TapeEntry){op, result, a, NULL, NULL, NULL, NULL};
        return result;
    }

    int out_dims[MAX_DIMS], out_ndims = op == ADD ? a->ndims : MAX(a->ndims, b->ndims);
    if (op == ADD) {
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else {
        for (int i = 0; i < out_ndims - 2; i++) out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[out_ndims-2] = a->dims[a->ndims-2];
        out_dims[out_ndims-1] = b->dims[b->ndims-1];
    }

    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    
    if (op == ADD) {
        for (int i = 0; i < result->size; i++) result->data[i] = a->data[i] + b->data[i];
    } else {
        int batch_size = result->size / (result->dims[out_ndims-2] * result->dims[out_ndims-1]);
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++) {
            float *out = result->data + batch * m * p;
            const float *a_data = a->data + batch * m * n;
            const float *b_data = b->data + batch * n * p;
            
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++) {
                    float aik = a_data[i * n + k];
                    for (int j = 0; j < p; j++)
                        out[i * p + j] += aik * b_data[k * p + j];
                }
        }
    }
    
    if (result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){op, result, a, b, NULL, NULL, NULL};
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)
#define tensor_relu(a) tensor_op(a, NULL, RELU)
#define tensor_sigmoid(a) tensor_op(a, NULL, SIGMOID)

Tensor* tensor_slice(Tensor* t, const int* start, const int* end) {
    int new_dims[MAX_DIMS];
    for (int i = 0; i < t->ndims; i++) {
        new_dims[i] = end[i] - start[i];
        if (start[i] < 0 || end[i] > t->dims[i] || start[i] >= end[i]) return NULL;
    }
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], src_coords[MAX_DIMS];
    for (int i = 0; i < result->size; i++) {
        index_to_coords(i, coords, result->dims, result->ndims);
        for (int j = 0; j < t->ndims; j++) src_coords[j] = coords[j] + start[j];
        result->data[i] = t->data[coords_to_index(src_coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int* slice_start = malloc(t->ndims * sizeof(int));
        int* slice_end = malloc(t->ndims * sizeof(int));
        memcpy(slice_start, start, t->ndims * sizeof(int));
        memcpy(slice_end, end, t->ndims * sizeof(int));
        tape.entries[tape.len++] = (TapeEntry){SLICE, result, t, NULL, slice_start, slice_end, NULL};
    }
    return result;
}

Tensor* tensor_reshape(Tensor* t, int new_ndims, const int* new_dims) {
    int new_size = 1;
    for (int i = 0; i < new_ndims; i++) new_size *= new_dims[i];
    if (new_size != t->size) return NULL;
    
    Tensor* result = tensor_new(new_ndims, new_dims, t->data, t->requires_grad);
    if (result->requires_grad)
        tape.entries[tape.len++] = (TapeEntry){RESHAPE, result, t, NULL, NULL, NULL, NULL};
    return result;
}

Tensor* tensor_permute(Tensor* t, const int* permutation) {
    int new_dims[MAX_DIMS];
    for (int i = 0; i < t->ndims; i++) new_dims[i] = t->dims[permutation[i]];
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], new_coords[MAX_DIMS];
    for (int i = 0; i < t->size; i++) {
        index_to_coords(i, new_coords, result->dims, result->ndims);
        for (int j = 0; j < t->ndims; j++) coords[permutation[j]] = new_coords[j];
        result->data[i] = t->data[coords_to_index(coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int* perm_copy = malloc(t->ndims * sizeof(int));
        memcpy(perm_copy, permutation, t->ndims * sizeof(int));
        tape.entries[tape.len++] = (TapeEntry){PERMUTE, result, t, NULL, NULL, NULL, perm_copy};
    }
    return result;
}

void backward() {
    if (!tape.len) return;
    
    Tensor* final = tape.entries[tape.len - 1].result;
    if (!final->grad) final->grad = calloc(final->size, sizeof(float));
    for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
    
    for (int i = tape.len - 1; i >= 0; i--) {
        TapeEntry* e = &tape.entries[i];
        Tensor *t = e->result, *a = e->input1, *b = e->input2;
        
        if (a->requires_grad && !a->grad) a->grad = calloc(a->size, sizeof(float));
        if (b && b->requires_grad && !b->grad) b->grad = calloc(b->size, sizeof(float));
        
        switch (e->op) {
            case PERMUTE: {
                if (a->requires_grad) {
                    int inverse_perm[MAX_DIMS];
                    for (int j = 0; j < t->ndims; j++) inverse_perm[e->permutation[j]] = j;
                    
                    int old_coords[MAX_DIMS], new_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, old_coords, t->dims, t->ndims);
                        for (int k = 0; k < t->ndims; k++) new_coords[inverse_perm[k]] = old_coords[k];
                        a->grad[coords_to_index(new_coords, a->dims, a->ndims)] += t->grad[j];
                    }
                }
                break;
            }
            case SLICE: {
                if (a->requires_grad) {
                    int coords[MAX_DIMS], src_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, coords, t->dims, t->ndims);
                        for (int k = 0; k < a->ndims; k++) src_coords[k] = coords[k] + e->slice_start[k];
                        a->grad[coords_to_index(src_coords, a->dims, a->ndims)] += t->grad[j];
                    }
                }
                break;
            }
            case RESHAPE:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++) a->grad[j] += t->grad[j];
                break;
            case RELU:
            case SIGMOID:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++)
                        a->grad[j] += t->grad[j] * (e->op == RELU ? d_relu(a->data[j]) : d_sigmoid(a->data[j]));
                break;
            case ADD:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++) a->grad[j] += t->grad[j];
                if (b && b->requires_grad)
                    for (int j = 0; j < t->size; j++) b->grad[j] += t->grad[j];
                break;
            case MATMUL: {
                int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
                int batch_size = t->size / (m * p);
                
                for (int batch = 0; batch < batch_size; batch++) {
                    float *t_grad = t->grad + batch * m * p;
                    float *a_data = a->data + batch * m * n;
                    float *b_data = b->data + batch * n * p;
                    
                    if (a->requires_grad) {
                        float *a_grad = a->grad + batch * m * n;
                        for (int i = 0; i < m; i++)
                            for (int k = 0; k < n; k++)
                                for (int j = 0; j < p; j++)
                                    a_grad[i * n + k] += t_grad[i * p + j] * b_data[k * p + j];
                    }
                    if (b->requires_grad) {
                        float *b_grad = b->grad + batch * n * p;
                        for (int k = 0; k < n; k++)
                            for (int j = 0; j < p; j++)
                                for (int i = 0; i < m; i++)
                                    b_grad[k * p + j] += t_grad[i * p + j] * a_data[i * n + k];
                    }
                }
                break;
            }
        }
    }
}

void cleanup_tape() {
    for (int i = 0; i < tape.len; i++) {
        free(tape.entries[i].slice_start);
        free(tape.entries[i].slice_end);
        free(tape.entries[i].permutation);
    }
    tape.len = 0;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->dims);
    free(t);
}

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

        printf("\nGradients in weight matrix:\n");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.4f ", weights->grad[i * 2 + j]);
            }
            printf("\n");
        }

        printf("\nGradients in input tensor (first slice, non-zero elements):\n");
        for (int i = 1; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                printf("%.4f ", input->grad[i * 3 + j]);
            }
            printf("\n");
        }

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
        // First, test with a simple 2x3x4 tensor
        int dims[] = {2, 3, 4};
        float* data = malloc(24 * sizeof(float));
        for (int i = 0; i < 24; i++) data[i] = i;
        Tensor* t = tensor_new(3, dims, data, 1);

        // Permute from (2,3,4) to (4,2,3)
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

        // Test gradient flow
        Tensor* activated = tensor_relu(permuted);
        backward();

        printf("Gradients in original tensor:\n");
        for (int i = 0; i < t->dims[0]; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < t->dims[1]; j++) {
                for (int k = 0; k < t->dims[2]; k++) {
                    printf("%2.1f ", t->grad[i * t->dims[1] * t->dims[2] + j * t->dims[2] + k]);
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

        // Add a simple 2D example
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

        tensor_free(permuted2d);
        tensor_free(t2d);
    }

    return 0;
}