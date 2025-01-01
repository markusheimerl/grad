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

Tensor* tensor_hadamard(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    return tensor_exp(tensor_add(tensor_log(a), tensor_log(b)));
}

Tensor* tensor_ones(int ndims, const int* dims) {
    Tensor* t = tensor_new(ndims, dims, NULL, 0);
    for (int i = 0; i < t->size; i++) t->data[i] = 1.0f;
    return t;
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

void print_tensor(Tensor* t, const char* name) {
    printf("%s: shape(", name);
    for (int i = 0; i < t->ndims; i++) printf("%d%s", t->dims[i], i < t->ndims - 1 ? "," : ")");
    printf(" first[%.4f,%.4f] grad[%.4f,%.4f]\n", t->data[0], t->data[1], t->requires_grad ? t->grad[0] : 0.0f, t->requires_grad ? t->grad[1] : 0.0f);
}

int main() {
    // Test 1: Detailed 2D permutation verification
    printf("\nTest 1: Detailed 2D permutation verification\n");
    float data1[] = {1,2,3,4,5,6};
    Tensor* t1 = tensor_new(2, (int[]){2,3}, data1, 1);
    Tensor* p1 = tensor_permute(t1, (int[]){1,0});
    printf("Original (2x3):\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) printf("%.1f ", t1->data[i*3 + j]);
        printf("\n");
    }
    printf("Permuted (3x2):\n");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 2; j++) printf("%.1f ", p1->data[i*2 + j]);
        printf("\n");
    }

    // Test 2: Complex 3D permutation with value verification
    printf("\nTest 2: Complex 3D permutation (2x3x4)\n");
    Tensor* t2 = tensor_new(3, (int[]){2,3,4}, NULL, 1);
    for(int i = 0; i < t2->size; i++) t2->data[i] = i;
    
    printf("Original (2x3x4):\n");
    for(int i = 0; i < 2; i++) {
        printf("Layer %d:\n", i);
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 4; k++) printf("%2.0f ", t2->data[i*12 + j*4 + k]);
            printf("\n");
        }
    }

    Tensor* p2 = tensor_permute(t2, (int[]){2,0,1});
    if (p2) {
        printf("\nPermuted (4x2x3):\n");
        for(int i = 0; i < 4; i++) {
            printf("Layer %d:\n", i);
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < 3; k++) printf("%2.0f ", p2->data[i*6 + j*3 + k]);
                printf("\n");
            }
        }

        // Verify dimensions
        printf("\nDimension verification:\n");
        printf("Original dims: %d x %d x %d\n", t2->dims[0], t2->dims[1], t2->dims[2]);
        printf("Permuted dims: %d x %d x %d\n", p2->dims[0], p2->dims[1], p2->dims[2]);
        printf("Original size: %d, Permuted size: %d\n", t2->size, p2->size);
    }

    // Test 3: Simple permutation verification
    printf("\nTest 3: Simple permutation verification\n");
    float data3[] = {1,2,3,4,5,6,7,8};
    Tensor* t3 = tensor_new(3, (int[]){2,2,2}, data3, 1);
    printf("Original (2x2x2):\n");
    for(int i = 0; i < 2; i++) {
        printf("Layer %d:\n", i);
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) printf("%.1f ", t3->data[i*4 + j*2 + k]);
            printf("\n");
        }
    }

    Tensor* p3 = tensor_permute(t3, (int[]){1,0,2});
    if (p3) {
        printf("\nPermuted (2x2x2):\n");
        for(int i = 0; i < 2; i++) {
            printf("Layer %d:\n", i);
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < 2; k++) printf("%.1f ", p3->data[i*4 + j*2 + k]);
                printf("\n");
            }
        }
    }

    // Test 4: Gradient propagation verification
    printf("\nTest 4: Gradient verification\n");
    float data4[] = {1,2,3,4};
    Tensor* t4 = tensor_new(2, (int[]){2,2}, data4, 1);
    Tensor* p4 = tensor_permute(t4, (int[]){1,0});
    if (p4) {
        for(int i = 0; i < 4; i++) p4->grad[i] = i + 1;
        backward();
        
        printf("Original gradients (2x2):\n");
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) printf("%.1f ", t4->grad[i*2 + j]);
            printf("\n");
        }
    }

    // Test 5: Complex permutation chain
printf("\nTest 5: Complex permutation chain\n");
float data5[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
Tensor* t5 = tensor_new(4, (int[]){2,2,2,2}, data5, 1);

printf("Original (2x2x2x2):\n");
for(int w = 0; w < 2; w++) {
    printf("W=%d:\n", w);
    for(int z = 0; z < 2; z++) {
        printf("Z=%d:\n", z);
        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                printf("%.0f ", t5->data[w*8 + z*4 + y*2 + x]);
            }
            printf("\n");
        }
    }
}

// Chain of permutations
int perms[][4] = {{3,1,2,0}, {0,2,1,3}, {2,3,0,1}};
Tensor* current = t5;
for(int i = 0; i < 3; i++) {
    current = tensor_permute(current, perms[i]);
    printf("\nAfter permutation %d (%d,%d,%d,%d):\n", i+1,
           current->dims[0], current->dims[1], current->dims[2], current->dims[3]);
    for(int w = 0; w < current->dims[0]; w++) {
        printf("W=%d:\n", w);
        for(int z = 0; z < current->dims[1]; z++) {
            printf("Z=%d:\n", z);
            for(int y = 0; y < current->dims[2]; y++) {
                for(int x = 0; x < current->dims[3]; x++) {
                    int idx = w*current->dims[1]*current->dims[2]*current->dims[3] + 
                             z*current->dims[2]*current->dims[3] + 
                             y*current->dims[3] + x;
                    printf("%.0f ", current->data[idx]);
                }
                printf("\n");
            }
        }
    }
}

    clean_registry();
    return 0;
}