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

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) if (a->dims[i] != b->dims[i]) return NULL;
    Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) result->data[i] = a->data[i] + b->data[i];
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){ADD, result, a, b};
    return result;
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
        int x = i, y = 0;
        for (int d = 0; d < a->ndims; d++) y += (x / t[0][d]) * t[1][p[d]], x %= t[0][d];
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
    // Test 1: Simple 2D permutation
    printf("\nTest 1: Simple 2D permutation [1,0]\n");
    float data1[] = {1,2,3,4,5,6};
    Tensor* t1 = tensor_new(2, (int[]){2,3}, data1, 1);
    int perm1[] = {1,0};  // Define permutation array with longer lifetime
    Tensor* p1 = tensor_permute(t1, perm1);
    print_tensor(t1, "Original");
    print_tensor(p1, "Permuted");
    
    // Test 2: 3D permutation
    printf("\nTest 2: 3D permutation [2,0,1]\n");
    float data2[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    Tensor* t2 = tensor_new(3, (int[]){2,2,3}, data2, 1);
    int perm2[] = {2,0,1};  // Define permutation array with longer lifetime
    Tensor* p2 = tensor_permute(t2, perm2);
    print_tensor(t2, "Original");
    print_tensor(p2, "Permuted");

    // Test 3: Identity permutation
    printf("\nTest 3: Identity permutation [0,1,2]\n");
    int perm3[] = {0,1,2};  // Define permutation array with longer lifetime
    Tensor* p3 = tensor_permute(t2, perm3);
    print_tensor(t2, "Original");
    if (p3) print_tensor(p3, "Permuted");

    // Test 4: Gradient propagation
    printf("\nTest 4: Gradient propagation\n");
    float data4[] = {1,2,3,4};
    Tensor* t4 = tensor_new(2, (int[]){2,2}, data4, 1);
    int perm4[] = {1,0};  // Define permutation array with longer lifetime
    Tensor* p4 = tensor_permute(t4, perm4);
    if (p4) {
        p4->grad[0] = 1.0f;
        p4->grad[1] = 2.0f;
        p4->grad[2] = 3.0f;
        p4->grad[3] = 4.0f;
        backward();
        print_tensor(t4, "Original with grad");
        print_tensor(p4, "Permuted with grad");
    }

    // Test 5: Invalid permutations
    printf("\nTest 5: Invalid permutations\n");
    int invalid_perm1[] = {0,0,1};
    int invalid_perm2[] = {0,1,3};
    Tensor* invalid1 = tensor_permute(t2, invalid_perm1);
    Tensor* invalid2 = tensor_permute(t2, invalid_perm2);
    printf("Invalid permutation 1 (duplicate): %s\n", invalid1 ? "Failed" : "Passed");
    printf("Invalid permutation 2 (out of range): %s\n", invalid2 ? "Failed" : "Passed");

    // Test 6: Chained permutations
    printf("\nTest 6: Chained permutations\n");
    Tensor* t6 = tensor_new(3, (int[]){2,3,4}, NULL, 1);
    for(int i = 0; i < t6->size; i++) t6->data[i] = i;
    int perm6a[] = {1,0,2};
    int perm6b[] = {2,1,0};
    Tensor* p6a = tensor_permute(t6, perm6a);
    Tensor* p6b = p6a ? tensor_permute(p6a, perm6b) : NULL;
    print_tensor(t6, "Original");
    if (p6a) print_tensor(p6a, "First permutation");
    if (p6b) print_tensor(p6b, "Second permutation");

    // Test 7: Edge cases
    printf("\nTest 7: Edge cases\n");
    float data7[] = {1,2,3};
    Tensor* t7a = tensor_new(1, (int[]){3}, data7, 1);
    int perm7[] = {0};
    Tensor* p7a = tensor_permute(t7a, perm7);
    Tensor* p7b = tensor_permute(NULL, perm7);
    if (t7a) print_tensor(t7a, "1D tensor");
    if (p7a) print_tensor(p7a, "1D permuted");
    printf("NULL tensor permutation: %s\n", p7b ? "Failed" : "Passed");

    clean_registry();
    return 0;
}