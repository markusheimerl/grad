#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f
#define MAX_DIMS 32

typedef enum { MATMUL, EXP, LOG, ADD, SUB, RESHAPE } OpType;

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

Tensor* tensor_reshape(Tensor* a, int ndims, const int* new_dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= new_dims[i];
    if (size != a->size) return NULL;
    Tensor* result = tensor_new(ndims, new_dims, a->data, a->requires_grad);
    if (result->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, result, a, NULL};
    return result;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    int max_d = fmax(a->ndims, b->ndims), rd[MAX_DIMS];
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        rd[max_d-1-i] = fmax(d1, d2);
    }
    Tensor* r = tensor_new(max_d, rd, NULL, a->requires_grad || b->requires_grad);
    if (!r) return NULL;

    for (int i = 0; i < r->size; i++) {
        int coords[MAX_DIMS], temp = i;
        for (int d = max_d-1; d >= 0; d--) {
            coords[d] = temp % rd[d];
            temp /= rd[d];
        }
        
        // Calculate indices for a
        int ai = 0, astride = 1;
        for (int d = a->ndims-1; d >= 0; d--) {
            int rd_idx = d + (max_d - a->ndims);
            ai += (a->dims[d] == 1 ? 0 : coords[rd_idx]) * astride;
            astride *= a->dims[d];
        }
        
        // Calculate indices for b
        int bi = 0, bstride = 1;
        for (int d = b->ndims-1; d >= 0; d--) {
            int rd_idx = d + (max_d - b->ndims);
            bi += (b->dims[d] == 1 ? 0 : coords[rd_idx]) * bstride;
            bstride *= b->dims[d];
        }
        
        r->data[i] = a->data[ai] + b->data[bi];
    }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){ADD, r, a, b};
    return r;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    int max_d = fmax(a->ndims, b->ndims), rd[MAX_DIMS];
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        rd[max_d-1-i] = fmax(d1, d2);
    }
    Tensor* r = tensor_new(max_d, rd, NULL, a->requires_grad || b->requires_grad);
    if (!r) return NULL;

    for (int i = 0; i < r->size; i++) {
        int coords[MAX_DIMS], temp = i;
        for (int d = max_d-1; d >= 0; d--) {
            coords[d] = temp % rd[d];
            temp /= rd[d];
        }
        
        // Calculate indices for a
        int ai = 0, astride = 1;
        for (int d = a->ndims-1; d >= 0; d--) {
            int rd_idx = d + (max_d - a->ndims);
            ai += (a->dims[d] == 1 ? 0 : coords[rd_idx]) * astride;
            astride *= a->dims[d];
        }
        
        // Calculate indices for b
        int bi = 0, bstride = 1;
        for (int d = b->ndims-1; d >= 0; d--) {
            int rd_idx = d + (max_d - b->ndims);
            bi += (b->dims[d] == 1 ? 0 : coords[rd_idx]) * bstride;
            bstride *= b->dims[d];
        }
        
        r->data[i] = a->data[ai] - b->data[bi];
    }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){SUB, r, a, b};
    return r;
}

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* e = &tape[t];
        Tensor *r = e->result, *a = e->input1, *b = e->input2;
        switch (e->op) {
            case MATMUL: {
                if (!a->requires_grad && !b->requires_grad) break;
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
                int batch = r->size / (M * N);
                for (int n = 0; n < batch; n++)
                    for (int i = 0; i < M; i++)
                        for (int j = 0; j < N; j++) {
                            float g = r->grad[n*M*N + i*N + j];
                            for (int k = 0; k < K; k++) {
                                if (a->requires_grad) a->grad[n*M*K + i*K + k] += g * b->data[n*K*N + k*N + j];
                                if (b->requires_grad) b->grad[n*K*N + k*N + j] += g * a->data[n*M*K + i*K + k];
                            }
                        }
                break;
            }
            case ADD: case SUB: {
                if (!a->requires_grad && !b->requires_grad) break;
                for (int i = 0; i < r->size; i++) {
                    int coords[MAX_DIMS], temp = i;
                    for (int d = r->ndims-1; d >= 0; d--) {
                        coords[d] = temp % r->dims[d];
                        temp /= r->dims[d];
                    }
                    
                    if (a->requires_grad) {
                        int ai = 0, astride = 1;
                        for (int d = a->ndims-1; d >= 0; d--) {
                            int rd_idx = d + (r->ndims - a->ndims);
                            ai += (a->dims[d] == 1 ? 0 : coords[rd_idx]) * astride;
                            astride *= a->dims[d];
                        }
                        a->grad[ai] += r->grad[i];
                    }
                    
                    if (b->requires_grad) {
                        int bi = 0, bstride = 1;
                        for (int d = b->ndims-1; d >= 0; d--) {
                            int rd_idx = d + (r->ndims - b->ndims);
                            bi += (b->dims[d] == 1 ? 0 : coords[rd_idx]) * bstride;
                            bstride *= b->dims[d];
                        }
                        b->grad[bi] += (e->op == ADD ? 1 : -1) * r->grad[i];
                    }
                }
                break;
            }
            case EXP:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) 
                    a->grad[i] += r->grad[i] * r->data[i];
                break;
            case LOG:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) 
                    a->grad[i] += r->grad[i] / fmaxf(a->data[i], MIN_LOG);
                break;
            case RESHAPE:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) 
                    a->grad[i] += r->grad[i];
                break;
        }
    }
    tape_len = 0;
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

int main() {
    void verify(const char* test, float* expected, float* actual, int size) {
        float eps = 1e-5;
        int pass = 1;
        for(int i = 0; i < size; i++) {
            if (fabs(expected[i] - actual[i]) > eps) {
                pass = 0;
                printf("FAIL: %s at index %d: expected %.1f, got %.1f\n", test, i, expected[i], actual[i]);
            }
        }
        if (pass) printf("PASS: %s\n", test);
    }

    // Test 1: Basic same-shape addition/subtraction
    float d1[] = {1,2,3};
    int dims1[] = {3};
    Tensor* t1 = tensor_new(1, dims1, d1, 1);
    Tensor* t2 = tensor_new(1, dims1, d1, 1);
    verify("Basic Add", (float[]){2,4,6}, tensor_add(t1, t2)->data, 3);
    verify("Basic Sub", (float[]){0,0,0}, tensor_sub(t1, t2)->data, 3);

    // Test 2: Scalar broadcasting
    float scalar[] = {2};
    int dims_scalar[] = {1};
    Tensor* ts = tensor_new(1, dims_scalar, scalar, 1);
    verify("Scalar Add", (float[]){3,4,5}, tensor_add(t1, ts)->data, 3);
    verify("Scalar Sub", (float[]){-1,0,1}, tensor_sub(t1, ts)->data, 3);

    // Test 3: Complex 3D broadcasting
    float d3[] = {1,2,3,4,5,6,7,8,9,10,11,12};  // (2,2,3)
    float d4[] = {1,10,100};                     // (1,1,3)
    int dims3[] = {2,2,3};
    int dims4[] = {1,1,3};
    Tensor* t3 = tensor_new(3, dims3, d3, 1);
    Tensor* t4 = tensor_new(3, dims4, d4, 1);
    float expected_add[] = {2,12,103, 5,15,106, 8,18,109, 11,21,112};
    verify("3D Broadcasting Add", expected_add, tensor_add(t3, t4)->data, 12);

    // Test 4: Broadcasting with empty dimensions
    float d5[] = {1,2,3};  // (1,1,3)
    float d6[] = {1};      // (1,1,1)
    int dims5[] = {1,1,3};
    int dims6[] = {1,1,1};
    Tensor* t5 = tensor_new(3, dims5, d5, 1);
    Tensor* t6 = tensor_new(3, dims6, d6, 1);
    verify("Empty Dims Add", (float[]){2,3,4}, tensor_add(t5, t6)->data, 3);

    // Test 5: Different dimensionality broadcasting
    float d7[] = {1,2,3,4,5,6};  // (2,3)
    float d8[] = {10,20,30};     // (3,)
    int dims7[] = {2,3};
    int dims8[] = {3};
    Tensor* t7 = tensor_new(2, dims7, d7, 1);
    Tensor* t8 = tensor_new(1, dims8, d8, 1);
    float expected_add2[] = {11,22,33, 14,25,36};
    verify("Different Dims Add", expected_add2, tensor_add(t7, t8)->data, 6);

    // Test 6: Edge case - all ones broadcasting
    float d9[] = {5};  // (1,1,1)
    float d10[] = {3}; // (1,1,1)
    int dims9[] = {1,1,1};
    Tensor* t9 = tensor_new(3, dims9, d9, 1);
    Tensor* t10 = tensor_new(3, dims9, d10, 1);
    verify("All Ones Add", (float[]){8}, tensor_add(t9, t10)->data, 1);

    // Test 7: Gradient flow through complex broadcasting
    Tensor* result = tensor_add(tensor_sub(t7, t8), t9);  // ((2,3) - (3,)) + (1,1,1)
    for(int i = 0; i < result->size; i++) result->grad[i] = 1.0;
    backward();
    verify("t7 gradients", (float[]){1,1,1,1,1,1}, t7->grad, 6);
    verify("t8 gradients", (float[]){-2,-2,-2}, t8->grad, 3);
    verify("t9 gradients", (float[]){6}, t9->grad, 1);

    // Test 8: Error cases
    float d11[] = {1,2,3,4};
    int dims11[] = {2,2};
    Tensor* t11 = tensor_new(2, dims11, d11, 1);
    if (tensor_add(t11, t3) != NULL) printf("FAIL: Should return NULL for incompatible shapes\n");
    if (tensor_add(NULL, t11) != NULL) printf("FAIL: Should return NULL for NULL input\n");
    if (tensor_add(t11, NULL) != NULL) printf("FAIL: Should return NULL for NULL input\n");

    // Test 9: Large dimension difference broadcasting
    float d12[] = {1};
    float d13[] = {1,2,3,4,5,6,7,8};
    int dims12[] = {1};
    int dims13[] = {2,2,2};
    Tensor* t12 = tensor_new(1, dims12, d12, 1);
    Tensor* t13 = tensor_new(3, dims13, d13, 1);
    float expected_add3[] = {2,3,4,5,6,7,8,9};
    verify("Large Dim Diff Add", expected_add3, tensor_add(t13, t12)->data, 8);

    clean_registry();
    return 0;
}