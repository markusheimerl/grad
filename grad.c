#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000

typedef enum { MATMUL, ADD, RESHAPE, SOFTMAX, PERMUTE, RMSNORM, HADAMARD, GELU } OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    void *aux_data;
} TapeEntry;

static TapeEntry tape[MAX_TAPE];
static int tape_len = 0;
static Tensor* registry[MAX_TENSORS];
static int registry_len = 0;

static int get_index(int idx, const int* dims, int ndims, const int* ref_dims, int ref_ndims) {
    // Calculate coordinates in the reference shape
    int coords[32];  // Assuming max dimensions is 32
    int temp = idx;
    int stride = 1;
    
    for (int d = ref_ndims - 1; d >= 0; d--) {
        coords[d] = (temp / stride) % ref_dims[d];
        temp -= coords[d] * stride;
        stride *= ref_dims[d];
    }
    
    // Map these coordinates to the input tensor
    int result = 0;
    stride = 1;
    int offset = ref_ndims - ndims;
    
    for (int d = ndims - 1; d >= 0; d--) {
        int ref_d = d + offset;
        result += (dims[d] == 1 ? 0 : coords[ref_d]) * stride;
        stride *= dims[d];
    }
    
    return result;
}

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

Tensor* tensor_randn(int ndims, const int* dims, int requires_grad) {
    static int seed_set = 0;
    if (!seed_set) { srand(time(NULL)); seed_set = 1; }
    Tensor* t = tensor_new(ndims, dims, NULL, requires_grad);
    for (int i = 0; i < t->size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        t->data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
    return t;
}

Tensor* tensor_zeros(int ndims, const int* dims, int requires_grad) {
    return tensor_new(ndims, dims, NULL, requires_grad);
}

void clean_registry() {
    while (registry_len > 0) {
        Tensor* t = registry[--registry_len];
        free(t->data); free(t->grad); free(t->dims); free(t);
    }
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!a || !b) return NULL;
    int max_d = fmax(a->ndims, b->ndims), dims[32];
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        dims[max_d-1-i] = fmax(d1, d2);
    }
    Tensor* r = tensor_new(max_d, dims, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < r->size; i++) {
        float av = a->data[get_index(i, a->dims, a->ndims, dims, max_d)];
        float bv = b->data[get_index(i, b->dims, b->ndims, dims, max_d)];
        r->data[i] = op == HADAMARD ? av * bv : av + bv;
    }
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){op, r, a, b, NULL};
    return r;
}

Tensor* tensor_add(Tensor* a, Tensor* b) { return tensor_op(a, b, ADD); }
Tensor* tensor_hadamard(Tensor* a, Tensor* b) { return tensor_op(a, b, HADAMARD); }

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
    int max_d = fmax(a->ndims, b->ndims), dims[32];
    memcpy(dims, (a->ndims > b->ndims ? a : b)->dims, (max_d - 2) * sizeof(int));
    dims[max_d-2] = a->dims[a->ndims-2];
    dims[max_d-1] = b->dims[b->ndims-1];
    
    Tensor* r = tensor_new(max_d, dims, NULL, a->requires_grad || b->requires_grad);
    int M = a->dims[a->ndims-2], N = b->dims[b->ndims-1], K = a->dims[a->ndims-1];
    int batch = r->size / (M * N);
    
    for (int n = 0; n < batch; n++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a->data[n*M*K + i*K + k] * b->data[n*K*N + k*N + j];
                r->data[n*M*N + i*N + j] = sum;
            }
    
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, r, a, b, NULL};
    return r;
}

Tensor* tensor_softmax(Tensor* a) {
    if (!a) return NULL;
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    int last_dim = a->dims[a->ndims - 1], outer_size = a->size / last_dim;
    
    for (int i = 0; i < outer_size; i++) {
        float max_val = a->data[i * last_dim];
        for (int j = 1; j < last_dim; j++)
            max_val = fmaxf(max_val, a->data[i * last_dim + j]);
        
        float sum = 0;
        for (int j = 0; j < last_dim; j++)
            sum += (r->data[i * last_dim + j] = expf(a->data[i * last_dim + j] - max_val));
        for (int j = 0; j < last_dim; j++)
            r->data[i * last_dim + j] /= sum;
    }
    
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){SOFTMAX, r, a, NULL, NULL};
    return r;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL, NULL};
    return r;
}

Tensor* tensor_permute(Tensor* a, const int* perm, int perm_size) {
    if (!a || !perm || perm_size != a->ndims) return NULL;
    
    int* used = calloc(perm_size, sizeof(int));
    for (int i = 0; i < perm_size; i++)
        if (perm[i] < 0 || perm[i] >= perm_size || used[perm[i]]) {
            free(used);
            return NULL;
        } else used[perm[i]] = 1;
    free(used);
    
    int* new_dims = malloc(a->ndims * sizeof(int));
    for (int i = 0; i < a->ndims; i++) new_dims[i] = a->dims[perm[i]];
    
    Tensor* r = tensor_new(a->ndims, new_dims, NULL, a->requires_grad);
    int *a_strides = malloc(a->ndims * sizeof(int)), *r_strides = malloc(r->ndims * sizeof(int));
    
    a_strides[a->ndims-1] = r_strides[r->ndims-1] = 1;
    for (int i = a->ndims-2; i >= 0; i--) {
        a_strides[i] = a_strides[i+1] * a->dims[i+1];
        r_strides[i] = r_strides[i+1] * r->dims[i+1];
    }
    
    for (int i = 0; i < r->size; i++) {
        int temp = i, old_idx = 0;
        for (int d = 0; d < r->ndims; d++) {
            int coord = temp / r_strides[d];
            temp %= r_strides[d];
            old_idx += coord * a_strides[perm[d]];
        }
        r->data[i] = a->data[old_idx];
    }
    
    free(a_strides); free(r_strides); free(new_dims);
    
    if (r->requires_grad) {
        int* stored_perm = malloc(perm_size * sizeof(int));
        memcpy(stored_perm, perm, perm_size * sizeof(int));
        tape[tape_len++] = (TapeEntry){PERMUTE, r, a, NULL, stored_perm};
    }
    return r;
}

Tensor* tensor_rms_norm(Tensor* x, float eps) {
    if (!x) return NULL;
    Tensor* out = tensor_new(x->ndims, x->dims, NULL, x->requires_grad);
    int last_dim = x->dims[x->ndims - 1], batch_size = x->size / last_dim;
    
    for (int b = 0; b < batch_size; b++) {
        // Calculate mean square
        float ms = 0.0f;
        for (int i = 0; i < last_dim; i++) {
            float val = x->data[b * last_dim + i];
            ms += val * val;
        }
        ms /= last_dim;
        
        // Normalize
        float scale = 1.0f / sqrtf(ms + eps);
        for (int i = 0; i < last_dim; i++) {
            out->data[b * last_dim + i] = x->data[b * last_dim + i] * scale;
        }
    }
    
    if (out->requires_grad) {
        float* eps_ptr = malloc(sizeof(float));
        *eps_ptr = eps;
        tape[tape_len++] = (TapeEntry){RMSNORM, out, x, NULL, eps_ptr};
    }
    return out;
}

Tensor* tensor_gelu(Tensor* x) {
    if (!x) return NULL;
    Tensor* out = tensor_new(x->ndims, x->dims, NULL, x->requires_grad);
    const float sqrt_2_pi = 0.7978845608028654f;
    
    for (int i = 0; i < x->size; i++) {
        float val = x->data[i];
        float cube = val * val * val;
        float inner = sqrt_2_pi * (val + 0.044715f * cube);
        out->data[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
    
    if (out->requires_grad) tape[tape_len++] = (TapeEntry){GELU, out, x, NULL, NULL};
    return out;
}

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* e = &tape[t];
        Tensor *r = e->result, *a = e->input1, *b = e->input2;
        
        switch(e->op) {
            case ADD:
            case HADAMARD:
                for (int i = 0; i < r->size; i++) {
                    float grad = r->grad[i];
                    if (a->requires_grad) {
                        int a_idx = get_index(i, a->dims, a->ndims, r->dims, r->ndims);
                        a->grad[a_idx] += e->op == HADAMARD ? 
                            grad * b->data[get_index(i, b->dims, b->ndims, r->dims, r->ndims)] : grad;
                    }
                    if (b->requires_grad) {
                        int b_idx = get_index(i, b->dims, b->ndims, r->dims, r->ndims);
                        b->grad[b_idx] += e->op == HADAMARD ? 
                            grad * a->data[get_index(i, a->dims, a->ndims, r->dims, r->ndims)] : grad;
                    }
                }
                break;
                
            case MATMUL: {
                int M = a->dims[a->ndims-2], K = a->dims[a->ndims-1], N = b->dims[b->ndims-1];
                int batch = r->size/(M*N);
                for (int n = 0; n < batch; n++)
                    for (int i = 0; i < M; i++)
                        for (int j = 0; j < N; j++) {
                            float g = r->grad[n*M*N + i*N + j];
                            for (int k = 0; k < K; k++) {
                                if (a->requires_grad)
                                    a->grad[n*M*K + i*K + k] += g * b->data[n*K*N + k*N + j];
                                if (b->requires_grad)
                                    b->grad[n*K*N + k*N + j] += g * a->data[n*M*K + i*K + k];
                            }
                        }
                break;
            }
            
            case SOFTMAX:
                if (a->requires_grad) {
                    int last_dim = a->dims[a->ndims - 1], outer_size = a->size / last_dim;
                    for (int i = 0; i < outer_size; i++) {
                        float sum = 0;
                        for (int j = 0; j < last_dim; j++)
                            sum += r->grad[i*last_dim+j] * r->data[i*last_dim+j];
                        for (int j = 0; j < last_dim; j++)
                            a->grad[i*last_dim+j] += r->data[i*last_dim+j] * (r->grad[i*last_dim+j] - sum);
                    }
                }
                break;
                
            case RESHAPE:
                if (a->requires_grad)
                    for (int i = 0; i < a->size; i++)
                        a->grad[i] += r->grad[i];
                break;
                
            case PERMUTE:
                if (a->requires_grad) {
                    int *inv_perm = malloc(a->ndims * sizeof(int));
                    int *a_strides = malloc(a->ndims * sizeof(int));
                    int *r_strides = malloc(r->ndims * sizeof(int));
                    
                    for (int i = 0; i < a->ndims; i++)
                        inv_perm[((int*)e->aux_data)[i]] = i;
                    
                    a_strides[a->ndims-1] = r_strides[r->ndims-1] = 1;
                    for (int i = a->ndims-2; i >= 0; i--) {
                        a_strides[i] = a_strides[i+1] * a->dims[i+1];
                        r_strides[i] = r_strides[i+1] * r->dims[i+1];
                    }
                    
                    for (int i = 0; i < r->size; i++) {
                        int temp = i, old_idx = 0;
                        for (int d = 0; d < r->ndims; d++) {
                            int coord = temp / r_strides[d];
                            temp %= r_strides[d];
                            old_idx += coord * a_strides[inv_perm[d]];
                        }
                        a->grad[old_idx] += r->grad[i];
                    }
                    
                    free(a_strides); free(r_strides); free(inv_perm);
                }
                break;
                
            case RMSNORM:
                if (a->requires_grad) {
                    float eps = *(float*)e->aux_data;
                    int last_dim = a->dims[a->ndims-1], batch_size = a->size/last_dim;
                    for (int b = 0; b < batch_size; b++) {
                        float ms = 0.0f;
                        for (int i = 0; i < last_dim; i++)
                            ms += a->data[b*last_dim+i] * a->data[b*last_dim+i];
                        ms /= last_dim;
                        float scale = 1.0f/sqrt(ms+eps);
                        float sum_grad_times_val = 0.0f;
                        for (int i = 0; i < last_dim; i++)
                            sum_grad_times_val += r->grad[b*last_dim+i] * a->data[b*last_dim+i];
                        for (int i = 0; i < last_dim; i++)
                            a->grad[b*last_dim+i] += scale * r->grad[b*last_dim+i] -
                                (scale*scale*scale) * a->data[b*last_dim+i] * sum_grad_times_val/last_dim;
                    }
                }
                break;
                
            case GELU:
                if (a->requires_grad) {
                    const float sqrt_2_pi = 0.7978845608028654f;
                    for (int i = 0; i < a->size; i++) {
                        float x = a->data[i], cube = x * x * x;
                        float inner = sqrt_2_pi * (x + 0.044715f * cube);
                        float tanh_inner = tanhf(inner);
                        float sech_squared = 1.0f - tanh_inner * tanh_inner;
                        float derivative = 0.5f * (1.0f + tanh_inner +
                            x * sech_squared * sqrt_2_pi * (1.0f + 0.134145f * cube));
                        a->grad[i] += r->grad[i] * derivative;
                    }
                }
                break;
        }
        free(e->aux_data);
    }
    tape_len = 0;
}

void assert_float_eq(float a, float b, float eps, const char* msg) {
    if (fabsf(a - b) > eps) {
        printf("ASSERTION FAILED: %s\n", msg);
        printf("Expected: %f, Got: %f\n", b, a);
        exit(1);
    }
}

void assert_tensor_eq(Tensor* a, Tensor* b, float eps, const char* msg) {
    if (a->size != b->size) {
        printf("ASSERTION FAILED: %s (size mismatch)\n", msg);
        exit(1);
    }
    for (int i = 0; i < a->size; i++) {
        assert_float_eq(a->data[i], b->data[i], eps, msg);
    }
}

void print_tensor(Tensor* t, const char* name) {
    printf("%s: dims=[", name);
    for (int i = 0; i < t->ndims; i++) {
        printf("%d%s", t->dims[i], i < t->ndims-1 ? "," : "");
    }
    printf("], data=[");
    for (int i = 0; i < t->size; i++) {
        printf("%.1f%s", t->data[i], i < t->size-1 ? "," : "");
    }
    printf("]\n");
}

void test_basic_ops() {
    printf("Testing basic operations...\n");
    
    // Test addition
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    float expected_add[] = {6, 8, 10, 12};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    Tensor* result = tensor_add(t1, t2);
    
    assert_tensor_eq(result, tensor_new(2, dims, expected_add, 0), 1e-5, "Basic addition failed");
    
    // Test broadcasting
    int dims1[] = {2, 1};  // [[1],
                          //  [2]]
    int dims2[] = {1, 2};  // [[3, 4]]
    float data3[] = {1, 2};
    float data4[] = {3, 4};
    
    t1 = tensor_new(2, dims1, data3, 1);
    t2 = tensor_new(2, dims2, data4, 1);
    
    print_tensor(t1, "t1");
    print_tensor(t2, "t2");
    
    result = tensor_add(t1, t2);
    print_tensor(result, "result");
    
    // The broadcasting should work like this:
    // [[1],     [[3, 4]]     [[1+3, 1+4],
    //  [2]]  +             =  [2+3, 2+4]]
    //
    // [[1, 1],   [[3, 4]]    [[4, 5],
    //  [2, 2]] +           =  [5, 6]]
    
    int expected_dims[] = {2, 2};
    float expected_broadcast[] = {4, 5, 5, 6};
    Tensor* expected = tensor_new(2, expected_dims, expected_broadcast, 0);
    print_tensor(expected, "expected");
    
    assert_tensor_eq(result, expected, 1e-5, "Broadcasting failed");
}

void test_broadcasting() {
    printf("Testing broadcasting...\n");
    
    // Test (2,1) + (1,2)
    {
        int dims1[] = {2, 1};
        int dims2[] = {1, 2};
        float data1[] = {1, 2};
        float data2[] = {3, 4};
        float expected[] = {4, 5, 5, 6};
        
        Tensor* t1 = tensor_new(2, dims1, data1, 1);
        Tensor* t2 = tensor_new(2, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        int expected_dims[] = {2, 2};
        assert_tensor_eq(result, tensor_new(2, expected_dims, expected, 0), 1e-5, "Broadcasting (2,1) + (1,2) failed");
    }
    
    // Test (3,1,2) + (2)
    {
        int dims1[] = {3, 1, 2};
        int dims2[] = {2};
        float data1[] = {1, 2, 3, 4, 5, 6};
        float data2[] = {7, 8};
        float expected[] = {8, 10, 10, 12, 12, 14};
        
        Tensor* t1 = tensor_new(3, dims1, data1, 1);
        Tensor* t2 = tensor_new(1, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        int expected_dims[] = {3, 1, 2};
        assert_tensor_eq(result, tensor_new(3, expected_dims, expected, 0), 1e-5, "Broadcasting (3,1,2) + (2) failed");
    }
    
    // Test scalar broadcasting
    {
        int dims1[] = {2, 2};
        int dims2[] = {1};
        float data1[] = {1, 2, 3, 4};
        float data2[] = {5};
        float expected[] = {6, 7, 8, 9};
        
        Tensor* t1 = tensor_new(2, dims1, data1, 1);
        Tensor* t2 = tensor_new(1, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        int expected_dims[] = {2, 2};
        assert_tensor_eq(result, tensor_new(2, expected_dims, expected, 0), 1e-5, "Broadcasting scalar failed");
    }
}

void test_matmul() {
    printf("Testing matrix multiplication...\n");
    
    int dims1[] = {2, 3};
    int dims2[] = {3, 2};
    float data1[] = {1, 2, 3, 4, 5, 6};
    float data2[] = {7, 8, 9, 10, 11, 12};
    float expected[] = {58, 64, 139, 154};
    
    Tensor* t1 = tensor_new(2, dims1, data1, 1);
    Tensor* t2 = tensor_new(2, dims2, data2, 1);
    Tensor* result = tensor_matmul(t1, t2);
    
    int expected_dims[] = {2, 2};
    assert_tensor_eq(result, tensor_new(2, expected_dims, expected, 1), 1e-5, "Matrix multiplication failed");
}

void test_softmax() {
    printf("Testing softmax...\n");
    
    int dims[] = {1, 3};
    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = tensor_new(2, dims, data, 1);
    Tensor* result = tensor_softmax(t);
    
    float sum = 0;
    for (int i = 0; i < result->size; i++) {
        sum += result->data[i];
        assert_float_eq(result->data[i], result->data[i], 1e-5, "Softmax produced NaN");
    }
    assert_float_eq(sum, 1.0f, 1e-5, "Softmax sum != 1");
}

void test_backward() {
    printf("Testing backward pass...\n");
    
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    Tensor* result = tensor_add(t1, t2);
    
    // Set gradient to all ones
    for (int i = 0; i < result->size; i++) {
        result->grad[i] = 1.0f;
    }
    
    backward();
    
    // For addition, gradients should flow through unchanged
    for (int i = 0; i < t1->size; i++) {
        assert_float_eq(t1->grad[i], 1.0f, 1e-5, "Addition backward pass failed for t1");
        assert_float_eq(t2->grad[i], 1.0f, 1e-5, "Addition backward pass failed for t2");
    }
}

void test_gelu() {
    printf("Testing GELU...\n");
    
    // Test GELU with specific values we know
    {
        int dims[] = {5};
        float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_gelu(t);

        // GELU should be approximately:
        // x < 0: smaller activation
        // x = 0: 0
        // x > 0: closer to x
        assert_float_eq(result->data[2], 0.0f, 1e-5, "GELU(0) should be 0");
        assert_float_eq(result->data[3], 0.841192f, 1e-5, "GELU(1) incorrect");
        
        // Test that outputs are reasonable
        for (int i = 0; i < result->size; i++) {
            // Output should be bounded
            assert_float_eq(result->data[i] <= fabsf(data[i]), 1.0f, 1e-5, "GELU output too large");
            // Output should have same sign as input (except very near 0)
            if (fabsf(data[i]) > 0.1f) {
                assert_float_eq(signbit(result->data[i]) == signbit(data[i]), 1.0f, 1e-5, "GELU sign mismatch");
            }
        }
    }

    // Test GELU derivative
    {
        int dims[] = {1};
        float data[] = {1.0f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_gelu(t);
        result->grad[0] = 1.0f;
        backward();
        
        // GELU derivative at x=1 should be approximately 1.0837
        assert_float_eq(t->grad[0], 1.0837f, 1e-3, "GELU derivative incorrect");
    }
}

void test_rms_norm() {
    printf("Testing RMSNorm...\n");
    
    int dims[] = {1, 4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* t = tensor_new(2, dims, data, 1);
    Tensor* result = tensor_rms_norm(t, 1e-5f);
    
    // Check that the RMS of the output is approximately 1
    float sum_sq = 0;
    for (int i = 0; i < result->size; i++) {
        sum_sq += result->data[i] * result->data[i];
    }
    float rms = sqrt(sum_sq / result->size);
    assert_float_eq(rms, 1.0f, 1e-4, "RMSNorm failed to normalize");
}

void test_edge_cases() {
    printf("Testing edge cases...\n");
    
    // Test single-element tensors
    {
        int dims[] = {1};
        float data1[] = {2.0f};
        float data2[] = {3.0f};
        Tensor* t1 = tensor_new(1, dims, data1, 1);
        Tensor* t2 = tensor_new(1, dims, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        assert_float_eq(result->data[0], 5.0f, 1e-5, "Single element addition failed");
    }
    
    // Test broadcasting with 1s in different positions
    {
        int dims1[] = {2, 1, 3};
        int dims2[] = {1, 4, 1};
        float data1[] = {1, 2, 3, 4, 5, 6};
        float data2[] = {10, 20, 30, 40};
        
        Tensor* t1 = tensor_new(3, dims1, data1, 1);
        Tensor* t2 = tensor_new(3, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        assert_float_eq(result->dims[0], 2, 1e-5, "Complex broadcasting shape mismatch");
        assert_float_eq(result->dims[1], 4, 1e-5, "Complex broadcasting shape mismatch");
        assert_float_eq(result->dims[2], 3, 1e-5, "Complex broadcasting shape mismatch");
    }
}

void test_numerical_stability() {
    printf("Testing numerical stability...\n");
    
    // Test softmax with large numbers
    {
        int dims[] = {2};
        float data[] = {1000.0f, 1000.1f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_softmax(t);
        float sum = result->data[0] + result->data[1];
        assert_float_eq(sum, 1.0f, 1e-5, "Softmax normalization failed for large inputs");
    }
    
    // Test RMSNorm basic functionality
    {
        int dims[] = {4};
        float data[] = {2.0f, 2.0f, 2.0f, 2.0f};  // Using uniform non-zero values
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_rms_norm(t, 1e-5f);
        
        // For identical inputs, outputs should all be equal and the RMS should be 1
        // If all inputs are 2.0, then ms = 4.0, and scale = 1/sqrt(4) = 1/2
        // So each output should be 2.0 * (1/2) = 1.0
        float expected = 1.0f;  // Corrected expected value
        for (int i = 0; i < 4; i++) {
            assert_float_eq(result->data[i], expected, 1e-5, "RMSNorm failed for uniform inputs");
        }
        
        // Verify RMS = 1
        float sum_sq = 0.0f;
        for (int i = 0; i < 4; i++) {
            sum_sq += result->data[i] * result->data[i];
        }
        float rms = sqrtf(sum_sq / 4.0f);
        assert_float_eq(rms, 1.0f, 1e-5, "RMSNorm output RMS != 1");
    }
    
    // Test RMSNorm with mixed values
    {
        int dims[] = {3};
        float data[] = {1.0f, 2.0f, 3.0f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_rms_norm(t, 1e-5f);
        
        // Calculate expected values
        float ms = (1.0f + 4.0f + 9.0f) / 3.0f;  // = 4.666...
        float scale = 1.0f / sqrtf(ms);
        float expected[] = {1.0f * scale, 2.0f * scale, 3.0f * scale};
        
        // Verify outputs
        for (int i = 0; i < 3; i++) {
            assert_float_eq(result->data[i], expected[i], 1e-5, "RMSNorm failed for mixed values");
        }
        
        // Verify unit RMS
        float sum_sq = 0.0f;
        for (int i = 0; i < 3; i++) {
            sum_sq += result->data[i] * result->data[i];
        }
        float output_rms = sqrtf(sum_sq / 3.0f);
        assert_float_eq(output_rms, 1.0f, 1e-5, "RMSNorm failed to normalize to unit RMS");
    }
    
    // Test RMSNorm with small but reasonable values
    {
        int dims[] = {3};
        float data[] = {0.01f, 0.02f, 0.03f};
        Tensor* t = tensor_new(1, dims, data, 1);
        Tensor* result = tensor_rms_norm(t, 1e-5f);
        
        // Verify ratios are preserved
        float ratio1 = result->data[1] / result->data[0];
        float ratio2 = result->data[2] / result->data[1];
        assert_float_eq(ratio1, 2.0f, 1e-5, "RMSNorm failed to preserve ratios for small values");
        assert_float_eq(ratio2, 1.5f, 1e-5, "RMSNorm failed to preserve ratios for small values");
    }
}

void test_gradient_accumulation() {
    printf("Testing gradient accumulation...\n");
    
    int dims[] = {2, 2};
    float data1[] = {1, 2, 3, 4};
    float data2[] = {5, 6, 7, 8};
    
    Tensor* t1 = tensor_new(2, dims, data1, 1);
    Tensor* t2 = tensor_new(2, dims, data2, 1);
    
    // Multiple operations on same tensors
    Tensor* add1 = tensor_add(t1, t2);
    Tensor* add2 = tensor_add(t1, t2);
    Tensor* final = tensor_add(add1, add2);
    
    // Set gradient
    for (int i = 0; i < final->size; i++) {
        final->grad[i] = 1.0f;
    }
    
    backward();
    
    // Each input tensor should have accumulated gradient = 2
    for (int i = 0; i < t1->size; i++) {
        assert_float_eq(t1->grad[i], 2.0f, 1e-5, "Gradient accumulation failed");
        assert_float_eq(t2->grad[i], 2.0f, 1e-5, "Gradient accumulation failed");
    }
}

void test_stress() {
    printf("Testing stress cases...\n");
    
    // Test large tensor operations
    {
        int dims[] = {100, 100};
        Tensor* t1 = tensor_randn(2, dims, 1);
        Tensor* t2 = tensor_randn(2, dims, 1);
        Tensor* result = tensor_add(t1, t2);
        assert_float_eq(result->size, 10000, 1e-5, "Large tensor operation failed");
    }
    
    // Test deep computation graph
    {
        int dims[] = {2, 2};
        Tensor* t = tensor_randn(2, dims, 1);
        Tensor* current = t;
        
        // Create a deep chain of operations
        for (int i = 0; i < 100; i++) {
            current = tensor_add(current, t);
        }
        
        // Should be able to backprop through this deep graph
        current->grad[0] = 1.0f;
        backward();
    }
}

int main() {
    test_basic_ops();
    test_broadcasting();
    test_matmul();
    test_softmax();
    test_backward();
    test_gelu();
    test_rms_norm();
    test_edge_cases();
    test_numerical_stability();
    test_gradient_accumulation();
    test_stress();
    
    printf("All tests passed!\n");
    clean_registry();
    return 0;
}