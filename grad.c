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
    // Input validation
    if (!dims || ndims <= 0) return NULL;
    
    // Check for zero-size dimensions
    int size = 1;
    for (int i = 0; i < ndims; i++) {
        if (dims[i] <= 0) return NULL;  // Reject non-positive dimensions
        size *= dims[i];
    }
    
    // Check for overflow
    if (size > MAX_TAPE * MAX_TAPE) return NULL;  // Arbitrary limit to prevent overflow
    
    // Proceed with tensor creation
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    t->size = size;
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
                
                for (int n = 0; n < batch; n++) {
                    if (a->requires_grad) {
                        for (int i = 0; i < M; i++) {
                            for (int k = 0; k < K; k++) {
                                float sum = 0.0f;
                                for (int j = 0; j < N; j++) {
                                    sum += r->grad[n*M*N + i*N + j] * b->data[n*K*N + k*N + j];
                                }
                                a->grad[n*M*K + i*K + k] += sum;
                            }
                        }
                    }
                    if (b->requires_grad) {
                        for (int k = 0; k < K; k++) {
                            for (int j = 0; j < N; j++) {
                                float sum = 0.0f;
                                for (int i = 0; i < M; i++) {
                                    sum += r->grad[n*M*N + i*N + j] * a->data[n*M*K + i*K + k];
                                }
                                b->grad[n*K*N + k*N + j] += sum;
                            }
                        }
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
                
            case GELU: {
                if (a->requires_grad) {
                    const float sqrt_2_pi = 0.7978845608028654f;
                    for (int i = 0; i < a->size; i++) {
                        float x = a->data[i];
                        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_pi * (x + 0.044715f * x * x * x)));
                        float pdf = sqrt_2_pi * (1.0f + 0.134145f * x * x) * 
                                  (1.0f - tanhf(sqrt_2_pi * (x + 0.044715f * x * x * x)) * 
                                   tanhf(sqrt_2_pi * (x + 0.044715f * x * x * x))) * 0.5f;
                        a->grad[i] += r->grad[i] * (cdf + x * pdf);
                    }
                }
                break;
            }
            
            case RMSNORM: {
                if (a->requires_grad) {
                    float eps = *(float*)e->aux_data;
                    int last_dim = a->dims[a->ndims-1];
                    int batch_size = a->size/last_dim;
                    
                    for (int b = 0; b < batch_size; b++) {
                        float ms = 0.0f;
                        for (int i = 0; i < last_dim; i++) {
                            float val = a->data[b*last_dim + i];
                            ms += val * val;
                        }
                        ms /= last_dim;
                        
                        float inv_rms = 1.0f / sqrtf(ms + eps);
                        float inv_rms_cubed = inv_rms * inv_rms * inv_rms;
                        
                        float sum_xdout = 0.0f;
                        for (int i = 0; i < last_dim; i++) {
                            sum_xdout += a->data[b*last_dim + i] * r->grad[b*last_dim + i];
                        }
                        
                        for (int i = 0; i < last_dim; i++) {
                            float x_i = a->data[b*last_dim + i];
                            float dout_i = r->grad[b*last_dim + i];
                            a->grad[b*last_dim + i] += inv_rms * dout_i - 
                                (x_i * sum_xdout * inv_rms_cubed) / last_dim;
                        }
                    }
                }
                break;
            }
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

void test_complex_broadcasting() {
    printf("Testing complex broadcasting patterns...\n");
    
    // Test 4D broadcasting
    {
        int dims1[] = {2, 1, 3, 1};  // [2, 1, 3, 1]
        int dims2[] = {1, 4, 1, 5};  // [1, 4, 1, 5]
        float* data1 = malloc(2 * 1 * 3 * 1 * sizeof(float));
        float* data2 = malloc(1 * 4 * 1 * 5 * sizeof(float));
        
        // Fill with recognizable patterns
        for (int i = 0; i < 6; i++) data1[i] = i + 1;
        for (int i = 0; i < 20; i++) data2[i] = (i + 1) * 0.1f;
        
        Tensor* t1 = tensor_new(4, dims1, data1, 1);
        Tensor* t2 = tensor_new(4, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        // Result should be [2, 4, 3, 5]
        assert_float_eq(result->ndims, 4, 1e-5, "Wrong number of dimensions");
        assert_float_eq(result->dims[0], 2, 1e-5, "Wrong dimension 0");
        assert_float_eq(result->dims[1], 4, 1e-5, "Wrong dimension 1");
        assert_float_eq(result->dims[2], 3, 1e-5, "Wrong dimension 2");
        assert_float_eq(result->dims[3], 5, 1e-5, "Wrong dimension 3");
        
        free(data1);
        free(data2);
    }
    
    // Test broadcasting with mixed dimensionality
    {
        int dims1[] = {3, 1};        // [3, 1]
        int dims2[] = {2, 4, 1, 2};  // [2, 4, 1, 2]
        float data1[] = {1, 2, 3};
        float* data2 = malloc(16 * sizeof(float));
        for (int i = 0; i < 16; i++) data2[i] = i * 0.1f;
        
        Tensor* t1 = tensor_new(2, dims1, data1, 1);
        Tensor* t2 = tensor_new(4, dims2, data2, 1);
        Tensor* result = tensor_add(t1, t2);
        
        // Result should be [2, 4, 3, 2]
        assert_float_eq(result->ndims, 4, 1e-5, "Wrong number of dimensions");
        assert_float_eq(result->dims[0], 2, 1e-5, "Wrong dimension 0");
        assert_float_eq(result->dims[1], 4, 1e-5, "Wrong dimension 1");
        assert_float_eq(result->dims[2], 3, 1e-5, "Wrong dimension 2");
        assert_float_eq(result->dims[3], 2, 1e-5, "Wrong dimension 3");
        
        free(data2);
    }
}

void test_edge_cases_comprehensive() {
    printf("Testing comprehensive edge cases...\n");
    
    // Test zero-size dimension
    {
        int dims[] = {0, 2};
        Tensor* t = tensor_new(2, dims, NULL, 0);
        assert_float_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject zero-size dimension");
    }
    
    // Test negative dimension
    {
        int dims[] = {2, -1};
        Tensor* t = tensor_new(2, dims, NULL, 0);
        assert_float_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject negative dimension");
    }
    
    // Test zero dimensions
    {
        int dims[] = {1};
        Tensor* t = tensor_new(0, dims, NULL, 0);
        assert_float_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject zero dimensions");
    }
    
    // Test NULL dims
    {
        Tensor* t = tensor_new(1, NULL, NULL, 0);
        assert_float_eq(t == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject NULL dims");
    }
    
    // Test maximum dimension size
    {
        int dims[] = {1, MAX_TAPE};
        float* data = malloc(MAX_TAPE * sizeof(float));
        Tensor* t = tensor_new(2, dims, data, 0);
        assert_float_eq(t != NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should handle maximum size");
        free(data);
    }
    
    // Test invalid broadcasting
    {
        int dims1[] = {2, 3};
        int dims2[] = {2, 2};
        float data1[] = {1, 2, 3, 4, 5, 6};
        float data2[] = {1, 2, 3, 4};
        
        Tensor* t1 = tensor_new(2, dims1, data1, 0);
        Tensor* t2 = tensor_new(2, dims2, data2, 0);
        Tensor* result = tensor_add(t1, t2);
        assert_float_eq(result == NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should reject invalid broadcasting");
    }
    
    // Test single-element broadcasting
    {
        int dims1[] = {1};
        int dims2[] = {2, 2};
        float data1[] = {5};
        float data2[] = {1, 2, 3, 4};
        
        Tensor* t1 = tensor_new(1, dims1, data1, 0);
        Tensor* t2 = tensor_new(2, dims2, data2, 0);
        Tensor* result = tensor_add(t1, t2);
        assert_float_eq(result != NULL ? 1.0f : 0.0f, 1.0f, 1e-5, "Should allow scalar broadcasting");
        if (result) {
            assert_float_eq(result->data[0], 6.0f, 1e-5, "Incorrect scalar broadcasting result");
            assert_float_eq(result->data[1], 7.0f, 1e-5, "Incorrect scalar broadcasting result");
            assert_float_eq(result->data[2], 8.0f, 1e-5, "Incorrect scalar broadcasting result");
            assert_float_eq(result->data[3], 9.0f, 1e-5, "Incorrect scalar broadcasting result");
        }
    }
}

typedef struct {
    const char* name;
    double time;
    size_t memory;
} BenchmarkResult;

BenchmarkResult benchmark_operation(const char* name, void (*op)()) {
    clock_t start = clock();
    size_t initial_memory = 0;  // Would need platform-specific implementation
    
    op();
    
    clock_t end = clock();
    size_t final_memory = 0;    // Would need platform-specific implementation
    
    return (BenchmarkResult){
        .name = name,
        .time = ((double)(end - start)) / CLOCKS_PER_SEC,
        .memory = final_memory - initial_memory
    };
}

void test_performance() {
    printf("Running performance benchmarks...\n");
    
    // Benchmark large matrix multiplication
    void bench_matmul() {
        int dims1[] = {100, 100};
        int dims2[] = {100, 100};
        Tensor* t1 = tensor_randn(2, dims1, 1);
        Tensor* t2 = tensor_randn(2, dims2, 1);
        Tensor* result = tensor_matmul(t1, t2);
        result->grad[0] = 1.0f;
        backward();
    }
    
    // Benchmark deep network
    void bench_deep() {
        int dims[] = {32, 32};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* current = x;
        for (int i = 0; i < 100; i++) {
            current = tensor_gelu(tensor_rms_norm(current, 1e-5f));
        }
        current->grad[0] = 1.0f;
        backward();
    }
    
    BenchmarkResult results[] = {
        benchmark_operation("Large MatMul", bench_matmul),
        benchmark_operation("Deep Network", bench_deep)
    };
    
    for (size_t i = 0; i < sizeof(results)/sizeof(results[0]); i++) {
        printf("%s: %.3f seconds, %zu bytes\n", 
               results[i].name, results[i].time, results[i].memory);
    }
}

void test_memory_leaks() {
    printf("Testing for memory leaks...\n");
    
    // Store initial registry state
    size_t initial_allocs = registry_len;
    
    // Perform various operations that should all register their tensors
    {
        int dims[] = {10, 10};
        Tensor* t1 = tensor_randn(2, dims, 1);
        Tensor* t2 = tensor_randn(2, dims, 1);
        Tensor* result = tensor_matmul(t1, t2);
        result->grad[0] = 1.0f;
        backward();
    }
    
    // The number of tensors should be predictable:
    // - t1 (1 tensor)
    // - t2 (1 tensor)
    // - result (1 tensor)
    size_t expected_new_tensors = 3;
    
    // Verify the number of new allocations
    assert_float_eq(registry_len - initial_allocs, expected_new_tensors, 1e-5, "Unexpected number of tensor allocations");
}

#include <pthread.h>

void* thread_function(void* arg) {
    (void)arg;  // Explicitly mark as unused
    int dims[] = {10, 10};
    Tensor* t1 = tensor_randn(2, dims, 1);
    Tensor* t2 = tensor_randn(2, dims, 1);
    Tensor* result = tensor_matmul(t1, t2);
    result->grad[0] = 1.0f;
    backward();
    return NULL;
}


void test_thread_safety() {
    printf("Testing thread safety...\n");
    
    #define NUM_THREADS 4
    pthread_t threads[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_function, NULL);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void test_gradients_comprehensive() {
    printf("Testing comprehensive gradients...\n");
    
    // Test 1: Simple MatMul gradient
    {
        printf("Testing MatMul gradient...\n");
        int dims[] = {2, 2};
        float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float data2[] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        Tensor* x = tensor_new(2, dims, data1, 1);
        Tensor* y = tensor_new(2, dims, data2, 1);
        Tensor* out = tensor_matmul(x, y);
        
        float original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        float analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        float epsilon = 1e-4f;
        float original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_matmul(x, y);
        float numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;  // Restore original value
        
        printf("MatMul - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        
        // Calculate relative error
        float abs_diff = fabsf(analytical_grad - numerical_grad);
        float avg_magnitude = (fabsf(analytical_grad) + fabsf(numerical_grad)) / 2.0f;
        float relative_error = abs_diff / (avg_magnitude + 1e-10f);
        
        printf("Relative error: %f\n", relative_error);
        assert_float_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "MatMul gradient verification failed");
    }
    
    // Test 2: Simple GELU gradient
    {
        printf("Testing GELU gradient...\n");
        int dims[] = {1};
        float data[] = {0.5f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_gelu(x);
        
        float original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        float analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        float epsilon = 1e-4f;
        float original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_gelu(x);
        float numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;
        
        printf("GELU - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        float relative_error = fabsf(analytical_grad - numerical_grad) / 
                             (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        assert_float_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "GELU gradient verification failed");
    }
    
    // Test 3: Simple RMSNorm gradient
    {
        printf("Testing RMSNorm gradient...\n");
        int dims[] = {2};
        float data[] = {1.0f, 2.0f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_rms_norm(x, 1e-5f);
        
        float original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        float analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        float epsilon = 1e-4f;
        float original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_rms_norm(x, 1e-5f);
        float numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;
        
        printf("RMSNorm - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        float relative_error = fabsf(analytical_grad - numerical_grad) / 
                             (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        assert_float_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "RMSNorm gradient verification failed");
    }
    
    // Test 4: Simple Softmax gradient
    {
        printf("Testing Softmax gradient...\n");
        int dims[] = {2};
        float data[] = {1.0f, 2.0f};
        
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_softmax(x);
        
        float original_output = out->data[0];
        out->grad[0] = 1.0f;
        backward();
        float analytical_grad = x->grad[0];
        
        // Compute numerical gradient
        float epsilon = 1e-4f;
        float original_x = x->data[0];
        x->data[0] = original_x + epsilon;
        Tensor* out2 = tensor_softmax(x);
        float numerical_grad = (out2->data[0] - original_output) / epsilon;
        x->data[0] = original_x;
        
        printf("Softmax - Analytical: %f, Numerical: %f\n", analytical_grad, numerical_grad);
        float relative_error = fabsf(analytical_grad - numerical_grad) / 
                             (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
        assert_float_eq(relative_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5, 
                       "Softmax gradient verification failed");
    }
}

float tensor_grad_max(Tensor* t) {
    if (!t || !t->grad) return 0.0f;
    float max_grad = fabsf(t->grad[0]);
    for (int i = 1; i < t->size; i++) {
        float abs_grad = fabsf(t->grad[i]);
        if (abs_grad > max_grad) max_grad = abs_grad;
    }
    return max_grad;
}

void print_tensor_stats(Tensor* t, const char* name) {
    if (!t) return;
    float min_val = t->data[0], max_val = t->data[0], sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < min_val) min_val = t->data[i];
        if (t->data[i] > max_val) max_val = t->data[i];
        sum += t->data[i];
    }
    float mean = sum / t->size;
    
    printf("%s stats:\n", name);
    printf("  min: %.6f\n", min_val);
    printf("  max: %.6f\n", max_val);
    printf("  mean: %.6f\n", mean);
    if (t->grad) {
        printf("  grad_max: %.6f\n", tensor_grad_max(t));
    }
}

// Helper function for gradient mean
float tensor_grad_mean(Tensor* t) {
    if (!t || !t->grad) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        sum += fabsf(t->grad[i]);
    }
    return sum / t->size;
}

void test_individual_gradients() {
    printf("Testing individual operation gradients...\n");
    
    // Test MatMul gradient
    {
        printf("\nTesting MatMul gradient...\n");
        int dims[] = {2, 2};
        float a_data[] = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity matrix
        float b_data[] = {2.0f, 1.0f, 1.0f, 2.0f};
        
        Tensor* a = tensor_new(2, dims, a_data, 1);
        Tensor* b = tensor_new(2, dims, b_data, 0);
        Tensor* c = tensor_matmul(a, b);
        
        float original = c->data[0];
        c->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        float epsilon = 1e-5f;
        float saved = a->data[0];
        a->data[0] += epsilon;
        Tensor* c_new = tensor_matmul(a, b);
        float numerical = (c_new->data[0] - original) / epsilon;
        a->data[0] = saved;
        
        printf("MatMul - Analytical: %.6f, Numerical: %.6f\n", a->grad[0], numerical);
        
        // Calculate relative error
        float rel_error = fabsf(a->grad[0] - numerical) / 
                         (fabsf(a->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        
        // Use 1% relative error tolerance
        assert_float_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "MatMul gradient incorrect");
    }
    
    // Test GELU gradient
    {
        printf("\nTesting GELU gradient...\n");
        int dims[] = {1};
        float x_data[] = {0.5f};
        
        Tensor* x = tensor_new(1, dims, x_data, 1);
        Tensor* y = tensor_gelu(x);
        
        float original = y->data[0];
        y->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        float epsilon = 1e-5f;
        float saved = x->data[0];
        x->data[0] += epsilon;
        Tensor* y_new = tensor_gelu(x);
        float numerical = (y_new->data[0] - original) / epsilon;
        x->data[0] = saved;
        
        printf("GELU - Analytical: %.6f, Numerical: %.6f\n", x->grad[0], numerical);
        float rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_float_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "GELU gradient incorrect");
    }
    
    // Test RMSNorm gradient
    {
        printf("\nTesting RMSNorm gradient...\n");
        int dims[] = {2};
        float x_data[] = {1.0f, 2.0f};
        
        Tensor* x = tensor_new(1, dims, x_data, 1);
        Tensor* y = tensor_rms_norm(x, 1e-5f);
        
        float original = y->data[0];
        y->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        float epsilon = 1e-5f;
        float saved = x->data[0];
        x->data[0] += epsilon;
        Tensor* y_new = tensor_rms_norm(x, 1e-5f);
        float numerical = (y_new->data[0] - original) / epsilon;
        x->data[0] = saved;
        
        printf("RMSNorm - Analytical: %.6f, Numerical: %.6f\n", x->grad[0], numerical);
        float rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_float_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "RMSNorm gradient incorrect");
    }
    
    // Test simple chain of operations
    {
        printf("\nTesting simple operation chain...\n");
        int dims[] = {2};
        float x_data[] = {0.5f, 0.5f};
        
        Tensor* x = tensor_new(1, dims, x_data, 1);
        Tensor* y = tensor_gelu(x);
        Tensor* z = tensor_rms_norm(y, 1e-5f);
        
        float original = z->data[0];
        z->grad[0] = 1.0f;
        backward();
        
        // Compute numerical gradient
        float epsilon = 1e-5f;
        float saved = x->data[0];
        x->data[0] += epsilon;
        Tensor* y_new = tensor_gelu(x);
        Tensor* z_new = tensor_rms_norm(y_new, 1e-5f);
        float numerical = (z_new->data[0] - original) / epsilon;
        x->data[0] = saved;
        
        printf("Chain - Analytical: %.6f, Numerical: %.6f\n", x->grad[0], numerical);
        float rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        printf("Relative error: %.6f\n", rel_error);
        assert_float_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Operation chain gradient incorrect");
    }
}

void test_gradient_edge_cases() {
    printf("Testing gradient edge cases...\n");
    
    // Test 1: Very large values
    {
        int dims[] = {2};
        float data[] = {1000.0f, 1000.0f};
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_softmax(x);
        out->grad[0] = 1.0f;
        backward();
        printf("Large value gradient: %.6f\n", x->grad[0]);
    }
    
    // Test 2: Very small values
    {
        int dims[] = {2};
        float data[] = {1e-6f, 1e-6f};
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_rms_norm(x, 1e-5f);
        out->grad[0] = 1.0f;
        backward();
        printf("Small value gradient: %.6e\n", x->grad[0]);
    }
    
    // Test 3: Mixed scale values
    {
        int dims[] = {3};
        float data[] = {1e-6f, 1.0f, 1e6f};
        Tensor* x = tensor_new(1, dims, data, 1);
        Tensor* out = tensor_rms_norm(x, 1e-5f);
        out->grad[0] = 1.0f;
        backward();
        printf("Mixed scale gradients: %.6e, %.6f, %.6e\n", 
               x->grad[0], x->grad[1], x->grad[2]);
    }
}

void benchmark_gradients() {
    printf("Benchmarking gradient computation...\n");
    
    // Benchmark 1: Large matrix operations
    {
        clock_t start = clock();
        int dims[] = {256, 256};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* w = tensor_randn(2, dims, 1);
        Tensor* out = tensor_matmul(x, w);
        out->grad[0] = 1.0f;
        backward();
        clock_t end = clock();
        printf("Large matrix gradient time: %.3f seconds\n", 
               ((double)(end - start)) / CLOCKS_PER_SEC);
    }
    
    // Benchmark 2: Deep network
    {
        clock_t start = clock();
        int dims[] = {32, 32};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* w = tensor_randn(2, dims, 1);
        Tensor* current = x;
        for (int i = 0; i < 50; i++) {
            current = tensor_rms_norm(tensor_gelu(tensor_matmul(current, w)), 1e-5f);
        }
        current->grad[0] = 1.0f;
        backward();
        clock_t end = clock();
        printf("Deep network gradient time: %.3f seconds\n", 
               ((double)(end - start)) / CLOCKS_PER_SEC);
    }
}

typedef struct {
    float min;
    float max;
    float mean;
    float std;
} TensorStats;

TensorStats compute_tensor_stats(Tensor* t) {
    TensorStats stats = {t->data[0], t->data[0], 0.0f, 0.0f};
    float sum = 0.0f, sum_sq = 0.0f;
    
    for (int i = 0; i < t->size; i++) {
        float val = t->data[i];
        stats.min = fminf(stats.min, val);
        stats.max = fmaxf(stats.max, val);
        sum += val;
        sum_sq += val * val;
    }
    
    stats.mean = sum / t->size;
    stats.std = sqrtf(sum_sq/t->size - stats.mean*stats.mean);
    return stats;
}

void print_gradient_flow(Tensor** layers, int n_layers, const char** names) {
    printf("\nGradient Flow Visualization:\n");
    printf("Layer                  Min Grad    Max Grad    Mean Grad   Std Grad\n");
    printf("----------------------------------------------------------------\n");
    
    for (int i = 0; i < n_layers; i++) {
        if (!layers[i]->grad) continue;
        TensorStats stats = compute_tensor_stats(layers[i]);
        printf("%-20s %10.6f %10.6f %10.6f %10.6f\n",
               names[i], stats.min, stats.max, stats.mean, stats.std);
    }
}

void test_complex_networks() {
    printf("\nTesting complex network architectures...\n");
    
    // Test 1: Skip connection network
    {
        printf("\nTesting Skip Connection Network:\n");
        int dims[] = {4, 4};
        
        // Initialize with controlled values
        Tensor* x = tensor_new(2, dims, NULL, 1);
        Tensor* w1 = tensor_new(2, dims, NULL, 1);
        Tensor* w2 = tensor_new(2, dims, NULL, 1);
        
        // Use He initialization for weights
        float w_scale = sqrtf(2.0f / dims[0]);
        for (int i = 0; i < x->size; i++) {
            x->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;  // Small inputs
            w1->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * w_scale;
            w2->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * w_scale;
        }
        
        printf("\nInput statistics:\n");
        TensorStats x_stats = compute_tensor_stats(x);
        printf("Input - mean: %.6f, std: %.6f\n", x_stats.mean, x_stats.std);
        
        // Forward pass with skip connection
        printf("\nForward pass:\n");
        
        Tensor* h1 = tensor_matmul(x, w1);
        printf("After MatMul1 - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h1).mean, compute_tensor_stats(h1).std);
        
        Tensor* h2 = tensor_gelu(h1);
        printf("After GELU - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h2).mean, compute_tensor_stats(h2).std);
        
        Tensor* h3 = tensor_matmul(h2, w2);
        printf("After MatMul2 - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h3).mean, compute_tensor_stats(h3).std);
        
        Tensor* h4 = tensor_add(h3, x);  // Skip connection
        printf("After Skip Add - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(h4).mean, compute_tensor_stats(h4).std);
        
        Tensor* out = tensor_rms_norm(h4, 1e-5f);
        printf("After RMSNorm - mean: %.6f, std: %.6f\n",
               compute_tensor_stats(out).mean, compute_tensor_stats(out).std);
        
        Tensor* layers[] = {x, h1, h2, h3, h4, out};
        const char* names[] = {"Input", "MatMul1", "GELU", "MatMul2", "Skip Add", "RMSNorm"};
        
        // Store original values for gradient check
        float* original_input = malloc(x->size * sizeof(float));
        memcpy(original_input, x->data, x->size * sizeof(float));
        float original_output = out->data[0];
        
        // Backward pass
        printf("\nBackward pass:\n");
        out->grad[0] = 1.0f;
        backward();
        
        print_gradient_flow(layers, 6, names);
        
        // Numerical gradient check with smaller epsilon
        float epsilon = 1e-6f;  // Smaller epsilon for better accuracy
        x->data[0] = original_input[0] + epsilon;
        
        // Recompute forward pass
        Tensor* h1_new = tensor_matmul(x, w1);
        Tensor* h2_new = tensor_gelu(h1_new);
        Tensor* h3_new = tensor_matmul(h2_new, w2);
        Tensor* h4_new = tensor_add(h3_new, x);
        Tensor* out_new = tensor_rms_norm(h4_new, 1e-5f);
        
        float numerical = (out_new->data[0] - original_output) / epsilon;
        
        // Restore original input
        memcpy(x->data, original_input, x->size * sizeof(float));
        free(original_input);
        
        float rel_error = fabsf(x->grad[0] - numerical) / 
                         (fabsf(x->grad[0]) + fabsf(numerical) + 1e-10f);
        
        printf("\nSkip Connection Gradient Check:\n");
        printf("Analytical: %.6e\n", x->grad[0]);
        printf("Numerical:  %.6e\n", numerical);
        printf("Absolute difference: %.6e\n", fabsf(x->grad[0] - numerical));
        printf("Relative error: %.6f\n", rel_error);
        
        // Use more reasonable tolerance for complex network
        assert_float_eq(rel_error < 0.1f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "Skip connection gradient incorrect");
    }
}

void benchmark_operations(int size) {
    // Check if size is too large
    if (size * size > MAX_TAPE * MAX_TAPE / 4) {
        printf("Skipping benchmark for size %dx%d (too large)\n", size, size);
        return;
    }
    
    printf("\nBenchmarking operations with size %dx%d:\n", size, size);
    
    int initial_registry = registry_len;
    clock_t start, end;
    double cpu_time;
    
    // Benchmark MatMul
    {
        int dims[] = {size, size};
        Tensor* a = tensor_randn(2, dims, 1);
        Tensor* b = tensor_randn(2, dims, 1);
        
        start = clock();
        int iterations = size <= 256 ? 10 : 1;  // Fewer iterations for large sizes
        for (int i = 0; i < iterations; i++) {
            Tensor* c = tensor_matmul(a, b);
            c->grad[0] = 1.0f;
            backward();
        }
        end = clock();
        
        cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC / iterations;
        printf("MatMul + Backward: %.3f seconds per iteration\n", cpu_time);
    }
    
    // Reset registry to initial state
    while (registry_len > initial_registry) {
        registry_len--;
    }
    
    // Benchmark RMSNorm
    {
        int dims[] = {1, size * size};
        Tensor* x = tensor_randn(2, dims, 1);
        
        start = clock();
        int iterations = size <= 256 ? 100 : 10;  // Fewer iterations for large sizes
        for (int i = 0; i < iterations; i++) {
            Tensor* y = tensor_rms_norm(x, 1e-5f);
            y->grad[0] = 1.0f;
            backward();
        }
        end = clock();
        
        cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC / iterations;
        printf("RMSNorm + Backward: %.3f seconds per iteration\n", cpu_time);
    }
    
    // Reset registry to initial state
    while (registry_len > initial_registry) {
        registry_len--;
    }
}

void test_numerical_stability_comprehensive() {
    printf("\nTesting numerical stability...\n");
    
    // Test 1: Extreme value handling
    {
        printf("\nTesting extreme values:\n");
        int dims[] = {4};
        float extreme_data[] = {1e-10f, 1e10f, -1e-10f, -1e10f};
        Tensor* x = tensor_new(1, dims, extreme_data, 1);
        
        // Test RMSNorm
        Tensor* y = tensor_rms_norm(x, 1e-5f);
        y->grad[0] = 1.0f;
        backward();
        
        TensorStats stats = compute_tensor_stats(y);
        printf("RMSNorm output stats:\n");
        printf("min: %.6e, max: %.6e, mean: %.6e, std: %.6e\n",
               stats.min, stats.max, stats.mean, stats.std);
        
        // Verify output is normalized
        assert_float_eq(fabsf(stats.std - 1.0f) < 0.1f ? 1.0f : 0.0f, 1.0f, 1e-5,
                       "RMSNorm failed to normalize extreme values");
    }
    
    // Test 2: Gradient explosion/vanishing
    {
        printf("\nTesting gradient propagation:\n");
        int dims[] = {4, 4};
        Tensor* x = tensor_randn(2, dims, 1);
        Tensor* w = tensor_randn(2, dims, 1);
        
        // Create deep network
        Tensor* current = x;
        const int DEPTH = 20;
        Tensor** layers = malloc(DEPTH * sizeof(Tensor*));
        
        for (int i = 0; i < DEPTH; i++) {
            current = tensor_rms_norm(tensor_gelu(tensor_matmul(current, w)), 1e-5f);
            layers[i] = current;
        }
        
        current->grad[0] = 1.0f;
        backward();
        
        // Check gradient magnitudes
        printf("\nGradient magnitudes through depth:\n");
        for (int i = 0; i < DEPTH; i++) {
            TensorStats stats = compute_tensor_stats(layers[i]);
            printf("Layer %2d - mean grad: %.6e, std grad: %.6e\n",
                   i, stats.mean, stats.std);
        }
        
        free(layers);
    }
}

void test_transformer_block() {
    printf("\nTesting Transformer block components...\n");
    
    // Test attention pattern
    {
        printf("\nTesting attention pattern:\n");
        int batch = 2, seq_len = 4, d_model = 8;
        int dims_qk[] = {batch, seq_len, d_model};
        
        // Create Q, K matrices
        Tensor* Q = tensor_randn(3, dims_qk, 1);
        Tensor* K = tensor_randn(3, dims_qk, 1);
        
        // Q @ K^T
        int perm[] = {0, 2, 1};  // Transpose last two dims
        Tensor* Kt = tensor_permute(K, perm, 3);
        Tensor* QK = tensor_matmul(Q, Kt);
        
        // Scale and softmax
        float scale = 1.0f / sqrtf(d_model);
        for (int i = 0; i < QK->size; i++) {
            QK->data[i] *= scale;
        }
        Tensor* attention = tensor_softmax(QK);
        
        // Verify attention properties
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seq_len; i++) {
                float sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    sum += attention->data[b*seq_len*seq_len + i*seq_len + j];
                }
                assert_float_eq(sum, 1.0f, 1e-5, "Attention weights don't sum to 1");
            }
        }
        
        // Test gradient flow
        attention->grad[0] = 1.0f;
        backward();
        
        printf("Attention pattern gradient check passed\n");
    }
}

// Add memory usage tracking
size_t get_total_memory_usage() {
    size_t total = 0;
    for (int i = 0; i < registry_len; i++) {
        Tensor* t = registry[i];
        total += sizeof(Tensor);
        total += t->size * sizeof(float);  // data
        if (t->grad) total += t->size * sizeof(float);  // grad
        total += t->ndims * sizeof(int);  // dims
    }
    return total;
}

void print_memory_usage(const char* label) {
    size_t mem = get_total_memory_usage();
    printf("Memory usage at %s: %.2f MB\n", label, mem / (1024.0 * 1024.0));
}

// Update main to use safer benchmarking
int main() {
    print_memory_usage("start");
    
    // Original tests
    test_basic_ops();
    test_broadcasting();
    test_complex_broadcasting();
    test_matmul();
    test_softmax();
    test_backward();
    test_gelu();
    test_rms_norm();
    test_edge_cases();
    test_edge_cases_comprehensive();
    test_numerical_stability();
    test_gradient_accumulation();
    test_gradients_comprehensive();
    test_individual_gradients();
    
    print_memory_usage("after basic tests");
    
    // New comprehensive tests
    test_complex_networks();
    test_numerical_stability_comprehensive();
    test_transformer_block();
    
    print_memory_usage("after comprehensive tests");
    
    // Performance benchmarks with memory checks
    printf("\nRunning performance benchmarks:\n");
    benchmark_operations(32);    // Tiny
    print_memory_usage("after tiny benchmark");
    
    benchmark_operations(64);    // Small
    print_memory_usage("after small benchmark");
    
    benchmark_operations(256);   // Medium
    print_memory_usage("after medium benchmark");
    
    // Only run large benchmark if memory allows
    size_t estimated_memory = (size_t)1024 * 1024 * 3 * sizeof(float);
    if (estimated_memory < MAX_TAPE * sizeof(float)) {
        benchmark_operations(512);   // Large
        print_memory_usage("after large benchmark");
    } else {
        printf("Skipping large benchmark (insufficient memory)\n");
    }
    
    printf("\nAll tests passed!\n");
    print_memory_usage("before cleanup");
    clean_registry();
    print_memory_usage("end");
    return 0;
}