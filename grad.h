#ifndef __GRAD_H__
#define __GRAD_H__

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
    int is_permanent;
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
static int* temp_buffer = NULL;
static int temp_buffer_size = 0;

#define REGISTRY_SAFETY_CHECK() \
    if (registry_len >= MAX_TENSORS) { \
        printf("Error: Tensor registry full!\n"); \
        return NULL; \
    }

static int* get_temp_buffer(int size) {
    if (size > temp_buffer_size) {
        free(temp_buffer);
        temp_buffer = malloc(size * sizeof(int));
        temp_buffer_size = size;
    }
    return temp_buffer;
}

static int get_index(int idx, const int* dims, int ndims, const int* ref_dims, int ref_ndims) {
    int coords[32];
    int temp = idx;
    int stride = 1;
    
    for (int d = ref_ndims - 1; d >= 0; d--) {
        coords[d] = (temp / stride) % ref_dims[d];
        temp -= coords[d] * stride;
        stride *= ref_dims[d];
    }
    
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

static Tensor* tensor_create(int ndims, const int* dims, const float* data, int requires_grad, int is_permanent) {
    if (!dims || ndims <= 0) return NULL;
    REGISTRY_SAFETY_CHECK();
    
    int size = 1;
    for (int i = 0; i < ndims; i++) {
        if (dims[i] <= 0) return NULL;
        size *= dims[i];
    }
    if (size > MAX_TAPE * MAX_TAPE) return NULL;
    
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    t->size = size;
    memcpy(t->dims, dims, ndims * sizeof(int));
    t->data = malloc(size * sizeof(float));
    if (data) memcpy(t->data, data, size * sizeof(float));
    if ((t->requires_grad = requires_grad)) t->grad = calloc(size, sizeof(float));
    t->is_permanent = is_permanent;
    registry[registry_len++] = t;
    return t;
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    return tensor_create(ndims, dims, data, requires_grad, 0);
}

Tensor* tensor_new_permanent(int ndims, const int* dims, const float* data, int requires_grad) {
    return tensor_create(ndims, dims, data, requires_grad, 1);
}

Tensor* tensor_zeros(int ndims, const int* dims, int requires_grad) {
    return tensor_create(ndims, dims, NULL, requires_grad, 0);
}

Tensor* tensor_zeros_permanent(int ndims, const int* dims, int requires_grad) {
    return tensor_create(ndims, dims, NULL, requires_grad, 1);
}

Tensor* tensor_randn(int ndims, const int* dims, int requires_grad) {
    static int seed_set = 0;
    if (!seed_set) { srand(time(NULL)); seed_set = 1; }
    
    Tensor* t = tensor_create(ndims, dims, NULL, requires_grad, 0);
    if (!t) return NULL;
    
    for (int i = 0; i < t->size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        t->data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
    return t;
}

Tensor* tensor_randn_permanent(int ndims, const int* dims, int requires_grad) {
    Tensor* t = tensor_randn(ndims, dims, requires_grad);
    if (t) t->is_permanent = 1;
    return t;
}

void clean_registry() {
    int write = 0;
    for (int read = 0; read < registry_len; read++) {
        Tensor* t = registry[read];
        if (t->is_permanent) {
            registry[write++] = t;
        } else {
            free(t->data);
            free(t->grad);
            free(t->dims);
            free(t);
        }
    }
    registry_len = write;
    
    free(temp_buffer);
    temp_buffer = NULL;
    temp_buffer_size = 0;
}

void free_permanent_tensor(Tensor* t) {
    if (!t || !t->is_permanent) return;
    
    // Find and remove from registry
    for (int i = 0; i < registry_len; i++) {
        if (registry[i] == t) {
            registry[i] = registry[--registry_len];
            break;
        }
    }
    
    free(t->data);
    free(t->grad);
    free(t->dims);
    free(t);
}

static void compute_strides(const int* dims, int ndims, int* strides) {
    strides[ndims-1] = 1;
    for (int i = ndims-2; i >= 0; i--) {
        strides[i] = strides[i+1] * dims[i+1];
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
    
    int M = a->dims[a->ndims-2];
    int K = a->dims[a->ndims-1];
    int N = b->dims[b->ndims-1];
    int a_batch_dims = a->ndims - 2;
    int b_batch_dims = b->ndims - 2;
    int max_batch_dims = fmax(a_batch_dims, b_batch_dims);
    
    int out_dims[32];
    int out_ndims = max_batch_dims + 2;
    int total_batch = 1;
    
    for (int i = 0; i < max_batch_dims; i++) {
        int a_dim = (i < a_batch_dims) ? a->dims[i] : 1;
        int b_dim = (i < b_batch_dims) ? b->dims[i] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) return NULL;
        out_dims[i] = fmax(a_dim, b_dim);
        total_batch *= out_dims[i];
    }
    
    out_dims[max_batch_dims] = M;
    out_dims[max_batch_dims + 1] = N;
    
    Tensor* r = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    if (!r) return NULL;
    
    int* a_strides = get_temp_buffer(32);
    int* b_strides = a_strides + 16;
    
    int a_stride = M * K;
    for (int i = a_batch_dims - 1; i >= 0; i--) {
        a_strides[i] = a_stride;
        a_stride *= a->dims[i];
    }
    
    int b_stride = K * N;
    for (int i = b_batch_dims - 1; i >= 0; i--) {
        b_strides[i] = b_stride;
        b_stride *= b->dims[i];
    }
    
    for (int batch = 0; batch < total_batch; batch++) {
        int a_offset = 0, b_offset = 0;
        int tmp = batch;
        
        for (int i = 0; i < max_batch_dims; i++) {
            int dim_idx = tmp / (total_batch / out_dims[i]);
            tmp %= (total_batch / out_dims[i]);
            
            if (i < a_batch_dims) {
                a_offset += (a->dims[i] == 1 ? 0 : dim_idx) * a_strides[i];
            }
            if (i < b_batch_dims) {
                b_offset += (b->dims[i] == 1 ? 0 : dim_idx) * b_strides[i];
            }
        }
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += a->data[a_offset + i*K + k] * b->data[b_offset + k*N + j];
                }
                r->data[batch*M*N + i*N + j] = sum;
            }
        }
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
    if (!a) return NULL;
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r && r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL, NULL};
    return r;
}

Tensor* tensor_permute(Tensor* a, const int* perm, int perm_size) {
    if (!a || !perm || perm_size != a->ndims) return NULL;
    
    int* used = get_temp_buffer(perm_size);
    memset(used, 0, perm_size * sizeof(int));
    
    for (int i = 0; i < perm_size; i++)
        if (perm[i] < 0 || perm[i] >= perm_size || used[perm[i]]++)
            return NULL;
    
    int* new_dims = malloc(a->ndims * sizeof(int));
    for (int i = 0; i < a->ndims; i++) new_dims[i] = a->dims[perm[i]];
    
    Tensor* r = tensor_new(a->ndims, new_dims, NULL, a->requires_grad);
    free(new_dims);
    if (!r) return NULL;
    
    int* strides = get_temp_buffer(2 * a->ndims);
    int* r_strides = strides + a->ndims;
    
    compute_strides(a->dims, a->ndims, strides);
    compute_strides(r->dims, r->ndims, r_strides);
    
    for (int i = 0; i < r->size; i++) {
        int temp = i, old_idx = 0;
        for (int d = 0; d < r->ndims; d++) {
            int coord = temp / r_strides[d];
            temp %= r_strides[d];
            old_idx += coord * strides[perm[d]];
        }
        r->data[i] = a->data[old_idx];
    }
    
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
        float ms = 0.0f;
        for (int i = 0; i < last_dim; i++) {
            float val = x->data[b * last_dim + i];
            ms += val * val;
        }
        ms /= last_dim;
        
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
                int a_batch_dims = a->ndims - 2;
                int b_batch_dims = b->ndims - 2;
                int r_batch_dims = r->ndims - 2;
                int total_batch = 1;
                for (int i = 0; i < r_batch_dims; i++) total_batch *= r->dims[i];
                
                int* a_strides = get_temp_buffer(32);
                int* b_strides = a_strides + 16;
                
                int a_stride = M * K;
                for (int i = a_batch_dims - 1; i >= 0; i--) {
                    a_strides[i] = a_stride;
                    a_stride *= a->dims[i];
                }
                
                int b_stride = K * N;
                for (int i = b_batch_dims - 1; i >= 0; i--) {
                    b_strides[i] = b_stride;
                    b_stride *= b->dims[i];
                }
                
                for (int batch = 0; batch < total_batch; batch++) {
                    int a_offset = 0, b_offset = 0;
                    int tmp = batch;
                    
                    for (int i = 0; i < r_batch_dims; i++) {
                        int dim_idx = tmp / (total_batch / r->dims[i]);
                        tmp %= (total_batch / r->dims[i]);
                        
                        if (i < a_batch_dims) {
                            a_offset += (a->dims[i] == 1 ? 0 : dim_idx) * a_strides[i];
                        }
                        if (i < b_batch_dims) {
                            b_offset += (b->dims[i] == 1 ? 0 : dim_idx) * b_strides[i];
                        }
                    }
                    
                    if (a->requires_grad) {
                        for (int i = 0; i < M; i++) {
                            for (int k = 0; k < K; k++) {
                                float sum = 0.0f;
                                for (int j = 0; j < N; j++) {
                                    sum += r->grad[batch*M*N + i*N + j] * 
                                        b->data[b_offset + k*N + j];
                                }
                                a->grad[a_offset + i*K + k] += sum;
                            }
                        }
                    }
                    
                    if (b->requires_grad) {
                        for (int k = 0; k < K; k++) {
                            for (int j = 0; j < N; j++) {
                                float sum = 0.0f;
                                for (int i = 0; i < M; i++) {
                                    sum += r->grad[batch*M*N + i*N + j] * 
                                        a->data[a_offset + i*K + k];
                                }
                                b->grad[b_offset + k*N + j] += sum;
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
                            a->grad[i*last_dim+j] += r->data[i*last_dim+j] * 
                                (r->grad[i*last_dim+j] - sum);
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
                    int* perm = (int*)e->aux_data;
                    int* inv_perm = get_temp_buffer(a->ndims);
                    int* strides = inv_perm + a->ndims;
                    int* r_strides = strides + a->ndims;
                    
                    for (int i = 0; i < a->ndims; i++)
                        inv_perm[perm[i]] = i;
                    
                    compute_strides(a->dims, a->ndims, strides);
                    compute_strides(r->dims, r->ndims, r_strides);
                    
                    for (int i = 0; i < r->size; i++) {
                        int temp = i, old_idx = 0;
                        for (int d = 0; d < r->ndims; d++) {
                            int coord = temp / r_strides[d];
                            temp %= r_strides[d];
                            old_idx += coord * strides[inv_perm[d]];
                        }
                        a->grad[old_idx] += r->grad[i];
                    }
                }
                break;
                
            case GELU:
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
                
            case RMSNORM:
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
        free(e->aux_data);
    }
    tape_len = 0;
}

#endif