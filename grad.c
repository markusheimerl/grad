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

static int get_broadcasted_index(int flat_idx, const int* dims, int ndims, const int* ref_dims, int ref_ndims) {
    int idx = 0, stride = 1;
    for (int d = ndims - 1; d >= 0; d--) {
        int ref_d = d + (ref_ndims - ndims);
        int coord = (flat_idx / stride) % ref_dims[ref_d];
        idx += (dims[d] == 1 ? 0 : coord) * stride;
        stride *= dims[d];
    }
    return idx;
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

void clean_registry() {
    for (int i = 0; i < registry_len; i++) {
        free(registry[i]->data);
        free(registry[i]->grad);
        free(registry[i]->dims);
        free(registry[i]);
    }
    registry_len = 0;
}

static Tensor* binary_op(Tensor* a, Tensor* b, OpType op) {
    if (!a || !b) return NULL;
    int max_d = fmax(a->ndims, b->ndims), rd[MAX_DIMS];
    
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        rd[max_d-1-i] = fmax(d1, d2);
    }

    Tensor* r = tensor_new(max_d, rd, NULL, a->requires_grad || b->requires_grad);
    
    for (int i = 0; i < r->size; i++) {
        int ai = get_broadcasted_index(i, a->dims, a->ndims, rd, max_d);
        int bi = get_broadcasted_index(i, b->dims, b->ndims, rd, max_d);
        r->data[i] = op == ADD ? a->data[ai] + b->data[bi] : a->data[ai] - b->data[bi];
    }
    
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){op, r, a, b};
    return r;
}

Tensor* tensor_add(Tensor* a, Tensor* b) { return binary_op(a, b, ADD); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return binary_op(a, b, SUB); }

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
        TapeEntry* e = &tape[t];
        Tensor *r = e->result, *a = e->input1, *b = e->input2;
        switch (e->op) {
            case ADD: case SUB:
                if (a->requires_grad || b->requires_grad)
                    for (int i = 0; i < r->size; i++) {
                        int ai = get_broadcasted_index(i, a->dims, a->ndims, r->dims, r->ndims);
                        int bi = get_broadcasted_index(i, b->dims, b->ndims, r->dims, r->ndims);
                        if (a->requires_grad) a->grad[ai] += r->grad[i];
                        if (b->requires_grad) b->grad[bi] += (e->op == ADD ? 1 : -1) * r->grad[i];
                    }
                break;
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
            case EXP:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += r->grad[i] * r->data[i];
                break;
            case LOG:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += r->grad[i] / fmaxf(a->data[i], MIN_LOG);
                break;
            case RESHAPE:
                if (a->requires_grad) for (int i = 0; i < a->size; i++) a->grad[i] += r->grad[i];
                break;
        }
    }
    tape_len = 0;
}

int main() {

    clean_registry();
    return 0;
}