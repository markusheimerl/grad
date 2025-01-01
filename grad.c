#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

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

static int get_index(int idx, const int* dims, int ndims, const int* ref_dims, int ref_ndims) {
    int result = 0, stride = 1;
    for (int d = ndims - 1; d >= 0; d--) {
        int coord = (idx / stride) % ref_dims[d + ref_ndims - ndims];
        result += (dims[d] == 1 ? 0 : coord) * stride;
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

void clean_registry() {
    while (registry_len > 0) {
        Tensor* t = registry[--registry_len];
        free(t->data); free(t->grad); free(t->dims); free(t);
    }
}

static Tensor* binary_op(Tensor* a, Tensor* b, OpType op) {
    if (!a || !b) return NULL;
    int max_d = fmax(a->ndims, b->ndims), rd[32];
    
    for (int i = 0; i < max_d; i++) {
        int d1 = i < a->ndims ? a->dims[a->ndims-1-i] : 1;
        int d2 = i < b->ndims ? b->dims[b->ndims-1-i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) return NULL;
        rd[max_d-1-i] = fmax(d1, d2);
    }

    Tensor* r = tensor_new(max_d, rd, NULL, a->requires_grad || b->requires_grad);
    for (int i = 0; i < r->size; i++) {
        float av = a->data[get_index(i, a->dims, a->ndims, rd, max_d)];
        float bv = b->data[get_index(i, b->dims, b->ndims, rd, max_d)];
        r->data[i] = op == ADD ? av + bv : av - bv;
    }
    
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){op, r, a, b};
    return r;
}

Tensor* tensor_add(Tensor* a, Tensor* b) { return binary_op(a, b, ADD); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return binary_op(a, b, SUB); }

Tensor* tensor_exp(Tensor* a) {
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) r->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){EXP, r, a};
    return r;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) r->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){LOG, r, a};
    return r;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
    int max_d = fmax(a->ndims, b->ndims);
    int dims[32];
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
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, r, a, b};
    return r;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a};
    return r;
}

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* e = &tape[t];
        Tensor *r = e->result, *a = e->input1, *b = e->input2;
        
        if (e->op == ADD || e->op == SUB) {
            for (int i = 0; i < r->size; i++) {
                if (a->requires_grad) a->grad[get_index(i, a->dims, a->ndims, r->dims, r->ndims)] += r->grad[i];
                if (b->requires_grad) b->grad[get_index(i, b->dims, b->ndims, r->dims, r->ndims)] += (e->op == ADD ? 1 : -1) * r->grad[i];
            }
        }
        else if (e->op == MATMUL) {
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
        }
        else if (e->op == EXP && a->requires_grad)
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i] * r->data[i];
        else if (e->op == LOG && a->requires_grad)
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i] / fmaxf(a->data[i], MIN_LOG);
        else if (e->op == RESHAPE && a->requires_grad)
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i];
    }
    tape_len = 0;
}

int main() {

    clean_registry();
    return 0;
}