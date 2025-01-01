#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TAPE 1000
#define MAX_TENSORS 1000
#define MIN_LOG 1e-7f
#define MAX_EXP 88.0f

typedef enum { MATMUL, EXP, LOG, ADD, SUB, RESHAPE, SOFTMAX } OpType;

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

Tensor* tensor_softmax(Tensor* a) {
    if (!a || a->ndims < 1) return NULL;
    
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    int last_dim = a->dims[a->ndims - 1];
    int outer_size = a->size / last_dim;
    
    for (int i = 0; i < outer_size; i++) {
        // Find max for this batch
        float max_val = a->data[i * last_dim];
        for (int j = 1; j < last_dim; j++) {
            max_val = fmaxf(max_val, a->data[i * last_dim + j]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            float val = a->data[i * last_dim + j] - max_val;
            r->data[i * last_dim + j] = expf(val);
            sum += r->data[i * last_dim + j];
        }
        
        // Normalize
        for (int j = 0; j < last_dim; j++) {
            r->data[i * last_dim + j] /= sum;
        }
    }
    
    if (r->requires_grad) {
        tape[tape_len++] = (TapeEntry){SOFTMAX, r, a, NULL};
    }
    
    return r;
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
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

Tensor* tensor_add(Tensor* a, Tensor* b) { return tensor_op(a, b, ADD); }
Tensor* tensor_sub(Tensor* a, Tensor* b) { return tensor_op(a, b, SUB); }

Tensor* tensor_exp(Tensor* a) {
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) r->data[i] = expf(fminf(a->data[i], MAX_EXP));
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){EXP, r, a, NULL};
    return r;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* r = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
    for (int i = 0; i < a->size; i++) r->data[i] = logf(fmaxf(a->data[i], MIN_LOG));
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){LOG, r, a, NULL};
    return r;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndims < 1 || b->ndims < 1 || a->dims[a->ndims-1] != b->dims[b->ndims-2]) return NULL;
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
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){MATMUL, r, a, b};
    return r;
}

Tensor* tensor_reshape(Tensor* a, int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    if (size != a->size) return NULL;
    Tensor* r = tensor_new(ndims, dims, a->data, a->requires_grad);
    if (r->requires_grad) tape[tape_len++] = (TapeEntry){RESHAPE, r, a, NULL};
    return r;
}

void backward() {
    for (int t = tape_len-1; t >= 0; t--) {
        TapeEntry* e = &tape[t];
        Tensor *r = e->result, *a = e->input1, *b = e->input2;
        
        if (e->op == ADD || e->op == SUB) {
            for (int i = 0; i < r->size; i++) {
                if (a->requires_grad) 
                    a->grad[get_index(i, a->dims, a->ndims, r->dims, r->ndims)] += r->grad[i];
                if (b->requires_grad) 
                    b->grad[get_index(i, b->dims, b->ndims, r->dims, r->ndims)] += 
                        (e->op == ADD ? 1 : -1) * r->grad[i];
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
                            if (a->requires_grad) 
                                a->grad[n*M*K + i*K + k] += g * b->data[n*K*N + k*N + j];
                            if (b->requires_grad) 
                                b->grad[n*K*N + k*N + j] += g * a->data[n*M*K + i*K + k];
                        }
                    }
        }
        else if (e->op == EXP && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i] * r->data[i];
        }
        else if (e->op == LOG && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i] / fmaxf(a->data[i], MIN_LOG);
        }
        else if (e->op == RESHAPE && a->requires_grad) {
            for (int i = 0; i < a->size; i++)
                a->grad[i] += r->grad[i];
        }
        else if (e->op == SOFTMAX && a->requires_grad) {
            int last_dim = a->dims[a->ndims - 1];
            int outer_size = a->size / last_dim;
            
            for (int i = 0; i < outer_size; i++) {
                float sum = 0.0f;
                for (int j = 0; j < last_dim; j++) {
                    sum += r->grad[i * last_dim + j] * r->data[i * last_dim + j];
                }
                
                for (int j = 0; j < last_dim; j++) {
                    float softmax_j = r->data[i * last_dim + j];
                    a->grad[i * last_dim + j] += softmax_j * (r->grad[i * last_dim + j] - sum);
                }
            }
        }
    }
    tape_len = 0;
}

int main() {
    // Test 1: Basic softmax properties
    printf("Test 1: Basic softmax properties\n");
    int dims1[] = {3};
    float data1[] = {1.0f, 2.0f, 3.0f};
    Tensor* t1 = tensor_new(1, dims1, data1, 1);
    Tensor* s1 = tensor_softmax(t1);
    
    printf("Input: [%.2f, %.2f, %.2f]\n", t1->data[0], t1->data[1], t1->data[2]);
    printf("Softmax: [%.6f, %.6f, %.6f]\n", s1->data[0], s1->data[1], s1->data[2]);
    float sum1 = s1->data[0] + s1->data[1] + s1->data[2];
    printf("Sum (should be 1.0): %.6f\n\n", sum1);

    // Test 2: Numerical stability test
    printf("Test 2: Numerical stability test\n");
    int dims2[] = {3};
    float data2[] = {1000.0f, 1000.0f, 1000.0f};  // Large numbers
    Tensor* t2 = tensor_new(1, dims2, data2, 1);
    Tensor* s2 = tensor_softmax(t2);
    printf("Large inputs: [%.1f, %.1f, %.1f]\n", t2->data[0], t2->data[1], t2->data[2]);
    printf("Softmax: [%.6f, %.6f, %.6f]\n", s2->data[0], s2->data[1], s2->data[2]);
    float sum2 = s2->data[0] + s2->data[1] + s2->data[2];
    printf("Sum (should be 1.0): %.6f\n\n", sum2);

    // Test 3: Batch processing
    printf("Test 3: Batch processing\n");
    int dims3[] = {2, 3};  // 2 batches, 3 classes each
    float data3[] = {1.0f, 2.0f, 3.0f, // first batch
                     10.0f, 11.0f, 12.0f}; // second batch (very different scale)
    Tensor* t3 = tensor_new(2, dims3, data3, 1);
    Tensor* s3 = tensor_softmax(t3);
    
    printf("Batch 1 input: [%.2f, %.2f, %.2f]\n", t3->data[0], t3->data[1], t3->data[2]);
    printf("Batch 1 softmax: [%.6f, %.6f, %.6f]\n", s3->data[0], s3->data[1], s3->data[2]);
    printf("Batch 1 sum: %.6f\n", s3->data[0] + s3->data[1] + s3->data[2]);
    
    printf("Batch 2 input: [%.2f, %.2f, %.2f]\n", t3->data[3], t3->data[4], t3->data[5]);
    printf("Batch 2 softmax: [%.6f, %.6f, %.6f]\n", s3->data[3], s3->data[4], s3->data[5]);
    printf("Batch 2 sum: %.6f\n\n", s3->data[3] + s3->data[4] + s3->data[5]);

    // Test 4: Gradient verification
    printf("Test 4: Gradient verification\n");
    // Reset gradients
    memset(t3->grad, 0, t3->size * sizeof(float));
    memset(s3->grad, 0, s3->size * sizeof(float));
    
    // Set gradient for first element in each batch
    s3->grad[0] = 1.0f;
    s3->grad[3] = 1.0f;
    
    backward();
    
    printf("Batch 1 gradients: [%.6f, %.6f, %.6f]\n", 
           t3->grad[0], t3->grad[1], t3->grad[2]);
    printf("Batch 2 gradients: [%.6f, %.6f, %.6f]\n", 
           t3->grad[3], t3->grad[4], t3->grad[5]);
    
    // Verify gradient properties
    float grad_sum1 = t3->grad[0] + t3->grad[1] + t3->grad[2];
    float grad_sum2 = t3->grad[3] + t3->grad[4] + t3->grad[5];
    printf("Gradient sums (should be ~0): %.6f, %.6f\n\n", grad_sum1, grad_sum2);


printf("\nTest 5: Softmax invariance property\n");
int dims5[] = {3};
float data5[] = {101.0f, 102.0f, 103.0f};  // Same differences as [1,2,3] but shifted by 100
Tensor* t5 = tensor_new(1, dims5, data5, 1);
Tensor* s5 = tensor_softmax(t5);
printf("Input: [%.2f, %.2f, %.2f]\n", t5->data[0], t5->data[1], t5->data[2]);
printf("Softmax: [%.6f, %.6f, %.6f]\n", s5->data[0], s5->data[1], s5->data[2]);
    clean_registry();
    return 0;
}