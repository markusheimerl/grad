#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA error checking macro.
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuBLAS error checking macro.
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if(status != CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// -----------------------------------------------------------------------------
// AdamW update kernel (used to update all parameters)
__global__ void adamw_update_kernel(
    float* weight,
    const float* grad,
    float* m,
    float* v,
    float beta1,
    float beta2,
    float epsilon,
    float learning_rate,
    float weight_decay,
    float alpha_t,
    int size,
    int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        float g = grad[idx] / batch_size;
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float update_val = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update_val;
    }
}

// -----------------------------------------------------------------------------
// Kernel: embed_input_kernel
// For each raw input, determine the bin and look up the corresponding embedding;
// also store the selected bin index for the backward pass.
__global__ void embed_input_kernel(
    const float* raw_input,           // [batch_size x raw_input_dim]
    const float* embedding_table,     // [raw_input_dim x (num_bins * embedding_dim)]
    float* embedded_output,           // [batch_size x (raw_input_dim * embedding_dim)]
    int* embed_indices,               // [batch_size x raw_input_dim]
    int batch_size,
    int raw_input_dim,
    int num_bins,
    int embedding_dim,
    float input_range_min,
    float input_range_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * raw_input_dim;
    if(idx < total){
        int sample = idx / raw_input_dim;
        int feature = idx % raw_input_dim;
        float val = raw_input[sample * raw_input_dim + feature];
        float range = input_range_max - input_range_min;
        int bin_idx = (int)(((val - input_range_min) / range) * num_bins);
        if(bin_idx < 0) bin_idx = 0;
        if(bin_idx >= num_bins) bin_idx = num_bins - 1;
        embed_indices[idx] = bin_idx;
        
        int emb_offset = feature * num_bins * embedding_dim + bin_idx * embedding_dim;
        int dest_offset = sample * (raw_input_dim * embedding_dim) + feature * embedding_dim;
        for(int k = 0; k < embedding_dim; k++){
            embedded_output[dest_offset + k] = embedding_table[emb_offset + k];
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: project_forward_kernel – computes Y = X * W.
// X: [batch_size, num_tokens, dim] (row–major, each token is contiguous)
// W: [dim, dim] (row–major)
// Y: same shape as X.
__global__ void project_forward_kernel(
    const float* X,
    const float* W,
    float* Y,
    int batch_size,
    int num_tokens,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_tokens;
    if(idx < total){
        int offset = idx * dim;
        for(int j = 0; j < dim; j++){
            float sum = 0.0f;
            for(int k = 0; k < dim; k++){
                sum += X[offset + k] * W[k * dim + j];
            }
            Y[offset + j] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: project_backward_dX_kernel – computes dX = dY * (W^T).
__global__ void project_backward_dX_kernel(
    const float* dY,
    const float* W,
    float* dX,
    int batch_size,
    int num_tokens,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_tokens;
    if(idx < total){
        int offset = idx * dim;
        for(int j = 0; j < dim; j++){
            float sum = 0.0f;
            for(int k = 0; k < dim; k++){
                // W^T element (j,k) = W[k,j]
                sum += dY[offset + k] * W[j * dim + k];
            }
            dX[offset + j] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: project_backward_dW_kernel – computes dW = X^T * dY.
// For each element (i,j) in W, sum over all samples and tokens.
__global__ void project_backward_dW_kernel(
    const float* X,
    const float* dY,
    float* dW,
    int batch_size,
    int num_tokens,
    int dim)
{
    int i = blockIdx.x;      // row index in W
    int j = threadIdx.x;     // column index in W
    if(i < dim && j < dim){
        float sum = 0.0f;
        for(int s = 0; s < batch_size; s++){
            for(int t = 0; t < num_tokens; t++){
                int idx = (s * num_tokens + t) * dim;
                sum += X[idx + i] * dY[idx + j];
            }
        }
        dW[i * dim + j] = sum;
    }
}

// -----------------------------------------------------------------------------
// Kernel: attention_forward_kernel – computes self-attention.
// Q, K, V: [batch_size, num_tokens, dim]
// attn_weights: [batch_size, num_tokens, num_tokens]
// O: [batch_size, num_tokens, dim] (the “attention output”)
__global__ void attention_forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* attn_weights,
    float* O,
    int num_tokens,
    int dim)
{
    int sample = blockIdx.x;       // one block per sample
    int token = threadIdx.x;       // each thread handles one query token (assume blockDim.x = num_tokens)
    if(token < num_tokens){
        const float* Q_s = Q + sample * num_tokens * dim;
        const float* K_s = K + sample * num_tokens * dim;
        const float* V_s = V + sample * num_tokens * dim;
        float* attn_s = attn_weights + sample * num_tokens * num_tokens;
        float* O_s = O + sample * num_tokens * dim;
        float scores[64];        // assuming num_tokens <= 64
        for(int j = 0; j < num_tokens; j++){
            float dot = 0.0f;
            for(int k = 0; k < dim; k++){
                dot += Q_s[token * dim + k] * K_s[j * dim + k];
            }
            scores[j] = dot / sqrtf((float)dim);
        }
        float max_score = scores[0];
        for(int j = 1; j < num_tokens; j++){
            if(scores[j] > max_score)
                max_score = scores[j];
        }
        float sum_exp = 0.0f;
        for(int j = 0; j < num_tokens; j++){
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for(int j = 0; j < num_tokens; j++){
            float weight = scores[j] / sum_exp;
            attn_s[token * num_tokens + j] = weight;
        }
        for(int k = 0; k < dim; k++){
            float out_val = 0.0f;
            for(int j = 0; j < num_tokens; j++){
                out_val += attn_s[token * num_tokens + j] * V_s[j * dim + k];
            }
            O_s[token * dim + k] = out_val;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: attention_backward_kernel – backward pass for attention.
// Computes gradients for Q, K and V from dO (the gradient of the attention output)
// and the saved attention weights. One block per sample.
__global__ void attention_backward_kernel(
    const float* dO,   // gradient from loss, shape: [batch_size, num_tokens, dim]
    const float* Q,    // saved Q from forward
    const float* K,    // saved K
    const float* V,    // saved V
    const float* attn_weights,  // saved attention weights, [batch_size, num_tokens, num_tokens]
    float* dQ,         // output gradient for Q, same shape as Q
    float* dK,         // output gradient for K, same shape as K
    float* dV,         // output gradient for V, same shape as V
    int num_tokens,
    int dim)
{
    int sample = blockIdx.x;   // one block per sample
    const float* Q_s = Q + sample * num_tokens * dim;
    const float* K_s = K + sample * num_tokens * dim;
    const float* V_s = V + sample * num_tokens * dim;
    const float* dO_s = dO + sample * num_tokens * dim;
    const float* attn_s = attn_weights + sample * num_tokens * num_tokens;
    float* dQ_s = dQ + sample * num_tokens * dim;
    float* dK_s = dK + sample * num_tokens * dim;
    float* dV_s = dV + sample * num_tokens * dim;
    // Initialize gradients for this sample to zero.
    for (int i = 0; i < num_tokens * dim; i++){
        dQ_s[i] = 0.0f;
        dK_s[i] = 0.0f;
        dV_s[i] = 0.0f;
    }
    // For each query token i compute intermediate gradients.
    for(int i = 0; i < num_tokens; i++){
        float temp[64];  // temporary buffer: dot(dO[i], V[j]) for each key j.
        for(int j = 0; j < num_tokens; j++){
            float dot = 0.0f;
            for(int k = 0; k < dim; k++){
                dot += dO_s[i*dim + k] * V_s[j*dim + k];
            }
            temp[j] = dot;
        }
        float z = 0.0f;
        for(int j = 0; j < num_tokens; j++){
            z += attn_s[i*num_tokens + j] * temp[j];
        }
        for(int j = 0; j < num_tokens; j++){
            float d_s = attn_s[i*num_tokens + j] * (temp[j] - z);
            for(int k = 0; k < dim; k++){
                dQ_s[i*dim + k] += (1.0f/sqrtf((float)dim)) * d_s * K_s[j*dim + k];
                dK_s[j*dim + k] += (1.0f/sqrtf((float)dim)) * d_s * Q_s[i*dim + k];
            }
        }
    }
    // Compute gradient for V: for each key token j, dV[j] = sum_i (attn[i,j] * dO[i])
    for(int j = 0; j < num_tokens; j++){
        for(int k = 0; k < dim; k++){
            float sum = 0.0f;
            for(int i = 0; i < num_tokens; i++){
                sum += attn_s[i*num_tokens + j] * dO_s[i*dim + k];
            }
            dV_s[j*dim + k] += sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: embedding_backward_kernel – scatters dX_grad (gradient from the projection)
// into the embedding gradient buffer using the stored embed_indices.
__global__ void embedding_backward_kernel(
    const float* dX_grad,       // [batch_size x raw_input_dim x embedding_dim]
    int* embed_indices,         // [batch_size x raw_input_dim]
    float* d_embedding_grad,    // [raw_input_dim x (num_bins * embedding_dim)]
    int batch_size,
    int raw_input_dim,
    int num_bins,
    int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * raw_input_dim;
    if(idx < total){
        int sample = idx / raw_input_dim;
        int feature = idx % raw_input_dim;
        int bin = embed_indices[idx];
        int emb_offset = feature * num_bins * embedding_dim + bin * embedding_dim;
        int dX_offset = idx * embedding_dim;
        for(int k = 0; k < embedding_dim; k++){
            atomicAdd(&d_embedding_grad[emb_offset + k], dX_grad[dX_offset + k]);
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: cross_entropy_loss_kernel – computes the softmax and cross–entropy loss
// for each token prediction and sets d_error = (softmax(prob) - one_hot(target)).
__global__ void cross_entropy_loss_kernel(
    const float* predictions, // [batch_size x (num_raw_targets * target_bins)]
    const int* target_labels, // [batch_size x num_raw_targets]
    float* out_loss,          // [batch_size x num_raw_targets]
    float* d_error,           // [batch_size x (num_raw_targets * target_bins)]
    int batch_size,
    int num_raw_targets,
    int target_bins,
    float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_raw_targets;
    if (i < total_groups) {
        int sample = i / num_raw_targets;
        int token = i % num_raw_targets;
        int offset = sample * (num_raw_targets * target_bins) + token * target_bins;
        float max_logit = predictions[offset];
        for (int j = 1; j < target_bins; j++) {
            float logit = predictions[offset + j];
            if (logit > max_logit)
                max_logit = logit;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < target_bins; j++) {
            sum_exp += expf(predictions[offset + j] - max_logit);
        }
        float loss_group = 0.0f;
        int target = target_labels[sample * num_raw_targets + token];
        for (int j = 0; j < target_bins; j++) {
            float exp_val = expf(predictions[offset + j] - max_logit);
            float prob = exp_val / sum_exp;
            d_error[offset + j] = prob - ((j == target) ? 1.0f : 0.0f);
            if (j == target) {
                loss_group = -logf(prob + epsilon);
            }
        }
        out_loss[i] = loss_group;
    }
}

// -----------------------------------------------------------------------------
// Structure: Net
typedef struct {
    // Embedding parameters.
    int raw_input_dim;    // number of tokens per sample
    int num_bins;         // number of bins per token (for embedding lookup)
    int embedding_dim;    // dimension of each embedding vector
    float* d_embedding_table;   // device embedding table [raw_input_dim x (num_bins*embedding_dim)]
    float* h_embedding_table;   // host copy (for saving/loading)
    float* d_embedding_grad;    // gradient for embedding table
    float* d_embedding_m;       // Adam momentum buffer for embedding
    float* d_embedding_v;       // Adam variance buffer for embedding

    // Projection parameters for attention.
    // Wq, Wk, Wv: each is [embedding_dim x embedding_dim] (row–major)
    float* d_Wq;       
    float* h_Wq;       
    float* d_Wq_grad;  
    float* d_Wq_m;     
    float* d_Wq_v;
    
    float* d_Wk;       
    float* h_Wk;       
    float* d_Wk_grad;  
    float* d_Wk_m;     
    float* d_Wk_v;
    
    float* d_Wv;       
    float* h_Wv;       
    float* d_Wv_grad;  
    float* d_Wv_m;     
    float* d_Wv_v;
    
    // Derived dimensions.
    int input_dim;        // = raw_input_dim * embedding_dim
    int num_raw_targets;  // = raw_input_dim  (one prediction per token)
    int target_bins;      // = embedding_dim (each token predicts one of embedding_dim classes)
    int output_dim;       // = input_dim
    int batch_size;
    
    // Device pointers for data.
    float* d_X;           // embedded input [batch_size x (raw_input_dim * embedding_dim)]
    // Projection outputs.
    float* d_Q;           // [batch_size x (raw_input_dim * embedding_dim)]
    float* d_K;           // [batch_size x (raw_input_dim * embedding_dim)]
    float* d_V;           // [batch_size x (raw_input_dim * embedding_dim)]
    
    // Buffers for gradients from attention.
    float* d_Q_grad;      // same shape as d_Q
    float* d_K_grad;      // same shape as d_K
    float* d_V_grad;      // same shape as d_V
    
    float* d_X_grad;      // gradient for embedded input (from projection backprop)
    
    float* d_predictions; // output logits (attention output) [batch_size x output_dim]
    float* d_error;       // gradient from loss [batch_size x output_dim]
    float* d_y;           // (target not used in cross-entropy loss)
    
    // Attention weights: [batch_size x (raw_input_dim x raw_input_dim)]
    float* d_attn_weights;
    
    // Embedding indices from lookup (for backward)
    int* d_embed_indices; // [batch_size x raw_input_dim]
    
    // Adam hyperparameters.
    float beta1;
    float beta2;
    float epsilon;
    int t;              // time step counter
    float weight_decay; // weight decay for AdamW
    
    // cuBLAS handle (unused in our custom kernels but kept for API consistency)
    cublasHandle_t cublas_handle;
} Net;

// -----------------------------------------------------------------------------
// Function prototypes (External API)
Net* init_net(int raw_input_dim, int embedding_num_bins, int embedding_dim, int batch_size);
void embed_input(Net* net, float* raw_input_h);
void forward_pass(Net* net);
float calculate_loss(Net* net, int* target_labels);
void zero_gradients(Net* net);
void backward_pass(Net* net);
void update_weights(Net* net, float learning_rate);
void save_model(Net* net, const char* filename);
Net* load_model(const char* filename);
void free_net(Net* net);

#endif