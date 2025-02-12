#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuBLAS Error checking macro
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef struct {
    // Device pointers for weights
    float* d_token_embedding;    // [num_tokens x model_dim]
    float* d_token_mixing;       // [num_tokens x num_tokens]
    float* d_channel_mixing;     // [model_dim x model_dim]
    float* d_output_proj;        // [model_dim x 1]
    
    // Weight gradients
    float* d_token_embedding_grad;
    float* d_token_mixing_grad;
    float* d_channel_mixing_grad;
    float* d_output_proj_grad;
    
    // Host copies of weights
    float* h_token_embedding;
    float* h_token_mixing;
    float* h_channel_mixing;
    float* h_output_proj;
    
    // Adam parameters for each weight matrix
    float* d_token_embedding_m;
    float* d_token_embedding_v;
    float* d_token_mixing_m;
    float* d_token_mixing_v;
    float* d_channel_mixing_m;
    float* d_channel_mixing_v;
    float* d_output_proj_m;
    float* d_output_proj_v;
    
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Intermediate activations
    float* d_embedded;           // After token embedding
    float* d_token_mixed;        // After token mixing
    float* d_channel_mixed;      // After channel mixing
    float* d_predictions;        // Final output
    
    // Intermediate gradients and temporary storage
    float* d_token_mixing_temp;
    float* d_channel_mixing_temp;
    float* d_embedding_temp;
    float* d_error;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int num_tokens;     // Number of input/output tokens (16 in your case)
    int model_dim;      // Hidden dimension for token representation
    int batch_size;
} Net;

// Initialize the network
Net* init_net(int num_tokens, int model_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->num_tokens = num_tokens;
    net->model_dim = model_dim;
    net->batch_size = batch_size;
    
    // Initialize Adam parameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.001f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Allocate host memory for weights
    net->h_token_embedding = (float*)malloc(num_tokens * model_dim * sizeof(float));
    net->h_token_mixing = (float*)malloc(num_tokens * num_tokens * sizeof(float));
    net->h_channel_mixing = (float*)malloc(model_dim * model_dim * sizeof(float));
    net->h_output_proj = (float*)malloc(model_dim * sizeof(float));
    
    // Initialize weights on host with scaled random values
    float scale_embed = sqrtf(2.0f / (num_tokens + model_dim));
    float scale_token = sqrtf(2.0f / (num_tokens + num_tokens));
    float scale_channel = sqrtf(2.0f / (model_dim + model_dim));
    float scale_output = sqrtf(2.0f / model_dim);
    
    for (int i = 0; i < num_tokens * model_dim; i++) {
        net->h_token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_embed;
    }
    
    for (int i = 0; i < num_tokens * num_tokens; i++) {
        net->h_token_mixing[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_token;
    }
    
    for (int i = 0; i < model_dim * model_dim; i++) {
        net->h_channel_mixing[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_channel;
    }
    
    for (int i = 0; i < model_dim; i++) {
        net->h_output_proj[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_output;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&net->d_token_embedding, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_mixing, num_tokens * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mixing, model_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_output_proj, model_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&net->d_token_embedding_grad, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_mixing_grad, num_tokens * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mixing_grad, model_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_output_proj_grad, model_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&net->d_token_embedding_m, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_embedding_v, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_mixing_m, num_tokens * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_mixing_v, num_tokens * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mixing_m, model_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mixing_v, model_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_output_proj_m, model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_output_proj_v, model_dim * sizeof(float)));
    
    // Allocate device memory for intermediate activations
    CHECK_CUDA(cudaMalloc(&net->d_embedded, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_mixed, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mixed, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_predictions, batch_size * num_tokens * sizeof(float)));
    
    // Allocate device memory for temporary storage and gradients
    CHECK_CUDA(cudaMalloc(&net->d_token_mixing_temp, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mixing_temp, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_embedding_temp, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error, batch_size * num_tokens * sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_token_embedding, net->h_token_embedding,
                         num_tokens * model_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_mixing, net->h_token_mixing,
                         num_tokens * num_tokens * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_mixing, net->h_channel_mixing,
                         model_dim * model_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_output_proj, net->h_output_proj,
                         model_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam accumulators to zero
    CHECK_CUDA(cudaMemset(net->d_token_embedding_m, 0, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_embedding_v, 0, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_mixing_m, 0, num_tokens * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_mixing_v, 0, num_tokens * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_channel_mixing_m, 0, model_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_channel_mixing_v, 0, model_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_output_proj_m, 0, model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_output_proj_v, 0, model_dim * sizeof(float)));
    
    return net;
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel(float* grad_output, float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float swish = x * sigmoid;
        grad_input[idx] = grad_output[idx] * (swish + sigmoid * (1.0f - swish));
    }
}

// Forward pass
void forward_pass(Net* net, float* X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int block_size = 256;
    
    // Copy input to device if not already there
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->num_tokens * sizeof(float), 
                         cudaMemcpyHostToDevice));

    // 1. Token Embedding: [batch_size, num_tokens] -> [batch_size, num_tokens, model_dim]
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->model_dim,           // n
                            net->batch_size * net->num_tokens, // m
                            1,                        // k
                            &alpha,
                            net->d_token_embedding,   // A
                            net->model_dim,           // lda
                            d_X,                      // B
                            1,                        // ldb
                            &beta,
                            net->d_embedded,          // C
                            net->model_dim));         // ldc

    // Apply Swish after embedding
    int embed_size = net->batch_size * net->num_tokens * net->model_dim;
    int num_blocks = (embed_size + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_embedded,
        net->d_embedded,
        embed_size
    );

    // 2. Token Mixing: mix across tokens for each channel
    // Reshape and transpose operations are handled implicitly in the matrix multiplication
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->num_tokens,          // n
                            net->batch_size * net->model_dim, // m
                            net->num_tokens,          // k
                            &alpha,
                            net->d_token_mixing,      // A
                            net->num_tokens,          // lda
                            net->d_embedded,          // B
                            net->num_tokens,          // ldb
                            &beta,
                            net->d_token_mixed,       // C
                            net->num_tokens));        // ldc

    // Apply Swish after token mixing
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_token_mixed,
        net->d_token_mixed,
        embed_size
    );

    // 3. Channel Mixing: mix across channels for each token
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->model_dim,           // n
                            net->batch_size * net->num_tokens, // m
                            net->model_dim,           // k
                            &alpha,
                            net->d_channel_mixing,    // A
                            net->model_dim,           // lda
                            net->d_token_mixed,       // B
                            net->model_dim,           // ldb
                            &beta,
                            net->d_channel_mixed,     // C
                            net->model_dim));         // ldc

    // Apply Swish after channel mixing
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_channel_mixed,
        net->d_channel_mixed,
        embed_size
    );

    // 4. Output Projection: project back to scalar per token
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            1,                        // n
                            net->batch_size * net->num_tokens, // m
                            net->model_dim,           // k
                            &alpha,
                            net->d_output_proj,       // A
                            net->model_dim,           // lda
                            net->d_channel_mixed,     // B
                            net->model_dim,           // ldb
                            &beta,
                            net->d_predictions,       // C
                            1));                      // ldc

    // Cleanup
    cudaFree(d_X);
}

// Custom kernel for calculating error and squared error
__global__ void calc_error_kernel(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// Backward pass
void backward_pass(Net* net, float* X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int block_size = 256;

    // Copy input to device if not already there
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));

    // 1. Output Projection Gradient
    // Gradient w.r.t. output projection weights
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->model_dim,           // n
                            1,                        // m
                            net->batch_size * net->num_tokens, // k
                            &alpha,
                            net->d_channel_mixed,     // A
                            net->model_dim,           // lda
                            net->d_error,             // B
                            1,                        // ldb
                            &beta,
                            net->d_output_proj_grad,  // C
                            net->model_dim));         // ldc

    // Gradient w.r.t. channel mixed output
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->model_dim,           // n
                            net->batch_size * net->num_tokens, // m
                            1,                        // k
                            &alpha,
                            net->d_output_proj,       // A
                            net->model_dim,           // lda
                            net->d_error,             // B
                            1,                        // ldb
                            &beta,
                            net->d_channel_mixing_temp, // C
                            net->model_dim));         // ldc

    // 2. Channel Mixing Gradient
    // Apply Swish derivative
    int embed_size = net->batch_size * net->num_tokens * net->model_dim;
    int num_blocks = (embed_size + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_channel_mixing_temp,
        net->d_channel_mixed,
        net->d_channel_mixing_temp,
        embed_size
    );

    // Gradient w.r.t. channel mixing weights
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->model_dim,           // n
                            net->model_dim,           // m
                            net->batch_size * net->num_tokens, // k
                            &alpha,
                            net->d_token_mixed,       // A
                            net->model_dim,           // lda
                            net->d_channel_mixing_temp, // B
                            net->model_dim,           // ldb
                            &beta,
                            net->d_channel_mixing_grad, // C
                            net->model_dim));         // ldc

    // Gradient w.r.t. token mixed output
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            net->model_dim,           // n
                            net->batch_size * net->num_tokens, // m
                            net->model_dim,           // k
                            &alpha,
                            net->d_channel_mixing,    // A
                            net->model_dim,           // lda
                            net->d_channel_mixing_temp, // B
                            net->model_dim,           // ldb
                            &beta,
                            net->d_token_mixing_temp, // C
                            net->model_dim));         // ldc

    // 3. Token Mixing Gradient
    // Apply Swish derivative
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_token_mixing_temp,
        net->d_token_mixed,
        net->d_token_mixing_temp,
        embed_size
    );

    // Gradient w.r.t. token mixing weights
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->num_tokens,          // n
                            net->num_tokens,          // m
                            net->batch_size * net->model_dim, // k
                            &alpha,
                            net->d_embedded,          // A
                            net->num_tokens,          // lda
                            net->d_token_mixing_temp, // B
                            net->num_tokens,          // ldb
                            &beta,
                            net->d_token_mixing_grad, // C
                            net->num_tokens));        // ldc

    // Gradient w.r.t. embedded output
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            net->num_tokens,          // n
                            net->batch_size * net->model_dim, // m
                            net->num_tokens,          // k
                            &alpha,
                            net->d_token_mixing,      // A
                            net->num_tokens,          // lda
                            net->d_token_mixing_temp, // B
                            net->num_tokens,          // ldb
                            &beta,
                            net->d_embedding_temp,    // C
                            net->num_tokens));        // ldc

    // 4. Token Embedding Gradient
    // Apply Swish derivative
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_embedding_temp,
        net->d_embedded,
        net->d_embedding_temp,
        embed_size
    );

    // Gradient w.r.t. token embedding weights
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->model_dim,           // n
                            1,                        // m
                            net->batch_size * net->num_tokens, // k
                            &alpha,
                            net->d_embedding_temp,    // A
                            net->model_dim,           // lda
                            d_X,                      // B
                            1,                        // ldb
                            &beta,
                            net->d_token_embedding_grad, // C
                            net->model_dim));         // ldc

    // Cleanup
    cudaFree(d_X);
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel(
    float* weight,
    float* grad,
    float* m,
    float* v,
    float beta1,
    float beta2,
    float epsilon,
    float learning_rate,
    float weight_decay,
    float alpha_t,
    int size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights(Net* net, float learning_rate) {
    net->t++;
    
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update token embedding weights
    int embed_size = net->num_tokens * net->model_dim;
    int embed_blocks = (embed_size + block_size - 1) / block_size;
    adamw_update_kernel<<<embed_blocks, block_size>>>(
        net->d_token_embedding,
        net->d_token_embedding_grad,
        net->d_token_embedding_m,
        net->d_token_embedding_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        embed_size,
        net->batch_size
    );
    
    // Update token mixing weights
    int token_mix_size = net->num_tokens * net->num_tokens;
    int token_mix_blocks = (token_mix_size + block_size - 1) / block_size;
    adamw_update_kernel<<<token_mix_blocks, block_size>>>(
        net->d_token_mixing,
        net->d_token_mixing_grad,
        net->d_token_mixing_m,
        net->d_token_mixing_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        token_mix_size,
        net->batch_size
    );
    
    // Update channel mixing weights
    int channel_mix_size = net->model_dim * net->model_dim;
    int channel_mix_blocks = (channel_mix_size + block_size - 1) / block_size;
    adamw_update_kernel<<<channel_mix_blocks, block_size>>>(
        net->d_channel_mixing,
        net->d_channel_mixing_grad,
        net->d_channel_mixing_m,
        net->d_channel_mixing_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        channel_mix_size,
        net->batch_size
    );
    
    // Update output projection weights
    int output_proj_blocks = (net->model_dim + block_size - 1) / block_size;
    adamw_update_kernel<<<output_proj_blocks, block_size>>>(
        net->d_output_proj,
        net->d_output_proj_grad,
        net->d_output_proj_m,
        net->d_output_proj_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        net->model_dim,
        net->batch_size
    );
}

// Zero gradients
void zero_gradients(Net* net) {
    CHECK_CUDA(cudaMemset(net->d_token_embedding_grad, 0, 
                         net->num_tokens * net->model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_mixing_grad, 0, 
                         net->num_tokens * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_channel_mixing_grad, 0, 
                         net->model_dim * net->model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_output_proj_grad, 0, 
                         net->model_dim * sizeof(float)));
}

// Calculate loss
float calculate_loss(Net* net, float* y) {
    float* d_y;
    CHECK_CUDA(cudaMalloc(&d_y, net->batch_size * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_y, y, net->batch_size * net->num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));

    int size = net->batch_size * net->num_tokens;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    calc_error_kernel<<<num_blocks, block_size>>>(
        net->d_error,
        net->d_predictions,
        d_y,
        size
    );

    float* h_error = (float*)malloc(size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_error, net->d_error, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss += h_error[i] * h_error[i];
    }

    free(h_error);
    cudaFree(d_y);

    return loss / size;
}

// Free network memory
void free_net(Net* net) {
    // Free device memory
    cudaFree(net->d_token_embedding);
    cudaFree(net->d_token_mixing);
    cudaFree(net->d_channel_mixing);
    cudaFree(net->d_output_proj);
    
    cudaFree(net->d_token_embedding_grad);
    cudaFree(net->d_token_mixing_grad);
    cudaFree(net->d_channel_mixing_grad);
    cudaFree(net->d_output_proj_grad);
    
    cudaFree(net->d_token_embedding_m);
    cudaFree(net->d_token_embedding_v);
    cudaFree(net->d_token_mixing_m);
    cudaFree(net->d_token_mixing_v);
    cudaFree(net->d_channel_mixing_m);
    cudaFree(net->d_channel_mixing_v);
    cudaFree(net->d_output_proj_m);
    cudaFree(net->d_output_proj_v);
    
    cudaFree(net->d_embedded);
    cudaFree(net->d_token_mixed);
    cudaFree(net->d_channel_mixed);
    cudaFree(net->d_predictions);
    cudaFree(net->d_error);
    
    cudaFree(net->d_token_mixing_temp);
    cudaFree(net->d_channel_mixing_temp);
    cudaFree(net->d_embedding_temp);
    
    // Free host memory
    free(net->h_token_embedding);
    free(net->h_token_mixing);
    free(net->h_channel_mixing);
    free(net->h_output_proj);
    
    // Destroy cuBLAS handle
    cublasDestroy(net->cublas_handle);
    
    free(net);
}

// Save model weights to binary file
void save_model(Net* net, const char* filename) {
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(net->h_token_embedding, net->d_token_embedding,
                         net->num_tokens * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_token_mixing, net->d_token_mixing,
                         net->num_tokens * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_channel_mixing, net->d_channel_mixing,
                         net->model_dim * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_output_proj, net->d_output_proj,
                         net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&net->num_tokens, sizeof(int), 1, file);
    fwrite(&net->model_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(net->h_token_embedding, sizeof(float), net->num_tokens * net->model_dim, file);
    fwrite(net->h_token_mixing, sizeof(float), net->num_tokens * net->num_tokens, file);
    fwrite(net->h_channel_mixing, sizeof(float), net->model_dim * net->model_dim, file);
    fwrite(net->h_output_proj, sizeof(float), net->model_dim, file);
    fwrite(&net->t, sizeof(int), 1, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int num_tokens, model_dim, batch_size;
    fread(&num_tokens, sizeof(int), 1, file);
    fread(&model_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    Net* net = init_net(num_tokens, model_dim, batch_size);
    
    fread(net->h_token_embedding, sizeof(float), num_tokens * model_dim, file);
    fread(net->h_token_mixing, sizeof(float), num_tokens * num_tokens, file);
    fread(net->h_channel_mixing, sizeof(float), model_dim * model_dim, file);
    fread(net->h_output_proj, sizeof(float), model_dim, file);
    fread(&net->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_token_embedding, net->h_token_embedding,
                         num_tokens * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_mixing, net->h_token_mixing,
                         num_tokens * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_mixing, net->h_channel_mixing,
                         model_dim * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_output_proj, net->h_output_proj,
                         model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif