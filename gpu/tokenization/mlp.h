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
    // Device pointers for weights and gradients
    float* d_token_embed;      // [num_tokens, model_dim]
    float* d_channel_fc;       // [model_dim, model_dim]
    float* d_token_fc;         // [num_tokens, num_tokens]
    float* d_final_proj;       // [model_dim, num_tokens]
    float* d_token_embed_grad;
    float* d_channel_fc_grad;
    float* d_token_fc_grad;
    float* d_final_proj_grad;
    
    // Host copies of weights
    float* h_token_embed;
    float* h_channel_fc;
    float* h_token_fc;
    float* h_final_proj;
    
    // Device pointers for Adam parameters
    float* d_token_embed_m;
    float* d_token_embed_v;
    float* d_channel_fc_m;
    float* d_channel_fc_v;
    float* d_token_fc_m;
    float* d_token_fc_v;
    float* d_final_proj_m;
    float* d_final_proj_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays
    float* d_embedded;           // After token embedding
    float* d_after_channel;      // After channel mixing
    float* d_after_token;        // After token mixing
    float* d_predictions;        // Final output
    float* d_error;             // Error storage
    float* d_grad_after_token;
    float* d_grad_after_channel;
    float* d_grad_embedded;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int num_tokens;
    int model_dim;
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
    net->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Allocate host memory for weights
    net->h_token_embed = (float*)malloc(num_tokens * model_dim * sizeof(float));
    net->h_channel_fc = (float*)malloc(model_dim * model_dim * sizeof(float));
    net->h_token_fc = (float*)malloc(num_tokens * num_tokens * sizeof(float));
    net->h_final_proj = (float*)malloc(model_dim * num_tokens * sizeof(float));
    
    // Initialize weights with scaled random values
    float token_embed_scale = 1.0f / sqrt(num_tokens);
    float channel_fc_scale = 1.0f / sqrt(model_dim);
    float token_fc_scale = 1.0f / sqrt(num_tokens);
    float final_proj_scale = 1.0f / sqrt(model_dim);
    
    for (int i = 0; i < num_tokens * model_dim; i++) {
        net->h_token_embed[i] = ((float)rand() / RAND_MAX * 2 - 1) * token_embed_scale;
    }
    for (int i = 0; i < model_dim * model_dim; i++) {
        net->h_channel_fc[i] = ((float)rand() / RAND_MAX * 2 - 1) * channel_fc_scale;
    }
    for (int i = 0; i < num_tokens * num_tokens; i++) {
        net->h_token_fc[i] = ((float)rand() / RAND_MAX * 2 - 1) * token_fc_scale;
    }
    for (int i = 0; i < model_dim * num_tokens; i++) {
        net->h_final_proj[i] = ((float)rand() / RAND_MAX * 2 - 1) * final_proj_scale;
    }
    
    // Allocate all device memory
    size_t token_embed_size = num_tokens * model_dim * sizeof(float);
    size_t channel_fc_size = model_dim * model_dim * sizeof(float);
    size_t token_fc_size = num_tokens * num_tokens * sizeof(float);
    size_t final_proj_size = model_dim * num_tokens * sizeof(float);
    size_t batch_token_size = batch_size * num_tokens * sizeof(float);
    size_t batch_model_size = batch_size * model_dim * sizeof(float);
    
    // Weights and gradients
    CHECK_CUDA(cudaMalloc(&net->d_token_embed, token_embed_size));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc, channel_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc, token_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj, final_proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_token_embed_grad, token_embed_size));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc_grad, channel_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc_grad, token_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj_grad, final_proj_size));
    
    // Adam states
    CHECK_CUDA(cudaMalloc(&net->d_token_embed_m, token_embed_size));
    CHECK_CUDA(cudaMalloc(&net->d_token_embed_v, token_embed_size));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc_m, channel_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc_v, channel_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc_m, token_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc_v, token_fc_size));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj_m, final_proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj_v, final_proj_size));
    
    // Helper arrays
    CHECK_CUDA(cudaMalloc(&net->d_embedded, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_after_channel, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_after_token, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_predictions, batch_token_size));
    CHECK_CUDA(cudaMalloc(&net->d_error, batch_token_size));
    CHECK_CUDA(cudaMalloc(&net->d_grad_after_token, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_grad_after_channel, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_grad_embedded, batch_size * num_tokens * model_dim * sizeof(float)));
    
    // Initialize all device memory
    CHECK_CUDA(cudaMemcpy(net->d_token_embed, net->h_token_embed, token_embed_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc, net->h_channel_fc, channel_fc_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_fc, net->h_token_fc, token_fc_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_final_proj, net->h_final_proj, final_proj_size, cudaMemcpyHostToDevice));
    
    // Initialize Adam states to zero
    cudaMemset(net->d_token_embed_m, 0, token_embed_size);
    cudaMemset(net->d_token_embed_v, 0, token_embed_size);
    cudaMemset(net->d_channel_fc_m, 0, channel_fc_size);
    cudaMemset(net->d_channel_fc_v, 0, channel_fc_size);
    cudaMemset(net->d_token_fc_m, 0, token_fc_size);
    cudaMemset(net->d_token_fc_v, 0, token_fc_size);
    cudaMemset(net->d_final_proj_m, 0, final_proj_size);
    cudaMemset(net->d_final_proj_v, 0, final_proj_size);
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    // Free device memory - weights and gradients
    cudaFree(net->d_token_embed);
    cudaFree(net->d_channel_fc);
    cudaFree(net->d_token_fc);
    cudaFree(net->d_final_proj);
    cudaFree(net->d_token_embed_grad);
    cudaFree(net->d_channel_fc_grad);
    cudaFree(net->d_token_fc_grad);
    cudaFree(net->d_final_proj_grad);
    
    // Free device memory - Adam states
    cudaFree(net->d_token_embed_m);
    cudaFree(net->d_token_embed_v);
    cudaFree(net->d_channel_fc_m);
    cudaFree(net->d_channel_fc_v);
    cudaFree(net->d_token_fc_m);
    cudaFree(net->d_token_fc_v);
    cudaFree(net->d_final_proj_m);
    cudaFree(net->d_final_proj_v);
    
    // Free device memory - helper arrays
    cudaFree(net->d_embedded);
    cudaFree(net->d_after_channel);
    cudaFree(net->d_after_token);
    cudaFree(net->d_predictions);
    cudaFree(net->d_error);
    cudaFree(net->d_grad_after_token);
    cudaFree(net->d_grad_after_channel);
    cudaFree(net->d_grad_embedded);
    
    // Free host memory
    free(net->h_token_embed);
    free(net->h_channel_fc);
    free(net->h_token_fc);
    free(net->h_final_proj);
    
    // Destroy cuBLAS handle
    cublasDestroy(net->cublas_handle);
    
    // Free network struct
    free(net);
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

// Custom kernel for calculating error
__global__ void calc_error_kernel(float* error, float* predictions, float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - targets[idx];
    }
}

// Forward pass
void forward_pass(Net* net, float* X) {
    // Copy input to device
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->num_tokens * sizeof(float), 
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Token embedding: [num_tokens, model_dim] x [batch_size, num_tokens] -> [batch_size, model_dim]
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->model_dim,      // n
                            net->batch_size,     // m
                            net->num_tokens,     // k
                            &alpha,
                            net->d_token_embed,  // A
                            net->model_dim,      // lda
                            d_X,                 // B
                            net->num_tokens,     // ldb
                            &beta,
                            net->d_embedded,     // C
                            net->model_dim));    // ldc

    // Channel mixing: [model_dim, model_dim] x [batch_size, model_dim] -> [batch_size, model_dim]
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->model_dim,      // n
                            net->batch_size,     // m
                            net->model_dim,      // k
                            &alpha,
                            net->d_channel_fc,   // A
                            net->model_dim,      // lda
                            net->d_embedded,     // B
                            net->model_dim,      // ldb
                            &beta,
                            net->d_after_channel,// C
                            net->model_dim));    // ldc

    // Apply Swish activation
    int block_size = 256;
    int num_blocks = (net->batch_size * net->model_dim + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_after_channel,
        net->d_after_channel,
        net->batch_size * net->model_dim
    );

    // Token mixing: [num_tokens, num_tokens] x [batch_size, num_tokens] -> [batch_size, num_tokens]
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->num_tokens,     // n
                            net->batch_size,     // m
                            net->num_tokens,     // k
                            &alpha,
                            net->d_token_fc,     // A
                            net->num_tokens,     // lda
                            net->d_after_channel,// B
                            net->num_tokens,     // ldb
                            &beta,
                            net->d_after_token,  // C
                            net->num_tokens));   // ldc

    // Apply Swish activation
    num_blocks = (net->batch_size * net->num_tokens + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_after_token,
        net->d_after_token,
        net->batch_size * net->num_tokens
    );

    // Final projection: [model_dim, num_tokens] x [batch_size, model_dim] -> [batch_size, num_tokens]
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->num_tokens,     // n
                            net->batch_size,     // m
                            net->model_dim,      // k
                            &alpha,
                            net->d_final_proj,   // A
                            net->num_tokens,     // lda
                            net->d_after_token,  // B
                            net->model_dim,      // ldb
                            &beta,
                            net->d_predictions,  // C
                            net->num_tokens));   // ldc

    // Cleanup
    cudaFree(d_X);
}

// Calculate loss
float calculate_loss(Net* net, float* y) {
    float* d_y;
    CHECK_CUDA(cudaMalloc(&d_y, net->batch_size * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_y, y, net->batch_size * net->num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Calculate error (predictions - y)
    int size = net->batch_size * net->num_tokens;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    calc_error_kernel<<<num_blocks, block_size>>>(
        net->d_error,
        net->d_predictions,
        d_y,
        size
    );

    // Calculate loss on CPU
    float* h_error = (float*)malloc(size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_error, net->d_error, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss += h_error[i] * h_error[i];
    }

    // Cleanup
    free(h_error);
    cudaFree(d_y);

    return loss / size;
}

// Zero gradients
void zero_gradients(Net* net) {
    CHECK_CUDA(cudaMemset(net->d_token_embed_grad, 0, 
                         net->num_tokens * net->model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_channel_fc_grad, 0,
                         net->model_dim * net->model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_fc_grad, 0,
                         net->num_tokens * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_final_proj_grad, 0,
                         net->model_dim * net->num_tokens * sizeof(float)));
}

// Backward pass
void backward_pass(Net* net, float* X) {
    // Copy input to device
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Gradient of final projection layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->num_tokens,     // n
                            net->model_dim,      // m
                            net->batch_size,     // k
                            &alpha,
                            net->d_error,        // A
                            net->num_tokens,     // lda
                            net->d_after_token,  // B
                            net->model_dim,      // ldb
                            &beta,
                            net->d_final_proj_grad, // C
                            net->num_tokens));   // ldc

    // Backpropagate error through final projection
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            net->model_dim,      // n
                            net->batch_size,     // m
                            net->num_tokens,     // k
                            &alpha,
                            net->d_final_proj,   // A
                            net->num_tokens,     // lda
                            net->d_error,        // B
                            net->num_tokens,     // ldb
                            &beta,
                            net->d_grad_after_token, // C
                            net->model_dim));    // ldc

    // Apply Swish derivative for token mixing
    int block_size = 256;
    int num_blocks = (net->batch_size * net->num_tokens + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_grad_after_token,
        net->d_after_token,
        net->d_grad_after_token,
        net->batch_size * net->num_tokens
    );

    // Gradient of token mixing layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->num_tokens,     // n
                            net->num_tokens,     // m
                            net->batch_size,     // k
                            &alpha,
                            net->d_grad_after_token, // A
                            net->num_tokens,     // lda
                            net->d_after_channel,// B
                            net->num_tokens,     // ldb
                            &beta,
                            net->d_token_fc_grad,// C
                            net->num_tokens));   // ldc

    // Backpropagate error through token mixing
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            net->model_dim,      // n
                            net->batch_size,     // m
                            net->num_tokens,     // k
                            &alpha,
                            net->d_token_fc,     // A
                            net->num_tokens,     // lda
                            net->d_grad_after_token, // B
                            net->num_tokens,     // ldb
                            &beta,
                            net->d_grad_after_channel, // C
                            net->model_dim));    // ldc

    // Apply Swish derivative for channel mixing
    num_blocks = (net->batch_size * net->model_dim + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_grad_after_channel,
        net->d_after_channel,
        net->d_grad_after_channel,
        net->batch_size * net->model_dim
    );

    // Gradient of channel mixing layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->model_dim,      // n
                            net->model_dim,      // m
                            net->batch_size,     // k
                            &alpha,
                            net->d_grad_after_channel, // A
                            net->model_dim,      // lda
                            net->d_embedded,     // B
                            net->model_dim,      // ldb
                            &beta,
                            net->d_channel_fc_grad, // C
                            net->model_dim));    // ldc

    // Backpropagate error through channel mixing
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            net->model_dim,      // n
                            net->batch_size,     // m
                            net->model_dim,      // k
                            &alpha,
                            net->d_channel_fc,   // A
                            net->model_dim,      // lda
                            net->d_grad_after_channel, // B
                            net->model_dim,      // ldb
                            &beta,
                            net->d_grad_embedded,// C
                            net->model_dim));    // ldc

    // Gradient of token embedding
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->num_tokens,     // n
                            net->model_dim,      // m
                            net->batch_size,     // k
                            &alpha,
                            d_X,                 // A
                            net->num_tokens,     // lda
                            net->d_grad_embedded,// B
                            net->model_dim,      // ldb
                            &beta,
                            net->d_token_embed_grad, // C
                            net->num_tokens));   // ldc

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
    int token_embed_size = net->num_tokens * net->model_dim;
    int blocks = (token_embed_size + block_size - 1) / block_size;
    adamw_update_kernel<<<blocks, block_size>>>(
        net->d_token_embed,
        net->d_token_embed_grad,
        net->d_token_embed_m,
        net->d_token_embed_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        token_embed_size, net->batch_size
    );
    
    // Update channel mixing weights
    int channel_fc_size = net->model_dim * net->model_dim;
    blocks = (channel_fc_size + block_size - 1) / block_size;
    adamw_update_kernel<<<blocks, block_size>>>(
        net->d_channel_fc,
        net->d_channel_fc_grad,
        net->d_channel_fc_m,
        net->d_channel_fc_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        channel_fc_size, net->batch_size
    );
    
    // Update token mixing weights
    int token_fc_size = net->num_tokens * net->num_tokens;
    blocks = (token_fc_size + block_size - 1) / block_size;
    adamw_update_kernel<<<blocks, block_size>>>(
        net->d_token_fc,
        net->d_token_fc_grad,
        net->d_token_fc_m,
        net->d_token_fc_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        token_fc_size, net->batch_size
    );
    
    // Update final projection weights
    int final_proj_size = net->model_dim * net->num_tokens;
    blocks = (final_proj_size + block_size - 1) / block_size;
    adamw_update_kernel<<<blocks, block_size>>>(
        net->d_final_proj,
        net->d_final_proj_grad,
        net->d_final_proj_m,
        net->d_final_proj_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        final_proj_size, net->batch_size
    );
}

// Save model weights to binary file
void save_model(Net* net, const char* filename) {
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(net->h_token_embed, net->d_token_embed,
                         net->num_tokens * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_channel_fc, net->d_channel_fc,
                         net->model_dim * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_token_fc, net->d_token_fc,
                         net->num_tokens * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_final_proj, net->d_final_proj,
                         net->model_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&net->num_tokens, sizeof(int), 1, file);
    fwrite(&net->model_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(net->h_token_embed, sizeof(float), net->num_tokens * net->model_dim, file);
    fwrite(net->h_channel_fc, sizeof(float), net->model_dim * net->model_dim, file);
    fwrite(net->h_token_fc, sizeof(float), net->num_tokens * net->num_tokens, file);
    fwrite(net->h_final_proj, sizeof(float), net->model_dim * net->num_tokens, file);
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
    
    fread(net->h_token_embed, sizeof(float), num_tokens * model_dim, file);
    fread(net->h_channel_fc, sizeof(float), model_dim * model_dim, file);
    fread(net->h_token_fc, sizeof(float), num_tokens * num_tokens, file);
    fread(net->h_final_proj, sizeof(float), model_dim * num_tokens, file);
    fread(&net->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_token_embed, net->h_token_embed,
                         num_tokens * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc, net->h_channel_fc,
                         model_dim * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_fc, net->h_token_fc,
                         num_tokens * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_final_proj, net->h_final_proj,
                         model_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif