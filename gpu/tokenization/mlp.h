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
    // Dimensions
    int num_tokens;     // number of input/output tokens (16)
    int model_dim;      // hidden dimension for tokens
    int mlp_dim;        // expansion dimension for mixing MLPs
    int batch_size;     // batch size

    // Device pointers for weights
    float* d_token_embed;     // [num_tokens, model_dim]
    float* d_channel_fc1;     // [model_dim, mlp_dim]
    float* d_channel_fc2;     // [mlp_dim, model_dim]
    float* d_token_fc1;       // [num_tokens, mlp_dim]
    float* d_token_fc2;       // [mlp_dim, num_tokens]
    float* d_final_proj;      // [model_dim, num_tokens]

    // Host copies of weights
    float* h_token_embed;
    float* h_channel_fc1;
    float* h_channel_fc2;
    float* h_token_fc1;
    float* h_token_fc2;
    float* h_final_proj;

    // Gradients
    float* d_token_embed_grad;
    float* d_channel_fc1_grad;
    float* d_channel_fc2_grad;
    float* d_token_fc1_grad;
    float* d_token_fc2_grad;
    float* d_final_proj_grad;

    // Intermediate activations
    float* d_embedded;        // After token embedding
    float* d_channel_mid;     // After first channel mixing
    float* d_after_channel;   // After channel mixing
    float* d_token_mid;       // After first token mixing
    float* d_after_token;     // After token mixing
    float* d_predictions;     // Final output

    // Pre-allocated memory for backward pass
    float* d_grad_after_token;
    float* d_grad_token_mid;
    float* d_grad_after_channel;
    float* d_grad_channel_mid;
    float* d_grad_embedded;
    float* d_X_device;        // Device copy of input
    float* d_error;           // Error storage

    // Adam optimizer parameters
    float* d_token_embed_m;
    float* d_token_embed_v;
    float* d_channel_fc1_m;
    float* d_channel_fc1_v;
    float* d_channel_fc2_m;
    float* d_channel_fc2_v;
    float* d_token_fc1_m;
    float* d_token_fc1_v;
    float* d_token_fc2_m;
    float* d_token_fc2_v;
    float* d_final_proj_m;
    float* d_final_proj_v;

    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;

    // cuBLAS handle
    cublasHandle_t cublas_handle;
} Net;

// Initialize the network
Net* init_net(int num_tokens, int model_dim, int mlp_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->num_tokens = num_tokens;
    net->model_dim = model_dim;
    net->mlp_dim = mlp_dim;
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
    net->h_channel_fc1 = (float*)malloc(model_dim * mlp_dim * sizeof(float));
    net->h_channel_fc2 = (float*)malloc(mlp_dim * model_dim * sizeof(float));
    net->h_token_fc1 = (float*)malloc(num_tokens * mlp_dim * sizeof(float));
    net->h_token_fc2 = (float*)malloc(mlp_dim * num_tokens * sizeof(float));
    net->h_final_proj = (float*)malloc(model_dim * num_tokens * sizeof(float));
    
    // Initialize weights with scaled random values
    float token_embed_scale = 1.0f / sqrt(num_tokens);
    float channel_fc1_scale = 1.0f / sqrt(model_dim);
    float channel_fc2_scale = 1.0f / sqrt(mlp_dim);
    float token_fc1_scale = 1.0f / sqrt(num_tokens);
    float token_fc2_scale = 1.0f / sqrt(mlp_dim);
    float final_proj_scale = 1.0f / sqrt(model_dim);
    
    for (int i = 0; i < num_tokens * model_dim; i++) {
        net->h_token_embed[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * token_embed_scale;
    }
    for (int i = 0; i < model_dim * mlp_dim; i++) {
        net->h_channel_fc1[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * channel_fc1_scale;
    }
    for (int i = 0; i < mlp_dim * model_dim; i++) {
        net->h_channel_fc2[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * channel_fc2_scale;
    }
    for (int i = 0; i < num_tokens * mlp_dim; i++) {
        net->h_token_fc1[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * token_fc1_scale;
    }
    for (int i = 0; i < mlp_dim * num_tokens; i++) {
        net->h_token_fc2[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * token_fc2_scale;
    }
    for (int i = 0; i < model_dim * num_tokens; i++) {
        net->h_final_proj[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * final_proj_scale;
    }
    
    // Allocate device memory for weights and copy from host
    CHECK_CUDA(cudaMalloc(&net->d_token_embed, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc1, model_dim * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc2, mlp_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc1, num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc2, mlp_dim * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj, model_dim * num_tokens * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&net->d_token_embed_grad, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc1_grad, model_dim * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc2_grad, mlp_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc1_grad, num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc2_grad, mlp_dim * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj_grad, model_dim * num_tokens * sizeof(float)));

    // Allocate device memory for intermediate activations
    CHECK_CUDA(cudaMalloc(&net->d_embedded, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_mid, batch_size * num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_after_channel, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_mid, batch_size * num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_after_token, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_predictions, batch_size * num_tokens * sizeof(float)));

    // Allocate memory for backward pass (pre-allocated)
    CHECK_CUDA(cudaMalloc(&net->d_grad_after_token, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_grad_token_mid, batch_size * num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_grad_after_channel, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_grad_channel_mid, batch_size * num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_grad_embedded, batch_size * num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_X_device, batch_size * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error, batch_size * num_tokens * sizeof(float)));

    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&net->d_token_embed_m, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_embed_v, num_tokens * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc1_m, model_dim * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc1_v, model_dim * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc2_m, mlp_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_channel_fc2_v, mlp_dim * model_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc1_m, num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc1_v, num_tokens * mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc2_m, mlp_dim * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_token_fc2_v, mlp_dim * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj_m, model_dim * num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_final_proj_v, model_dim * num_tokens * sizeof(float)));

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_token_embed, net->h_token_embed, 
                         num_tokens * model_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc1, net->h_channel_fc1,
                         model_dim * mlp_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc2, net->h_channel_fc2,
                         mlp_dim * model_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_fc1, net->h_token_fc1,
                         num_tokens * mlp_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_fc2, net->h_token_fc2,
                         mlp_dim * num_tokens * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_final_proj, net->h_final_proj,
                         model_dim * num_tokens * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize Adam states to zero
    cudaMemset(net->d_token_embed_m, 0, num_tokens * model_dim * sizeof(float));
    cudaMemset(net->d_token_embed_v, 0, num_tokens * model_dim * sizeof(float));
    cudaMemset(net->d_channel_fc1_m, 0, model_dim * mlp_dim * sizeof(float));
    cudaMemset(net->d_channel_fc1_v, 0, model_dim * mlp_dim * sizeof(float));
    cudaMemset(net->d_channel_fc2_m, 0, mlp_dim * model_dim * sizeof(float));
    cudaMemset(net->d_channel_fc2_v, 0, mlp_dim * model_dim * sizeof(float));
    cudaMemset(net->d_token_fc1_m, 0, num_tokens * mlp_dim * sizeof(float));
    cudaMemset(net->d_token_fc1_v, 0, num_tokens * mlp_dim * sizeof(float));
    cudaMemset(net->d_token_fc2_m, 0, mlp_dim * num_tokens * sizeof(float));
    cudaMemset(net->d_token_fc2_v, 0, mlp_dim * num_tokens * sizeof(float));
    cudaMemset(net->d_final_proj_m, 0, model_dim * num_tokens * sizeof(float));
    cudaMemset(net->d_final_proj_v, 0, model_dim * num_tokens * sizeof(float));
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    // Free device memory - weights
    cudaFree(net->d_token_embed);
    cudaFree(net->d_channel_fc1);
    cudaFree(net->d_channel_fc2);
    cudaFree(net->d_token_fc1);
    cudaFree(net->d_token_fc2);
    cudaFree(net->d_final_proj);
    
    // Free device memory - gradients
    cudaFree(net->d_token_embed_grad);
    cudaFree(net->d_channel_fc1_grad);
    cudaFree(net->d_channel_fc2_grad);
    cudaFree(net->d_token_fc1_grad);
    cudaFree(net->d_token_fc2_grad);
    cudaFree(net->d_final_proj_grad);
    
    // Free device memory - intermediate activations
    cudaFree(net->d_embedded);
    cudaFree(net->d_channel_mid);
    cudaFree(net->d_after_channel);
    cudaFree(net->d_token_mid);
    cudaFree(net->d_after_token);
    cudaFree(net->d_predictions);
    
    // Free device memory - backward pass pre-allocated
    cudaFree(net->d_grad_after_token);
    cudaFree(net->d_grad_token_mid);
    cudaFree(net->d_grad_after_channel);
    cudaFree(net->d_grad_channel_mid);
    cudaFree(net->d_grad_embedded);
    cudaFree(net->d_X_device);
    cudaFree(net->d_error);
    
    // Free device memory - Adam states
    cudaFree(net->d_token_embed_m);
    cudaFree(net->d_token_embed_v);
    cudaFree(net->d_channel_fc1_m);
    cudaFree(net->d_channel_fc1_v);
    cudaFree(net->d_channel_fc2_m);
    cudaFree(net->d_channel_fc2_v);
    cudaFree(net->d_token_fc1_m);
    cudaFree(net->d_token_fc1_v);
    cudaFree(net->d_token_fc2_m);
    cudaFree(net->d_token_fc2_v);
    cudaFree(net->d_final_proj_m);
    cudaFree(net->d_final_proj_v);
    
    // Free host memory
    free(net->h_token_embed);
    free(net->h_channel_fc1);
    free(net->h_channel_fc2);
    free(net->h_token_fc1);
    free(net->h_token_fc2);
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
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Copy input to pre-allocated device memory
    CHECK_CUDA(cudaMemcpy(net->d_X_device, X, 
                         net->batch_size * net->num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Token embedding
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->model_dim, net->batch_size * net->num_tokens, net->num_tokens,
        &alpha, net->d_token_embed, net->model_dim,
        net->d_X_device, net->num_tokens,
        &beta, net->d_embedded, net->model_dim));

    // Channel mixing (first layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->mlp_dim, net->batch_size * net->num_tokens, net->model_dim,
        &alpha, net->d_channel_fc1, net->mlp_dim,
        net->d_embedded, net->model_dim,
        &beta, net->d_channel_mid, net->mlp_dim));

    // Apply Swish
    int size = net->batch_size * net->num_tokens * net->mlp_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_channel_mid, net->d_channel_mid, size);

    // Channel mixing (second layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->model_dim, net->batch_size * net->num_tokens, net->mlp_dim,
        &alpha, net->d_channel_fc2, net->model_dim,
        net->d_channel_mid, net->mlp_dim,
        &beta, net->d_after_channel, net->model_dim));

    // Token mixing (first layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->mlp_dim, net->batch_size * net->num_tokens, net->num_tokens,
        &alpha, net->d_token_fc1, net->mlp_dim,
        net->d_after_channel, net->num_tokens,
        &beta, net->d_token_mid, net->mlp_dim));

    // Apply Swish
    size = net->batch_size * net->num_tokens * net->mlp_dim;
    num_blocks = (size + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_token_mid, net->d_token_mid, size);

    // Token mixing (second layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->num_tokens, net->batch_size * net->num_tokens, net->mlp_dim,
        &alpha, net->d_token_fc2, net->num_tokens,
        net->d_token_mid, net->mlp_dim,
        &beta, net->d_after_token, net->num_tokens));

    // Final projection
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->num_tokens, net->batch_size, net->model_dim,
        &alpha, net->d_final_proj, net->num_tokens,
        net->d_after_token, net->model_dim,
        &beta, net->d_predictions, net->num_tokens));
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
    CHECK_CUDA(cudaMemset(net->d_channel_fc1_grad, 0,
                         net->model_dim * net->mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_channel_fc2_grad, 0,
                         net->mlp_dim * net->model_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_fc1_grad, 0,
                         net->num_tokens * net->mlp_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_token_fc2_grad, 0,
                         net->mlp_dim * net->num_tokens * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_final_proj_grad, 0,
                         net->model_dim * net->num_tokens * sizeof(float)));
}

// Backward pass
void backward_pass(Net* net, float* X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Copy input to device (using pre-allocated memory)
    CHECK_CUDA(cudaMemcpy(net->d_X_device, X, 
                         net->batch_size * net->num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Backward through final projection
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->model_dim, net->batch_size, net->num_tokens,
        &alpha, net->d_final_proj, net->num_tokens,
        net->d_error, net->num_tokens,
        &beta, net->d_grad_after_token, net->model_dim));

    // Gradient for final projection
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->model_dim, net->num_tokens, net->batch_size,
        &alpha, net->d_grad_after_token, net->model_dim,
        net->d_after_token, net->model_dim,
        &beta, net->d_final_proj_grad, net->model_dim));

    // Backward through token mixing (second layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->mlp_dim, net->batch_size * net->num_tokens, net->num_tokens,
        &alpha, net->d_token_fc2, net->num_tokens,
        net->d_grad_after_token, net->num_tokens,
        &beta, net->d_grad_token_mid, net->mlp_dim));

    // Gradient for token_fc2
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->mlp_dim, net->num_tokens, net->batch_size * net->num_tokens,
        &alpha, net->d_token_mid, net->mlp_dim,
        net->d_grad_after_token, net->num_tokens,
        &beta, net->d_token_fc2_grad, net->mlp_dim));

    // Backward through Swish in token mixing
    int size = net->batch_size * net->num_tokens * net->mlp_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_grad_token_mid,
        net->d_token_mid,
        net->d_grad_token_mid,
        size);

    // Backward through token mixing (first layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->num_tokens, net->batch_size * net->num_tokens, net->mlp_dim,
        &alpha, net->d_token_fc1, net->mlp_dim,
        net->d_grad_token_mid, net->mlp_dim,
        &beta, net->d_grad_after_channel, net->num_tokens));

    // Gradient for token_fc1
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->num_tokens, net->mlp_dim, net->batch_size * net->num_tokens,
        &alpha, net->d_after_channel, net->num_tokens,
        net->d_grad_token_mid, net->mlp_dim,
        &beta, net->d_token_fc1_grad, net->num_tokens));

    // Backward through channel mixing (second layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->mlp_dim, net->batch_size * net->num_tokens, net->model_dim,
        &alpha, net->d_channel_fc2, net->model_dim,
        net->d_grad_after_channel, net->model_dim,
        &beta, net->d_grad_channel_mid, net->mlp_dim));

    // Gradient for channel_fc2
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->mlp_dim, net->model_dim, net->batch_size * net->num_tokens,
        &alpha, net->d_channel_mid, net->mlp_dim,
        net->d_grad_after_channel, net->model_dim,
        &beta, net->d_channel_fc2_grad, net->mlp_dim));

    // Backward through Swish in channel mixing
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_grad_channel_mid,
        net->d_channel_mid,
        net->d_grad_channel_mid,
        size);

    // Backward through channel mixing (first layer)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->model_dim, net->batch_size * net->num_tokens, net->mlp_dim,
        &alpha, net->d_channel_fc1, net->mlp_dim,
        net->d_grad_channel_mid, net->mlp_dim,
        &beta, net->d_grad_embedded, net->model_dim));

    // Gradient for channel_fc1
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->model_dim, net->mlp_dim, net->batch_size * net->num_tokens,
        &alpha, net->d_embedded, net->model_dim,
        net->d_grad_channel_mid, net->mlp_dim,
        &beta, net->d_channel_fc1_grad, net->model_dim));

    // Gradient for token embedding
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->num_tokens, net->model_dim, net->batch_size * net->num_tokens,
        &alpha, net->d_X_device, net->num_tokens,
        net->d_grad_embedded, net->model_dim,
        &beta, net->d_token_embed_grad, net->num_tokens));
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
    int token_embed_blocks = (token_embed_size + block_size - 1) / block_size;
    adamw_update_kernel<<<token_embed_blocks, block_size>>>(
        net->d_token_embed,
        net->d_token_embed_grad,
        net->d_token_embed_m,
        net->d_token_embed_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        token_embed_size,
        net->batch_size
    );
    
    // Update channel mixing weights (fc1)
    int channel_fc1_size = net->model_dim * net->mlp_dim;
    int channel_fc1_blocks = (channel_fc1_size + block_size - 1) / block_size;
    adamw_update_kernel<<<channel_fc1_blocks, block_size>>>(
        net->d_channel_fc1,
        net->d_channel_fc1_grad,
        net->d_channel_fc1_m,
        net->d_channel_fc1_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        channel_fc1_size,
        net->batch_size
    );
    
    // Update channel mixing weights (fc2)
    int channel_fc2_size = net->mlp_dim * net->model_dim;
    int channel_fc2_blocks = (channel_fc2_size + block_size - 1) / block_size;
    adamw_update_kernel<<<channel_fc2_blocks, block_size>>>(
        net->d_channel_fc2,
        net->d_channel_fc2_grad,
        net->d_channel_fc2_m,
        net->d_channel_fc2_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        channel_fc2_size,
        net->batch_size
    );
    
    // Update token mixing weights (fc1)
    int token_fc1_size = net->num_tokens * net->mlp_dim;
    int token_fc1_blocks = (token_fc1_size + block_size - 1) / block_size;
    adamw_update_kernel<<<token_fc1_blocks, block_size>>>(
        net->d_token_fc1,
        net->d_token_fc1_grad,
        net->d_token_fc1_m,
        net->d_token_fc1_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        token_fc1_size,
        net->batch_size
    );
    
    // Update token mixing weights (fc2)
    int token_fc2_size = net->mlp_dim * net->num_tokens;
    int token_fc2_blocks = (token_fc2_size + block_size - 1) / block_size;
    adamw_update_kernel<<<token_fc2_blocks, block_size>>>(
        net->d_token_fc2,
        net->d_token_fc2_grad,
        net->d_token_fc2_m,
        net->d_token_fc2_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        token_fc2_size,
        net->batch_size
    );
    
    // Update final projection weights
    int final_proj_size = net->model_dim * net->num_tokens;
    int final_proj_blocks = (final_proj_size + block_size - 1) / block_size;
    adamw_update_kernel<<<final_proj_blocks, block_size>>>(
        net->d_final_proj,
        net->d_final_proj_grad,
        net->d_final_proj_m,
        net->d_final_proj_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        final_proj_size,
        net->batch_size
    );
}

// Save model weights to binary file
void save_model(Net* net, const char* filename) {
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(net->h_token_embed, net->d_token_embed,
                         net->num_tokens * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_channel_fc1, net->d_channel_fc1,
                         net->model_dim * net->mlp_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_channel_fc2, net->d_channel_fc2,
                         net->mlp_dim * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_token_fc1, net->d_token_fc1,
                         net->num_tokens * net->mlp_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_token_fc2, net->d_token_fc2,
                         net->mlp_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_final_proj, net->d_final_proj,
                         net->model_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write model architecture
    fwrite(&net->num_tokens, sizeof(int), 1, file);
    fwrite(&net->model_dim, sizeof(int), 1, file);
    fwrite(&net->mlp_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    
    // Write optimizer state
    fwrite(&net->t, sizeof(int), 1, file);
    fwrite(&net->beta1, sizeof(float), 1, file);
    fwrite(&net->beta2, sizeof(float), 1, file);
    fwrite(&net->epsilon, sizeof(float), 1, file);
    fwrite(&net->weight_decay, sizeof(float), 1, file);
    
    // Write model weights
    fwrite(net->h_token_embed, sizeof(float), net->num_tokens * net->model_dim, file);
    fwrite(net->h_channel_fc1, sizeof(float), net->model_dim * net->mlp_dim, file);
    fwrite(net->h_channel_fc2, sizeof(float), net->mlp_dim * net->model_dim, file);
    fwrite(net->h_token_fc1, sizeof(float), net->num_tokens * net->mlp_dim, file);
    fwrite(net->h_token_fc2, sizeof(float), net->mlp_dim * net->num_tokens, file);
    fwrite(net->h_final_proj, sizeof(float), net->model_dim * net->num_tokens, file);
    
    // Allocate temporary buffer for Adam states
    float* h_buffer = (float*)malloc(net->model_dim * net->mlp_dim * sizeof(float));  // Largest size needed

    // Save Adam states for token_embed
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_token_embed_m,
                         net->num_tokens * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->num_tokens * net->model_dim, file);
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_token_embed_v,
                         net->num_tokens * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->num_tokens * net->model_dim, file);

    // Save Adam states for channel_fc1
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_channel_fc1_m,
                         net->model_dim * net->mlp_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->model_dim * net->mlp_dim, file);
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_channel_fc1_v,
                         net->model_dim * net->mlp_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->model_dim * net->mlp_dim, file);

    // Save Adam states for channel_fc2
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_channel_fc2_m,
                         net->mlp_dim * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->mlp_dim * net->model_dim, file);
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_channel_fc2_v,
                         net->mlp_dim * net->model_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->mlp_dim * net->model_dim, file);

    // Save Adam states for token_fc1
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_token_fc1_m,
                         net->num_tokens * net->mlp_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->num_tokens * net->mlp_dim, file);
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_token_fc1_v,
                         net->num_tokens * net->mlp_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->num_tokens * net->mlp_dim, file);

    // Save Adam states for token_fc2
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_token_fc2_m,
                         net->mlp_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->mlp_dim * net->num_tokens, file);
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_token_fc2_v,
                         net->mlp_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->mlp_dim * net->num_tokens, file);

    // Save Adam states for final_proj
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_final_proj_m,
                         net->model_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->model_dim * net->num_tokens, file);
    CHECK_CUDA(cudaMemcpy(h_buffer, net->d_final_proj_v,
                         net->model_dim * net->num_tokens * sizeof(float),
                         cudaMemcpyDeviceToHost));
    fwrite(h_buffer, sizeof(float), net->model_dim * net->num_tokens, file);
    
    free(h_buffer);
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
    
    // Read model architecture
    int num_tokens, model_dim, mlp_dim, batch_size;
    fread(&num_tokens, sizeof(int), 1, file);
    fread(&model_dim, sizeof(int), 1, file);
    fread(&mlp_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize network with loaded dimensions
    Net* net = init_net(num_tokens, model_dim, mlp_dim, batch_size);
    
    // Read optimizer state
    fread(&net->t, sizeof(int), 1, file);
    fread(&net->beta1, sizeof(float), 1, file);
    fread(&net->beta2, sizeof(float), 1, file);
    fread(&net->epsilon, sizeof(float), 1, file);
    fread(&net->weight_decay, sizeof(float), 1, file);
    
    // Read weights
    fread(net->h_token_embed, sizeof(float), num_tokens * model_dim, file);
    fread(net->h_channel_fc1, sizeof(float), model_dim * mlp_dim, file);
    fread(net->h_channel_fc2, sizeof(float), mlp_dim * model_dim, file);
    fread(net->h_token_fc1, sizeof(float), num_tokens * mlp_dim, file);
    fread(net->h_token_fc2, sizeof(float), mlp_dim * num_tokens, file);
    fread(net->h_final_proj, sizeof(float), model_dim * num_tokens, file);
    
    // Allocate temporary buffer for Adam states
    float* h_buffer = (float*)malloc(model_dim * mlp_dim * sizeof(float));  // Largest size needed

    // Load Adam states for token_embed
    fread(h_buffer, sizeof(float), num_tokens * model_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_token_embed_m, h_buffer,
                         num_tokens * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    fread(h_buffer, sizeof(float), num_tokens * model_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_token_embed_v, h_buffer,
                         num_tokens * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Load Adam states for channel_fc1
    fread(h_buffer, sizeof(float), model_dim * mlp_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc1_m, h_buffer,
                         model_dim * mlp_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    fread(h_buffer, sizeof(float), model_dim * mlp_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc1_v, h_buffer,
                         model_dim * mlp_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Load Adam states for channel_fc2
    fread(h_buffer, sizeof(float), mlp_dim * model_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc2_m, h_buffer,
                         mlp_dim * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    fread(h_buffer, sizeof(float), mlp_dim * model_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc2_v, h_buffer,
                         mlp_dim * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Load Adam states for token_fc1
    fread(h_buffer, sizeof(float), num_tokens * mlp_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_token_fc1_m, h_buffer,
                         num_tokens * mlp_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    fread(h_buffer, sizeof(float), num_tokens * mlp_dim, file);
    CHECK_CUDA(cudaMemcpy(net->d_token_fc1_v, h_buffer,
                         num_tokens * mlp_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Load Adam states for token_fc2
    fread(h_buffer, sizeof(float), mlp_dim * num_tokens, file);
    CHECK_CUDA(cudaMemcpy(net->d_token_fc2_m, h_buffer,
                         mlp_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    fread(h_buffer, sizeof(float), mlp_dim * num_tokens, file);
    CHECK_CUDA(cudaMemcpy(net->d_token_fc2_v, h_buffer,
                         mlp_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Load Adam states for final_proj
    fread(h_buffer, sizeof(float), model_dim * num_tokens, file);
    CHECK_CUDA(cudaMemcpy(net->d_final_proj_m, h_buffer,
                         model_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    fread(h_buffer, sizeof(float), model_dim * num_tokens, file);
    CHECK_CUDA(cudaMemcpy(net->d_final_proj_v, h_buffer,
                         model_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_token_embed, net->h_token_embed,
                         num_tokens * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc1, net->h_channel_fc1,
                         model_dim * mlp_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_channel_fc2, net->h_channel_fc2,
                         mlp_dim * model_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_fc1, net->h_token_fc1,
                         num_tokens * mlp_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_token_fc2, net->h_token_fc2,
                         mlp_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_final_proj, net->h_final_proj,
                         model_dim * num_tokens * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    free(h_buffer);
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif