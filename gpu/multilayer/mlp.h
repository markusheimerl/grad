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

// The network now has three weight sets:
//  • fc1: first hidden layer (hidden_dim x input_dim)
//  • fc2: second hidden layer (hidden_dim x hidden_dim)
//  • fc3: final output layer (output_dim x hidden_dim)
// Also, we add extra helper arrays for the second hidden layer and for the residual connection.
typedef struct {
    // Device pointers for weights and gradients
    float* d_fc1_weight;      // [hidden_dim x input_dim]
    float* d_fc2_weight;      // [hidden_dim x hidden_dim] (second hidden layer)
    float* d_fc3_weight;      // [output_dim x hidden_dim] (final output layer)
    float* d_fc1_weight_grad; // [hidden_dim x input_dim]
    float* d_fc2_weight_grad; // [hidden_dim x hidden_dim]
    float* d_fc3_weight_grad; // [output_dim x hidden_dim]
    
    // Host copies (for saving and loading)
    float* h_fc1_weight;
    float* h_fc2_weight;
    float* h_fc3_weight;
    float* h_error;
    
    // Device pointers for Adam parameters
    float* d_fc1_m;
    float* d_fc1_v;
    float* d_fc2_m;
    float* d_fc2_v;
    float* d_fc3_m;
    float* d_fc3_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays
    float* d_layer1_output;    // First hidden layer’s activated output [batch_size x hidden_dim]
    float* d_pre_activation;   // Pre-activation values for layer1 [batch_size x hidden_dim]
    float* d_layer2_output;    // Second hidden layer’s output (after activation and residual add) [batch_size x hidden_dim]
    float* d_pre_activation2;  // Pre-activation values for layer2 [batch_size x hidden_dim]
    float* d_predictions;      // Final output predictions [batch_size x output_dim]
    float* d_error;            // (predictions - y) [batch_size x output_dim]
    float* d_error_hidden;     // Backpropagated error for fc1 [batch_size x hidden_dim]
    float* d_error_hidden2;    // Backpropagated error for fc2 (the non‐residual branch) [batch_size x hidden_dim]
    float* d_error_residual;   // Temporary buffer to store the residual branch gradient [batch_size x hidden_dim]
    float* d_X;                // Input data [batch_size x input_dim]
    float* d_y;                // Labels [batch_size x output_dim]
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel(float* output, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = pre_activation[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel(float* error_hidden, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = pre_activation[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        error_hidden[idx] *= sigmoid + x * sigmoid * (1.0f - sigmoid);
    }
}

// Custom kernel for calculating error (predictions-y)
__global__ void calc_error_kernel(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// CUDA kernel for AdamW update (unchanged)
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

// New kernel to add the residual connection (elementwise add)
__global__ void add_residual_kernel(float* target, const float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target[idx] += residual[idx];
    }
}

// Initialize the network with the new three‐layer (2 hidden + output) architecture
Net* init_net(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Initialize Adam parameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Allocate host memory for weights and error
    net->h_fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->h_fc2_weight = (float*)malloc(hidden_dim * hidden_dim * sizeof(float));
    net->h_fc3_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    net->h_error = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize weights on host
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    float scale3 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        net->h_fc1_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale1;
    }
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        net->h_fc2_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale2;
    }
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->h_fc3_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale3;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc3_weight, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight_grad, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight_grad, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc3_weight_grad, output_dim * hidden_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&net->d_fc1_m, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_v, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_m, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_v, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc3_m, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc3_v, output_dim * hidden_dim * sizeof(float)));
    
    // Allocate device memory for helper arrays
    CHECK_CUDA(cudaMalloc(&net->d_layer1_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_pre_activation, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_layer2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_pre_activation2, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error_hidden2, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error_residual, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_y, batch_size * output_dim * sizeof(float)));
    
    // Copy host weights to device
    CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, net->h_fc1_weight, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, net->h_fc2_weight, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc3_weight, net->h_fc3_weight, output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters (set to zero)
    CHECK_CUDA(cudaMemset(net->d_fc1_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc1_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_m, 0, hidden_dim * hidden_dim * sizeof(float)));
   	CHECK_CUDA(cudaMemset(net->d_fc2_v, 0, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc3_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc3_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    // Free device memory
    cudaFree(net->d_fc1_weight);
    cudaFree(net->d_fc2_weight);
    cudaFree(net->d_fc3_weight);
    cudaFree(net->d_fc1_weight_grad);
    cudaFree(net->d_fc2_weight_grad);
    cudaFree(net->d_fc3_weight_grad);
    cudaFree(net->d_fc1_m);
    cudaFree(net->d_fc1_v);
    cudaFree(net->d_fc2_m);
    cudaFree(net->d_fc2_v);
    cudaFree(net->d_fc3_m);
    cudaFree(net->d_fc3_v);
    cudaFree(net->d_layer1_output);
    cudaFree(net->d_pre_activation);
    cudaFree(net->d_layer2_output);
    cudaFree(net->d_pre_activation2);
    cudaFree(net->d_predictions);
    cudaFree(net->d_error);
    cudaFree(net->d_error_hidden);
    cudaFree(net->d_error_hidden2);
    cudaFree(net->d_error_residual);
    cudaFree(net->d_X);
    cudaFree(net->d_y);
    
    // Free host memory
    free(net->h_fc1_weight);
    free(net->h_fc2_weight);
    free(net->h_fc3_weight);
    free(net->h_error);
    
    // Destroy cuBLAS handle
    cublasDestroy(net->cublas_handle);
    
    free(net);
}

// Forward pass with residual connection
void forward_pass(Net* net, float* X) {
    CHECK_CUDA(cudaMemcpy(net->d_X, X, net->batch_size * net->input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // First hidden layer (fc1)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->hidden_dim,    // rows of fc1 weight
                            net->batch_size,    // number of samples
                            net->input_dim,     // inner dimension
                            &alpha,
                            net->d_fc1_weight,  // fc1 weight matrix [hidden_dim x input_dim]
                            net->hidden_dim,
                            net->d_X,           // input [input_dim x batch_size]
                            net->input_dim,
                            &beta,
                            net->d_layer1_output, // temporary storage [hidden_dim x batch_size]
                            net->hidden_dim));
    
    // Save pre-activation values for fc1 and then apply Swish
    CHECK_CUDA(cudaMemcpy(net->d_pre_activation, net->d_layer1_output,
                         net->batch_size * net->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    int block_size = 256;
    int num_blocks = (net->batch_size * net->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(net->d_layer1_output,
                                                     net->d_pre_activation,
                                                     net->batch_size * net->hidden_dim);
    
    // Second hidden layer (fc2)
    // Compute pre-activation from fc2 weight and first hidden layer output
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->hidden_dim,    // rows [hidden_dim]
                            net->batch_size,    // columns [batch_size]
                            net->hidden_dim,    // inner dimension
                            &alpha,
                            net->d_fc2_weight,  // fc2 weight [hidden_dim x hidden_dim]
                            net->hidden_dim,
                            net->d_layer1_output, // input is output from fc1 [hidden_dim x batch_size]
                            net->hidden_dim,
                            &beta,
                            net->d_layer2_output, // pre-activation for layer2 [hidden_dim x batch_size]
                            net->hidden_dim));
    
    // Save pre-activation values for fc2 and apply Swish activation
    CHECK_CUDA(cudaMemcpy(net->d_pre_activation2, net->d_layer2_output,
                         net->batch_size * net->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    num_blocks = (net->batch_size * net->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(net->d_layer2_output,
                                                     net->d_pre_activation2,
                                                     net->batch_size * net->hidden_dim);
    
    // Residual connection: add fc1’s activated output to fc2’s output
    add_residual_kernel<<<num_blocks, block_size>>>(net->d_layer2_output,
                                                    net->d_layer1_output,
                                                    net->batch_size * net->hidden_dim);
    
    // Final output layer (fc3)
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->output_dim,     // rows [output_dim]
                            net->batch_size,
                            net->hidden_dim,
                            &alpha,
                            net->d_fc3_weight,   // fc3 weight [output_dim x hidden_dim]
                            net->output_dim,
                            net->d_layer2_output, // input from second hidden layer [hidden_dim x batch_size]
                            net->hidden_dim,
                            &beta,
                            net->d_predictions,  // predictions [output_dim x batch_size]
                            net->output_dim));
}

// Calculate loss (mean squared error)
float calculate_loss(Net* net, float* y) {
    CHECK_CUDA(cudaMemcpy(net->d_y, y, net->batch_size * net->output_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    int size = net->batch_size * net->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    calc_error_kernel<<<num_blocks, block_size>>>(net->d_error,
                                                  net->d_predictions,
                                                  net->d_y,
                                                  size);
    
    CHECK_CUDA(cudaMemcpy(net->h_error, net->d_error, size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss += net->h_error[i] * net->h_error[i];
    }
    
    return loss / size;
}

// Zero gradients for all weight matrices
void zero_gradients(Net* net) {
    CHECK_CUDA(cudaMemset(net->d_fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_weight_grad, 0, net->hidden_dim * net->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc3_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float)));
}

// Backward pass with residual connection and two hidden layers
void backward_pass(Net* net, float* X) {
    CHECK_CUDA(cudaMemcpy(net->d_X, X, net->batch_size * net->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // --- Final layer (fc3) ---
    // Compute gradient for fc3: d_fc3_weight_grad = error * (layer2_output)^T
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             net->output_dim,
                             net->hidden_dim,
                             net->batch_size,
                             &alpha,
                             net->d_error,
                             net->output_dim,
                             net->d_layer2_output,
                             net->hidden_dim,
                             &beta,
                             net->d_fc3_weight_grad,
                             net->output_dim));
    
    // Backpropagate error from output to second hidden layer:
    // d_error_hidden2 = (fc3_weight)^T * error
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             net->hidden_dim,
                             net->batch_size,
                             net->output_dim,
                             &alpha,
                             net->d_fc3_weight,
                             net->output_dim,
                             net->d_error,
                             net->output_dim,
                             &beta,
                             net->d_error_hidden2,
                             net->hidden_dim));
    
    // Save a copy for the residual branch
    CHECK_CUDA(cudaMemcpy(net->d_error_residual, net->d_error_hidden2,
                          net->batch_size * net->hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Apply Swish derivative on the second hidden layer’s non‐residual branch
    int block_size = 256;
    int num_blocks = (net->batch_size * net->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(net->d_error_hidden2,
                                                      net->d_pre_activation2,
                                                      net->batch_size * net->hidden_dim);
    
    // Gradient for fc2: d_fc2_weight_grad = d_error_hidden2 * (layer1_output)^T
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             net->hidden_dim,
                             net->hidden_dim,
                             net->batch_size,
                             &alpha,
                             net->d_error_hidden2,
                             net->hidden_dim,
                             net->d_layer1_output,
                             net->hidden_dim,
                             &beta,
                             net->d_fc2_weight_grad,
                             net->hidden_dim));
    
    // Propagate error from fc2 branch back to first hidden layer:
    // d_error_hidden = (fc2_weight)^T * d_error_hidden2
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             net->hidden_dim,
                             net->batch_size,
                             net->hidden_dim,
                             &alpha,
                             net->d_fc2_weight,
                             net->hidden_dim,
                             net->d_error_hidden2,
                             net->hidden_dim,
                             &beta,
                             net->d_error_hidden,
                             net->hidden_dim));
    
    // Add the contribution from the residual branch (the skip connection)
    num_blocks = (net->batch_size * net->hidden_dim + block_size - 1) / block_size;
    add_residual_kernel<<<num_blocks, block_size>>>(net->d_error_hidden,
                                                    net->d_error_residual,
                                                    net->batch_size * net->hidden_dim);
    
    // Gradient for fc1: d_fc1_weight_grad = d_error_hidden * (X)^T
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             net->hidden_dim,
                             net->input_dim,
                             net->batch_size,
                             &alpha,
                             net->d_error_hidden,
                             net->hidden_dim,
                             net->d_X,
                             net->input_dim,
                             &beta,
                             net->d_fc1_weight_grad,
                             net->hidden_dim));
}

// Update weights using AdamW for all three weight sets
void update_weights(Net* net, float learning_rate) {
    net->t++;
    
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update fc1 weights
    int fc1_size = net->hidden_dim * net->input_dim;
    int fc1_blocks = (fc1_size + block_size - 1) / block_size;
    adamw_update_kernel<<<fc1_blocks, block_size>>>(
        net->d_fc1_weight,
        net->d_fc1_weight_grad,
        net->d_fc1_m,
        net->d_fc1_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        fc1_size,
        net->batch_size
    );
    
    // Update fc2 weights
    int fc2_size = net->hidden_dim * net->hidden_dim;
    int fc2_blocks = (fc2_size + block_size - 1) / block_size;
    adamw_update_kernel<<<fc2_blocks, block_size>>>(
        net->d_fc2_weight,
        net->d_fc2_weight_grad,
        net->d_fc2_m,
        net->d_fc2_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        fc2_size,
        net->batch_size
    );
    
    // Update fc3 weights
    int fc3_size = net->output_dim * net->hidden_dim;
    int fc3_blocks = (fc3_size + block_size - 1) / block_size;
    adamw_update_kernel<<<fc3_blocks, block_size>>>(
        net->d_fc3_weight,
        net->d_fc3_weight_grad,
        net->d_fc3_m,
        net->d_fc3_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        fc3_size,
        net->batch_size
    );
}

// Save model weights to binary file (now saves fc1, fc2, and fc3 weights)
void save_model(Net* net, const char* filename) {
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(net->h_fc1_weight, net->d_fc1_weight,
                         net->hidden_dim * net->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_fc2_weight, net->d_fc2_weight,
                         net->hidden_dim * net->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_fc3_weight, net->d_fc3_weight,
                         net->output_dim * net->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(net->h_fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->h_fc2_weight, sizeof(float), net->hidden_dim * net->hidden_dim, file);
    fwrite(net->h_fc3_weight, sizeof(float), net->output_dim * net->hidden_dim, file);
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
    
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    fread(net->h_fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->h_fc2_weight, sizeof(float), hidden_dim * hidden_dim, file);
    fread(net->h_fc3_weight, sizeof(float), output_dim * hidden_dim, file);
    fread(&net->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, net->h_fc1_weight,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, net->h_fc2_weight,
                         hidden_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc3_weight, net->h_fc3_weight,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif