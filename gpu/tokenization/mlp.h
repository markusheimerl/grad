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
    float* d_fc1_weight;     // hidden_dim x input_dim
    float* d_fc2_weight;     // output_dim x hidden_dim
    float* d_fc1_weight_grad; // hidden_dim x input_dim
    float* d_fc2_weight_grad; // output_dim x hidden_dim
    
    // Host copies of weights
    float* h_fc1_weight;
    float* h_fc2_weight;
    
    // Device pointers for Adam parameters
    float* d_fc1_m;  // First moment for fc1
    float* d_fc1_v;  // Second moment for fc1
    float* d_fc2_m;  // First moment for fc2
    float* d_fc2_v;  // Second moment for fc2
    float beta1;   // Exponential decay rate for first moment
    float beta2;   // Exponential decay rate for second moment
    float epsilon; // Small constant for numerical stability
    int t;         // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Device pointers for helper arrays
    float* d_layer1_output;   // batch_size x hidden_dim
    float* d_predictions;     // batch_size x output_dim
    float* d_error;          // batch_size x output_dim
    float* d_pre_activation; // batch_size x hidden_dim
    float* d_error_hidden;   // batch_size x hidden_dim
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

// Initialize the network with configurable dimensions
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
    
    // Allocate host memory for weights
    net->h_fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->h_fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Initialize weights on host
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        net->h_fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->h_fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight_grad, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight_grad, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&net->d_fc1_m, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_v, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_m, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_v, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&net->d_layer1_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_pre_activation, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error_hidden, batch_size * hidden_dim * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, net->h_fc1_weight, 
                         hidden_dim * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, net->h_fc2_weight, 
                         output_dim * hidden_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(net->d_fc1_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc1_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    // Free device memory
    cudaFree(net->d_fc1_weight);
    cudaFree(net->d_fc2_weight);
    cudaFree(net->d_fc1_weight_grad);
    cudaFree(net->d_fc2_weight_grad);
    cudaFree(net->d_fc1_m);
    cudaFree(net->d_fc1_v);
    cudaFree(net->d_fc2_m);
    cudaFree(net->d_fc2_v);
    cudaFree(net->d_layer1_output);
    cudaFree(net->d_predictions);
    cudaFree(net->d_error);
    cudaFree(net->d_pre_activation);
    cudaFree(net->d_error_hidden);
    
    // Free host memory
    free(net->h_fc1_weight);
    free(net->h_fc2_weight);
    
    // Destroy cuBLAS handle
    cublasDestroy(net->cublas_handle);
    
    free(net);
}

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

// Forward pass
void forward_pass(Net* net, float* X) {
    // Copy input to device if not already there
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // First layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->hidden_dim,    // n
                            net->batch_size,    // m
                            net->input_dim,     // k
                            &alpha,
                            net->d_fc1_weight,  // A
                            net->hidden_dim,    // lda
                            d_X,                // B
                            net->input_dim,     // ldb
                            &beta,
                            net->d_layer1_output, // C
                            net->hidden_dim));    // ldc

    // Store pre-activation values
    CHECK_CUDA(cudaMemcpy(net->d_pre_activation, net->d_layer1_output,
                         net->batch_size * net->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Apply Swish activation
    int block_size = 256;
    int num_blocks = (net->batch_size * net->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        net->d_layer1_output,
        net->d_pre_activation,
        net->batch_size * net->hidden_dim
    );

    // Second layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            net->output_dim,     // n
                            net->batch_size,     // m
                            net->hidden_dim,     // k
                            &alpha,
                            net->d_fc2_weight,   // A
                            net->output_dim,     // lda
                            net->d_layer1_output,// B
                            net->hidden_dim,     // ldb
                            &beta,
                            net->d_predictions,  // C
                            net->output_dim));   // ldc

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

// Calculate loss
float calculate_loss(Net* net, float* y) {
    float* d_y;
    CHECK_CUDA(cudaMalloc(&d_y, net->batch_size * net->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_y, y, net->batch_size * net->output_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Calculate error (predictions - y)
    int size = net->batch_size * net->output_dim;
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
    CHECK_CUDA(cudaMemset(net->d_fc1_weight_grad, 0, 
                         net->hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_weight_grad, 0, 
                         net->output_dim * net->hidden_dim * sizeof(float)));
}

// Backward pass
void backward_pass(Net* net, float* X) {
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Gradient of second layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->output_dim,     // n
                            net->hidden_dim,     // m
                            net->batch_size,     // k
                            &alpha,
                            net->d_error,        // A
                            net->output_dim,     // lda
                            net->d_layer1_output,// B
                            net->hidden_dim,     // ldb
                            &beta,
                            net->d_fc2_weight_grad, // C
                            net->output_dim));   // ldc

    // Backpropagate error through second layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            net->hidden_dim,     // n
                            net->batch_size,     // m
                            net->output_dim,     // k
                            &alpha,
                            net->d_fc2_weight,   // A
                            net->output_dim,     // lda
                            net->d_error,        // B
                            net->output_dim,     // ldb
                            &beta,
                            net->d_error_hidden, // C
                            net->hidden_dim));   // ldc

    // Apply Swish derivative
    int block_size = 256;
    int num_blocks = (net->batch_size * net->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(
        net->d_error_hidden,
        net->d_pre_activation,
        net->batch_size * net->hidden_dim
    );

    // Gradient of first layer
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            net->hidden_dim,     // n
                            net->input_dim,      // m
                            net->batch_size,     // k
                            &alpha,
                            net->d_error_hidden, // A
                            net->hidden_dim,     // lda
                            d_X,                 // B
                            net->input_dim,      // ldb
                            &beta,
                            net->d_fc1_weight_grad, // C
                            net->hidden_dim));   // ldc

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
    int fc2_size = net->output_dim * net->hidden_dim;
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
}

// Save model weights to binary file
void save_model(Net* net, const char* filename) {
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(net->h_fc1_weight, net->d_fc1_weight,
                         net->hidden_dim * net->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_fc2_weight, net->d_fc2_weight,
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
    fwrite(net->h_fc2_weight, sizeof(float), net->output_dim * net->hidden_dim, file);
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
    fread(net->h_fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    fread(&net->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, net->h_fc1_weight,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, net->h_fc2_weight,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif