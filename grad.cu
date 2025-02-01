#include "data.cuh"
#include <time.h>
#include <curand_kernel.h>

// Model configuration and structure definitions
typedef struct {
    int batch_size;
    int sequence_length;
    int n_inputs;
    int n_outputs;
    int n_conv1_filters;
    int n_conv2_filters;
    int dense_size;
    int conv_kernel_size;
} ModelConfig;

typedef struct {
    // Dimensions
    int batch_size;
    int sequence_length;
    int n_inputs;
    int n_outputs;
    int n_conv1_filters;
    int n_conv2_filters;
    int dense_size;
    int conv_kernel_size;
    
    // Layer weights and biases
    float *conv1_weights;  // [n_conv1_filters, kernel_size, n_inputs]
    float *conv1_bias;     // [n_conv1_filters]
    float *conv2_weights;  // [n_conv2_filters, kernel_size, n_conv1_filters]
    float *conv2_bias;     // [n_conv2_filters]
    float *dense1_weights; // [n_conv2_filters, dense_size]
    float *dense1_bias;    // [dense_size]
    float *dense2_weights; // [dense_size, n_outputs]
    float *dense2_bias;    // [n_outputs]
    
    // Intermediate activations (for backprop)
    float *conv1_output;   // After first conv + ReLU
    float *conv2_output;   // After second conv + ReLU
    float *pool_output;    // After global average pooling
    float *dense1_output;  // After first dense + ReLU
} Model;

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Initialize weights with Xavier/Glorot initialization
__global__ void init_weights_kernel(float* weights, int fan_in, int fan_out, 
                                  int total_elements, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        float limit = sqrt(6.0f / (fan_in + fan_out));
        weights[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * limit;
    }
}

// Create and initialize the model
Model* create_model(ModelConfig config) {
    Model* model = (Model*)malloc(sizeof(Model));
    
    // Copy configuration
    model->batch_size = config.batch_size;
    model->sequence_length = config.sequence_length;
    model->n_inputs = config.n_inputs;
    model->n_outputs = config.n_outputs;
    model->n_conv1_filters = config.n_conv1_filters;
    model->n_conv2_filters = config.n_conv2_filters;
    model->dense_size = config.dense_size;
    model->conv_kernel_size = config.conv_kernel_size;
    
    // Calculate sizes for weight matrices
    int conv1_weights_size = config.n_conv1_filters * config.conv_kernel_size * config.n_inputs;
    int conv2_weights_size = config.n_conv2_filters * config.conv_kernel_size * config.n_conv1_filters;
    int dense1_weights_size = config.n_conv2_filters * config.dense_size;
    int dense2_weights_size = config.dense_size * config.n_outputs;
    
    // Allocate memory for weights and biases
    CUDA_CHECK(cudaMalloc(&model->conv1_weights, conv1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->conv1_bias, config.n_conv1_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->conv2_weights, conv2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->conv2_bias, config.n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->dense1_weights, dense1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->dense1_bias, config.dense_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->dense2_weights, dense2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->dense2_bias, config.n_outputs * sizeof(float)));
    
    // Allocate memory for intermediate activations
    CUDA_CHECK(cudaMalloc(&model->conv1_output, 
        config.batch_size * (config.sequence_length - config.conv_kernel_size + 1) * 
        config.n_conv1_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->conv2_output,
        config.batch_size * (config.sequence_length - 2*config.conv_kernel_size + 2) * 
        config.n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->pool_output,
        config.batch_size * config.n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->dense1_output,
        config.batch_size * config.dense_size * sizeof(float)));
    
    // Initialize weights
    int threads_per_block = 256;
    
    // Conv1 weights
    int blocks = (conv1_weights_size + threads_per_block - 1) / threads_per_block;
    init_weights_kernel<<<blocks, threads_per_block>>>(
        model->conv1_weights, config.n_inputs, config.n_conv1_filters,
        conv1_weights_size, time(NULL));
    
    // Conv2 weights
    blocks = (conv2_weights_size + threads_per_block - 1) / threads_per_block;
    init_weights_kernel<<<blocks, threads_per_block>>>(
        model->conv2_weights, config.n_conv1_filters, config.n_conv2_filters,
        conv2_weights_size, time(NULL) + 1);
    
    // Dense1 weights
    blocks = (dense1_weights_size + threads_per_block - 1) / threads_per_block;
    init_weights_kernel<<<blocks, threads_per_block>>>(
        model->dense1_weights, config.n_conv2_filters, config.dense_size,
        dense1_weights_size, time(NULL) + 2);
    
    // Dense2 weights
    blocks = (dense2_weights_size + threads_per_block - 1) / threads_per_block;
    init_weights_kernel<<<blocks, threads_per_block>>>(
        model->dense2_weights, config.dense_size, config.n_outputs,
        dense2_weights_size, time(NULL) + 3);
    
    // Initialize biases to zero
    CUDA_CHECK(cudaMemset(model->conv1_bias, 0, config.n_conv1_filters * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->conv2_bias, 0, config.n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->dense1_bias, 0, config.dense_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->dense2_bias, 0, config.n_outputs * sizeof(float)));
    
    return model;
}

// Free model memory
void free_model(Model* model) {
    if (model) {
        CUDA_CHECK(cudaFree(model->conv1_weights));
        CUDA_CHECK(cudaFree(model->conv1_bias));
        CUDA_CHECK(cudaFree(model->conv2_weights));
        CUDA_CHECK(cudaFree(model->conv2_bias));
        CUDA_CHECK(cudaFree(model->dense1_weights));
        CUDA_CHECK(cudaFree(model->dense1_bias));
        CUDA_CHECK(cudaFree(model->dense2_weights));
        CUDA_CHECK(cudaFree(model->dense2_bias));
        
        CUDA_CHECK(cudaFree(model->conv1_output));
        CUDA_CHECK(cudaFree(model->conv2_output));
        CUDA_CHECK(cudaFree(model->pool_output));
        CUDA_CHECK(cudaFree(model->dense1_output));
        
        free(model);
    }
}

// ReLU activation kernel
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// 1D Convolution kernel
__global__ void conv1d_kernel(
    const float* input,      // [batch_size, sequence_length, in_channels]
    const float* weights,    // [out_channels, kernel_size, in_channels]
    const float* bias,       // [out_channels]
    float* output,          // [batch_size, sequence_length - kernel_size + 1, out_channels]
    int batch_size,
    int sequence_length,
    int in_channels,
    int out_channels,
    int kernel_size
) {
    int batch_idx = blockIdx.x;
    int out_seq_idx = blockIdx.y;
    int out_channel = threadIdx.x;
    
    if (batch_idx >= batch_size || out_seq_idx >= (sequence_length - kernel_size + 1) || 
        out_channel >= out_channels) return;
        
    float sum = bias[out_channel];
    
    // Compute convolution for this output position
    for (int k = 0; k < kernel_size; k++) {
        for (int in_c = 0; in_c < in_channels; in_c++) {
            int input_idx = batch_idx * sequence_length * in_channels + 
                          (out_seq_idx + k) * in_channels + in_c;
            int weight_idx = out_channel * kernel_size * in_channels + 
                           k * in_channels + in_c;
            sum += input[input_idx] * weights[weight_idx];
        }
    }
    
    int output_idx = batch_idx * (sequence_length - kernel_size + 1) * out_channels + 
                     out_seq_idx * out_channels + out_channel;
    output[output_idx] = sum;
}

// Global Average Pooling kernel
__global__ void global_avg_pooling_kernel(
    const float* input,     // [batch_size, sequence_length, channels]
    float* output,         // [batch_size, channels]
    int batch_size,
    int sequence_length,
    int channels
) {
    int batch_idx = blockIdx.x;
    int channel = threadIdx.x;
    
    if (batch_idx >= batch_size || channel >= channels) return;
    
    float sum = 0.0f;
    for (int i = 0; i < sequence_length; i++) {
        int idx = batch_idx * sequence_length * channels + i * channels + channel;
        sum += input[idx];
    }
    
    output[batch_idx * channels + channel] = sum / sequence_length;
}

// Dense layer kernel
__global__ void dense_kernel(
    const float* input,     // [batch_size, in_features]
    const float* weights,   // [in_features, out_features]
    const float* bias,      // [out_features]
    float* output,         // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_feature = threadIdx.x;
    
    if (batch_idx >= batch_size || out_feature >= out_features) return;
    
    float sum = bias[out_feature];
    for (int i = 0; i < in_features; i++) {
        sum += input[batch_idx * in_features + i] * 
               weights[i * out_features + out_feature];
    }
    
    output[batch_idx * out_features + out_feature] = sum;
}

// Forward pass function
void forward_pass(Model* model, float* input_data) {
    dim3 conv1_grid(model->batch_size, 
                   model->sequence_length - model->conv_kernel_size + 1);
    dim3 conv1_block(model->n_conv1_filters);
    
    // First convolution layer
    conv1d_kernel<<<conv1_grid, conv1_block>>>(
        input_data,
        model->conv1_weights,
        model->conv1_bias,
        model->conv1_output,
        model->batch_size,
        model->sequence_length,
        model->n_inputs,
        model->n_conv1_filters,
        model->conv_kernel_size
    );
    
    // ReLU after first conv
    int conv1_output_size = model->batch_size * 
                           (model->sequence_length - model->conv_kernel_size + 1) * 
                           model->n_conv1_filters;
    int threads_per_block = 256;
    int blocks = (conv1_output_size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(model->conv1_output, conv1_output_size);
    
    // Second convolution layer
    dim3 conv2_grid(model->batch_size, 
                   model->sequence_length - 2*model->conv_kernel_size + 2);
    dim3 conv2_block(model->n_conv2_filters);
    
    conv1d_kernel<<<conv2_grid, conv2_block>>>(
        model->conv1_output,
        model->conv2_weights,
        model->conv2_bias,
        model->conv2_output,
        model->batch_size,
        model->sequence_length - model->conv_kernel_size + 1,
        model->n_conv1_filters,
        model->n_conv2_filters,
        model->conv_kernel_size
    );
    
    // ReLU after second conv
    int conv2_output_size = model->batch_size * 
                           (model->sequence_length - 2*model->conv_kernel_size + 2) * 
                           model->n_conv2_filters;
    blocks = (conv2_output_size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(model->conv2_output, conv2_output_size);
    
    // Global Average Pooling
    dim3 pool_grid(model->batch_size);
    dim3 pool_block(model->n_conv2_filters);
    
    global_avg_pooling_kernel<<<pool_grid, pool_block>>>(
        model->conv2_output,
        model->pool_output,
        model->batch_size,
        model->sequence_length - 2*model->conv_kernel_size + 2,
        model->n_conv2_filters
    );
    
    // First dense layer
    dim3 dense1_grid(model->batch_size);
    dim3 dense1_block(model->dense_size);
    
    dense_kernel<<<dense1_grid, dense1_block>>>(
        model->pool_output,
        model->dense1_weights,
        model->dense1_bias,
        model->dense1_output,
        model->batch_size,
        model->n_conv2_filters,
        model->dense_size
    );
    
    // ReLU after first dense
    int dense1_output_size = model->batch_size * model->dense_size;
    blocks = (dense1_output_size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(model->dense1_output, dense1_output_size);
    
    // Final dense layer (output layer)
    dim3 dense2_grid(model->batch_size);
    dim3 dense2_block(model->n_outputs);
    
    dense_kernel<<<dense2_grid, dense2_block>>>(
        model->dense1_output,
        model->dense2_weights,
        model->dense2_bias,
        model->pool_output,  // Reuse pool_output as final output storage
        model->batch_size,
        model->dense_size,
        model->n_outputs
    );
}

// MSE Loss gradient kernel
__global__ void mse_gradient_kernel(
    const float* predictions,  // [batch_size, n_outputs]
    const float* targets,      // [batch_size, n_outputs]
    float* gradient,           // [batch_size, n_outputs]
    int batch_size,
    int n_outputs
) {
    int batch_idx = blockIdx.x;
    int output_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || output_idx >= n_outputs) return;
    
    int idx = batch_idx * n_outputs + output_idx;
    gradient[idx] = 2.0f * (predictions[idx] - targets[idx]) / batch_size;
}

// ReLU gradient kernel
__global__ void relu_gradient_kernel(
    const float* input,
    const float* grad_output,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

// Dense layer backward kernel (gradient w.r.t. input)
__global__ void dense_backward_input_kernel(
    const float* grad_output,  // [batch_size, out_features]
    const float* weights,      // [in_features, out_features]
    float* grad_input,         // [batch_size, in_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int in_feature = threadIdx.x;
    
    if (batch_idx >= batch_size || in_feature >= in_features) return;
    
    float sum = 0.0f;
    for (int i = 0; i < out_features; i++) {
        sum += grad_output[batch_idx * out_features + i] * 
               weights[in_feature * out_features + i];
    }
    
    grad_input[batch_idx * in_features + in_feature] = sum;
}

// Dense layer backward kernel (gradient w.r.t. weights)
__global__ void dense_backward_weights_kernel(
    const float* input,        // [batch_size, in_features]
    const float* grad_output,  // [batch_size, out_features]
    float* grad_weights,       // [in_features, out_features]
    float* grad_bias,          // [out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int in_feature = blockIdx.x;
    int out_feature = threadIdx.x;
    
    if (in_feature >= in_features || out_feature >= out_features) return;
    
    float weight_grad = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        weight_grad += input[b * in_features + in_feature] * 
                      grad_output[b * out_features + out_feature];
    }
    
    grad_weights[in_feature * out_features + out_feature] = weight_grad / batch_size;
    
    if (in_feature == 0) {
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_grad += grad_output[b * out_features + out_feature];
        }
        grad_bias[out_feature] = bias_grad / batch_size;
    }
}

// Global Average Pooling backward kernel
__global__ void global_avg_pooling_backward_kernel(
    const float* grad_output,  // [batch_size, channels]
    float* grad_input,         // [batch_size, sequence_length, channels]
    int batch_size,
    int sequence_length,
    int channels
) {
    int batch_idx = blockIdx.x;
    int channel = threadIdx.x;
    
    if (batch_idx >= batch_size || channel >= channels) return;
    
    float grad_value = grad_output[batch_idx * channels + channel] / sequence_length;
    
    for (int i = 0; i < sequence_length; i++) {
        grad_input[batch_idx * sequence_length * channels + i * channels + channel] = grad_value;
    }
}

// Conv1D backward kernel (gradient w.r.t. input)
__global__ void conv1d_backward_input_kernel(
    const float* grad_output,  // [batch_size, out_seq_len, out_channels]
    const float* weights,      // [out_channels, kernel_size, in_channels]
    float* grad_input,         // [batch_size, in_seq_len, in_channels]
    int batch_size,
    int in_seq_len,
    int out_seq_len,
    int in_channels,
    int out_channels,
    int kernel_size
) {
    int batch_idx = blockIdx.x;
    int in_seq_idx = blockIdx.y;
    int in_channel = threadIdx.x;
    
    if (batch_idx >= batch_size || in_seq_idx >= in_seq_len || 
        in_channel >= in_channels) return;
    
    float sum = 0.0f;
    
    for (int out_c = 0; out_c < out_channels; out_c++) {
        for (int k = 0; k < kernel_size; k++) {
            int out_seq_idx = in_seq_idx - k;
            if (out_seq_idx >= 0 && out_seq_idx < out_seq_len) {
                sum += grad_output[batch_idx * out_seq_len * out_channels + 
                                 out_seq_idx * out_channels + out_c] *
                       weights[out_c * kernel_size * in_channels + 
                              k * in_channels + in_channel];
            }
        }
    }
    
    grad_input[batch_idx * in_seq_len * in_channels + 
               in_seq_idx * in_channels + in_channel] = sum;
}

// Conv1D backward kernel (gradient w.r.t. weights)
__global__ void conv1d_backward_weights_kernel(
    const float* input,        // [batch_size, in_seq_len, in_channels]
    const float* grad_output,  // [batch_size, out_seq_len, out_channels]
    float* grad_weights,       // [out_channels, kernel_size, in_channels]
    float* grad_bias,          // [out_channels]
    int batch_size,
    int in_seq_len,
    int out_seq_len,
    int in_channels,
    int out_channels,
    int kernel_size
) {
    int out_channel = blockIdx.x;
    int k = blockIdx.y;
    int in_channel = threadIdx.x;
    
    if (out_channel >= out_channels || k >= kernel_size || 
        in_channel >= in_channels) return;
    
    float weight_grad = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int out_seq_idx = 0; out_seq_idx < out_seq_len; out_seq_idx++) {
            int in_seq_idx = out_seq_idx + k;
            if (in_seq_idx < in_seq_len) {
                weight_grad += input[b * in_seq_len * in_channels + 
                                   in_seq_idx * in_channels + in_channel] *
                              grad_output[b * out_seq_len * out_channels + 
                                        out_seq_idx * out_channels + out_channel];
            }
        }
    }
    
    grad_weights[out_channel * kernel_size * in_channels + 
                k * in_channels + in_channel] = weight_grad / batch_size;
    
    if (k == 0 && in_channel == 0) {
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int out_seq_idx = 0; out_seq_idx < out_seq_len; out_seq_idx++) {
                bias_grad += grad_output[b * out_seq_len * out_channels + 
                                      out_seq_idx * out_channels + out_channel];
            }
        }
        grad_bias[out_channel] = bias_grad / (batch_size * out_seq_len);
    }
}

// Weight update kernel
__global__ void update_weights(
    float* weights,
    const float* gradients,
    float learning_rate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// Backward pass function
void backward_pass(Model* model, float* input_data, float* target_data, 
                  float* output_data, float learning_rate) {
    // Allocate temporary gradient storage
    float *grad_dense1, *grad_pool, *grad_conv2, *grad_conv1;
    float *grad_weights_dense2, *grad_bias_dense2;
    float *grad_weights_dense1, *grad_bias_dense1;
    float *grad_weights_conv2, *grad_bias_conv2;
    float *grad_weights_conv1, *grad_bias_conv1;
    
    // Allocate memory for gradients
    CUDA_CHECK(cudaMalloc(&grad_dense1, 
        model->batch_size * model->dense_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_pool, 
        model->batch_size * model->n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv2, 
        model->batch_size * (model->sequence_length - 2*model->conv_kernel_size + 2) * 
        model->n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1, 
        model->batch_size * (model->sequence_length - model->conv_kernel_size + 1) * 
        model->n_conv1_filters * sizeof(float)));
    
    // Allocate memory for weight gradients
    CUDA_CHECK(cudaMalloc(&grad_weights_dense2, 
        model->dense_size * model->n_outputs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_bias_dense2, 
        model->n_outputs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_weights_dense1, 
        model->n_conv2_filters * model->dense_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_bias_dense1, 
        model->dense_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_weights_conv2, 
        model->n_conv2_filters * model->conv_kernel_size * 
        model->n_conv1_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_bias_conv2, 
        model->n_conv2_filters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_weights_conv1, 
        model->n_conv1_filters * model->conv_kernel_size * 
        model->n_inputs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_bias_conv1, 
        model->n_conv1_filters * sizeof(float)));

    // Compute initial gradient (MSE loss gradient)
    float* grad_output;
    CUDA_CHECK(cudaMalloc(&grad_output, 
        model->batch_size * model->n_outputs * sizeof(float)));
    
    dim3 loss_grid(model->batch_size);
    dim3 loss_block(model->n_outputs);
    mse_gradient_kernel<<<loss_grid, loss_block>>>(
        output_data, target_data, grad_output,
        model->batch_size, model->n_outputs
    );

    // Backward through dense2 layer
    dim3 dense2_grid(model->dense_size);
    dim3 dense2_block(model->n_outputs);
    dense_backward_weights_kernel<<<dense2_grid, dense2_block>>>(
        model->dense1_output,
        grad_output,
        grad_weights_dense2,
        grad_bias_dense2,
        model->batch_size,
        model->dense_size,
        model->n_outputs
    );
    
    dim3 dense2_input_grid(model->batch_size);
    dim3 dense2_input_block(model->dense_size);
    dense_backward_input_kernel<<<dense2_input_grid, dense2_input_block>>>(
        grad_output,
        model->dense2_weights,
        grad_dense1,
        model->batch_size,
        model->dense_size,
        model->n_outputs
    );

    // ReLU gradient for dense1
    int dense1_size = model->batch_size * model->dense_size;
    int threads_per_block = 256;
    int blocks = (dense1_size + threads_per_block - 1) / threads_per_block;
    relu_gradient_kernel<<<blocks, threads_per_block>>>(
        model->dense1_output,
        grad_dense1,
        grad_dense1,
        dense1_size
    );

    // Backward through dense1 layer
    dim3 dense1_grid(model->n_conv2_filters);
    dim3 dense1_block(model->dense_size);
    dense_backward_weights_kernel<<<dense1_grid, dense1_block>>>(
        model->pool_output,
        grad_dense1,
        grad_weights_dense1,
        grad_bias_dense1,
        model->batch_size,
        model->n_conv2_filters,
        model->dense_size
    );
    
    dim3 dense1_input_grid(model->batch_size);
    dim3 dense1_input_block(model->n_conv2_filters);
    dense_backward_input_kernel<<<dense1_input_grid, dense1_input_block>>>(
        grad_dense1,
        model->dense1_weights,
        grad_pool,
        model->batch_size,
        model->n_conv2_filters,
        model->dense_size
    );

    // Backward through global average pooling
    dim3 pool_grid(model->batch_size);
    dim3 pool_block(model->n_conv2_filters);
    global_avg_pooling_backward_kernel<<<pool_grid, pool_block>>>(
        grad_pool,
        grad_conv2,
        model->batch_size,
        model->sequence_length - 2*model->conv_kernel_size + 2,
        model->n_conv2_filters
    );

    // ReLU gradient for conv2
    int conv2_size = model->batch_size * 
                    (model->sequence_length - 2*model->conv_kernel_size + 2) * 
                    model->n_conv2_filters;
    blocks = (conv2_size + threads_per_block - 1) / threads_per_block;
    relu_gradient_kernel<<<blocks, threads_per_block>>>(
        model->conv2_output,
        grad_conv2,
        grad_conv2,
        conv2_size
    );

    // Backward through conv2 layer
    dim3 conv2_grid(model->n_conv2_filters, model->conv_kernel_size);
    dim3 conv2_block(model->n_conv1_filters);
    conv1d_backward_weights_kernel<<<conv2_grid, conv2_block>>>(
        model->conv1_output,
        grad_conv2,
        grad_weights_conv2,
        grad_bias_conv2,
        model->batch_size,
        model->sequence_length - model->conv_kernel_size + 1,
        model->sequence_length - 2*model->conv_kernel_size + 2,
        model->n_conv1_filters,
        model->n_conv2_filters,
        model->conv_kernel_size
    );
    
    dim3 conv2_input_grid(model->batch_size, 
                         model->sequence_length - model->conv_kernel_size + 1);
    dim3 conv2_input_block(model->n_conv1_filters);
    conv1d_backward_input_kernel<<<conv2_input_grid, conv2_input_block>>>(
        grad_conv2,
        model->conv2_weights,
        grad_conv1,
        model->batch_size,
        model->sequence_length - model->conv_kernel_size + 1,
        model->sequence_length - 2*model->conv_kernel_size + 2,
        model->n_conv1_filters,
        model->n_conv2_filters,
        model->conv_kernel_size
    );

    // ReLU gradient for conv1
    int conv1_size = model->batch_size * 
                    (model->sequence_length - model->conv_kernel_size + 1) * 
                    model->n_conv1_filters;
    blocks = (conv1_size + threads_per_block - 1) / threads_per_block;
    relu_gradient_kernel<<<blocks, threads_per_block>>>(
        model->conv1_output,
        grad_conv1,
        grad_conv1,
        conv1_size
    );

    // Backward through conv1 layer
    dim3 conv1_grid(model->n_conv1_filters, model->conv_kernel_size);
    dim3 conv1_block(model->n_inputs);
    conv1d_backward_weights_kernel<<<conv1_grid, conv1_block>>>(
        input_data,
        grad_conv1,
        grad_weights_conv1,
        grad_bias_conv1,
        model->batch_size,
        model->sequence_length,
        model->sequence_length - model->conv_kernel_size + 1,
        model->n_inputs,
        model->n_conv1_filters,
        model->conv_kernel_size
    );

    // Update weights using SGD
    update_weights<<<blocks, threads_per_block>>>(
        model->conv1_weights, grad_weights_conv1, learning_rate,
        model->n_conv1_filters * model->conv_kernel_size * model->n_inputs
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->conv1_bias, grad_bias_conv1, learning_rate,
        model->n_conv1_filters
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->conv2_weights, grad_weights_conv2, learning_rate,
        model->n_conv2_filters * model->conv_kernel_size * model->n_conv1_filters
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->conv2_bias, grad_bias_conv2, learning_rate,
        model->n_conv2_filters
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->dense1_weights, grad_weights_dense1, learning_rate,
        model->n_conv2_filters * model->dense_size
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->dense1_bias, grad_bias_dense1, learning_rate,
        model->dense_size
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->dense2_weights, grad_weights_dense2, learning_rate,
        model->dense_size * model->n_outputs
    );
    update_weights<<<blocks, threads_per_block>>>(
        model->dense2_bias, grad_bias_dense2, learning_rate,
        model->n_outputs
    );

    // Free temporary gradient storage
    cudaFree(grad_dense1);
    cudaFree(grad_pool);
    cudaFree(grad_conv2);
    cudaFree(grad_conv1);
    cudaFree(grad_weights_dense2);
    cudaFree(grad_bias_dense2);
    cudaFree(grad_weights_dense1);
    cudaFree(grad_bias_dense1);
    cudaFree(grad_weights_conv2);
    cudaFree(grad_bias_conv2);
    cudaFree(grad_weights_conv1);
    cudaFree(grad_bias_conv1);
    cudaFree(grad_output);
}

// Compute MSE loss
__global__ void compute_mse_loss_kernel(
    const float* predictions,
    const float* targets,
    float* loss,
    int batch_size,
    int n_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * n_outputs) {
        float diff = predictions[idx] - targets[idx];
        atomicAdd(loss, diff * diff / (batch_size * n_outputs));
    }
}

float compute_loss(const float* predictions, const float* targets, 
                  int batch_size, int n_outputs) {
    float* d_loss;
    float h_loss = 0.0f;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    int total_elements = batch_size * n_outputs;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    compute_mse_loss_kernel<<<blocks, threads_per_block>>>(
        predictions, targets, d_loss, batch_size, n_outputs);
    
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    
    return h_loss;
}

// Prepare batch data
void prepare_batch(Dataset* dataset, int batch_idx, int batch_size,
                  float* d_batch_inputs, float* d_batch_targets) {
    int sequence_size = dataset->sequence_length * dataset->n_inputs;
    int target_size = dataset->n_outputs;
    
    // Temporary host storage
    float* h_batch_inputs = (float*)malloc(batch_size * sequence_size * sizeof(float));
    float* h_batch_targets = (float*)malloc(batch_size * target_size * sizeof(float));
    
    // Copy data to contiguous array
    for(int i = 0; i < batch_size; i++) {
        int dataset_idx = batch_idx * batch_size + i;
        if(dataset_idx >= dataset->n_sequences) break;
        
        // Copy inputs
        for(int t = 0; t < dataset->sequence_length; t++) {
            for(int f = 0; f < dataset->n_inputs; f++) {
                h_batch_inputs[i * sequence_size + t * dataset->n_inputs + f] = 
                    dataset->inputs[dataset_idx][t][f];
            }
        }
        
        // Copy targets
        for(int f = 0; f < dataset->n_outputs; f++) {
            h_batch_targets[i * target_size + f] = dataset->targets[dataset_idx][f];
        }
    }
    
    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(d_batch_inputs, h_batch_inputs, 
        batch_size * sequence_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_targets, h_batch_targets,
        batch_size * target_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_batch_inputs);
    free(h_batch_targets);
}


// Evaluation function
float evaluate_model(Model* model, Dataset* data, int batch_size) {
    float *d_batch_inputs, *d_batch_targets;
    int sequence_size = data->sequence_length * data->n_inputs;
    int target_size = data->n_outputs;
    
    CUDA_CHECK(cudaMalloc(&d_batch_inputs, 
        batch_size * sequence_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_targets, 
        batch_size * target_size * sizeof(float)));
    
    int n_batches = (data->n_sequences + batch_size - 1) / batch_size;
    float total_loss = 0.0f;
    
    for(int batch = 0; batch < n_batches; batch++) {
        prepare_batch(data, batch, batch_size, d_batch_inputs, d_batch_targets);
        
        forward_pass(model, d_batch_inputs);
        
        float batch_loss = compute_loss(model->pool_output, d_batch_targets,
                                     batch_size, model->n_outputs);
        total_loss += batch_loss;
    }
    
    CUDA_CHECK(cudaFree(d_batch_inputs));
    CUDA_CHECK(cudaFree(d_batch_targets));
    
    return total_loss / n_batches;
}

// Training function
void train_model(Model* model, Dataset* train_data, Dataset* val_data,
                int n_epochs, int batch_size, float learning_rate) {
    // Allocate GPU memory for batch data
    float *d_batch_inputs, *d_batch_targets, *d_batch_outputs;
    int sequence_size = train_data->sequence_length * train_data->n_inputs;
    int target_size = train_data->n_outputs;
    
    CUDA_CHECK(cudaMalloc(&d_batch_inputs, 
        batch_size * sequence_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_targets, 
        batch_size * target_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_outputs, 
        batch_size * target_size * sizeof(float)));
    
    int n_batches = (train_data->n_sequences + batch_size - 1) / batch_size;
    
    for(int epoch = 0; epoch < n_epochs; epoch++) {
        float total_loss = 0.0f;
        
        // Training loop
        for(int batch = 0; batch < n_batches; batch++) {
            // Prepare batch data
            prepare_batch(train_data, batch, batch_size, 
                        d_batch_inputs, d_batch_targets);
            
            // Forward pass
            forward_pass(model, d_batch_inputs);
            
            // Compute loss
            float batch_loss = compute_loss(model->pool_output, d_batch_targets,
                                         batch_size, model->n_outputs);
            total_loss += batch_loss;
            
            // Backward pass and update weights
            backward_pass(model, d_batch_inputs, d_batch_targets,
                        model->pool_output, learning_rate);
        }
        
        // Validation
        float val_loss = evaluate_model(model, val_data, batch_size);
        
        printf("Epoch %d/%d - train_loss: %.4f - val_loss: %.4f\n",
               epoch + 1, n_epochs, total_loss / n_batches, val_loss);
    }
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_batch_inputs));
    CUDA_CHECK(cudaFree(d_batch_targets));
    CUDA_CHECK(cudaFree(d_batch_outputs));
}

int main() {
    // Set random seed
    srand(time(NULL));
    
    // Generate synthetic dataset
    int n_sequences = 1000;
    int sequence_length = 32;
    int n_inputs = 6;
    int n_outputs = 4;
    float noise_level = 0.1;
    
    Dataset* train_data = generate_data(n_sequences, sequence_length, 
                                      n_inputs, n_outputs, noise_level);
    Dataset* val_data = generate_data(200, sequence_length, 
                                    n_inputs, n_outputs, noise_level);
    
    // Create model configuration
    ModelConfig config = {
        .batch_size = 32,
        .sequence_length = sequence_length,
        .n_inputs = n_inputs,
        .n_outputs = n_outputs,
        .n_conv1_filters = 32,
        .n_conv2_filters = 64,
        .dense_size = 128,
        .conv_kernel_size = 3
    };
    
    // Create and initialize model
    Model* model = create_model(config);
    
    // Training parameters
    int n_epochs = 50;
    int batch_size = config.batch_size;
    float learning_rate = 0.001f;
    
    // Train the model
    printf("Starting training...\n");
    train_model(model, train_data, val_data, n_epochs, batch_size, learning_rate);
    
    // Save final predictions
    printf("Generating final predictions...\n");
    float* predictions = (float*)malloc(val_data->n_sequences * 
                                      val_data->n_outputs * sizeof(float));
    evaluate_model(model, val_data, batch_size);
    
    // Save results
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_predictions.csv", 
             localtime(&now));
    
    FILE* fp = fopen(fname, "w");
    if(fp) {
        fprintf(fp, "sequence");
        for(int i = 0; i < n_outputs; i++)
            fprintf(fp, ",y%d_true,y%d_pred", i, i);
        fprintf(fp, "\n");
        
        for(int i = 0; i < val_data->n_sequences; i++) {
            fprintf(fp, "%d", i);
            for(int j = 0; j < n_outputs; j++) {
                fprintf(fp, ",%.6f,%.6f", 
                        val_data->targets[i][j],
                        predictions[i * n_outputs + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("Predictions saved to: %s\n", fname);
    }
    
    // Cleanup
    free(predictions);
    free_model(model);
    free_dataset(train_data);
    free_dataset(val_data);
    
    return 0;
}