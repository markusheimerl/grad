#include "data.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

// Utility macros
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_NULL(ptr) { \
    if (ptr == NULL) { \
        printf("Memory allocation failed %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Layer structures
typedef struct {
    float *weights;      // [out_channels, in_channels, kernel_size]
    float *bias;         // [out_channels]
    float *d_weights;    // gradients for weights
    float *d_bias;       // gradients for bias
    int in_channels;
    int out_channels;
    int kernel_size;
} Conv1DLayer;

typedef struct {
    float *weights;      // [out_features, in_features]
    float *bias;         // [out_features]
    float *d_weights;    // gradients for weights
    float *d_bias;       // gradients for bias
    int in_features;
    int out_features;
} DenseLayer;

typedef struct {
    Conv1DLayer conv1;
    Conv1DLayer conv2;
    DenseLayer dense1;
    DenseLayer dense2;
    float learning_rate;
    int batch_size;
    
    // Intermediate activations for backprop
    float *conv1_output;
    float *conv2_output;
    float *pool_output;
    float *dense1_output;
} Model;

// Helper functions for initialization
static float* cuda_malloc_float(size_t size) {
    float *ptr;
    CHECK_CUDA(cudaMalloc(&ptr, size * sizeof(float)));
    return ptr;
}

static void init_conv_layer(Conv1DLayer *layer, int in_channels, int out_channels, int kernel_size) {
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    
    size_t weights_size = out_channels * in_channels * kernel_size;
    layer->weights = cuda_malloc_float(weights_size);
    layer->bias = cuda_malloc_float(out_channels);
    layer->d_weights = cuda_malloc_float(weights_size);
    layer->d_bias = cuda_malloc_float(out_channels);
    
    float *h_weights = (float*)malloc(weights_size * sizeof(float));
    float *h_bias = (float*)malloc(out_channels * sizeof(float));
    CHECK_NULL(h_weights);
    CHECK_NULL(h_bias);
    
    // Modified Xavier initialization
    float scale = sqrt(2.0f / (in_channels * kernel_size + out_channels));
    for(size_t i = 0; i < weights_size; i++) {
        h_weights[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    for(int i = 0; i < out_channels; i++) {
        h_bias[i] = 0.0f;
    }
    
    CHECK_CUDA(cudaMemcpy(layer->weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(layer->bias, h_bias, out_channels * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_weights);
    free(h_bias);
}

static void init_dense_layer(DenseLayer *layer, int in_features, int out_features) {
    layer->in_features = in_features;
    layer->out_features = out_features;
    
    size_t weights_size = out_features * in_features;
    layer->weights = cuda_malloc_float(weights_size);
    layer->bias = cuda_malloc_float(out_features);
    layer->d_weights = cuda_malloc_float(weights_size);
    layer->d_bias = cuda_malloc_float(out_features);
    
    float *h_weights = (float*)malloc(weights_size * sizeof(float));
    float *h_bias = (float*)malloc(out_features * sizeof(float));
    CHECK_NULL(h_weights);
    CHECK_NULL(h_bias);
    
    // Modified Xavier initialization
    float scale = sqrt(2.0f / (in_features + out_features));
    for(size_t i = 0; i < weights_size; i++) {
        h_weights[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    for(int i = 0; i < out_features; i++) {
        h_bias[i] = 0.0f;
    }
    
    CHECK_CUDA(cudaMemcpy(layer->weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(layer->bias, h_bias, out_features * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_weights);
    free(h_bias);
}

static Model* create_model(int window_size, int n_inputs, int n_outputs, int batch_size) {
    Model *model = (Model*)malloc(sizeof(Model));
    CHECK_NULL(model);
    
    model->batch_size = batch_size;
    model->learning_rate = 1e-4f;
    
    // Initialize layers
    init_conv_layer(&model->conv1, n_inputs, 32, 3);
    init_conv_layer(&model->conv2, 32, 64, 3);
    
    // Calculate sizes for dense layers
    int conv1_output_size = (window_size - 2);  // After first conv
    int conv2_output_size = (conv1_output_size - 2);  // After second conv
    int pool_size = 64;  // After global average pooling
    
    init_dense_layer(&model->dense1, pool_size, 128);
    init_dense_layer(&model->dense2, 128, n_outputs);
    
    // Allocate memory for intermediate activations
    model->conv1_output = cuda_malloc_float(batch_size * 32 * conv1_output_size);
    model->conv2_output = cuda_malloc_float(batch_size * 64 * conv2_output_size);
    model->pool_output = cuda_malloc_float(batch_size * 64);
    model->dense1_output = cuda_malloc_float(batch_size * 128);
    
    return model;
}

// Forward pass kernels and helper functions

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void conv1d_forward_kernel(
    const float* input,      // [batch_size, in_channels, input_length]
    const float* weights,    // [out_channels, in_channels, kernel_size]
    const float* bias,       // [out_channels]
    float* output,          // [batch_size, out_channels, output_length]
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size
) {
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_pos = threadIdx.x;
    
    int output_length = input_length - kernel_size + 1;
    if (out_pos >= output_length) return;
    
    float sum = bias[out_channel];
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = out_pos + k;
            float in_val = input[
                batch_idx * (in_channels * input_length) +
                in_c * input_length +
                in_pos
            ];
            float weight = weights[
                out_channel * (in_channels * kernel_size) +
                in_c * kernel_size +
                k
            ];
            sum += in_val * weight;
        }
    }
    
    output[
        batch_idx * (out_channels * output_length) +
        out_channel * output_length +
        out_pos
    ] = relu(sum);
}

__global__ void global_avg_pooling_kernel(
    const float* input,     // [batch_size, channels, length]
    float* output,         // [batch_size, channels]
    int batch_size,
    int channels,
    int length
) {
    int batch_idx = blockIdx.x;
    int channel = threadIdx.x;
    
    if (channel >= channels) return;
    
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += input[
            batch_idx * (channels * length) +
            channel * length +
            i
        ];
    }
    
    output[batch_idx * channels + channel] = sum / length;
}

__global__ void dense_forward_kernel(
    const float* input,     // [batch_size, in_features]
    const float* weights,   // [out_features, in_features]
    const float* bias,      // [out_features]
    float* output,         // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu
) {
    int batch_idx = blockIdx.x;
    int out_feature = threadIdx.x;
    
    if (out_feature >= out_features) return;
    
    float sum = bias[out_feature];
    
    for (int in_f = 0; in_f < in_features; in_f++) {
        sum += input[batch_idx * in_features + in_f] *
               weights[out_feature * in_features + in_f];
    }
    
    output[batch_idx * out_features + out_feature] = 
        apply_relu ? relu(sum) : sum;
}

// Forward pass wrapper functions
static void conv1d_forward(
    const float* input,
    const Conv1DLayer* layer,
    float* output,
    int batch_size,
    int input_length
) {
    int output_length = input_length - layer->kernel_size + 1;
    
    dim3 grid(batch_size, layer->out_channels);
    dim3 block(output_length);
    
    conv1d_forward_kernel<<<grid, block>>>(
        input,
        layer->weights,
        layer->bias,
        output,
        batch_size,
        layer->in_channels,
        layer->out_channels,
        input_length,
        layer->kernel_size
    );
    CHECK_CUDA(cudaGetLastError());
}

static void global_avg_pooling(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int length
) {
    dim3 grid(batch_size);
    dim3 block(channels);
    
    global_avg_pooling_kernel<<<grid, block>>>(
        input,
        output,
        batch_size,
        channels,
        length
    );
    CHECK_CUDA(cudaGetLastError());
}

static void dense_forward(
    const float* input,
    const DenseLayer* layer,
    float* output,
    int batch_size,
    bool apply_relu
) {
    dim3 grid(batch_size);
    dim3 block(layer->out_features);
    
    dense_forward_kernel<<<grid, block>>>(
        input,
        layer->weights,
        layer->bias,
        output,
        batch_size,
        layer->in_features,
        layer->out_features,
        apply_relu
    );
    CHECK_CUDA(cudaGetLastError());
}

// Complete forward pass function
static void forward_pass(
    Model* model,
    const float* input,     // [batch_size, n_inputs, window_size]
    float* output          // [batch_size, n_outputs]
) {
    int window_size = 32;  // Hardcoded for now
    
    // Conv1D layer 1
    conv1d_forward(
        input,
        &model->conv1,
        model->conv1_output,
        model->batch_size,
        window_size
    );
    
    // Conv1D layer 2
    conv1d_forward(
        model->conv1_output,
        &model->conv2,
        model->conv2_output,
        model->batch_size,
        window_size - 2  // After first conv
    );
    
    // Global Average Pooling
    global_avg_pooling(
        model->conv2_output,
        model->pool_output,
        model->batch_size,
        64,  // conv2 out_channels
        window_size - 4  // After both convs
    );
    
    // Dense layer 1 with ReLU
    dense_forward(
        model->pool_output,
        &model->dense1,
        model->dense1_output,
        model->batch_size,
        true
    );
    
    // Dense layer 2 (output) without ReLU
    dense_forward(
        model->dense1_output,
        &model->dense2,
        output,
        model->batch_size,
        false
    );
}

// Backward pass kernels and helper functions

__device__ float relu_gradient(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

__global__ void dense_backward_kernel(
    const float* input,         // [batch_size, in_features]
    const float* weights,       // [out_features, in_features]
    const float* output,        // [batch_size, out_features]
    const float* grad_output,   // [batch_size, out_features]
    float* grad_input,         // [batch_size, in_features]
    float* grad_weights,       // [out_features, in_features]
    float* grad_bias,         // [out_features]
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu
) {
    extern __shared__ float shared_grad_bias[];
    
    int tid = threadIdx.x;
    if (tid < out_features) {
        shared_grad_bias[tid] = 0.0f;
    }
    __syncthreads();
    
    // Each thread handles one input feature for all batches
    for (int in_f = tid; in_f < in_features; in_f += blockDim.x) {
        float grad_in = 0.0f;
        
        for (int out_f = 0; out_f < out_features; out_f++) {
            float weight = weights[out_f * in_features + in_f];
            
            for (int b = 0; b < batch_size; b++) {
                float out_val = output[b * out_features + out_f];
                float grad_out = grad_output[b * out_features + out_f];
                
                if (apply_relu) {
                    grad_out *= relu_gradient(out_val);
                }
                
                // Gradient with respect to input
                grad_in += weight * grad_out;
                
                // Gradient with respect to weights
                float in_val = input[b * in_features + in_f];
                atomicAdd(&grad_weights[out_f * in_features + in_f], 
                         in_val * grad_out);
                
                // Accumulate gradient with respect to bias
                if (in_f == 0) {
                    atomicAdd(&shared_grad_bias[out_f], grad_out);
                }
            }
        }
        
        if (grad_input != nullptr) {
            for (int b = 0; b < batch_size; b++) {
                grad_input[b * in_features + in_f] = grad_in;
            }
        }
    }
    
    __syncthreads();
    
    // Write accumulated bias gradients
    if (tid < out_features) {
        atomicAdd(&grad_bias[tid], shared_grad_bias[tid]);
    }
}

__global__ void global_avg_pooling_backward_kernel(
    const float* grad_output,   // [batch_size, channels]
    float* grad_input,         // [batch_size, channels, length]
    int batch_size,
    int channels,
    int length
) {
    int batch_idx = blockIdx.x;
    int channel = threadIdx.x;
    
    if (channel >= channels) return;
    
    float grad_val = grad_output[batch_idx * channels + channel] / length;
    
    for (int i = 0; i < length; i++) {
        grad_input[
            batch_idx * (channels * length) +
            channel * length +
            i
        ] = grad_val;
    }
}

__global__ void conv1d_backward_kernel(
    const float* input,         // [batch_size, in_channels, input_length]
    const float* weights,       // [out_channels, in_channels, kernel_size]
    const float* output,        // [batch_size, out_channels, output_length]
    const float* grad_output,   // [batch_size, out_channels, output_length]
    float* grad_input,         // [batch_size, in_channels, input_length]
    float* grad_weights,       // [out_channels, in_channels, kernel_size]
    float* grad_bias,         // [out_channels]
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size
) {
    extern __shared__ float shared_mem[];
    float* shared_grad_bias = shared_mem;
    
    int batch_idx = blockIdx.x;
    int out_c = blockIdx.y;
    int tid = threadIdx.x;
    
    int output_length = input_length - kernel_size + 1;
    
    // Initialize shared memory
    if (tid < out_channels) {
        shared_grad_bias[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute gradients
    for (int pos = tid; pos < output_length; pos += blockDim.x) {
        float grad_out = grad_output[
            batch_idx * (out_channels * output_length) +
            out_c * output_length +
            pos
        ];
        float out_val = output[
            batch_idx * (out_channels * output_length) +
            out_c * output_length +
            pos
        ];
        
        grad_out *= relu_gradient(out_val);
        
        // Accumulate bias gradient
        atomicAdd(&shared_grad_bias[out_c], grad_out);
        
        // Compute gradients for weights and input
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int k = 0; k < kernel_size; k++) {
                // Gradient for weights
                float in_val = input[
                    batch_idx * (in_channels * input_length) +
                    in_c * input_length +
                    pos + k
                ];
                atomicAdd(&grad_weights[
                    out_c * (in_channels * kernel_size) +
                    in_c * kernel_size +
                    k
                ], in_val * grad_out);
                
                // Gradient for input
                float weight = weights[
                    out_c * (in_channels * kernel_size) +
                    in_c * kernel_size +
                    k
                ];
                atomicAdd(&grad_input[
                    batch_idx * (in_channels * input_length) +
                    in_c * input_length +
                    pos + k
                ], weight * grad_out);
            }
        }
    }
    
    __syncthreads();
    
    // Write accumulated bias gradients
    if (tid < out_channels && batch_idx == 0) {
        atomicAdd(&grad_bias[tid], shared_grad_bias[tid]);
    }
}

// Backward pass wrapper functions
static void dense_backward(
    const float* input,
    const DenseLayer* layer,
    const float* output,
    const float* grad_output,
    float* grad_input,
    int batch_size,
    bool apply_relu
) {
    dim3 grid(1);
    dim3 block(256);  // Adjust based on GPU capabilities
    
    size_t shared_mem_size = layer->out_features * sizeof(float);
    
    dense_backward_kernel<<<grid, block, shared_mem_size>>>(
        input,
        layer->weights,
        output,
        grad_output,
        grad_input,
        layer->d_weights,
        layer->d_bias,
        batch_size,
        layer->in_features,
        layer->out_features,
        apply_relu
    );
    CHECK_CUDA(cudaGetLastError());
}

static void global_avg_pooling_backward(
    const float* grad_output,
    float* grad_input,
    int batch_size,
    int channels,
    int length
) {
    dim3 grid(batch_size);
    dim3 block(channels);
    
    global_avg_pooling_backward_kernel<<<grid, block>>>(
        grad_output,
        grad_input,
        batch_size,
        channels,
        length
    );
    CHECK_CUDA(cudaGetLastError());
}

static void conv1d_backward(
    const float* input,
    const Conv1DLayer* layer,
    const float* output,
    const float* grad_output,
    float* grad_input,
    int batch_size,
    int input_length
) {
    dim3 grid(batch_size, layer->out_channels);
    dim3 block(256);  // Adjust based on GPU capabilities
    
    size_t shared_mem_size = layer->out_channels * sizeof(float);
    
    conv1d_backward_kernel<<<grid, block, shared_mem_size>>>(
        input,
        layer->weights,
        output,
        grad_output,
        grad_input,
        layer->d_weights,
        layer->d_bias,
        batch_size,
        layer->in_channels,
        layer->out_channels,
        input_length,
        layer->kernel_size
    );
    CHECK_CUDA(cudaGetLastError());
}

// Complete backward pass function
static void backward_pass(
    Model* model,
    const float* input,
    const float* grad_output,
    float* grad_input,
    int batch_size
) {
    int window_size = 32;  // Hardcoded for now
    
    // Temporary storage for gradients
    float *grad_dense1, *grad_pool, *grad_conv2, *grad_conv1;
    CHECK_CUDA(cudaMalloc(&grad_dense1, batch_size * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grad_pool, batch_size * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grad_conv2, 
        batch_size * 64 * (window_size - 2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grad_conv1, 
        batch_size * 32 * window_size * sizeof(float)));
    
    // Backward through dense2 (output layer)
    dense_backward(
        model->dense1_output,
        &model->dense2,
        nullptr,  // No activation
        grad_output,
        grad_dense1,
        batch_size,
        false
    );
    
    // Backward through dense1
    dense_backward(
        model->pool_output,
        &model->dense1,
        model->dense1_output,
        grad_dense1,
        grad_pool,
        batch_size,
        true
    );
    
    // Backward through global average pooling
    global_avg_pooling_backward(
        grad_pool,
        grad_conv2,
        batch_size,
        64,
        window_size - 4
    );
    
    // Backward through conv2
    conv1d_backward(
        model->conv1_output,
        &model->conv2,
        model->conv2_output,
        grad_conv2,
        grad_conv1,
        batch_size,
        window_size - 2
    );
    
    // Backward through conv1
    conv1d_backward(
        input,
        &model->conv1,
        model->conv1_output,
        grad_conv1,
        grad_input,
        batch_size,
        window_size
    );
    
    // Free temporary storage
    cudaFree(grad_dense1);
    cudaFree(grad_pool);
    cudaFree(grad_conv2);
    cudaFree(grad_conv1);
}

// Loss function and optimization kernels

__global__ void mse_loss_kernel(
    const float* predictions,
    const float* targets,
    float* loss,
    float* grad_output,
    int batch_size,
    int n_outputs
) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float local_loss = 0.0f;
    
    // Each thread processes multiple elements
    for (int idx = tid; idx < batch_size * n_outputs; idx += stride) {
        float diff = predictions[idx] - targets[idx];
        local_loss += diff * diff;
        grad_output[idx] = 2.0f * diff / (batch_size * n_outputs);
    }
    
    // Reduce loss in shared memory
    extern __shared__ float shared_loss[];
    shared_loss[tid] = local_loss;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, shared_loss[0] / (batch_size * n_outputs));
    }
}

__global__ void sgd_update_kernel(
    float* weights,
    const float* gradients,
    int size,
    float learning_rate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        weights[tid] -= learning_rate * gradients[tid];
    }
}

// Training utility functions
static void compute_loss_and_gradients(
    const float* predictions,
    const float* targets,
    float* loss,
    float* grad_output,
    int batch_size,
    int n_outputs
) {
    float* d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    dim3 block(256);
    size_t shared_mem_size = block.x * sizeof(float);
    
    mse_loss_kernel<<<1, block, shared_mem_size>>>(
        predictions,
        targets,
        d_loss,
        grad_output,
        batch_size,
        n_outputs
    );
    
    CHECK_CUDA(cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_loss);
}

static void update_parameters(Model* model) {
    // Update conv1 parameters
    int conv1_weights_size = model->conv1.out_channels * 
                            model->conv1.in_channels * 
                            model->conv1.kernel_size;
    sgd_update_kernel<<<(conv1_weights_size + 255)/256, 256>>>(
        model->conv1.weights,
        model->conv1.d_weights,
        conv1_weights_size,
        model->learning_rate
    );
    sgd_update_kernel<<<(model->conv1.out_channels + 255)/256, 256>>>(
        model->conv1.bias,
        model->conv1.d_bias,
        model->conv1.out_channels,
        model->learning_rate
    );
    
    // Update conv2 parameters
    int conv2_weights_size = model->conv2.out_channels * 
                            model->conv2.in_channels * 
                            model->conv2.kernel_size;
    sgd_update_kernel<<<(conv2_weights_size + 255)/256, 256>>>(
        model->conv2.weights,
        model->conv2.d_weights,
        conv2_weights_size,
        model->learning_rate
    );
    sgd_update_kernel<<<(model->conv2.out_channels + 255)/256, 256>>>(
        model->conv2.bias,
        model->conv2.d_bias,
        model->conv2.out_channels,
        model->learning_rate
    );
    
    // Update dense1 parameters
    int dense1_weights_size = model->dense1.in_features * 
                             model->dense1.out_features;
    sgd_update_kernel<<<(dense1_weights_size + 255)/256, 256>>>(
        model->dense1.weights,
        model->dense1.d_weights,
        dense1_weights_size,
        model->learning_rate
    );
    sgd_update_kernel<<<(model->dense1.out_features + 255)/256, 256>>>(
        model->dense1.bias,
        model->dense1.d_bias,
        model->dense1.out_features,
        model->learning_rate
    );
    
    // Update dense2 parameters
    int dense2_weights_size = model->dense2.in_features * 
                             model->dense2.out_features;
    sgd_update_kernel<<<(dense2_weights_size + 255)/256, 256>>>(
        model->dense2.weights,
        model->dense2.d_weights,
        dense2_weights_size,
        model->learning_rate
    );
    sgd_update_kernel<<<(model->dense2.out_features + 255)/256, 256>>>(
        model->dense2.bias,
        model->dense2.d_bias,
        model->dense2.out_features,
        model->learning_rate
    );
}

__global__ void clip_gradients_kernel(
    float* gradients,
    int size,
    float max_norm
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float val = gradients[tid];
        if (val > max_norm) {
            gradients[tid] = max_norm;
        } else if (val < -max_norm) {
            gradients[tid] = -max_norm;
        }
    }
}

// Training loop function
static void train_model(
    Model* model,
    Dataset* data,
    int epochs,
    int batch_size
) {
    float *d_batch_input, *d_batch_targets;
    float *d_predictions, *d_grad_output;
    
    size_t batch_input_size = batch_size * data->n_inputs * data->window_size;
    size_t batch_output_size = batch_size * data->n_outputs;
    
    CHECK_CUDA(cudaMalloc(&d_batch_input, batch_input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_targets, batch_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_predictions, batch_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, batch_output_size * sizeof(float)));
    
    float* h_batch_input = (float*)malloc(batch_input_size * sizeof(float));
    float* h_batch_targets = (float*)malloc(batch_output_size * sizeof(float));
    
    int n_batches = data->n_sequences / batch_size;
    const float gradient_clip = 1.0f;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < n_batches; batch++) {
            // Prepare batch data
            for (int i = 0; i < batch_size; i++) {
                int seq_idx = batch * batch_size + i;
                
                for (int t = 0; t < data->window_size; t++) {
                    for (int f = 0; f < data->n_inputs; f++) {
                        h_batch_input[i * data->n_inputs * data->window_size + 
                                    t * data->n_inputs + f] = 
                            data->inputs[seq_idx + t][f];
                    }
                }
                
                for (int f = 0; f < data->n_outputs; f++) {
                    h_batch_targets[i * data->n_outputs + f] = 
                        data->targets[seq_idx + data->window_size - 1][f];
                }
            }
            
            // Copy to device
            CHECK_CUDA(cudaMemcpy(d_batch_input, h_batch_input,
                                batch_input_size * sizeof(float),
                                cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_batch_targets, h_batch_targets,
                                batch_output_size * sizeof(float),
                                cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass(model, d_batch_input, d_predictions);
            
            // Compute loss and gradients
            float batch_loss;
            compute_loss_and_gradients(
                d_predictions,
                d_batch_targets,
                &batch_loss,
                d_grad_output,
                batch_size,
                data->n_outputs
            );
            
            // Clip gradients
            clip_gradients_kernel<<<(batch_output_size + 255)/256, 256>>>(
                d_grad_output,
                batch_output_size,
                gradient_clip
            );
            
            // Backward pass
            backward_pass(
                model,
                d_batch_input,
                d_grad_output,
                d_batch_input,
                batch_size
            );
            
            // Clip gradients for all layers
            int conv1_weights_size = model->conv1.out_channels * 
                                   model->conv1.in_channels * 
                                   model->conv1.kernel_size;
            clip_gradients_kernel<<<(conv1_weights_size + 255)/256, 256>>>(
                model->conv1.d_weights,
                conv1_weights_size,
                gradient_clip
            );
            
            int conv2_weights_size = model->conv2.out_channels * 
                                   model->conv2.in_channels * 
                                   model->conv2.kernel_size;
            clip_gradients_kernel<<<(conv2_weights_size + 255)/256, 256>>>(
                model->conv2.d_weights,
                conv2_weights_size,
                gradient_clip
            );
            
            int dense1_weights_size = model->dense1.in_features * 
                                    model->dense1.out_features;
            clip_gradients_kernel<<<(dense1_weights_size + 255)/256, 256>>>(
                model->dense1.d_weights,
                dense1_weights_size,
                gradient_clip
            );
            
            int dense2_weights_size = model->dense2.in_features * 
                                    model->dense2.out_features;
            clip_gradients_kernel<<<(dense2_weights_size + 255)/256, 256>>>(
                model->dense2.d_weights,
                dense2_weights_size,
                gradient_clip
            );
            
            // Update parameters
            update_parameters(model);
            
            epoch_loss += batch_loss;
        }
        
        printf("Epoch %d/%d - Loss: %f\n", 
               epoch + 1, epochs, epoch_loss / n_batches);
    }
    
    cudaFree(d_batch_input);
    cudaFree(d_batch_targets);
    cudaFree(d_predictions);
    cudaFree(d_grad_output);
    free(h_batch_input);
    free(h_batch_targets);
}

static void free_model(Model* model) {
    // Free conv1 memory
    cudaFree(model->conv1.weights);
    cudaFree(model->conv1.bias);
    cudaFree(model->conv1.d_weights);
    cudaFree(model->conv1.d_bias);
    
    // Free conv2 memory
    cudaFree(model->conv2.weights);
    cudaFree(model->conv2.bias);
    cudaFree(model->conv2.d_weights);
    cudaFree(model->conv2.d_bias);
    
    // Free dense1 memory
    cudaFree(model->dense1.weights);
    cudaFree(model->dense1.bias);
    cudaFree(model->dense1.d_weights);
    cudaFree(model->dense1.d_bias);
    
    // Free dense2 memory
    cudaFree(model->dense2.weights);
    cudaFree(model->dense2.bias);
    cudaFree(model->dense2.d_weights);
    cudaFree(model->dense2.d_bias);
    
    // Free intermediate activations
    cudaFree(model->conv1_output);
    cudaFree(model->conv2_output);
    cudaFree(model->pool_output);
    cudaFree(model->dense1_output);
    
    free(model);
}

// Main function
int main() {
    srand(time(NULL));
    
    // Generate synthetic data
    Dataset* data = generate_data(1000, 32, 6, 4, 0.1);
    
    // Create and train model with smaller batch size
    Model* model = create_model(data->window_size, data->n_inputs, 
                              data->n_outputs, 64);
    
    train_model(model, data, 10, 64);
    
    // Save results
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));
    save_csv(fname, data);
    printf("Data saved to: %s\n", fname);
    
    // Cleanup
    free_dataset(data);
    free_model(model);
    
    return 0;
}