#include "data.cuh"
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Hyperparameters
#define LEARNING_RATE 1e-3f
#define BATCH_SIZE 32
#define N_CONV_LAYERS 3
#define N_FILTERS 64
#define KERNEL_SIZE 3
#define EPSILON 1e-5f
#define MAX_GRAD_NORM 1.0f

typedef struct {
    float *weights;
    float *bias;
    
    // Temporary storage for backprop
    float *d_weights;
    float *d_bias;
} ConvLayer;

typedef struct {
    float *weights;
    float *bias;
    
    // Temporary storage for backprop
    float *d_weights;
    float *d_bias;
} DenseLayer;

typedef struct {
    ConvLayer *conv_layers;
    DenseLayer dense_layer;
    int n_conv_layers;
    int kernel_size;
    int n_filters;
    int sequence_length;
    int n_inputs;
    int n_outputs;
} Model;

__global__ void init_weights_kernel(float *weights, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        weights[idx] = (curand_uniform(&state) - 0.5f) * scale;
    }
}

static Model* create_model(int sequence_length, int n_inputs, int n_outputs) {
    Model *model = (Model*)malloc(sizeof(Model));
    model->n_conv_layers = N_CONV_LAYERS;
    model->kernel_size = KERNEL_SIZE;
    model->n_filters = N_FILTERS;
    model->sequence_length = sequence_length;
    model->n_inputs = n_inputs;
    model->n_outputs = n_outputs;
    
    // Allocate conv layers
    model->conv_layers = (ConvLayer*)malloc(N_CONV_LAYERS * sizeof(ConvLayer));
    
    for (int i = 0; i < N_CONV_LAYERS; i++) {
        int in_channels = (i == 0) ? n_inputs : N_FILTERS;
        int weights_size = N_FILTERS * in_channels * KERNEL_SIZE;
        
        // Allocate layer parameters
        CHECK_CUDA(cudaMalloc(&model->conv_layers[i].weights, weights_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&model->conv_layers[i].bias, N_FILTERS * sizeof(float)));
        
        // Allocate gradients
        CHECK_CUDA(cudaMalloc(&model->conv_layers[i].d_weights, weights_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&model->conv_layers[i].d_bias, N_FILTERS * sizeof(float)));
        
        // Initialize weights
        float scale = sqrtf(2.0f / (in_channels * KERNEL_SIZE)); // He initialization
        init_weights_kernel<<<(weights_size + 255) / 256, 256>>>(
            model->conv_layers[i].weights, weights_size, scale);
        
        // Initialize bias
        CHECK_CUDA(cudaMemset(model->conv_layers[i].bias, 0, N_FILTERS * sizeof(float)));
    }
    
    // Initialize dense layer
    int dense_weights_size = N_FILTERS * n_outputs;
    CHECK_CUDA(cudaMalloc(&model->dense_layer.weights, dense_weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->dense_layer.bias, n_outputs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->dense_layer.d_weights, dense_weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->dense_layer.d_bias, n_outputs * sizeof(float)));
    
    float dense_scale = sqrtf(2.0f / N_FILTERS);
    init_weights_kernel<<<(dense_weights_size + 255) / 256, 256>>>(
        model->dense_layer.weights, dense_weights_size, dense_scale);
    CHECK_CUDA(cudaMemset(model->dense_layer.bias, 0, n_outputs * sizeof(float)));
    
    return model;
}

// Structure to hold intermediate activations
typedef struct {
    float *conv_inputs;     // Input to each conv layer
    float *conv_outputs;    // Output after conv
    float *norm_outputs;    // Output after RMSNorm
    float *relu_outputs;    // Output after ReLU
    float *residual_outputs; // Residual connections
    float *pooled_output;   // After global average pooling
    float *final_output;    // Network output
} Activations;

static Activations* create_activations(Model *model, int batch_size) {
    Activations *acts = (Activations*)malloc(sizeof(Activations));
    int seq_features = batch_size * model->sequence_length * model->n_filters;
    int input_features = batch_size * model->sequence_length * model->n_inputs;
    
    CHECK_CUDA(cudaMalloc(&acts->conv_inputs, input_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->conv_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->norm_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->relu_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->residual_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->pooled_output, 
        batch_size * model->n_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->final_output, 
        batch_size * model->n_outputs * sizeof(float)));
    
    return acts;
}

static void free_activations(Activations *acts) {
    CHECK_CUDA(cudaFree(acts->conv_inputs));
    CHECK_CUDA(cudaFree(acts->conv_outputs));
    CHECK_CUDA(cudaFree(acts->norm_outputs));
    CHECK_CUDA(cudaFree(acts->relu_outputs));
    CHECK_CUDA(cudaFree(acts->residual_outputs));
    CHECK_CUDA(cudaFree(acts->pooled_output));
    CHECK_CUDA(cudaFree(acts->final_output));
    free(acts);
}

__global__ void conv1d_forward_kernel(
    const float *input, float *output,
    const float *weights, const float *bias,
    int batch_size, int sequence_length, int n_inputs, 
    int n_filters, int kernel_size
) {
    int batch_idx = blockIdx.x;
    int filter_idx = blockIdx.y;
    int seq_idx = threadIdx.x;
    
    if (seq_idx < sequence_length) {
        float sum = bias[filter_idx];
        
        for (int k = 0; k < kernel_size; k++) {
            int seq_pos = seq_idx - kernel_size/2 + k;
            if (seq_pos >= 0 && seq_pos < sequence_length) {
                for (int c = 0; c < n_inputs; c++) {
                    float input_val = input[
                        batch_idx * sequence_length * n_inputs + 
                        seq_pos * n_inputs + c
                    ];
                    float weight = weights[
                        filter_idx * n_inputs * kernel_size + 
                        c * kernel_size + k
                    ];
                    sum += input_val * weight;
                }
            }
        }
        
        output[
            batch_idx * sequence_length * n_filters + 
            seq_idx * n_filters + filter_idx
        ] = sum;
    }
}

__global__ void rms_norm_forward_kernel(
    float *input, float *output,
    int batch_size, int sequence_length, int n_filters
) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;
    
    if (seq_idx < sequence_length) {
        // Compute RMS for this position
        float sum_squared = 0.0f;
        for (int f = 0; f < n_filters; f++) {
            int idx = batch_idx * sequence_length * n_filters + 
                     seq_idx * n_filters + f;
            float val = input[idx];
            sum_squared += val * val;
        }
        
        float rms = sqrtf(sum_squared / n_filters + EPSILON);
        
        // Normalize using RMS
        for (int f = 0; f < n_filters; f++) {
            int idx = batch_idx * sequence_length * n_filters + 
                     seq_idx * n_filters + f;
            output[idx] = input[idx] / rms;
        }
    }
}

__global__ void relu_forward_kernel(
    float *input, float *output,
    int batch_size, int sequence_length, int n_filters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * sequence_length * n_filters;
    
    if (idx < total_elements) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void residual_add_kernel(
    float *input, float *residual,
    int batch_size, int sequence_length, int n_filters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * sequence_length * n_filters;
    
    if (idx < total_elements) {
        input[idx] += residual[idx];
    }
}

__global__ void global_avg_pool_kernel(
    const float *input, float *output,
    int batch_size, int sequence_length, int n_filters
) {
    int batch_idx = blockIdx.x;
    int filter_idx = threadIdx.x;
    
    if (filter_idx < n_filters) {
        float sum = 0.0f;
        for (int s = 0; s < sequence_length; s++) {
            sum += input[
                batch_idx * sequence_length * n_filters + 
                s * n_filters + filter_idx
            ];
        }
        output[batch_idx * n_filters + filter_idx] = sum / sequence_length;
    }
}

__global__ void dense_forward_kernel(
    const float *input, float *output,
    const float *weights, const float *bias,
    int batch_size, int n_filters, int n_outputs
) {
    int batch_idx = blockIdx.x;
    int output_idx = threadIdx.x;
    
    if (output_idx < n_outputs) {
        float sum = bias[output_idx];
        for (int f = 0; f < n_filters; f++) {
            sum += input[batch_idx * n_filters + f] * 
                   weights[f * n_outputs + output_idx];
        }
        output[batch_idx * n_outputs + output_idx] = sum;
    }
}

typedef struct {
    float *d_conv_inputs;
    float *d_conv_outputs;
    float *d_norm_outputs;
    float *d_relu_outputs;
    float *d_pooled_output;
    float *d_final_output;
} Gradients;

static Gradients* create_gradients(Model *model, int batch_size) {
    Gradients *grads = (Gradients*)malloc(sizeof(Gradients));
    int seq_features = batch_size * model->sequence_length * model->n_filters;
    int input_features = batch_size * model->sequence_length * model->n_inputs;
    
    CHECK_CUDA(cudaMalloc(&grads->d_conv_inputs, input_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_conv_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_norm_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_relu_outputs, seq_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_pooled_output, 
        batch_size * model->n_filters * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_final_output, 
        batch_size * model->n_outputs * sizeof(float)));
    
    return grads;
}

__global__ void rms_norm_backward_kernel(
    const float *d_output, const float *input,
    float *d_input,
    int batch_size, int sequence_length, int n_filters
) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;
    
    if (seq_idx < sequence_length) {
        // Compute RMS for this position
        float sum_squared = 0.0f;
        for (int f = 0; f < n_filters; f++) {
            int idx = batch_idx * sequence_length * n_filters + 
                     seq_idx * n_filters + f;
            float val = input[idx];
            sum_squared += val * val;
        }
        float rms = sqrtf(sum_squared / n_filters + EPSILON);
        float inv_rms = 1.0f / rms;
        
        // Compute gradients
        for (int f = 0; f < n_filters; f++) {
            int idx = batch_idx * sequence_length * n_filters + 
                     seq_idx * n_filters + f;
            float x_i = input[idx];
            float dy_i = d_output[idx];
            
            // Gradient for input considering the RMS normalization
            float dx = inv_rms * (dy_i - 
                (x_i * inv_rms / n_filters) * 
                (x_i * dy_i * inv_rms));
            
            d_input[idx] = dx;
        }
    }
}

__global__ void dense_backward_kernel(
    const float *d_output,
    const float *input,
    const float *weights,
    float *d_input,
    float *d_weights,
    float *d_bias,
    int batch_size,
    int n_filters,
    int n_outputs
) {
    int batch_idx = blockIdx.x;
    int filter_idx = threadIdx.x;
    
    if (filter_idx < n_filters) {
        float d_input_val = 0.0f;
        for (int o = 0; o < n_outputs; o++) {
            float d_output_val = d_output[batch_idx * n_outputs + o];
            d_input_val += d_output_val * weights[filter_idx * n_outputs + o];
            
            atomicAdd(&d_weights[filter_idx * n_outputs + o],
                     d_output_val * input[batch_idx * n_filters + filter_idx]);
        }
        d_input[batch_idx * n_filters + filter_idx] = d_input_val;
    }
    
    if (batch_idx == 0 && filter_idx < n_outputs) {
        float d_bias_val = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            d_bias_val += d_output[b * n_outputs + filter_idx];
        }
        d_bias[filter_idx] = d_bias_val;
    }
}

__global__ void global_avg_pool_backward_kernel(
    const float *d_output,
    float *d_input,
    int batch_size,
    int sequence_length,
    int n_filters
) {
    int batch_idx = blockIdx.x;
    int filter_idx = threadIdx.x;
    
    if (filter_idx < n_filters) {
        float d_val = d_output[batch_idx * n_filters + filter_idx] / sequence_length;
        for (int s = 0; s < sequence_length; s++) {
            d_input[batch_idx * sequence_length * n_filters + 
                   s * n_filters + filter_idx] = d_val;
        }
    }
}

__global__ void relu_backward_kernel(
    const float *d_output,
    const float *input,
    float *d_input,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        d_input[idx] = input[idx] > 0 ? d_output[idx] : 0;
    }
}

__global__ void conv1d_backward_kernel(
    const float *d_output,
    const float *input,
    const float *weights,
    float *d_input,
    float *d_weights,
    float *d_bias,
    int batch_size,
    int sequence_length,
    int n_inputs,
    int n_filters,
    int kernel_size
) {
    int batch_idx = blockIdx.x;
    int filter_idx = blockIdx.y;
    int seq_idx = threadIdx.x;
    
    if (seq_idx < sequence_length) {
        if (batch_idx == 0 && seq_idx == 0) {
            float d_bias_val = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < sequence_length; s++) {
                    d_bias_val += d_output[b * sequence_length * n_filters +
                                         s * n_filters + filter_idx];
                }
            }
            d_bias[filter_idx] = d_bias_val;
        }
        
        for (int k = 0; k < kernel_size; k++) {
            int seq_pos = seq_idx - kernel_size/2 + k;
            if (seq_pos >= 0 && seq_pos < sequence_length) {
                for (int c = 0; c < n_inputs; c++) {
                    float d_output_val = d_output[
                        batch_idx * sequence_length * n_filters +
                        seq_idx * n_filters + filter_idx
                    ];
                    
                    atomicAdd(&d_input[
                        batch_idx * sequence_length * n_inputs +
                        seq_pos * n_inputs + c
                    ], d_output_val * weights[
                        filter_idx * n_inputs * kernel_size +
                        c * kernel_size + k
                    ]);
                    
                    atomicAdd(&d_weights[
                        filter_idx * n_inputs * kernel_size +
                        c * kernel_size + k
                    ], d_output_val * input[
                        batch_idx * sequence_length * n_inputs +
                        seq_pos * n_inputs + c
                    ]);
                }
            }
        }
    }
}

__global__ void clip_gradients_kernel(float *grads, int size, float max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = grads[idx];
        if (grad > max_norm) {
            grads[idx] = max_norm;
        } else if (grad < -max_norm) {
            grads[idx] = -max_norm;
        }
    }
}

__global__ void update_parameters_kernel(
    float *params,
    const float *grads,
    int size,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * grads[idx];
    }
}

static void forward_pass(
    Model *model,
    Activations *acts,
    const float *input,
    int batch_size,
    bool is_training
) {
    CHECK_CUDA(cudaMemcpy(acts->conv_inputs, input,
        batch_size * model->sequence_length * model->n_inputs * sizeof(float),
        cudaMemcpyHostToDevice));
    
    float *layer_input = acts->conv_inputs;
    
    for (int i = 0; i < model->n_conv_layers; i++) {
        ConvLayer *layer = &model->conv_layers[i];
        int in_channels = (i == 0) ? model->n_inputs : model->n_filters;
        
        if (i > 0) {
            CHECK_CUDA(cudaMemcpy(acts->residual_outputs, layer_input,
                batch_size * model->sequence_length * model->n_filters * sizeof(float),
                cudaMemcpyDeviceToDevice));
        }
        
        dim3 conv_blocks(batch_size, model->n_filters);
        dim3 conv_threads(model->sequence_length);
        conv1d_forward_kernel<<<conv_blocks, conv_threads>>>(
            layer_input,
            acts->conv_outputs,
            layer->weights,
            layer->bias,
            batch_size,
            model->sequence_length,
            in_channels,
            model->n_filters,
            model->kernel_size
        );
        
        // RMSNorm
        dim3 norm_blocks(batch_size);
        dim3 norm_threads(model->sequence_length);
        rms_norm_forward_kernel<<<norm_blocks, norm_threads>>>(
            acts->conv_outputs,
            acts->norm_outputs,
            batch_size,
            model->sequence_length,
            model->n_filters
        );
        
        int total_elements = batch_size * model->sequence_length * model->n_filters;
        int block_size = 256;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        
        relu_forward_kernel<<<num_blocks, block_size>>>(
            acts->norm_outputs,
            acts->relu_outputs,
            batch_size,
            model->sequence_length,
            model->n_filters
        );
        
        if (i > 0) {
            residual_add_kernel<<<num_blocks, block_size>>>(
                acts->relu_outputs,
                acts->residual_outputs,
                batch_size,
                model->sequence_length,
                model->n_filters
            );
        }
        
        layer_input = acts->relu_outputs;
    }
    
    global_avg_pool_kernel<<<batch_size, model->n_filters>>>(
        layer_input,
        acts->pooled_output,
        batch_size,
        model->sequence_length,
        model->n_filters
    );
    
    dense_forward_kernel<<<batch_size, model->n_outputs>>>(
        acts->pooled_output,
        acts->final_output,
        model->dense_layer.weights,
        model->dense_layer.bias,
        batch_size,
        model->n_filters,
        model->n_outputs
    );
}

static void backward_pass(
    Model *model,
    Activations *acts,
    Gradients *grads,
    const float *targets,
    int batch_size
) {
    float *d_layer_output = grads->d_pooled_output;
    
    dense_backward_kernel<<<batch_size, model->n_filters>>>(
        grads->d_final_output,
        acts->pooled_output,
        model->dense_layer.weights,
        grads->d_pooled_output,
        model->dense_layer.d_weights,
        model->dense_layer.d_bias,
        batch_size,
        model->n_filters,
        model->n_outputs
    );
    
    for (int i = model->n_conv_layers - 1; i >= 0; i--) {
        ConvLayer *layer = &model->conv_layers[i];
        int in_channels = (i == 0) ? model->n_inputs : model->n_filters;
        
        if (i == model->n_conv_layers - 1) {
            global_avg_pool_backward_kernel<<<batch_size, model->n_filters>>>(
                d_layer_output,
                grads->d_relu_outputs,
                batch_size,
                model->sequence_length,
                model->n_filters
            );
            d_layer_output = grads->d_relu_outputs;
        }
        
        int total_elements = batch_size * model->sequence_length * model->n_filters;
        int block_size = 256;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        
        relu_backward_kernel<<<num_blocks, block_size>>>(
            d_layer_output,
            acts->norm_outputs,
            grads->d_norm_outputs,
            total_elements
        );
        
        dim3 norm_blocks(batch_size);
        dim3 norm_threads(model->sequence_length);
        rms_norm_backward_kernel<<<norm_blocks, norm_threads>>>(
            grads->d_norm_outputs,
            acts->conv_outputs,
            grads->d_conv_outputs,
            batch_size,
            model->sequence_length,
            model->n_filters
        );
        
        dim3 conv_blocks(batch_size, model->n_filters);
        dim3 conv_threads(model->sequence_length);
        conv1d_backward_kernel<<<conv_blocks, conv_threads>>>(
            grads->d_conv_outputs,
            acts->conv_inputs,
            layer->weights,
            grads->d_conv_inputs,
            layer->d_weights,
            layer->d_bias,
            batch_size,
            model->sequence_length,
            in_channels,
            model->n_filters,
            model->kernel_size
        );
        
        d_layer_output = grads->d_conv_inputs;
    }
}

static void update_parameters(Model *model, float learning_rate) {
    int block_size = 256;
    
    for (int i = 0; i < model->n_conv_layers; i++) {
        ConvLayer *layer = &model->conv_layers[i];
        int in_channels = (i == 0) ? model->n_inputs : model->n_filters;
        
        int weights_size = model->n_filters * in_channels * model->kernel_size;
        int num_blocks = (weights_size + block_size - 1) / block_size;
        
        clip_gradients_kernel<<<num_blocks, block_size>>>(
            layer->d_weights, weights_size, MAX_GRAD_NORM);
        update_parameters_kernel<<<num_blocks, block_size>>>(
            layer->weights, layer->d_weights, weights_size, learning_rate);
        
        num_blocks = (model->n_filters + block_size - 1) / block_size;
        clip_gradients_kernel<<<num_blocks, block_size>>>(
            layer->d_bias, model->n_filters, MAX_GRAD_NORM);
        update_parameters_kernel<<<num_blocks, block_size>>>(
            layer->bias, layer->d_bias, model->n_filters, learning_rate);
    }
    
    int dense_weights_size = model->n_filters * model->n_outputs;
    int num_blocks = (dense_weights_size + block_size - 1) / block_size;
    
    clip_gradients_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.d_weights, dense_weights_size, MAX_GRAD_NORM);
    update_parameters_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.weights, model->dense_layer.d_weights,
        dense_weights_size, learning_rate);
    
    num_blocks = (model->n_outputs + block_size - 1) / block_size;
    clip_gradients_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.d_bias, model->n_outputs, MAX_GRAD_NORM);
    update_parameters_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.bias, model->dense_layer.d_bias,
        model->n_outputs, learning_rate);
}

__global__ void mse_loss_kernel(
    const float *predictions,
    const float *targets,
    float *loss,
    float *d_predictions,
    int batch_size,
    int n_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * n_outputs) {
        float diff = predictions[idx] - targets[idx];
        d_predictions[idx] = 2.0f * diff / (batch_size * n_outputs);
        atomicAdd(loss, diff * diff / (batch_size * n_outputs));
    }
}

static void train(
    Model *model,
    Dataset *data,
    int n_epochs,
    int batch_size,
    float learning_rate
) {
    Activations *acts = create_activations(model, batch_size);
    Gradients *grads = create_gradients(model, batch_size);
    
    float *d_batch_inputs, *d_batch_targets, *d_loss;
    int batch_input_size = batch_size * model->sequence_length * model->n_inputs;
    int batch_target_size = batch_size * model->n_outputs;
    
    CHECK_CUDA(cudaMalloc(&d_batch_inputs, batch_input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_targets, batch_target_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    
    int n_batches = data->n_sequences / batch_size;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < n_batches; batch++) {
            float *batch_inputs = (float*)malloc(batch_input_size * sizeof(float));
            float *batch_targets = (float*)malloc(batch_target_size * sizeof(float));
            
            for (int i = 0; i < batch_size; i++) {
                int seq_idx = batch * batch_size + i;
                
                for (int t = 0; t < model->sequence_length; t++) {
                    for (int f = 0; f < model->n_inputs; f++) {
                        batch_inputs[i * model->sequence_length * model->n_inputs +
                                   t * model->n_inputs + f] = 
                            data->inputs[seq_idx][t][f];
                    }
                }
                
                for (int f = 0; f < model->n_outputs; f++) {
                    batch_targets[i * model->n_outputs + f] = 
                        data->targets[seq_idx][f];
                }
            }
            
            CHECK_CUDA(cudaMemcpy(d_batch_inputs, batch_inputs,
                batch_input_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_batch_targets, batch_targets,
                batch_target_size * sizeof(float), cudaMemcpyHostToDevice));
            
            forward_pass(model, acts, d_batch_inputs, batch_size, true);
            
            CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
            int total_elements = batch_size * model->n_outputs;
            int block_size = 256;
            int num_blocks = (total_elements + block_size - 1) / block_size;
            
            mse_loss_kernel<<<num_blocks, block_size>>>(
                acts->final_output,
                d_batch_targets,
                d_loss,
                grads->d_final_output,
                batch_size,
                model->n_outputs
            );
            
            backward_pass(model, acts, grads, d_batch_targets, batch_size);
            update_parameters(model, learning_rate);
            
            float batch_loss;
            CHECK_CUDA(cudaMemcpy(&batch_loss, d_loss, sizeof(float),
                cudaMemcpyDeviceToHost));
            total_loss += batch_loss;
            
            free(batch_inputs);
            free(batch_targets);
        }
        
        printf("Epoch %d/%d - Loss: %f\n", epoch + 1, n_epochs,
               total_loss / n_batches);
    }
    
    CHECK_CUDA(cudaFree(d_batch_inputs));
    CHECK_CUDA(cudaFree(d_batch_targets));
    CHECK_CUDA(cudaFree(d_loss));
    free_activations(acts);
    free(grads);
}

static void free_model(Model *model) {
    for (int i = 0; i < model->n_conv_layers; i++) {
        ConvLayer *layer = &model->conv_layers[i];
        CHECK_CUDA(cudaFree(layer->weights));
        CHECK_CUDA(cudaFree(layer->bias));
        CHECK_CUDA(cudaFree(layer->d_weights));
        CHECK_CUDA(cudaFree(layer->d_bias));
    }
    
    free(model->conv_layers);
    
    CHECK_CUDA(cudaFree(model->dense_layer.weights));
    CHECK_CUDA(cudaFree(model->dense_layer.bias));
    CHECK_CUDA(cudaFree(model->dense_layer.d_weights));
    CHECK_CUDA(cudaFree(model->dense_layer.d_bias));
    
    free(model);
}

int main() {
    srand(time(NULL));
    
    Dataset* data = generate_data(1000, 32, 6, 4, 0.1);
    Model* model = create_model(data->sequence_length, data->n_inputs,
                              data->n_outputs);
    
    int n_epochs = 50;
    int batch_size = 32;
    float learning_rate = LEARNING_RATE;
    
    train(model, data, n_epochs, batch_size, learning_rate);
    
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_data.csv",
             localtime(&now));
    save_csv(fname, data);
    printf("Data saved to: %s\n", fname);
    
    free_dataset(data);
    free_model(model);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
}