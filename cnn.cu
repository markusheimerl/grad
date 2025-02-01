#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "data_open_loop.cuh"
#include <time.h>

#define KERNEL_SIZE 3
#define CONV1_FILTERS 64
#define CONV2_FILTERS 128
#define CONV3_FILTERS 256

typedef struct {
    float *conv1_weights, *conv1_bias;
    float *conv1_weight_grads, *conv1_bias_grads;
    float *conv1_output, *conv1_delta;
    
    float *conv2_weights, *conv2_bias;
    float *conv2_weight_grads, *conv2_bias_grads;
    float *conv2_output, *conv2_delta;
    
    float *conv3_weights, *conv3_bias;
    float *conv3_weight_grads, *conv3_bias_grads;
    float *conv3_output, *conv3_delta;
    
    float *dense_weights, *dense_bias;
    float *dense_weight_grads, *dense_bias_grads;
    float *pool_output, *pool_delta;
    
    int batch_size;
    int timesteps;
    int input_features;
    int output_features;
    float learning_rate;
} CNNModel;

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

__global__ void xavier_init_kernel(float* weights, int fan_in, int fan_out, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    
    if (idx < fan_in * fan_out) {
        float limit = sqrtf(6.0f / (fan_in + fan_out));
        weights[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * limit;
    }
}

__global__ void zero_init_kernel(float* arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = 0.0f;
    }
}

__global__ void conv1d_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int timesteps,
    int input_channels,
    int output_channels,
    int kernel_size
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int f = threadIdx.x;
    
    if (b < batch_size && t < timesteps-kernel_size+1 && f < output_channels) {
        float sum = bias[f];
        
        for (int k = 0; k < kernel_size; k++) {
            for (int c = 0; c < input_channels; c++) {
                int input_idx = b * (timesteps * input_channels) + 
                              (t + k) * input_channels + c;
                int weight_idx = f * (kernel_size * input_channels) + 
                               k * input_channels + c;
                sum += input[input_idx] * weights[weight_idx];
            }
        }
        
        int output_idx = b * ((timesteps-kernel_size+1) * output_channels) + 
                        t * output_channels + f;
        output[output_idx] = relu(sum);
    }
}

__global__ void global_avg_pooling_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int timesteps,
    int channels
) {
    int b = blockIdx.x;
    int c = threadIdx.x;
    
    if (b < batch_size && c < channels) {
        float sum = 0;
        for (int t = 0; t < timesteps; t++) {
            int idx = b * (timesteps * channels) + t * channels + c;
            sum += input[idx];
        }
        output[b * channels + c] = sum / timesteps;
    }
}

__global__ void dense_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int output_size
) {
    int b = blockIdx.x;
    int o = threadIdx.x;
    
    if (b < batch_size && o < output_size) {
        float sum = bias[o];
        for (int i = 0; i < input_size; i++) {
            sum += input[b * input_size + i] * weights[o * input_size + i];
        }
        output[b * output_size + o] = sum;
    }
}

__global__ void conv1d_backward_kernel(
    const float* input,
    const float* weights,
    const float* output,
    const float* output_grad,
    float* input_grad,
    float* weight_grad,
    float* bias_grad,
    int batch_size,
    int timesteps,
    int input_channels,
    int output_channels,
    int kernel_size
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int f = threadIdx.x;
    
    if (b < batch_size && t < timesteps-kernel_size+1 && f < output_channels) {
        int out_idx = b * ((timesteps-kernel_size+1) * output_channels) + 
                     t * output_channels + f;
        float grad = output_grad[out_idx] * relu_derivative(output[out_idx]);
        
        atomicAdd(&bias_grad[f], grad);
        
        for (int k = 0; k < kernel_size; k++) {
            for (int c = 0; c < input_channels; c++) {
                int input_idx = b * (timesteps * input_channels) + 
                              (t + k) * input_channels + c;
                int weight_idx = f * (kernel_size * input_channels) + 
                               k * input_channels + c;
                
                atomicAdd(&weight_grad[weight_idx], 
                         grad * input[input_idx]);
                atomicAdd(&input_grad[input_idx], 
                         grad * weights[weight_idx]);
            }
        }
    }
}

__global__ void global_avg_pooling_backward_kernel(
    const float* output_grad,
    float* input_grad,
    int batch_size,
    int timesteps,
    int channels
) {
    int b = blockIdx.x;
    int c = threadIdx.x;
    
    if (b < batch_size && c < channels) {
        float grad = output_grad[b * channels + c] / timesteps;
        for (int t = 0; t < timesteps; t++) {
            int idx = b * (timesteps * channels) + t * channels + c;
            input_grad[idx] = grad;
        }
    }
}

__global__ void dense_backward_kernel(
    const float* input,
    const float* weights,
    const float* output_grad,
    float* input_grad,
    float* weight_grad,
    float* bias_grad,
    int batch_size,
    int input_size,
    int output_size
) {
    int b = blockIdx.x;
    int o = threadIdx.x;
    
    if (b < batch_size && o < output_size) {
        float grad = output_grad[b * output_size + o];
        
        atomicAdd(&bias_grad[o], grad);
        
        for (int i = 0; i < input_size; i++) {
            atomicAdd(&weight_grad[o * input_size + i], 
                     grad * input[b * input_size + i]);
            atomicAdd(&input_grad[b * input_size + i], 
                     grad * weights[o * input_size + i]);
        }
    }
}

__global__ void update_weights_kernel(
    float* weights,
    const float* weight_grad,
    float learning_rate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * weight_grad[idx];
    }
}

CNNModel* init_model(int batch_size, int timesteps, int input_features, int output_features, float learning_rate) {
    CNNModel* model = (CNNModel*)malloc(sizeof(CNNModel));
    
    model->batch_size = batch_size;
    model->timesteps = timesteps;
    model->input_features = input_features;
    model->output_features = output_features;
    model->learning_rate = learning_rate;
    
    int conv1_out_time = timesteps - KERNEL_SIZE + 1;
    int conv2_out_time = conv1_out_time - KERNEL_SIZE + 1;
    int conv3_out_time = conv2_out_time - KERNEL_SIZE + 1;
    
    cudaMalloc(&model->conv1_weights, KERNEL_SIZE * input_features * CONV1_FILTERS * sizeof(float));
    cudaMalloc(&model->conv1_bias, CONV1_FILTERS * sizeof(float));
    cudaMalloc(&model->conv1_weight_grads, KERNEL_SIZE * input_features * CONV1_FILTERS * sizeof(float));
    cudaMalloc(&model->conv1_bias_grads, CONV1_FILTERS * sizeof(float));
    cudaMalloc(&model->conv1_output, batch_size * conv1_out_time * CONV1_FILTERS * sizeof(float));
    cudaMalloc(&model->conv1_delta, batch_size * conv1_out_time * CONV1_FILTERS * sizeof(float));
    
    cudaMalloc(&model->conv2_weights, KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS * sizeof(float));
    cudaMalloc(&model->conv2_bias, CONV2_FILTERS * sizeof(float));
    cudaMalloc(&model->conv2_weight_grads, KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS * sizeof(float));
    cudaMalloc(&model->conv2_bias_grads, CONV2_FILTERS * sizeof(float));
    cudaMalloc(&model->conv2_output, batch_size * conv2_out_time * CONV2_FILTERS * sizeof(float));
    cudaMalloc(&model->conv2_delta, batch_size * conv2_out_time * CONV2_FILTERS * sizeof(float));
    
    cudaMalloc(&model->conv3_weights, KERNEL_SIZE * CONV2_FILTERS * CONV3_FILTERS * sizeof(float));
    cudaMalloc(&model->conv3_bias, CONV3_FILTERS * sizeof(float));
    cudaMalloc(&model->conv3_weight_grads, KERNEL_SIZE * CONV2_FILTERS * CONV3_FILTERS * sizeof(float));
    cudaMalloc(&model->conv3_bias_grads, CONV3_FILTERS * sizeof(float));
    cudaMalloc(&model->conv3_output, batch_size * conv3_out_time * CONV3_FILTERS * sizeof(float));
    cudaMalloc(&model->conv3_delta, batch_size * conv3_out_time * CONV3_FILTERS * sizeof(float));
    
    cudaMalloc(&model->dense_weights, CONV3_FILTERS * output_features * sizeof(float));
    cudaMalloc(&model->dense_bias, output_features * sizeof(float));
    cudaMalloc(&model->dense_weight_grads, CONV3_FILTERS * output_features * sizeof(float));
    cudaMalloc(&model->dense_bias_grads, output_features * sizeof(float));
    cudaMalloc(&model->pool_output, batch_size * CONV3_FILTERS * sizeof(float));
    cudaMalloc(&model->pool_delta, batch_size * CONV3_FILTERS * sizeof(float));
    
    int threads = 256;
    xavier_init_kernel<<<(KERNEL_SIZE * input_features * CONV1_FILTERS + threads - 1) / threads, threads>>>(
        model->conv1_weights, KERNEL_SIZE * input_features, CONV1_FILTERS, time(NULL));
    xavier_init_kernel<<<(KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS + threads - 1) / threads, threads>>>(
        model->conv2_weights, KERNEL_SIZE * CONV1_FILTERS, CONV2_FILTERS, time(NULL) + 1);
    xavier_init_kernel<<<(KERNEL_SIZE * CONV2_FILTERS * CONV3_FILTERS + threads - 1) / threads, threads>>>(
        model->conv3_weights, KERNEL_SIZE * CONV2_FILTERS, CONV3_FILTERS, time(NULL) + 2);
    xavier_init_kernel<<<(CONV3_FILTERS * output_features + threads - 1) / threads, threads>>>(
        model->dense_weights, CONV3_FILTERS, output_features, time(NULL) + 3);
    
    zero_init_kernel<<<(CONV1_FILTERS + threads - 1) / threads, threads>>>(model->conv1_bias, CONV1_FILTERS);
    zero_init_kernel<<<(CONV2_FILTERS + threads - 1) / threads, threads>>>(model->conv2_bias, CONV2_FILTERS);
    zero_init_kernel<<<(CONV3_FILTERS + threads - 1) / threads, threads>>>(model->conv3_bias, CONV3_FILTERS);
    zero_init_kernel<<<(output_features + threads - 1) / threads, threads>>>(model->dense_bias, output_features);
    
    return model;
}

void forward(CNNModel* model, const float* input, float* output) {
    int conv1_out_time = model->timesteps - KERNEL_SIZE + 1;
    int conv2_out_time = conv1_out_time - KERNEL_SIZE + 1;
    int conv3_out_time = conv2_out_time - KERNEL_SIZE + 1;
    
    dim3 conv1_grid(model->batch_size, conv1_out_time);
    conv1d_forward_kernel<<<conv1_grid, CONV1_FILTERS>>>(
        input, model->conv1_weights, model->conv1_bias, model->conv1_output,
        model->batch_size, model->timesteps, model->input_features, 
        CONV1_FILTERS, KERNEL_SIZE);
    
    dim3 conv2_grid(model->batch_size, conv2_out_time);
    conv1d_forward_kernel<<<conv2_grid, CONV2_FILTERS>>>(
        model->conv1_output, model->conv2_weights, model->conv2_bias, model->conv2_output,
        model->batch_size, conv1_out_time, CONV1_FILTERS, 
        CONV2_FILTERS, KERNEL_SIZE);
    
    dim3 conv3_grid(model->batch_size, conv3_out_time);
    conv1d_forward_kernel<<<conv3_grid, CONV3_FILTERS>>>(
        model->conv2_output, model->conv3_weights, model->conv3_bias, model->conv3_output,
        model->batch_size, conv2_out_time, CONV2_FILTERS, 
        CONV3_FILTERS, KERNEL_SIZE);
    
    global_avg_pooling_forward_kernel<<<model->batch_size, CONV3_FILTERS>>>(
        model->conv3_output, model->pool_output,
        model->batch_size, conv3_out_time, CONV3_FILTERS);
    
    dense_forward_kernel<<<model->batch_size, model->output_features>>>(
        model->pool_output, model->dense_weights, model->dense_bias, output,
        model->batch_size, CONV3_FILTERS, model->output_features);
}

void backward(CNNModel* model, const float* input, const float* output_grad) {
    int conv1_out_time = model->timesteps - KERNEL_SIZE + 1;
    int conv2_out_time = conv1_out_time - KERNEL_SIZE + 1;
    int conv3_out_time = conv2_out_time - KERNEL_SIZE + 1;
    
    dense_backward_kernel<<<model->batch_size, model->output_features>>>(
        model->pool_output, model->dense_weights, output_grad,
        model->pool_delta, model->dense_weight_grads, model->dense_bias_grads,
        model->batch_size, CONV3_FILTERS, model->output_features);
    
    global_avg_pooling_backward_kernel<<<model->batch_size, CONV3_FILTERS>>>(
        model->pool_delta, model->conv3_delta,
        model->batch_size, conv3_out_time, CONV3_FILTERS);
    
    dim3 conv3_grid(model->batch_size, conv3_out_time);
    conv1d_backward_kernel<<<conv3_grid, CONV3_FILTERS>>>(
        model->conv2_output, model->conv3_weights, model->conv3_output,
        model->conv3_delta, model->conv2_delta, model->conv3_weight_grads,
        model->conv3_bias_grads, model->batch_size, conv2_out_time,
        CONV2_FILTERS, CONV3_FILTERS, KERNEL_SIZE);
    
    dim3 conv2_grid(model->batch_size, conv2_out_time);
    conv1d_backward_kernel<<<conv2_grid, CONV2_FILTERS>>>(
        model->conv1_output, model->conv2_weights, model->conv2_output,
        model->conv2_delta, model->conv1_delta, model->conv2_weight_grads,
        model->conv2_bias_grads, model->batch_size, conv1_out_time,
        CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE);
    
    dim3 conv1_grid(model->batch_size, conv1_out_time);
    conv1d_backward_kernel<<<conv1_grid, CONV1_FILTERS>>>(
        input, model->conv1_weights, model->conv1_output,
        model->conv1_delta, NULL, model->conv1_weight_grads,
        model->conv1_bias_grads, model->batch_size, model->timesteps,
        model->input_features, CONV1_FILTERS, KERNEL_SIZE);
    
    int threads = 256;
    update_weights_kernel<<<(KERNEL_SIZE * model->input_features * CONV1_FILTERS + threads - 1) / threads, threads>>>(
        model->conv1_weights, model->conv1_weight_grads, model->learning_rate,
        KERNEL_SIZE * model->input_features * CONV1_FILTERS);
    update_weights_kernel<<<(CONV1_FILTERS + threads - 1) / threads, threads>>>(
        model->conv1_bias, model->conv1_bias_grads, model->learning_rate, CONV1_FILTERS);
        
    update_weights_kernel<<<(KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS + threads - 1) / threads, threads>>>(
        model->conv2_weights, model->conv2_weight_grads, model->learning_rate,
        KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS);
    update_weights_kernel<<<(CONV2_FILTERS + threads - 1) / threads, threads>>>(
        model->conv2_bias, model->conv2_bias_grads, model->learning_rate, CONV2_FILTERS);
        
    update_weights_kernel<<<(KERNEL_SIZE * CONV2_FILTERS * CONV3_FILTERS + threads - 1) / threads, threads>>>(
        model->conv3_weights, model->conv3_weight_grads, model->learning_rate,
        KERNEL_SIZE * CONV2_FILTERS * CONV3_FILTERS);
    update_weights_kernel<<<(CONV3_FILTERS + threads - 1) / threads, threads>>>(
        model->conv3_bias, model->conv3_bias_grads, model->learning_rate, CONV3_FILTERS);
        
    update_weights_kernel<<<(CONV3_FILTERS * model->output_features + threads - 1) / threads, threads>>>(
        model->dense_weights, model->dense_weight_grads, model->learning_rate,
        CONV3_FILTERS * model->output_features);
    update_weights_kernel<<<(model->output_features + threads - 1) / threads, threads>>>(
        model->dense_bias, model->dense_bias_grads, model->learning_rate, model->output_features);
}

float compute_mse(float* pred, float* target, int size) {
    float mse = 0.0f;
    for(int i = 0; i < size; i++) {
        float diff = pred[i] - target[i];
        mse += diff * diff;
    }
    return mse / size;
}

int main() {
    srand(time(NULL));
    
    // Generate synthetic data
    OpenLoopData* data = generate_open_loop_data(100, 50, 3, 2, 0.1);
    
    // Save data with timestamp
    time_t current_time = time(NULL);
    struct tm* timeinfo = localtime(&current_time);
    char data_fname[64];
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_open_loop_data.csv", timeinfo);
    
    save_open_loop_csv(data_fname, data);
    printf("Data saved to: %s\n", data_fname);
    
    // Print example sequence
    printf("\nExample sequence (first 5 timesteps of first sequence):\n");
    printf("Timestep | Input Features\n");
    printf("---------+---------------\n");
    
    for(int t = 0; t < 5; t++) {
        printf("%8d | ", t);
        for(int f = 0; f < data->input_features; f++) {
            printf("%6.3f ", data->windows[0][t][f]);
        }
        printf("\n");
    }
    
    printf("\nTarget outputs for this sequence:\n");
    for(int f = 0; f < data->output_features; f++) {
        printf("Output %d: %.3f\n", f, data->outputs[0][f]);
    }
    
    // Initialize model
    CNNModel* model = init_model(1, 50, 3, 2, 0.001f);
    
    // Allocate GPU memory for input and output
    float *d_input, *d_output, *d_target;
    cudaMalloc(&d_input, 50 * 3 * sizeof(float));
    cudaMalloc(&d_output, 2 * sizeof(float));
    cudaMalloc(&d_target, 2 * sizeof(float));
    
    // Training loop
    int epochs = 200;
    float* h_output = (float*)malloc(2 * sizeof(float));
    float* h_target = (float*)malloc(2 * sizeof(float));
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for(int i = 0; i < data->n; i++) {
            // Copy input sequence to GPU
            float* h_input = (float*)malloc(50 * 3 * sizeof(float));
            for(int t = 0; t < 50; t++) {
                for(int f = 0; f < 3; f++) {
                    h_input[t * 3 + f] = (float)data->windows[i][t][f];
                }
            }
            cudaMemcpy(d_input, h_input, 50 * 3 * sizeof(float), cudaMemcpyHostToDevice);
            free(h_input);
            
            // Copy target to GPU
            for(int f = 0; f < 2; f++) {
                h_target[f] = (float)data->outputs[i][f];
            }
            cudaMemcpy(d_target, h_target, 2 * sizeof(float), cudaMemcpyHostToDevice);
            
            // Forward and backward pass
            forward(model, d_input, d_output);
            backward(model, d_input, d_target);
            
            // Compute loss
            cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost);
            epoch_loss += compute_mse(h_output, h_target, 2);
        }
        
        epoch_loss /= data->n;
        if(epoch % 10 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, epoch_loss);
        }
    }
    
    // Save model with timestamp
    char model_fname[64];
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_cnn.bin", timeinfo);
    
    FILE* fp = fopen(model_fname, "wb");
    if(fp) {
        fwrite(&model->batch_size, sizeof(int), 1, fp);
        fwrite(&model->timesteps, sizeof(int), 1, fp);
        fwrite(&model->input_features, sizeof(int), 1, fp);
        fwrite(&model->output_features, sizeof(int), 1, fp);
        fwrite(&model->learning_rate, sizeof(float), 1, fp);
        fclose(fp);
    }
    printf("Model saved to: %s\n", model_fname);
    
    // Cleanup
    free(h_output);
    free(h_target);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_target);
    free_open_loop_data(data);
    
    return 0;
}