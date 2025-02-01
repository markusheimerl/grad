#include "data_open_loop.cuh"
#include <time.h>

#define LEARNING_RATE 0.001
#define NUM_EPOCHS 1000
#define NUM_FILTERS 16
#define KERNEL_SIZE 5
#define BLOCK_SIZE 256
#define BATCH_SIZE 32

typedef struct {
    // Host memory
    double ***conv_filters;  // [num_filters][kernel_size][input_features]
    double *conv_bias;       // [num_filters]
    double **fc_weights;     // [output_features][num_filters]
    double *fc_bias;         // [output_features]
    
    // Device memory
    double *d_conv_filters;  // flattened [num_filters * kernel_size * input_features]
    double *d_conv_bias;
    double *d_fc_weights;    // flattened [output_features * num_filters]
    double *d_fc_bias;
    
    int num_filters;
    int kernel_size;
} ConvModel;

// Helper functions for CPU memory
static double** malloc_2d(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        arr[i] = (double*)calloc(cols, sizeof(double));
    }
    return arr;
}

static void free_2d(double** arr, int rows) {
    for(int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

static double he_init() {
    double rand_normal = sqrt(-2.0 * log((double)rand() / RAND_MAX)) * 
                        cos(2.0 * M_PI * (double)rand() / RAND_MAX);
    return rand_normal * sqrt(2.0 / KERNEL_SIZE);
}

// CUDA kernels
__global__ void conv_forward_kernel(
    const double* input,      // [batch_size][window_size][input_features]
    const double* conv_filters,
    const double* conv_bias,
    double* conv_out,         // [batch_size][conv_out_size][num_filters]
    int window_size,
    int input_features,
    int num_filters,
    int kernel_size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int conv_out_size = window_size - kernel_size + 1;
    int total_elements = batch_size * conv_out_size * num_filters;
    
    if(idx < total_elements) {
        int batch_idx = idx / (conv_out_size * num_filters);
        int remainder = idx % (conv_out_size * num_filters);
        int t = remainder / num_filters;
        int f = remainder % num_filters;
        
        double sum = conv_bias[f];
        int input_offset = batch_idx * window_size * input_features;
        
        for(int k = 0; k < kernel_size; k++) {
            for(int c = 0; c < input_features; c++) {
                sum += conv_filters[f * kernel_size * input_features + k * input_features + c] * 
                       input[input_offset + (t + k) * input_features + c];
            }
        }
        conv_out[idx] = sum > 0 ? sum : 0; // ReLU
    }
}

__global__ void pooling_kernel(
    const double* conv_out,   // [batch_size][conv_out_size][num_filters]
    double* pooled,           // [batch_size][num_filters]
    int conv_out_size,
    int num_filters,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_filters;
    
    if(idx < total_elements) {
        int batch_idx = idx / num_filters;
        int f = idx % num_filters;
        
        double sum = 0;
        int conv_out_offset = batch_idx * conv_out_size * num_filters;
        
        for(int t = 0; t < conv_out_size; t++) {
            sum += conv_out[conv_out_offset + t * num_filters + f];
        }
        pooled[idx] = sum / conv_out_size;
    }
}

__global__ void fc_forward_kernel(
    const double* pooled,     // [batch_size][num_filters]
    const double* fc_weights,
    const double* fc_bias,
    double* output,           // [batch_size][output_features]
    int num_filters,
    int output_features,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_features;
    
    if(idx < total_elements) {
        int batch_idx = idx / output_features;
        int out_idx = idx % output_features;
        
        double sum = fc_bias[out_idx];
        int pooled_offset = batch_idx * num_filters;
        
        for(int f = 0; f < num_filters; f++) {
            sum += fc_weights[out_idx * num_filters + f] * pooled[pooled_offset + f];
        }
        output[idx] = sum;
    }
}

__global__ void backward_kernel(
    const double* input,
    const double* conv_out,
    const double* pooled,
    const double* fc_weights,
    const double* y_pred,
    const double* y_true,
    double* grad_conv_filters,
    double* grad_conv_bias,
    double* grad_fc_weights,
    double* grad_fc_bias,
    int window_size,
    int input_features,
    int num_filters,
    int kernel_size,
    int output_features,
    int conv_out_size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_features;
    
    if(idx < total_elements) {
        int batch_idx = idx / output_features;
        int out_idx = idx % output_features;
        
        double d_output = 2.0 * (y_pred[idx] - y_true[idx]) / (output_features * batch_size);
        
        // FC layer gradients
        atomicAdd(&grad_fc_bias[out_idx], d_output);
        
        int pooled_offset = batch_idx * num_filters;
        for(int f = 0; f < num_filters; f++) {
            atomicAdd(&grad_fc_weights[out_idx * num_filters + f], 
                     d_output * pooled[pooled_offset + f]);
            
            // Propagate to conv layer
            double d_pool = d_output * fc_weights[out_idx * num_filters + f] / conv_out_size;
            
            int conv_out_offset = batch_idx * conv_out_size * num_filters;
            int input_offset = batch_idx * window_size * input_features;
            
            for(int t = 0; t < conv_out_size; t++) {
                double d_relu = conv_out[conv_out_offset + t * num_filters + f] > 0 ? 1.0 : 0.0;
                double d_conv = d_pool * d_relu;
                
                atomicAdd(&grad_conv_bias[f], d_conv);
                
                for(int k = 0; k < kernel_size; k++) {
                    for(int c = 0; c < input_features; c++) {
                        atomicAdd(&grad_conv_filters[f * kernel_size * input_features + k * input_features + c],
                                d_conv * input[input_offset + (t + k) * input_features + c]);
                    }
                }
            }
        }
    }
}

__global__ void update_parameters_kernel(
    double* param,
    const double* grad,
    int size,
    double learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        param[idx] -= learning_rate * grad[idx];
    }
}

// Initialize model parameters
ConvModel* init_model(int input_features, int output_features) {
    ConvModel* model = (ConvModel*)malloc(sizeof(ConvModel));
    model->num_filters = NUM_FILTERS;
    model->kernel_size = KERNEL_SIZE;
    
    // Allocate and initialize host memory
    int conv_filters_size = NUM_FILTERS * KERNEL_SIZE * input_features;
    double* conv_filters_flat = (double*)malloc(conv_filters_size * sizeof(double));
    
    model->conv_filters = (double***)malloc(NUM_FILTERS * sizeof(double**));
    for(int f = 0; f < NUM_FILTERS; f++) {
        model->conv_filters[f] = malloc_2d(KERNEL_SIZE, input_features);
        for(int k = 0; k < KERNEL_SIZE; k++) {
            for(int c = 0; c < input_features; c++) {
                double val = he_init();
                model->conv_filters[f][k][c] = val;
                conv_filters_flat[f * KERNEL_SIZE * input_features + k * input_features + c] = val;
            }
        }
    }
    
    model->conv_bias = (double*)calloc(NUM_FILTERS, sizeof(double));
    
    int fc_weights_size = output_features * NUM_FILTERS;
    double* fc_weights_flat = (double*)malloc(fc_weights_size * sizeof(double));
    
    model->fc_weights = malloc_2d(output_features, NUM_FILTERS);
    for(int i = 0; i < output_features; i++) {
        for(int f = 0; f < NUM_FILTERS; f++) {
            double val = he_init();
            model->fc_weights[i][f] = val;
            fc_weights_flat[i * NUM_FILTERS + f] = val;
        }
    }
    
    model->fc_bias = (double*)calloc(output_features, sizeof(double));
    
    // Allocate and copy to device memory
    cudaMalloc(&model->d_conv_filters, conv_filters_size * sizeof(double));
    cudaMalloc(&model->d_conv_bias, NUM_FILTERS * sizeof(double));
    cudaMalloc(&model->d_fc_weights, fc_weights_size * sizeof(double));
    cudaMalloc(&model->d_fc_bias, output_features * sizeof(double));
    
    cudaMemcpy(model->d_conv_filters, conv_filters_flat, 
               conv_filters_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(model->d_conv_bias, model->conv_bias, 
               NUM_FILTERS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(model->d_fc_weights, fc_weights_flat, 
               fc_weights_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(model->d_fc_bias, model->fc_bias, 
               output_features * sizeof(double), cudaMemcpyHostToDevice);
    
    free(conv_filters_flat);
    free(fc_weights_flat);
    
    return model;
}

void update_parameters(ConvModel* model, double** x, double* y_pred, double* y_true,
    int window_size, int input_features, int output_features,
    double* d_input, double* d_conv_out, double* d_pooled,
    int batch_size) {

    int conv_out_size = window_size - model->kernel_size + 1;
    int conv_filters_size = NUM_FILTERS * KERNEL_SIZE * input_features;
    int fc_weights_size = output_features * NUM_FILTERS;

    // Allocate gradient buffers
    double *d_grad_conv_filters, *d_grad_conv_bias, *d_grad_fc_weights, *d_grad_fc_bias;
    cudaMalloc(&d_grad_conv_filters, conv_filters_size * sizeof(double));
    cudaMalloc(&d_grad_conv_bias, NUM_FILTERS * sizeof(double));
    cudaMalloc(&d_grad_fc_weights, fc_weights_size * sizeof(double));
    cudaMalloc(&d_grad_fc_bias, output_features * sizeof(double));

    cudaMemset(d_grad_conv_filters, 0, conv_filters_size * sizeof(double));
    cudaMemset(d_grad_conv_bias, 0, NUM_FILTERS * sizeof(double));
    cudaMemset(d_grad_fc_weights, 0, fc_weights_size * sizeof(double));
    cudaMemset(d_grad_fc_bias, 0, output_features * sizeof(double));

    // Copy predictions and targets to device
    double *d_y_pred, *d_y_true;
    cudaMalloc(&d_y_pred, batch_size * output_features * sizeof(double));  // Modified for batch
    cudaMalloc(&d_y_true, batch_size * output_features * sizeof(double));  // Modified for batch
    cudaMemcpy(d_y_pred, y_pred, batch_size * output_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, y_true, batch_size * output_features * sizeof(double), cudaMemcpyHostToDevice);

    // Compute gradients
    int threads = BLOCK_SIZE;
    int blocks = (batch_size * output_features + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Modified for batch

    backward_kernel<<<blocks, threads>>>(
    d_input, d_conv_out, d_pooled, model->d_fc_weights,
    d_y_pred, d_y_true, d_grad_conv_filters, d_grad_conv_bias,
    d_grad_fc_weights, d_grad_fc_bias, window_size, input_features,
    NUM_FILTERS, KERNEL_SIZE, output_features, conv_out_size, batch_size
    );

    // Update parameters
    update_parameters_kernel<<<(conv_filters_size + BLOCK_SIZE - 1) / BLOCK_SIZE, threads>>>(
    model->d_conv_filters, d_grad_conv_filters, conv_filters_size, LEARNING_RATE);

    update_parameters_kernel<<<(NUM_FILTERS + BLOCK_SIZE - 1) / BLOCK_SIZE, threads>>>(
    model->d_conv_bias, d_grad_conv_bias, NUM_FILTERS, LEARNING_RATE);

    update_parameters_kernel<<<(fc_weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE, threads>>>(
    model->d_fc_weights, d_grad_fc_weights, fc_weights_size, LEARNING_RATE);

    update_parameters_kernel<<<(output_features + BLOCK_SIZE - 1) / BLOCK_SIZE, threads>>>(
    model->d_fc_bias, d_grad_fc_bias, output_features, LEARNING_RATE);

    // Cleanup
    cudaFree(d_grad_conv_filters);
    cudaFree(d_grad_conv_bias);
    cudaFree(d_grad_fc_weights);
    cudaFree(d_grad_fc_bias);
    cudaFree(d_y_pred);
    cudaFree(d_y_true);
}

void free_model(ConvModel* model) {
    // Free host memory
    for(int f = 0; f < NUM_FILTERS; f++) {
        free_2d(model->conv_filters[f], KERNEL_SIZE);
    }
    free(model->conv_filters);
    free(model->conv_bias);
    free_2d(model->fc_weights, NUM_FILTERS);
    free(model->fc_bias);
    
    // Free device memory
    cudaFree(model->d_conv_filters);
    cudaFree(model->d_conv_bias);
    cudaFree(model->d_fc_weights);
    cudaFree(model->d_fc_bias);
    
    free(model);
}

// Mean squared error loss
static double compute_loss(double* y_pred, double* y_true, int output_features) {
    double loss = 0.0;
    for(int i = 0; i < output_features; i++) {
        double diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / output_features;
}

void train_batch(
    ConvModel* model,
    OpenLoopData* data,
    int batch_start,
    int batch_size,
    double* d_input,
    double* d_conv_out,
    double* d_pooled,
    double* d_y_pred,
    double* d_y_true
) {
    // Prepare batch data
    double* batch_input = (double*)malloc(
        batch_size * data->window_size * data->input_features * sizeof(double));
    double* batch_output = (double*)malloc(
        batch_size * data->output_features * sizeof(double));
    
    for(int i = 0; i < batch_size; i++) {
        int idx = batch_start + i;
        // Flatten input
        for(int t = 0; t < data->window_size; t++) {
            for(int f = 0; f < data->input_features; f++) {
                batch_input[i * data->window_size * data->input_features + 
                          t * data->input_features + f] = 
                    data->windows[idx][t][f];
            }
        }
        // Copy output
        memcpy(&batch_output[i * data->output_features],
               data->outputs[idx],
               data->output_features * sizeof(double));
    }
    
    // Copy batch data to GPU
    cudaMemcpy(d_input, batch_input,
               batch_size * data->window_size * data->input_features * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, batch_output,
               batch_size * data->output_features * sizeof(double),
               cudaMemcpyHostToDevice);
    
    // Forward pass
    int conv_out_size = data->window_size - KERNEL_SIZE + 1;
    
    int conv_threads = BLOCK_SIZE;
    int conv_blocks = (batch_size * conv_out_size * NUM_FILTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv_forward_kernel<<<conv_blocks, conv_threads>>>(
        d_input, model->d_conv_filters, model->d_conv_bias, d_conv_out,
        data->window_size, data->input_features, NUM_FILTERS, KERNEL_SIZE, batch_size);
    
    int pool_threads = BLOCK_SIZE;
    int pool_blocks = (batch_size * NUM_FILTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pooling_kernel<<<pool_blocks, pool_threads>>>(
        d_conv_out, d_pooled, conv_out_size, NUM_FILTERS, batch_size);
    
    int fc_threads = BLOCK_SIZE;
    int fc_blocks = (batch_size * data->output_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fc_forward_kernel<<<fc_blocks, fc_threads>>>(
        d_pooled, model->d_fc_weights, model->d_fc_bias, d_y_pred,
        NUM_FILTERS, data->output_features, batch_size);
    
    // Backward pass and parameter update
    update_parameters(model, data->windows[batch_start], d_y_pred, d_y_true,
                     data->window_size, data->input_features, data->output_features,
                     d_input, d_conv_out, d_pooled, batch_size);
    
    free(batch_input);
    free(batch_output);
}

int main() {
    srand(time(NULL));
    
    // Generate synthetic data
    OpenLoopData* data = generate_open_loop_data(1000, 50, 3, 2, 0.1);
    
    // Save data to CSV
    time_t current_time = time(NULL);
    struct tm* timeinfo = localtime(&current_time);
    char data_fname[64];
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_open_loop_data.csv", timeinfo);
    save_open_loop_csv(data_fname, data);
    printf("Data saved to: %s\n", data_fname);

    // Initialize model
    ConvModel* model = init_model(data->input_features, data->output_features);
    
    // Allocate GPU memory for batch processing
    double *d_input, *d_conv_out, *d_pooled, *d_y_pred, *d_y_true;
    cudaMalloc(&d_input, BATCH_SIZE * data->window_size * data->input_features * sizeof(double));
    cudaMalloc(&d_conv_out, BATCH_SIZE * (data->window_size - KERNEL_SIZE + 1) * 
               NUM_FILTERS * sizeof(double));
    cudaMalloc(&d_pooled, BATCH_SIZE * NUM_FILTERS * sizeof(double));
    cudaMalloc(&d_y_pred, BATCH_SIZE * data->output_features * sizeof(double));
    cudaMalloc(&d_y_true, BATCH_SIZE * data->output_features * sizeof(double));
    
    printf("Training started...\n");
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        
        // Process data in batches
        for(int batch_start = 0; batch_start < data->n; batch_start += BATCH_SIZE) {
            int batch_size = min(BATCH_SIZE, data->n - batch_start);
            
            train_batch(model, data, batch_start, batch_size,
                       d_input, d_conv_out, d_pooled, d_y_pred, d_y_true);
            
            // Compute loss for logging
            double* batch_pred = (double*)malloc(
                batch_size * data->output_features * sizeof(double));
            cudaMemcpy(batch_pred, d_y_pred,
                      batch_size * data->output_features * sizeof(double),
                      cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < batch_size; i++) {
                epoch_loss += compute_loss(
                    &batch_pred[i * data->output_features],
                    data->outputs[batch_start + i],
                    data->output_features);
            }
            
            free(batch_pred);
        }
        
        epoch_loss /= data->n;
        if(epoch % 2 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, epoch_loss);
        }
    }
    
    // Print example predictions
    printf("\nExample predictions:\n");
    double* batch_pred = (double*)malloc(
        BATCH_SIZE * data->output_features * sizeof(double));
    
    for(int i = 0; i < min(5, data->n); i++) {
        // Prepare single sample as a batch
        double* single_input = (double*)malloc(
            data->window_size * data->input_features * sizeof(double));
        
        for(int t = 0; t < data->window_size; t++) {
            for(int f = 0; f < data->input_features; f++) {
                single_input[t * data->input_features + f] = 
                    data->windows[i][t][f];
            }
        }
        
        cudaMemcpy(d_input, single_input,
                   data->window_size * data->input_features * sizeof(double),
                   cudaMemcpyHostToDevice);
        
        // Forward pass for single sample
        int conv_out_size = data->window_size - KERNEL_SIZE + 1;
        
        conv_forward_kernel<<<1, BLOCK_SIZE>>>(
            d_input, model->d_conv_filters, model->d_conv_bias, d_conv_out,
            data->window_size, data->input_features, NUM_FILTERS, KERNEL_SIZE, 1);
        
        pooling_kernel<<<1, BLOCK_SIZE>>>(
            d_conv_out, d_pooled, conv_out_size, NUM_FILTERS, 1);
        
        fc_forward_kernel<<<1, BLOCK_SIZE>>>(
            d_pooled, model->d_fc_weights, model->d_fc_bias, d_y_pred,
            NUM_FILTERS, data->output_features, 1);
        
        cudaMemcpy(batch_pred, d_y_pred,
                   data->output_features * sizeof(double),
                   cudaMemcpyDeviceToHost);
        
        printf("Sample %d:\n", i);
        printf("  True:");
        for(int j = 0; j < data->output_features; j++) {
            printf(" %.6f", data->outputs[i][j]);
        }
        printf("\n  Pred:");
        for(int j = 0; j < data->output_features; j++) {
            printf(" %.6f", batch_pred[j]);
        }
        printf("\n");
        
        free(single_input);
    }
    
    // Cleanup
    free(batch_pred);
    cudaFree(d_input);
    cudaFree(d_conv_out);
    cudaFree(d_pooled);
    cudaFree(d_y_pred);
    cudaFree(d_y_true);
    free_model(model);
    free_open_loop_data(data);
    
    return 0;
}