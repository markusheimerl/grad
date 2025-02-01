#include "data_open_loop.cuh"
#include <time.h>

#define LR 0.001
#define EPOCHS 1000
#define N_FILTERS 16
#define K_SIZE 5
#define B_SIZE 256
#define BATCH_SIZE 32

typedef struct {
    double *d_conv_f, *d_conv_b;    // Conv filters and bias
    double *d_fc_w, *d_fc_b;        // FC weights and bias
    int n_filters, k_size;
} Model;

__global__ void forward_kernel(
    const double* x,         // input
    const double* conv_f,    // conv filters
    const double* conv_b,    // conv bias
    const double* fc_w,      // fc weights
    const double* fc_b,      // fc bias
    double* conv_out,        // conv output
    double* pooled,          // pooled output
    double* y,              // final output
    int win_size, int in_feat, int out_feat, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int conv_size = win_size - K_SIZE + 1;
    
    // Conv + ReLU
    if (idx < batch_size * conv_size * N_FILTERS) {
        int b = idx / (conv_size * N_FILTERS);
        int t = (idx % (conv_size * N_FILTERS)) / N_FILTERS;
        int f = idx % N_FILTERS;
        
        double sum = conv_b[f];
        for(int k = 0; k < K_SIZE; k++)
            for(int c = 0; c < in_feat; c++)
                sum += conv_f[f * K_SIZE * in_feat + k * in_feat + c] * 
                       x[b * win_size * in_feat + (t + k) * in_feat + c];
        conv_out[idx] = sum > 0 ? sum : 0;
    }
    
    __syncthreads();
    
    // Average pooling
    if (idx < batch_size * N_FILTERS) {
        int b = idx / N_FILTERS;
        int f = idx % N_FILTERS;
        double sum = 0;
        for(int t = 0; t < conv_size; t++)
            sum += conv_out[b * conv_size * N_FILTERS + t * N_FILTERS + f];
        pooled[idx] = sum / conv_size;
    }
    
    __syncthreads();
    
    // FC layer
    if (idx < batch_size * out_feat) {
        int b = idx / out_feat;
        int o = idx % out_feat;
        double sum = fc_b[o];
        for(int f = 0; f < N_FILTERS; f++)
            sum += fc_w[o * N_FILTERS + f] * pooled[b * N_FILTERS + f];
        y[idx] = sum;
    }
}

__global__ void backward_kernel(
    const double* x,
    const double* conv_out,
    const double* pooled,
    const double* y_pred,
    const double* y_true,
    const double* conv_f,
    const double* fc_w,
    double* grad_conv_f,
    double* grad_conv_b,
    double* grad_fc_w,
    double* grad_fc_b,
    int win_size, int in_feat, int out_feat, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int conv_size = win_size - K_SIZE + 1;
    
    if (idx < batch_size * out_feat) {
        int b = idx / out_feat;
        int o = idx % out_feat;
        
        // Output gradient
        double d_out = 2.0 * (y_pred[idx] - y_true[idx]) / (out_feat * batch_size);
        
        // FC gradients
        atomicAdd(&grad_fc_b[o], d_out);
        for(int f = 0; f < N_FILTERS; f++) {
            atomicAdd(&grad_fc_w[o * N_FILTERS + f], 
                     d_out * pooled[b * N_FILTERS + f]);
            
            // Propagate to conv layer
            double d_pool = d_out * fc_w[o * N_FILTERS + f] / conv_size;
            
            for(int t = 0; t < conv_size; t++) {
                double d_relu = conv_out[b * conv_size * N_FILTERS + t * N_FILTERS + f] > 0 ? 1.0 : 0.0;
                double d_conv = d_pool * d_relu;
                
                atomicAdd(&grad_conv_b[f], d_conv);
                
                for(int k = 0; k < K_SIZE; k++) {
                    for(int c = 0; c < in_feat; c++) {
                        atomicAdd(&grad_conv_f[f * K_SIZE * in_feat + k * in_feat + c],
                                d_conv * x[b * win_size * in_feat + (t + k) * in_feat + c]);
                    }
                }
            }
        }
    }
}

__global__ void update_params_kernel(double* param, const double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) param[idx] -= LR * grad[idx];
}

Model* init_model(int in_feat, int out_feat) {
    Model* m = (Model*)malloc(sizeof(Model));
    m->n_filters = N_FILTERS;
    m->k_size = K_SIZE;
    
    int conv_size = N_FILTERS * K_SIZE * in_feat;
    int fc_size = out_feat * N_FILTERS;
    
    // Initialize host memory with He initialization
    double *h_conv_f = (double*)malloc(conv_size * sizeof(double));
    double *h_fc_w = (double*)malloc(fc_size * sizeof(double));
    
    double conv_scale = sqrt(2.0 / (K_SIZE * in_feat));
    double fc_scale = sqrt(2.0 / N_FILTERS);
    
    for(int i = 0; i < conv_size; i++)
        h_conv_f[i] = ((double)rand()/RAND_MAX * 2 - 1) * conv_scale;
    for(int i = 0; i < fc_size; i++)
        h_fc_w[i] = ((double)rand()/RAND_MAX * 2 - 1) * fc_scale;
    
    // Allocate and copy to device
    cudaMalloc(&m->d_conv_f, conv_size * sizeof(double));
    cudaMalloc(&m->d_conv_b, N_FILTERS * sizeof(double));
    cudaMalloc(&m->d_fc_w, fc_size * sizeof(double));
    cudaMalloc(&m->d_fc_b, out_feat * sizeof(double));
    
    cudaMemcpy(m->d_conv_f, h_conv_f, conv_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m->d_fc_w, h_fc_w, fc_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(m->d_conv_b, 0, N_FILTERS * sizeof(double));
    cudaMemset(m->d_fc_b, 0, out_feat * sizeof(double));
    
    free(h_conv_f);
    free(h_fc_w);
    return m;
}

void free_model(Model* m) {
    cudaFree(m->d_conv_f);
    cudaFree(m->d_conv_b);
    cudaFree(m->d_fc_w);
    cudaFree(m->d_fc_b);
    free(m);
}

int main() {
    srand(time(NULL));
    OpenLoopData* data = generate_open_loop_data(1000, 50, 3, 2, 0.1);
    
    // Save data
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_open_loop_csv(fname, data);
    printf("Data saved to: %s\n", fname);

    Model* model = init_model(data->input_features, data->output_features);
    
    // Allocate device memory for training
    double *d_x, *d_y_pred, *d_y_true, *d_conv_out, *d_pooled;
    double *d_grad_conv_f, *d_grad_conv_b, *d_grad_fc_w, *d_grad_fc_b;
    
    int conv_size = N_FILTERS * K_SIZE * data->input_features;
    int fc_size = data->output_features * N_FILTERS;
    
    cudaMalloc(&d_x, BATCH_SIZE * data->window_size * data->input_features * sizeof(double));
    cudaMalloc(&d_y_pred, BATCH_SIZE * data->output_features * sizeof(double));
    cudaMalloc(&d_y_true, BATCH_SIZE * data->output_features * sizeof(double));
    cudaMalloc(&d_conv_out, BATCH_SIZE * (data->window_size - K_SIZE + 1) * N_FILTERS * sizeof(double));
    cudaMalloc(&d_pooled, BATCH_SIZE * N_FILTERS * sizeof(double));
    
    cudaMalloc(&d_grad_conv_f, conv_size * sizeof(double));
    cudaMalloc(&d_grad_conv_b, N_FILTERS * sizeof(double));
    cudaMalloc(&d_grad_fc_w, fc_size * sizeof(double));
    cudaMalloc(&d_grad_fc_b, data->output_features * sizeof(double));
    
    printf("Training started...\n");
    for(int epoch = 0; epoch < EPOCHS; epoch += 2) {
        double loss = 0;
        
        for(int b = 0; b < data->n; b += BATCH_SIZE) {
            int batch_size = min(BATCH_SIZE, data->n - b);
            
            // Prepare batch data
            double* batch_x = (double*)malloc(
                batch_size * data->window_size * data->input_features * sizeof(double));
            double* batch_y = (double*)malloc(
                batch_size * data->output_features * sizeof(double));
            
            for(int i = 0; i < batch_size; i++) {
                for(int t = 0; t < data->window_size; t++)
                    for(int f = 0; f < data->input_features; f++)
                        batch_x[i * data->window_size * data->input_features + 
                               t * data->input_features + f] = data->windows[b + i][t][f];
                
                memcpy(&batch_y[i * data->output_features], data->outputs[b + i],
                       data->output_features * sizeof(double));
            }
            
            cudaMemcpy(d_x, batch_x, 
                      batch_size * data->window_size * data->input_features * sizeof(double),
                      cudaMemcpyHostToDevice);
            cudaMemcpy(d_y_true, batch_y,
                      batch_size * data->output_features * sizeof(double),
                      cudaMemcpyHostToDevice);
            
            // Zero gradients
            cudaMemset(d_grad_conv_f, 0, conv_size * sizeof(double));
            cudaMemset(d_grad_conv_b, 0, N_FILTERS * sizeof(double));
            cudaMemset(d_grad_fc_w, 0, fc_size * sizeof(double));
            cudaMemset(d_grad_fc_b, 0, data->output_features * sizeof(double));
            
            // Forward pass
            forward_kernel<<<(batch_size * data->window_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                d_x, model->d_conv_f, model->d_conv_b, model->d_fc_w, model->d_fc_b,
                d_conv_out, d_pooled, d_y_pred,
                data->window_size, data->input_features, data->output_features, batch_size);
            
            // Backward pass
            backward_kernel<<<(batch_size * data->output_features + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                d_x, d_conv_out, d_pooled, d_y_pred, d_y_true,
                model->d_conv_f, model->d_fc_w,
                d_grad_conv_f, d_grad_conv_b, d_grad_fc_w, d_grad_fc_b,
                data->window_size, data->input_features, data->output_features, batch_size);
            
            // Update parameters
            update_params_kernel<<<(conv_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_conv_f, d_grad_conv_f, conv_size);
            update_params_kernel<<<(N_FILTERS + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_conv_b, d_grad_conv_b, N_FILTERS);
            update_params_kernel<<<(fc_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_fc_w, d_grad_fc_w, fc_size);
            update_params_kernel<<<(data->output_features + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_fc_b, d_grad_fc_b, data->output_features);
            
            // Compute loss
            double* h_y_pred = (double*)malloc(batch_size * data->output_features * sizeof(double));
            cudaMemcpy(h_y_pred, d_y_pred,
                      batch_size * data->output_features * sizeof(double),
                      cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < batch_size; i++) {
                for(int j = 0; j < data->output_features; j++) {
                    double diff = h_y_pred[i * data->output_features + j] - data->outputs[b + i][j];
                    loss += diff * diff;
                }
            }
            
            free(batch_x);
            free(batch_y);
            free(h_y_pred);
        }
        
        printf("Epoch %d, Loss: %.6f\n", epoch, loss/data->n);
    }
    
    // Print example predictions
    printf("\nExample predictions:\n");
    double* x = (double*)malloc(data->window_size * data->input_features * sizeof(double));
    double* y_pred = (double*)malloc(data->output_features * sizeof(double));
    
    for(int i = 0; i < min(5, data->n); i++) {
        for(int t = 0; t < data->window_size; t++)
            for(int f = 0; f < data->input_features; f++)
                x[t * data->input_features + f] = data->windows[i][t][f];
        
        cudaMemcpy(d_x, x, 
                  data->window_size * data->input_features * sizeof(double),
                  cudaMemcpyHostToDevice);
        
        forward_kernel<<<(data->window_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
            d_x, model->d_conv_f, model->d_conv_b, model->d_fc_w, model->d_fc_b,
            d_conv_out, d_pooled, d_y_pred,
            data->window_size, data->input_features, data->output_features, 1);
        
        cudaMemcpy(y_pred, d_y_pred,
                  data->output_features * sizeof(double),
                  cudaMemcpyDeviceToHost);
        
        printf("Sample %d:\n", i);
        printf("  True:");
        for(int j = 0; j < data->output_features; j++)
            printf(" %.6f", data->outputs[i][j]);
        printf("\n  Pred:");
        for(int j = 0; j < data->output_features; j++)
            printf(" %.6f", y_pred[j]);
        printf("\n");
    }
    
    // Cleanup
    free(x);
    free(y_pred);
    cudaFree(d_x);
    cudaFree(d_y_pred);
    cudaFree(d_y_true);
    cudaFree(d_conv_out);
    cudaFree(d_pooled);
    cudaFree(d_grad_conv_f);
    cudaFree(d_grad_conv_b);
    cudaFree(d_grad_fc_w);
    cudaFree(d_grad_fc_b);
    free_model(model);
    free_open_loop_data(data);
    
    return 0;
}