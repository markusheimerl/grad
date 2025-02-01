#ifndef GRAD_CUH
#define GRAD_CUH

#include "data.cuh"
#include <cuda_runtime.h>

// CUDA Error checking helper
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

typedef struct {
    double *x;      // layer values (device)
    double *dx;     // gradients (device)
    int size;       // size of this layer
} CudaLayer;

typedef struct {
    CudaLayer *layers;  // array of layers (host array of device pointers)
    double **W;     // weights between layers (device)
    double **dW;    // weight gradients (device)
    double **b;     // biases for each layer (device)
    double **db;    // bias gradients (device)
    double **m_W;   // momentum for weights (device)
    double **v_W;   // velocity for weights (device)
    double **m_b;   // momentum for biases (device)
    double **v_b;   // velocity for biases (device)
    int n_layers;   // number of layers
    int t;          // timestep for AdamW
    double lr;      // initial learning rate
} CudaNet;

__host__ double get_learning_rate(int epoch, int total_epochs, double initial_lr) {
    return initial_lr * (1 + cos(M_PI * epoch / total_epochs)) / 2;
}

__host__ double get_weight_decay(int epoch, int total_epochs) {
    return 0.01 * (1 - epoch / (double)total_epochs);
}

__host__ double get_warmup_lr(int epoch, int warmup_epochs, double initial_lr) {
    if (epoch < warmup_epochs) {
        return initial_lr * epoch / warmup_epochs;
    }
    return initial_lr;
}

__device__ double cuda_gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

__device__ double cuda_gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

__global__ void forward_kernel(double* curr_layer, double* next_layer, 
                             double* W, double* b, int curr_size, int next_size,
                             bool apply_activation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < next_size) {
        double sum = 0;
        for (int i = 0; i < curr_size; i++) {
            sum += curr_layer[i] * W[idx * curr_size + i];
        }
        sum += b[idx];
        next_layer[idx] = apply_activation ? cuda_gelu(sum) : sum;
    }
}

__global__ void backward_kernel(double* curr_dx, double* next_dx, 
                              double* W, double* dW, double* db,
                              double* curr_x, int curr_size, int next_size,
                              bool apply_activation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < curr_size) {
        double sum = 0;
        for (int i = 0; i < next_size; i++) {
            sum += next_dx[i] * W[i * curr_size + idx];
            atomicAdd(&dW[i * curr_size + idx], next_dx[i] * curr_x[idx]);
            if (idx == 0) {
                atomicAdd(&db[i], next_dx[i]);
            }
        }
        if (apply_activation) {
            curr_dx[idx] = sum * cuda_gelu_derivative(curr_x[idx]);
        } else {
            curr_dx[idx] = sum;
        }
    }
}

__global__ void adamw_update_kernel(double* W, double* dW, double* m_W, double* v_W,
                                  double* b, double* db, double* m_b, double* v_b,
                                  double lr, double weight_decay, int t,
                                  int size, int prev_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;

    if (idx < size * prev_size) {
        // Update weights
        m_W[idx] = beta1 * m_W[idx] + (1 - beta1) * dW[idx];
        v_W[idx] = beta2 * v_W[idx] + (1 - beta2) * dW[idx] * dW[idx];
        
        double m_hat = m_W[idx] / (1 - pow(beta1, t));
        double v_hat = v_W[idx] / (1 - pow(beta2, t));
        
        W[idx] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * W[idx]);
        dW[idx] = 0.0; // Reset gradient
    }
    
    if (idx < size) {
        // Update biases
        m_b[idx] = beta1 * m_b[idx] + (1 - beta1) * db[idx];
        v_b[idx] = beta2 * v_b[idx] + (1 - beta2) * db[idx] * db[idx];
        
        double m_hat = m_b[idx] / (1 - pow(beta1, t));
        double v_hat = v_b[idx] / (1 - pow(beta2, t));
        
        b[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
        db[idx] = 0.0; // Reset gradient
    }
}

CudaNet* cuda_init_net(int n_layers, int* sizes, double lr) {
    CudaNet* net = (CudaNet*)malloc(sizeof(CudaNet));
    net->n_layers = n_layers;
    net->t = 1;
    net->lr = lr;
    
    // Allocate host array of layers
    net->layers = (CudaLayer*)malloc(n_layers * sizeof(CudaLayer));
    
    // Allocate device arrays for each layer
    for(int i = 0; i < n_layers; i++) {
        net->layers[i].size = sizes[i];
        CUDA_CHECK(cudaMalloc(&net->layers[i].x, sizes[i] * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->layers[i].dx, sizes[i] * sizeof(double)));
    }
    
    // Allocate arrays for weights and related parameters
    net->W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->dW = (double**)malloc((n_layers-1) * sizeof(double*));
    net->b = (double**)malloc((n_layers-1) * sizeof(double*));
    net->db = (double**)malloc((n_layers-1) * sizeof(double*));
    net->m_W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->v_W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->m_b = (double**)malloc((n_layers-1) * sizeof(double*));
    net->v_b = (double**)malloc((n_layers-1) * sizeof(double*));
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        int w_size = rows * cols;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&net->W[i], w_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->dW[i], w_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->b[i], rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->db[i], rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->m_W[i], w_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->v_W[i], w_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->m_b[i], rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&net->v_b[i], rows * sizeof(double)));
        
        // Initialize weights on host
        double* h_W = (double*)malloc(w_size * sizeof(double));
        double scale = sqrt(2.0 / (sizes[i] + sizes[i+1]));
        for(int j = 0; j < w_size; j++) {
            h_W[j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(net->W[i], h_W, w_size * sizeof(double), 
                            cudaMemcpyHostToDevice));
        free(h_W);
        
        // Initialize other arrays to zero
        CUDA_CHECK(cudaMemset(net->dW[i], 0, w_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(net->b[i], 0, rows * sizeof(double)));
        CUDA_CHECK(cudaMemset(net->db[i], 0, rows * sizeof(double)));
        CUDA_CHECK(cudaMemset(net->m_W[i], 0, w_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(net->v_W[i], 0, w_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(net->m_b[i], 0, rows * sizeof(double)));
        CUDA_CHECK(cudaMemset(net->v_b[i], 0, rows * sizeof(double)));
    }
    
    return net;
}

void cuda_forward(CudaNet* net, double* input) {
    // Copy input to first layer
    CUDA_CHECK(cudaMemcpy(net->layers[0].x, input, 
                         net->layers[0].size * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    // Forward propagation
    for(int i = 0; i < net->n_layers-1; i++) {
        dim3 block(256);
        dim3 grid((net->layers[i+1].size + block.x - 1) / block.x);
        
        forward_kernel<<<grid, block>>>(
            net->layers[i].x,
            net->layers[i+1].x,
            net->W[i],
            net->b[i],
            net->layers[i].size,
            net->layers[i+1].size,
            i < net->n_layers-2  // apply activation except for last layer
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

void cuda_backward(CudaNet* net, double* target, int epoch, int total_epochs) {
    int last = net->n_layers-1;
    
    // Compute output gradient
    double* h_output = (double*)malloc(net->layers[last].size * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_output, net->layers[last].x, 
                         net->layers[last].size * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    double* h_gradient = (double*)malloc(net->layers[last].size * sizeof(double));
    for(int i = 0; i < net->layers[last].size; i++) {
        h_gradient[i] = h_output[i] - target[i];
    }
    
    CUDA_CHECK(cudaMemcpy(net->layers[last].dx, h_gradient, 
                         net->layers[last].size * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    free(h_output);
    free(h_gradient);
    
    // Backward propagation
    for(int i = last-1; i >= 0; i--) {
        dim3 block(256);
        dim3 grid((net->layers[i].size + block.x - 1) / block.x);
        
        backward_kernel<<<grid, block>>>(
            net->layers[i].dx,
            net->layers[i+1].dx,
            net->W[i],
            net->dW[i],
            net->db[i],
            net->layers[i].x,
            net->layers[i].size,
            net->layers[i+1].size,
            i > 0  // apply activation gradient except for input layer
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Update weights and biases
    double current_lr = get_learning_rate(epoch, total_epochs, net->lr);
    double weight_decay = get_weight_decay(epoch, total_epochs);
    
    for(int i = 0; i < net->n_layers-1; i++) {
        dim3 block(256);
        dim3 grid((max(net->layers[i+1].size * net->layers[i].size,
                      net->layers[i+1].size) + block.x - 1) / block.x);
        
        adamw_update_kernel<<<grid, block>>>(
            net->W[i], net->dW[i], net->m_W[i], net->v_W[i],
            net->b[i], net->db[i], net->m_b[i], net->v_b[i],
            current_lr, weight_decay, net->t,
            net->layers[i+1].size, net->layers[i].size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    net->t++;
}

double cuda_mse(CudaNet* net, Data* data) {
    double total_error = 0.0;
    double* h_output = (double*)malloc(net->layers[net->n_layers-1].size * sizeof(double));
    
    for(int i = 0; i < data->n; i++) {
        cuda_forward(net, data->X[i]);
        
        CUDA_CHECK(cudaMemcpy(h_output, net->layers[net->n_layers-1].x,
                            net->layers[net->n_layers-1].size * sizeof(double),
                            cudaMemcpyDeviceToHost));
        
        for(int j = 0; j < data->fy; j++) {
            double diff = h_output[j] - data->y[i][j];
            total_error += diff * diff;
        }
    }
    
    free(h_output);
    return total_error / data->n;
}

void cuda_save_net(const char* f, CudaNet* net) {
    FILE* fp = fopen(f, "wb");
    if(!fp) return;
    
    fwrite(&net->n_layers, sizeof(int), 1, fp);
    fwrite(&net->lr, sizeof(double), 1, fp);
    fwrite(&net->t, sizeof(int), 1, fp);
    
    for(int i = 0; i < net->n_layers; i++) {
        fwrite(&net->layers[i].size, sizeof(int), 1, fp);
    }
    
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        int w_size = rows * cols;
        
        double* h_W = (double*)malloc(w_size * sizeof(double));
        double* h_m_W = (double*)malloc(w_size * sizeof(double));
        double* h_v_W = (double*)malloc(w_size * sizeof(double));
        double* h_b = (double*)malloc(rows * sizeof(double));
        double* h_m_b = (double*)malloc(rows * sizeof(double));
        double* h_v_b = (double*)malloc(rows * sizeof(double));
        
        CUDA_CHECK(cudaMemcpy(h_W, net->W[i], w_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m_W, net->m_W[i], w_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v_W, net->v_W[i], w_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b, net->b[i], rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m_b, net->m_b[i], rows * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v_b, net->v_b[i], rows * sizeof(double), cudaMemcpyDeviceToHost));
        
        fwrite(h_W, sizeof(double), w_size, fp);
        fwrite(h_m_W, sizeof(double), w_size, fp);
        fwrite(h_v_W, sizeof(double), w_size, fp);
        fwrite(h_b, sizeof(double), rows, fp);
        fwrite(h_m_b, sizeof(double), rows, fp);
        fwrite(h_v_b, sizeof(double), rows, fp);
        
        free(h_W);
        free(h_m_W);
        free(h_v_W);
        free(h_b);
        free(h_m_b);
        free(h_v_b);
    }
    
    fclose(fp);
}

CudaNet* cuda_load_net(const char* f) {
    FILE* fp = fopen(f, "rb");
    if(!fp) return NULL;
    
    int n_layers;
    double learning_rate;
    int timestep;
    fread(&n_layers, sizeof(int), 1, fp);
    fread(&learning_rate, sizeof(double), 1, fp);
    fread(&timestep, sizeof(int), 1, fp);
    
    int* sizes = (int*)malloc(n_layers * sizeof(int));
    for(int i = 0; i < n_layers; i++) {
        fread(&sizes[i], sizeof(int), 1, fp);
    }
    
    CudaNet* net = cuda_init_net(n_layers, sizes, learning_rate);
    net->t = timestep;
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        int w_size = rows * cols;
        
        double* h_W = (double*)malloc(w_size * sizeof(double));
        double* h_m_W = (double*)malloc(w_size * sizeof(double));
        double* h_v_W = (double*)malloc(w_size * sizeof(double));
        double* h_b = (double*)malloc(rows * sizeof(double));
        double* h_m_b = (double*)malloc(rows * sizeof(double));
        double* h_v_b = (double*)malloc(rows * sizeof(double));
        
        fread(h_W, sizeof(double), w_size, fp);
        fread(h_m_W, sizeof(double), w_size, fp);
        fread(h_v_W, sizeof(double), w_size, fp);
        fread(h_b, sizeof(double), rows, fp);
        fread(h_m_b, sizeof(double), rows, fp);
        fread(h_v_b, sizeof(double), rows, fp);
        
        CUDA_CHECK(cudaMemcpy(net->W[i], h_W, w_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->m_W[i], h_m_W, w_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->v_W[i], h_v_W, w_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->b[i], h_b, rows * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->m_b[i], h_m_b, rows * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->v_b[i], h_v_b, rows * sizeof(double), cudaMemcpyHostToDevice));
        
        free(h_W);
        free(h_m_W);
        free(h_v_W);
        free(h_b);
        free(h_m_b);
        free(h_v_b);
    }
    
    free(sizes);
    fclose(fp);
    return net;
}

void cuda_free_net(CudaNet* net) {
    for(int i = 0; i < net->n_layers; i++) {
        CUDA_CHECK(cudaFree(net->layers[i].x));
        CUDA_CHECK(cudaFree(net->layers[i].dx));
    }
    
    for(int i = 0; i < net->n_layers-1; i++) {
        CUDA_CHECK(cudaFree(net->W[i]));
        CUDA_CHECK(cudaFree(net->dW[i]));
        CUDA_CHECK(cudaFree(net->b[i]));
        CUDA_CHECK(cudaFree(net->db[i]));
        CUDA_CHECK(cudaFree(net->m_W[i]));
        CUDA_CHECK(cudaFree(net->v_W[i]));
        CUDA_CHECK(cudaFree(net->m_b[i]));
        CUDA_CHECK(cudaFree(net->v_b[i]));
    }
    
    free(net->W);
    free(net->dW);
    free(net->b);
    free(net->db);
    free(net->m_W);
    free(net->v_W);
    free(net->m_b);
    free(net->v_b);
    free(net->layers);
    free(net);
}

#endif