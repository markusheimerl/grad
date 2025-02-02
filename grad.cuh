#ifndef GRAD_CUH
#define GRAD_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include "data.cuh"

// CUDA Error checking helper
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// File operation helper
#define CHECK_FILE_OP(op, fp) { \
    size_t ret = op; \
    if (ret == 0) { \
        fclose(fp); \
        return NULL; \
    } \
}

typedef struct {
    double *x;      // host layer values
    double *dx;     // host gradients
    double *d_x;    // device layer values
    double *d_dx;   // device gradients
    int size;       // size of this layer
} Layer;

typedef struct {
    Layer *layers;  // array of layers
    double **W;     // host weights between layers
    double **dW;    // host weight gradients
    double **b;     // host biases for each layer
    double **db;    // host bias gradients
    double **m_W;   // host momentum for weights (AdamW)
    double **v_W;   // host velocity for weights (AdamW)
    double **m_b;   // host momentum for biases (AdamW)
    double **v_b;   // host velocity for biases (AdamW)
    
    // Device pointers
    double **d_W;   // device weights
    double **d_dW;  // device weight gradients
    double **d_b;   // device biases
    double **d_db;  // device bias gradients
    double **d_m_W; // device momentum for weights
    double **d_v_W; // device velocity for weights
    double **d_m_b; // device momentum for biases
    double **d_v_b; // device velocity for biases
    
    int n_layers;   // number of layers
    int t;          // timestep for AdamW
    double lr;      // initial learning rate
} Net;

// Device functions
__device__ double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

__device__ double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

// CUDA Kernels
__global__ void forward_kernel(double* W, double* b, double* input, double* output,
                             int input_size, int output_size, bool apply_gelu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        double sum = 0.0;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * W[idx * input_size + i];
        }
        sum += b[idx];
        output[idx] = apply_gelu ? gelu(sum) : sum;
    }
}

__global__ void backward_kernel(double* W, double* dW, double* dx, double* prev_x,
                              double* next_dx, int input_size, int output_size,
                              bool apply_gelu_derivative) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        double sum = 0.0;
        for (int i = 0; i < output_size; i++) {
            sum += next_dx[i] * W[i * input_size + idx];
            dW[i * input_size + idx] = next_dx[i] * prev_x[idx];
        }
        dx[idx] = apply_gelu_derivative ? sum * gelu_derivative(prev_x[idx]) : sum;
    }
}

__global__ void adamw_update_kernel(double* W, double* dW, double* m_W, double* v_W,
                                  double beta1, double beta2, double eps,
                                  double lr, double weight_decay, int size, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        m_W[idx] = beta1 * m_W[idx] + (1 - beta1) * dW[idx];
        v_W[idx] = beta2 * v_W[idx] + (1 - beta2) * dW[idx] * dW[idx];
        
        double m_hat = m_W[idx] / (1 - pow(beta1, t));
        double v_hat = v_W[idx] / (1 - pow(beta2, t));
        
        W[idx] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * W[idx]);
    }
}

// Host-side functions
static void free_partial_net(Net* net, int allocated_layers) {
    if (!net) return;
    
    if (net->layers) {
        for(int i = 0; i < allocated_layers; i++) {
            if (net->layers[i].x) free(net->layers[i].x);
            if (net->layers[i].dx) free(net->layers[i].dx);
            if (net->layers[i].d_x) cudaFree(net->layers[i].d_x);
            if (net->layers[i].d_dx) cudaFree(net->layers[i].d_dx);
        }
        free(net->layers);
    }
    
    for(int i = 0; i < allocated_layers-1; i++) {
        if (net->W && net->W[i]) free(net->W[i]);
        if (net->dW && net->dW[i]) free(net->dW[i]);
        if (net->b && net->b[i]) free(net->b[i]);
        if (net->db && net->db[i]) free(net->db[i]);
        if (net->m_W && net->m_W[i]) free(net->m_W[i]);
        if (net->v_W && net->v_W[i]) free(net->v_W[i]);
        if (net->m_b && net->m_b[i]) free(net->m_b[i]);
        if (net->v_b && net->v_b[i]) free(net->v_b[i]);
        
        if (net->d_W && net->d_W[i]) cudaFree(net->d_W[i]);
        if (net->d_dW && net->d_dW[i]) cudaFree(net->d_dW[i]);
        if (net->d_b && net->d_b[i]) cudaFree(net->d_b[i]);
        if (net->d_db && net->d_db[i]) cudaFree(net->d_db[i]);
        if (net->d_m_W && net->d_m_W[i]) cudaFree(net->d_m_W[i]);
        if (net->d_v_W && net->d_v_W[i]) cudaFree(net->d_v_W[i]);
        if (net->d_m_b && net->d_m_b[i]) cudaFree(net->d_m_b[i]);
        if (net->d_v_b && net->d_v_b[i]) cudaFree(net->d_v_b[i]);
    }
    
    if (net->W) free(net->W);
    if (net->dW) free(net->dW);
    if (net->b) free(net->b);
    if (net->db) free(net->db);
    if (net->m_W) free(net->m_W);
    if (net->v_W) free(net->v_W);
    if (net->m_b) free(net->m_b);
    if (net->v_b) free(net->v_b);
    
    if (net->d_W) free(net->d_W);
    if (net->d_dW) free(net->d_dW);
    if (net->d_b) free(net->d_b);
    if (net->d_db) free(net->d_db);
    if (net->d_m_W) free(net->d_m_W);
    if (net->d_v_W) free(net->d_v_W);
    if (net->d_m_b) free(net->d_m_b);
    if (net->d_v_b) free(net->d_v_b);
    
    free(net);
}

Net* init_net(int n_layers, int* sizes, double lr) {
    if (n_layers <= 0 || !sizes || lr <= 0) return NULL;
    
    Net* net = (Net*)calloc(1, sizeof(Net));
    if (!net) return NULL;
    
    net->n_layers = n_layers;
    net->t = 1;
    net->lr = lr;
    
    // Allocate layers
    net->layers = (Layer*)calloc(n_layers, sizeof(Layer));
    if (!net->layers) {
        free_partial_net(net, 0);
        return NULL;
    }
    
    // Allocate array pointers
    net->W = (double**)calloc(n_layers-1, sizeof(double*));
    net->dW = (double**)calloc(n_layers-1, sizeof(double*));
    net->b = (double**)calloc(n_layers-1, sizeof(double*));
    net->db = (double**)calloc(n_layers-1, sizeof(double*));
    net->m_W = (double**)calloc(n_layers-1, sizeof(double*));
    net->v_W = (double**)calloc(n_layers-1, sizeof(double*));
    net->m_b = (double**)calloc(n_layers-1, sizeof(double*));
    net->v_b = (double**)calloc(n_layers-1, sizeof(double*));
    
    net->d_W = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_dW = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_b = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_db = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_m_W = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_v_W = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_m_b = (double**)calloc(n_layers-1, sizeof(double*));
    net->d_v_b = (double**)calloc(n_layers-1, sizeof(double*));
    
    if (!net->W || !net->dW || !net->b || !net->db || 
        !net->m_W || !net->v_W || !net->m_b || !net->v_b ||
        !net->d_W || !net->d_dW || !net->d_b || !net->d_db ||
        !net->d_m_W || !net->d_v_W || !net->d_m_b || !net->d_v_b) {
        free_partial_net(net, 0);
        return NULL;
    }
    
    // Initialize layers and allocate memory
    for(int i = 0; i < n_layers; i++) {
        net->layers[i].size = sizes[i];
        net->layers[i].x = (double*)calloc(sizes[i], sizeof(double));
        net->layers[i].dx = (double*)calloc(sizes[i], sizeof(double));
        
        if (!net->layers[i].x || !net->layers[i].dx) {
            free_partial_net(net, i);
            return NULL;
        }
        
        if (cudaMalloc(&net->layers[i].d_x, sizes[i] * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->layers[i].d_dx, sizes[i] * sizeof(double)) != cudaSuccess) {
            free_partial_net(net, i);
            return NULL;
        }
    }
    
    // Initialize weights and allocate memory
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        int w_size = rows * cols;
        
        // Allocate host memory
        net->W[i] = (double*)malloc(w_size * sizeof(double));
        net->dW[i] = (double*)calloc(w_size, sizeof(double));
        net->b[i] = (double*)calloc(rows, sizeof(double));
        net->db[i] = (double*)calloc(rows, sizeof(double));
        net->m_W[i] = (double*)calloc(w_size, sizeof(double));
        net->v_W[i] = (double*)calloc(w_size, sizeof(double));
        net->m_b[i] = (double*)calloc(rows, sizeof(double));
        net->v_b[i] = (double*)calloc(rows, sizeof(double));
        
        if (!net->W[i] || !net->dW[i] || !net->b[i] || !net->db[i] ||
            !net->m_W[i] || !net->v_W[i] || !net->m_b[i] || !net->v_b[i]) {
            free_partial_net(net, n_layers);
            return NULL;
        }
        
        // Allocate device memory
        if (cudaMalloc(&net->d_W[i], w_size * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_dW[i], w_size * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_b[i], rows * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_db[i], rows * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_m_W[i], w_size * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_v_W[i], w_size * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_m_b[i], rows * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&net->d_v_b[i], rows * sizeof(double)) != cudaSuccess) {
            free_partial_net(net, n_layers);
            return NULL;
        }
        
        // Xavier initialization
        double scale = sqrt(2.0 / (sizes[i] + sizes[i+1]));
        for(int j = 0; j < w_size; j++) {
            net->W[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        }
        
        // Copy to device
        if (cudaMemcpy(net->d_W[i], net->W[i], w_size * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(net->d_b[i], net->b[i], rows * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess) {
            free_partial_net(net, n_layers);
            return NULL;
        }
    }
    
    return net;
}

void forward(Net* net, double* input) {
    // Copy input to first layer's device memory
    CHECK_CUDA(cudaMemcpy(net->layers[0].d_x, input, 
                         net->layers[0].size * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    // Forward propagation through layers
    for(int i = 0; i < net->n_layers-1; i++) {
        int input_size = net->layers[i].size;
        int output_size = net->layers[i+1].size;
        
        int block_size = 256;
        int num_blocks = (output_size + block_size - 1) / block_size;
        
        forward_kernel<<<num_blocks, block_size>>>(
            net->d_W[i], net->d_b[i],
            net->layers[i].d_x,
            net->layers[i+1].d_x,
            input_size, output_size,
            i < net->n_layers-2  // apply GELU except for last layer
        );
        
        CHECK_CUDA(cudaGetLastError());
    }
    
    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(net->layers[net->n_layers-1].x,
                         net->layers[net->n_layers-1].d_x,
                         net->layers[net->n_layers-1].size * sizeof(double),
                         cudaMemcpyDeviceToHost));
}

double get_learning_rate(int epoch, int total_epochs, double initial_lr) {
    return initial_lr * (1 + cos(M_PI * epoch / total_epochs)) / 2;
}

double get_weight_decay(int epoch, int total_epochs) {
    return 0.01 * (1 - epoch / (double)total_epochs);
}

double get_warmup_lr(int epoch, int warmup_epochs, double initial_lr) {
    if (epoch < warmup_epochs) {
        return initial_lr * epoch / warmup_epochs;
    }
    return initial_lr;
}

void backward(Net* net, double* target, int epoch, int total_epochs) {
    int last = net->n_layers-1;
    
    // Compute output gradient
    for(int i = 0; i < net->layers[last].size; i++) {
        net->layers[last].dx[i] = net->layers[last].x[i] - target[i];
    }
    
    // Copy output gradient to device
    CHECK_CUDA(cudaMemcpy(net->layers[last].d_dx, net->layers[last].dx,
                         net->layers[last].size * sizeof(double),
                         cudaMemcpyHostToDevice));
    
    // Backward propagation
    for(int i = last-1; i >= 0; i--) {
        int input_size = net->layers[i].size;
        int output_size = net->layers[i+1].size;
        
        int block_size = 256;
        int num_blocks = (input_size + block_size - 1) / block_size;
        
        backward_kernel<<<num_blocks, block_size>>>(
            net->d_W[i], net->d_dW[i],
            net->layers[i].d_dx,
            net->layers[i].d_x,
            net->layers[i+1].d_dx,
            input_size, output_size,
            i > 0  // apply GELU derivative except for input layer
        );
        
        CHECK_CUDA(cudaGetLastError());
    }
    
    // Update weights using AdamW
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    
    const int warmup_epochs = 5;
    double current_lr = get_warmup_lr(epoch, warmup_epochs, net->lr);
    current_lr = get_learning_rate(epoch, total_epochs, current_lr);
    double weight_decay = get_weight_decay(epoch, total_epochs);
    
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        int w_size = rows * cols;
        
        int block_size = 256;
        int num_blocks = (w_size + block_size - 1) / block_size;
        
        adamw_update_kernel<<<num_blocks, block_size>>>(
            net->d_W[i], net->d_dW[i],
            net->d_m_W[i], net->d_v_W[i],
            beta1, beta2, eps,
            current_lr, weight_decay,
            w_size, net->t
        );
        
        num_blocks = (rows + block_size - 1) / block_size;
        
        adamw_update_kernel<<<num_blocks, block_size>>>(
            net->d_b[i], net->d_db[i],
            net->d_m_b[i], net->d_v_b[i],
            beta1, beta2, eps,
            current_lr, 0.0,  // no weight decay for biases
            rows, net->t
        );
        
        CHECK_CUDA(cudaGetLastError());
    }
    
    net->t++;
}

double mse(Net* net, Data* data) {
    double error = 0;
    for(int i = 0; i < data->n; i++) {
        forward(net, data->X[i]);
        for(int j = 0; j < data->fy; j++) {
            double diff = net->layers[net->n_layers-1].x[j] - data->y[i][j];
            error += diff * diff;
        }
    }
    return error / data->n;
}

Net* load_net(const char* f) {
    if (!f) return NULL;
    
    FILE* fp = fopen(f, "rb");
    if (!fp) return NULL;
    
    int n_layers;
    double learning_rate;
    int timestep;
    
    if (fread(&n_layers, sizeof(int), 1, fp) != 1 ||
        fread(&learning_rate, sizeof(double), 1, fp) != 1 ||
        fread(&timestep, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }
    
    int* sizes = (int*)malloc(n_layers * sizeof(int));
    if (!sizes) {
        fclose(fp);
        return NULL;
    }
    
    for(int i = 0; i < n_layers; i++) {
        if (fread(&sizes[i], sizeof(int), 1, fp) != 1) {
            free(sizes);
            fclose(fp);
            return NULL;
        }
    }
    
    Net* net = init_net(n_layers, sizes, learning_rate);
    if (!net) {
        free(sizes);
        fclose(fp);
        return NULL;
    }
    
    net->t = timestep;
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        int w_size = rows * cols;
        
        if (fread(net->W[i], sizeof(double), w_size, fp) != w_size ||
            fread(net->m_W[i], sizeof(double), w_size, fp) != w_size ||
            fread(net->v_W[i], sizeof(double), w_size, fp) != w_size ||
            fread(net->b[i], sizeof(double), rows, fp) != rows ||
            fread(net->m_b[i], sizeof(double), rows, fp) != rows ||
            fread(net->v_b[i], sizeof(double), rows, fp) != rows) {
            free(sizes);
            free_partial_net(net, n_layers);
            fclose(fp);
            return NULL;
        }
        
        // Copy to device
        if (cudaMemcpy(net->d_W[i], net->W[i], w_size * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(net->d_m_W[i], net->m_W[i], w_size * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(net->d_v_W[i], net->v_W[i], w_size * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(net->d_b[i], net->b[i], rows * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(net->d_m_b[i], net->m_b[i], rows * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(net->d_v_b[i], net->v_b[i], rows * sizeof(double), 
                      cudaMemcpyHostToDevice) != cudaSuccess) {
            free(sizes);
            free_partial_net(net, n_layers);
            fclose(fp);
            return NULL;
        }
    }
    
    free(sizes);
    fclose(fp);
    return net;
}

void save_net(const char* f, Net* net) {
    if (!f || !net) return;
    
    FILE* fp = fopen(f, "wb");
    if (!fp) return;
    
    if (fwrite(&net->n_layers, sizeof(int), 1, fp) != 1 ||
        fwrite(&net->lr, sizeof(double), 1, fp) != 1 ||
        fwrite(&net->t, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return;
    }
    
    for(int i = 0; i < net->n_layers; i++) {
        if (fwrite(&net->layers[i].size, sizeof(int), 1, fp) != 1) {
            fclose(fp);
            return;
        }
    }
    
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        int w_size = rows * cols;
        
        // Copy from device to host before saving
        if (cudaMemcpy(net->W[i], net->d_W[i], w_size * sizeof(double), 
                      cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(net->m_W[i], net->d_m_W[i], w_size * sizeof(double), 
                      cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(net->v_W[i], net->d_v_W[i], w_size * sizeof(double), 
                      cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(net->b[i], net->d_b[i], rows * sizeof(double), 
                      cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(net->m_b[i], net->d_m_b[i], rows * sizeof(double), 
                      cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(net->v_b[i], net->d_v_b[i], rows * sizeof(double), 
                      cudaMemcpyDeviceToHost) != cudaSuccess) {
            fclose(fp);
            return;
        }
        
        if (fwrite(net->W[i], sizeof(double), w_size, fp) != w_size ||
            fwrite(net->m_W[i], sizeof(double), w_size, fp) != w_size ||
            fwrite(net->v_W[i], sizeof(double), w_size, fp) != w_size ||
            fwrite(net->b[i], sizeof(double), rows, fp) != rows ||
            fwrite(net->m_b[i], sizeof(double), rows, fp) != rows ||
            fwrite(net->v_b[i], sizeof(double), rows, fp) != rows) {
            fclose(fp);
            return;
        }
    }
    
    fclose(fp);
}

void free_net(Net* net) {
    if (!net) return;
    free_partial_net(net, net->n_layers);
}

#endif