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

// Define a structure for each layer
typedef struct {
    float* d_weight;         // weight matrix
    float* d_weight_grad;    // weight gradients
    float* d_output;         // layer output
    float* d_pre_activation; // pre-activation values
    float* d_error;          // layer error
    
    // Adam parameters
    float* d_m;             // First moment
    float* d_v;             // Second moment
    
    // Host copy of weights for saving/loading
    float* h_weight;
    
    int input_dim;          // input dimension
    int output_dim;         // output dimension
} Layer;

typedef struct {
    Layer** layers;         // Array of layer pointers
    int num_layers;         // Number of layers
    
    // Network parameters
    float beta1;            // Adam beta1
    float beta2;            // Adam beta2
    float epsilon;          // Adam epsilon
    int t;                  // Time step
    float weight_decay;     // Weight decay parameter
    
    // Dimensions
    int input_dim;          // Network input dimension
    int output_dim;         // Network output dimension
    int batch_size;         // Batch size
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} Net;

// Function declarations
Layer* init_layer(int input_dim, int output_dim, int batch_size);
void free_layer(Layer* layer);
Net* init_net(int* layer_dims, int num_layers, int batch_size);
void free_net(Net* net);
void forward_pass(Net* net, float* X);
float calculate_loss(Net* net, float* y);
void zero_gradients(Net* net);
void backward_pass(Net* net, float* X);
void update_weights(Net* net, float learning_rate);
void save_model(Net* net, const char* filename);
Net* load_model(const char* filename);

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

// Initialize a single layer
Layer* init_layer(int input_dim, int output_dim, int batch_size) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    
    // Initialize weights with Xavier initialization
    float scale = 1.0f / sqrt(input_dim);
    layer->h_weight = (float*)malloc(output_dim * input_dim * sizeof(float));
    for (int i = 0; i < output_dim * input_dim; i++) {
        layer->h_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&layer->d_weight, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_weight_grad, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_output, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_pre_activation, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_v, output_dim * input_dim * sizeof(float)));
    
    // Copy weights to device and initialize Adam parameters
    CHECK_CUDA(cudaMemcpy(layer->d_weight, layer->h_weight, 
                         output_dim * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(layer->d_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer->d_v, 0, output_dim * input_dim * sizeof(float)));
    
    return layer;
}

// Initialize network with arbitrary number of layers
Net* init_net(int* layer_dims, int num_layers, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    net->num_layers = num_layers - 1;  // number of weight matrices
    net->layers = (Layer**)malloc(net->num_layers * sizeof(Layer*));
    
    net->input_dim = layer_dims[0];
    net->output_dim = layer_dims[num_layers - 1];
    net->batch_size = batch_size;
    
    // Initialize Adam parameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Initialize each layer
    for (int i = 0; i < net->num_layers; i++) {
        net->layers[i] = init_layer(layer_dims[i], layer_dims[i + 1], batch_size);
    }
    
    return net;
}

void free_layer(Layer* layer) {
    cudaFree(layer->d_weight);
    cudaFree(layer->d_weight_grad);
    cudaFree(layer->d_output);
    cudaFree(layer->d_pre_activation);
    cudaFree(layer->d_error);
    cudaFree(layer->d_m);
    cudaFree(layer->d_v);
    free(layer->h_weight);
    free(layer);
}

void free_net(Net* net) {
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(net->layers[i]);
    }
    free(net->layers);
    cublasDestroy(net->cublas_handle);
    free(net);
}

void forward_pass(Net* net, float* X) {
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    float* current_input = d_X;
    
    for (int i = 0; i < net->num_layers; i++) {
        Layer* layer = net->layers[i];
        
        CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                layer->output_dim,     // n
                                net->batch_size,       // m
                                layer->input_dim,      // k
                                &alpha,
                                layer->d_weight,       // A
                                layer->output_dim,     // lda
                                current_input,         // B
                                layer->input_dim,      // ldb
                                &beta,
                                layer->d_output,       // C
                                layer->output_dim));   // ldc
        
        CHECK_CUDA(cudaMemcpy(layer->d_pre_activation, layer->d_output,
                             net->batch_size * layer->output_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        
        if (i < net->num_layers - 1) {
            int block_size = 256;
            int num_blocks = (net->batch_size * layer->output_dim + block_size - 1) / block_size;
            swish_forward_kernel<<<num_blocks, block_size>>>(
                layer->d_output,
                layer->d_pre_activation,
                net->batch_size * layer->output_dim
            );
        }
        
        current_input = layer->d_output;
    }
    
    cudaFree(d_X);
}

// Custom kernel for calculating error
__global__ void calc_error_kernel(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

float calculate_loss(Net* net, float* y) {
    Layer* output_layer = net->layers[net->num_layers - 1];
    
    float* d_y;
    CHECK_CUDA(cudaMalloc(&d_y, net->batch_size * net->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_y, y, net->batch_size * net->output_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    int size = net->batch_size * net->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    calc_error_kernel<<<num_blocks, block_size>>>(
        output_layer->d_error,
        output_layer->d_output,
        d_y,
        size
    );

    float* h_error = (float*)malloc(size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_error, output_layer->d_error, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss += h_error[i] * h_error[i];
    }

    free(h_error);
    cudaFree(d_y);

    return loss / size;
}

void zero_gradients(Net* net) {
    for (int i = 0; i < net->num_layers; i++) {
        Layer* layer = net->layers[i];
        CHECK_CUDA(cudaMemset(layer->d_weight_grad, 0, 
                             layer->output_dim * layer->input_dim * sizeof(float)));
    }
}

void backward_pass(Net* net, float* X) {
    float* d_X;
    CHECK_CUDA(cudaMalloc(&d_X, net->batch_size * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, net->batch_size * net->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer* layer = net->layers[i];
        float* input = (i == 0) ? d_X : net->layers[i-1]->d_output;

        // Calculate weight gradients
        CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                layer->output_dim,     // n
                                layer->input_dim,      // m
                                net->batch_size,       // k
                                &alpha,
                                layer->d_error,        // A
                                layer->output_dim,     // lda
                                input,                 // B
                                layer->input_dim,      // ldb
                                &beta,
                                layer->d_weight_grad,  // C
                                layer->output_dim));   // ldc

        if (i > 0) {
            Layer* prev_layer = net->layers[i-1];
            
            // Propagate error to previous layer
            CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    layer->input_dim,      // n
                                    net->batch_size,       // m
                                    layer->output_dim,     // k
                                    &alpha,
                                    layer->d_weight,       // A
                                    layer->output_dim,     // lda
                                    layer->d_error,        // B
                                    layer->output_dim,     // ldb
                                    &beta,
                                    prev_layer->d_error,   // C
                                    layer->input_dim));    // ldc

            // Apply activation derivative
            int block_size = 256;
            int num_blocks = (net->batch_size * prev_layer->output_dim + block_size - 1) / block_size;
            swish_backward_kernel<<<num_blocks, block_size>>>(
                prev_layer->d_error,
                prev_layer->d_pre_activation,
                net->batch_size * prev_layer->output_dim
            );
        }
    }

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

void update_weights(Net* net, float learning_rate) {
    net->t++;
    
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int i = 0; i < net->num_layers; i++) {
        Layer* layer = net->layers[i];
        int size = layer->output_dim * layer->input_dim;
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        
        adamw_update_kernel<<<num_blocks, block_size>>>(
            layer->d_weight,
            layer->d_weight_grad,
            layer->d_m,
            layer->d_v,
            net->beta1,
            net->beta2,
            net->epsilon,
            learning_rate,
            net->weight_decay,
            alpha_t,
            size,
            net->batch_size
        );
    }
}

void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save network structure
    fwrite(&net->num_layers, sizeof(int), 1, file);
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(&net->t, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int i = 0; i < net->num_layers; i++) {
        Layer* layer = net->layers[i];
        fwrite(&layer->input_dim, sizeof(int), 1, file);
        fwrite(&layer->output_dim, sizeof(int), 1, file);
        
        // Copy weights from device to host
        CHECK_CUDA(cudaMemcpy(layer->h_weight, layer->d_weight,
                             layer->output_dim * layer->input_dim * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        // Save weights
        fwrite(layer->h_weight, sizeof(float), 
               layer->output_dim * layer->input_dim, file);
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read network structure
    int num_layers, input_dim, output_dim, batch_size, t;
    fread(&num_layers, sizeof(int), 1, file);
    fread(&input_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    fread(&t, sizeof(int), 1, file);
    
    // Create array of layer dimensions
    int* layer_dims = (int*)malloc((num_layers + 1) * sizeof(int));
    layer_dims[0] = input_dim;
    
    // Initialize network
    Net* net = init_net(layer_dims, num_layers + 1, batch_size);
    net->t = t;
    
    // Load weights for each layer
    for (int i = 0; i < num_layers; i++) {
        Layer* layer = net->layers[i];
        int layer_input_dim, layer_output_dim;
        
        fread(&layer_input_dim, sizeof(int), 1, file);
        fread(&layer_output_dim, sizeof(int), 1, file);
        
        // Read weights
        fread(layer->h_weight, sizeof(float), 
              layer->output_dim * layer->input_dim, file);
        
        // Copy weights to device
        CHECK_CUDA(cudaMemcpy(layer->d_weight, layer->h_weight,
                             layer->output_dim * layer->input_dim * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
    
    free(layer_dims);
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif