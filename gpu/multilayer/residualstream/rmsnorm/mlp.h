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
    float** d_weights;         // [depth+1] arrays of weights
    float** d_weight_grads;    // [depth+1] arrays of gradients
    
    // Host copies of weights
    float** h_weights;         // [depth+1] arrays of weights
    
    // Device pointers for Adam parameters
    float** d_m;              // [depth+1] arrays of first moments
    float** d_v;              // [depth+1] arrays of second moments
    float beta1;              // Exponential decay rate for first moment
    float beta2;              // Exponential decay rate for second moment
    float epsilon;            // Small constant for numerical stability
    int t;                    // Time step
    float weight_decay;       // Weight decay parameter for AdamW
    
    // Device pointers for helper arrays
    float** d_layer_outputs;   // [depth+1] arrays of layer outputs
    float** d_pre_activations; // [depth] arrays of pre-activations
    float** d_errors;          // [depth+1] arrays of errors
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int* layer_dims;          // Array of dimensions for each layer
    int output_dim;
    int batch_size;
    int depth;                // Number of hidden layers
} Net;

// Initialize the network with configurable dimensions
Net* init_net(int input_dim, int hidden_dim, int depth, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->input_dim = input_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    net->depth = depth;
    
    // Allocate and store layer dimensions
    net->layer_dims = (int*)malloc((depth + 2) * sizeof(int));
    net->layer_dims[0] = input_dim;
    for(int i = 0; i < depth; i++) {
        net->layer_dims[i + 1] = hidden_dim;
    }
    net->layer_dims[depth + 1] = output_dim;
    
    // Initialize Adam parameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));

    // Allocate arrays of pointers
    net->d_weights = (float**)malloc((depth + 1) * sizeof(float*));
    net->d_weight_grads = (float**)malloc((depth + 1) * sizeof(float*));
    net->h_weights = (float**)malloc((depth + 1) * sizeof(float*));
    net->d_m = (float**)malloc((depth + 1) * sizeof(float*));
    net->d_v = (float**)malloc((depth + 1) * sizeof(float*));
    net->d_layer_outputs = (float**)malloc((depth + 2) * sizeof(float*));
    net->d_pre_activations = (float**)malloc((depth + 1) * sizeof(float*));
    net->d_errors = (float**)malloc((depth + 1) * sizeof(float*));

    // Allocate host memory for weights and initialize
    for(int i = 0; i <= depth; i++) {
        int in_dim = net->layer_dims[i];
        int out_dim = net->layer_dims[i + 1];
        
        net->h_weights[i] = (float*)malloc(out_dim * in_dim * sizeof(float));
        
        // Xavier initialization
        float scale = 1.0f / sqrt(in_dim);
        for(int j = 0; j < out_dim * in_dim; j++) {
            net->h_weights[i][j] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale;
        }
        
        // Allocate device memory for weights and gradients
        CHECK_CUDA(cudaMalloc(&net->d_weights[i], out_dim * in_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&net->d_weight_grads[i], out_dim * in_dim * sizeof(float)));
        
        // Allocate device memory for Adam parameters
        CHECK_CUDA(cudaMalloc(&net->d_m[i], out_dim * in_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&net->d_v[i], out_dim * in_dim * sizeof(float)));
        
        // Initialize device memory
        CHECK_CUDA(cudaMemcpy(net->d_weights[i], net->h_weights[i],
                             out_dim * in_dim * sizeof(float),
                             cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(net->d_weight_grads[i], 0, 
                             out_dim * in_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(net->d_m[i], 0, out_dim * in_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(net->d_v[i], 0, out_dim * in_dim * sizeof(float)));
    }

    // Allocate memory for layer outputs and errors
    for(int i = 0; i <= depth + 1; i++) {
        int dim = net->layer_dims[i];
        CHECK_CUDA(cudaMalloc(&net->d_layer_outputs[i], 
                             batch_size * dim * sizeof(float)));
        if(i > 0 && i <= depth) {
            CHECK_CUDA(cudaMalloc(&net->d_pre_activations[i-1], 
                                 batch_size * dim * sizeof(float)));
        }
        if(i > 0) {
            CHECK_CUDA(cudaMalloc(&net->d_errors[i-1], 
                                 batch_size * dim * sizeof(float)));
        }
    }

    return net;
}

// Free network memory
void free_net(Net* net) {
    // Free device memory
    for(int i = 0; i <= net->depth; i++) {
        cudaFree(net->d_weights[i]);
        cudaFree(net->d_weight_grads[i]);
        cudaFree(net->d_m[i]);
        cudaFree(net->d_v[i]);
    }
    
    for(int i = 0; i <= net->depth + 1; i++) {
        cudaFree(net->d_layer_outputs[i]);
        if(i > 0 && i <= net->depth) {
            cudaFree(net->d_pre_activations[i-1]);
        }
        if(i > 0) {
            cudaFree(net->d_errors[i-1]);
        }
    }
    
    // Free host memory
    for(int i = 0; i <= net->depth; i++) {
        free(net->h_weights[i]);
    }
    
    free(net->d_weights);
    free(net->d_weight_grads);
    free(net->h_weights);
    free(net->d_m);
    free(net->d_v);
    free(net->d_layer_outputs);
    free(net->d_pre_activations);
    free(net->d_errors);
    free(net->layer_dims);
    
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

// CUDA kernel for residual connection
__global__ void add_residual_connection(float* output, float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += residual[idx];
    }
}

// CUDA kernel for RMS normalization
__global__ void rms_norm_kernel(float* output, float* input, int batch_size, int dim, float eps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Calculate RMS for this sample
    float sum_squared = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = input[tid * dim + i];
        sum_squared += val * val;
    }
    float rms = sqrtf(sum_squared / dim + eps);

    // Normalize the values
    for (int i = 0; i < dim; i++) {
        output[tid * dim + i] = input[tid * dim + i] / rms;
    }
}

// CUDA kernel for RMS normalization backward pass
__global__ void rms_norm_backward_kernel(float* grad_output, float* grad_input, 
                                       float* original_input, int batch_size, int dim, float eps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Calculate RMS for this sample
    float sum_squared = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = original_input[tid * dim + i];
        sum_squared += val * val;
    }
    float rms = sqrtf(sum_squared / dim + eps);
    float rms_cubed = rms * rms * rms;

    // Calculate sum of (x_i * grad_i) for this sample
    float sum_grad_times_input = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_grad_times_input += original_input[tid * dim + i] * grad_output[tid * dim + i];
    }

    // Calculate gradient
    for (int i = 0; i < dim; i++) {
        int idx = tid * dim + i;
        float x_i = original_input[idx];
        float grad_i = grad_output[idx];
        
        grad_input[idx] = (grad_i * rms - x_i * sum_grad_times_input / dim) / rms_cubed;
    }
}

// Forward pass with residual connections and RMS normalization
void forward_pass(Net* net, float* X) {
    // Copy input to first layer output
    CHECK_CUDA(cudaMemcpy(net->d_layer_outputs[0], X,
                         net->batch_size * net->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Forward propagation through all layers
    for(int i = 0; i < net->depth + 1; i++) {
        int in_dim = net->layer_dims[i];
        int out_dim = net->layer_dims[i + 1];

        // Matrix multiplication
        CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                out_dim,           // n
                                net->batch_size,   // m
                                in_dim,            // k
                                &alpha,
                                net->d_weights[i], // A
                                out_dim,           // lda
                                net->d_layer_outputs[i], // B
                                in_dim,            // ldb
                                &beta,
                                net->d_layer_outputs[i + 1], // C
                                out_dim));         // ldc

        // Apply activation, normalization and residual connection for hidden layers
        if(i < net->depth) {
            // Store pre-activation values
            CHECK_CUDA(cudaMemcpy(net->d_pre_activations[i],
                                 net->d_layer_outputs[i + 1],
                                 net->batch_size * out_dim * sizeof(float),
                                 cudaMemcpyDeviceToDevice));

            int block_size = 256;
            int num_blocks;

            // Apply Swish activation
            num_blocks = (net->batch_size * out_dim + block_size - 1) / block_size;
            swish_forward_kernel<<<num_blocks, block_size>>>(
                net->d_layer_outputs[i + 1],
                net->d_pre_activations[i],
                net->batch_size * out_dim
            );

            // Apply RMS normalization
            num_blocks = (net->batch_size + block_size - 1) / block_size;
            rms_norm_kernel<<<num_blocks, block_size>>>(
                net->d_layer_outputs[i + 1],
                net->d_layer_outputs[i + 1],
                net->batch_size,
                out_dim,
                1e-6f  // epsilon value
            );

            // Add residual connection if dimensions match
            if (in_dim == out_dim) {
                num_blocks = (net->batch_size * out_dim + block_size - 1) / block_size;
                add_residual_connection<<<num_blocks, block_size>>>(
                    net->d_layer_outputs[i + 1],
                    net->d_layer_outputs[i],
                    net->batch_size * out_dim
                );
            }
        }
    }
}

// Custom kernel for calculating error
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

    int size = net->batch_size * net->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    calc_error_kernel<<<num_blocks, block_size>>>(
        net->d_errors[net->depth],
        net->d_layer_outputs[net->depth + 1],
        d_y,
        size
    );

    float* h_error = (float*)malloc(size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_error, net->d_errors[net->depth], 
                         size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for(int i = 0; i < size; i++) {
        loss += h_error[i] * h_error[i];
    }

    free(h_error);
    cudaFree(d_y);

    return loss / size;
}

// Zero gradients
void zero_gradients(Net* net) {
    for(int i = 0; i <= net->depth; i++) {
        int out_dim = net->layer_dims[i + 1];
        int in_dim = net->layer_dims[i];
        CHECK_CUDA(cudaMemset(net->d_weight_grads[i], 0,
                             out_dim * in_dim * sizeof(float)));
    }
}

// Backward pass
void backward_pass(Net* net, float* X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Backward propagation through all layers
    for(int i = net->depth; i >= 0; i--) {
        int out_dim = net->layer_dims[i + 1];
        int in_dim = net->layer_dims[i];

        // Calculate weight gradients
        CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                out_dim,           // n
                                in_dim,            // m
                                net->batch_size,   // k
                                &alpha,
                                net->d_errors[i],  // A
                                out_dim,           // lda
                                net->d_layer_outputs[i], // B
                                in_dim,            // ldb
                                &beta,
                                net->d_weight_grads[i], // C
                                out_dim));         // ldc

        // Propagate error backward (except for input layer)
        if(i > 0) {
            CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    in_dim,            // n
                                    net->batch_size,   // m
                                    out_dim,           // k
                                    &alpha,
                                    net->d_weights[i], // A
                                    out_dim,           // lda
                                    net->d_errors[i],  // B
                                    out_dim,           // ldb
                                    &beta,
                                    net->d_errors[i-1],// C
                                    in_dim));          // ldc

            int block_size = 256;
            int num_blocks;

            // Add residual gradient if dimensions match
            if (in_dim == out_dim) {
                num_blocks = (net->batch_size * in_dim + block_size - 1) / block_size;
                add_residual_connection<<<num_blocks, block_size>>>(
                    net->d_errors[i-1],
                    net->d_errors[i],
                    net->batch_size * in_dim
                );
            }

            // Apply RMS norm gradient
            num_blocks = (net->batch_size + block_size - 1) / block_size;
            rms_norm_backward_kernel<<<num_blocks, block_size>>>(
                net->d_errors[i-1],
                net->d_errors[i-1],
                net->d_layer_outputs[i],
                net->batch_size,
                in_dim,
                1e-6f
            );

            // Apply Swish derivative
            num_blocks = (net->batch_size * in_dim + block_size - 1) / block_size;
            swish_backward_kernel<<<num_blocks, block_size>>>(
                net->d_errors[i-1],
                net->d_pre_activations[i-1],
                net->batch_size * in_dim
            );
        }
    }
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
    
    // Update weights for all layers
    for(int i = 0; i <= net->depth; i++) {
        int out_dim = net->layer_dims[i + 1];
        int in_dim = net->layer_dims[i];
        int size = out_dim * in_dim;
        int num_blocks = (size + block_size - 1) / block_size;
        
        adamw_update_kernel<<<num_blocks, block_size>>>(
            net->d_weights[i],
            net->d_weight_grads[i],
            net->d_m[i],
            net->d_v[i],
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

// Save model weights to binary file
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save network architecture
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->depth, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    
    int hidden_dim = net->layer_dims[1];
    fwrite(&hidden_dim, sizeof(int), 1, file);
    
    // Copy weights from device to host and save
    for(int i = 0; i <= net->depth; i++) {
        int out_dim = net->layer_dims[i + 1];
        int in_dim = net->layer_dims[i];
        
        CHECK_CUDA(cudaMemcpy(net->h_weights[i], net->d_weights[i],
                             out_dim * in_dim * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        fwrite(net->h_weights[i], sizeof(float), out_dim * in_dim, file);
    }
    
    // Save optimizer state
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
    
    // Load network architecture
    int input_dim, depth, output_dim, batch_size, hidden_dim;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&depth, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    
    // Initialize network
    Net* net = init_net(input_dim, hidden_dim, depth, output_dim, batch_size);
    
    // Load weights
    for(int i = 0; i <= depth; i++) {
        int out_dim = net->layer_dims[i + 1];
        int in_dim = net->layer_dims[i];
        
        fread(net->h_weights[i], sizeof(float), out_dim * in_dim, file);
        
        // Copy weights to device
        CHECK_CUDA(cudaMemcpy(net->d_weights[i], net->h_weights[i],
                             out_dim * in_dim * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
    
    // Load optimizer state
    fread(&net->t, sizeof(int), 1, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif