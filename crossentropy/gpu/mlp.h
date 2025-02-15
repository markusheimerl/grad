#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro.
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuBLAS Error checking macro.
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*
  This file defines a multilayer perceptron (MLP) with an embedding lookup 
  adapted for GPU training using cross‑entropy loss for classification.
  Each raw input (a float) is binned and an embedding vector is looked up.
  The concatenated embedding then flows through a hidden layer with Swish
  activation into an output layer that produces logits. The cross‑entropy loss
  is computed by applying a softmax on the logits.
*/

typedef struct {

    // Embedding parameters.
    int raw_input_dim;    // Number of raw (continuous) input features.
    int num_bins;         // Number of bins per feature.
    int embedding_dim;    // Dimension of each embedding vector.
    // The effective input dimension = raw_input_dim * embedding_dim.
    // The embedding table is stored as a contiguous array of size: 
    // raw_input_dim * num_bins * embedding_dim.
    float* d_embedding_table;   // Device pointer.
    float* h_embedding_table;   // Host copy (for saving/loading).

    // Network weights and gradients.
    int input_dim;    // = raw_input_dim * embedding_dim.
    int hidden_dim;
    int output_dim;   // For classification, this equals the number of classes.
    int batch_size;
    
    // Device pointers for fully‑connected layer weights & gradients.
    // fc1: dimensions hidden_dim x input_dim; fc2: dimensions output_dim x hidden_dim.
    float* d_fc1_weight;
    float* d_fc2_weight;
    float* d_fc1_weight_grad;
    float* d_fc2_weight_grad;
    
    // Host copies (for model saving/loading).
    float* h_fc1_weight;
    float* h_fc2_weight;
    
    // Adam optimizer buffers on device.
    float* d_fc1_m;
    float* d_fc1_v;
    float* d_fc2_m;
    float* d_fc2_v;
    
    // Adam hyperparameters.
    float beta1;   
    float beta2;   
    float epsilon; 
    int   t;         // Time step counter.
    float weight_decay; // Weight decay (lambda) for AdamW.
    
    // Device pointers for helper arrays.
    float* d_layer1_output;   // Hidden layer output (batch_size x hidden_dim).
    float* d_predictions;     // Output logits (batch_size x output_dim).
    float* d_error;           // Gradient/error at output (batch_size x output_dim).
    float* d_pre_activation;  // Pre‑activation of hidden layer (batch_size x hidden_dim).
    float* d_error_hidden;    // Error backpropagated into hidden layer (batch_size x hidden_dim).
    
    // For transferring input and (if needed) labels.
    // d_X holds the embedded input (batch_size x input_dim).
    // d_y is not used for cross–entropy loss.
    float* d_X;
    float* d_y;
    
    // cuBLAS handle.
    cublasHandle_t cublas_handle;
    
} Net;

// ----------------------------------------------------------------------------
// GPU Kernel: embed_input_kernel
// For each sample and feature, determine the bin (from INPUT_RANGE_MIN/MAX)
// and write the corresponding embedding vector (of length embedding_dim)
// into the concatenated output (of shape batch_size x (raw_input_dim * embedding_dim)).
 // ----------------------------------------------------------------------------
__global__ void embed_input_kernel(
    const float* raw_input,           // shape: (batch_size x raw_input_dim)
    const float* embedding_table,     // shape: (raw_input_dim x (num_bins*embedding_dim))
    float* embedded_output,           // output: (batch_size x (raw_input_dim*embedding_dim))
    int batch_size,
    int raw_input_dim,
    int num_bins,
    int embedding_dim,
    float input_range_min,
    float input_range_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * raw_input_dim;
    if (idx < total) {
        int sample = idx / raw_input_dim;
        int feature = idx % raw_input_dim;
        float val = raw_input[sample * raw_input_dim + feature];
        float range = input_range_max - input_range_min;
        int bin_idx = (int)(((val - input_range_min) / range) * num_bins);
        if (bin_idx < 0) bin_idx = 0;
        if (bin_idx >= num_bins) bin_idx = num_bins - 1;
        
        int emb_offset = feature * num_bins * embedding_dim + bin_idx * embedding_dim;
        int dest_offset = sample * (raw_input_dim * embedding_dim) + feature * embedding_dim;
        for (int k = 0; k < embedding_dim; k++) {
            embedded_output[dest_offset + k] = embedding_table[emb_offset + k];
        }
    }
}

// ----------------------------------------------------------------------------
// Host function: embed_input
// Copies raw input (host pointer) to device temporarily and launches the embedding
// kernel to fill net->d_X (embedded representation).
// ----------------------------------------------------------------------------
void embed_input(Net* net, float* raw_input_h) {
    int raw_input_size = net->batch_size * net->raw_input_dim * sizeof(float);
    float* d_raw_input;
    CHECK_CUDA(cudaMalloc(&d_raw_input, raw_input_size));
    CHECK_CUDA(cudaMemcpy(d_raw_input, raw_input_h, raw_input_size, cudaMemcpyHostToDevice));
    
    int total = net->batch_size * net->raw_input_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    // Note: INPUT_RANGE_MIN and INPUT_RANGE_MAX are assumed to be defined (e.g., via a header).
    embed_input_kernel<<<num_blocks, block_size>>>(
        d_raw_input,
        net->d_embedding_table,
        net->d_X,  
        net->batch_size,
        net->raw_input_dim,
        net->num_bins,
        net->embedding_dim,
        INPUT_RANGE_MIN,
        INPUT_RANGE_MAX
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_raw_input);
}

// ----------------------------------------------------------------------------
// Function: init_net
// Initializes the network – allocations, random weights, Adam buffers, embedding table.
// ----------------------------------------------------------------------------
Net* init_net(int raw_input_dim, int num_bins, int embedding_dim,
              int hidden_dim, int output_dim, int batch_size)
{
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate Net structure.\n");
        exit(EXIT_FAILURE);
    }
    
    net->raw_input_dim = raw_input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    net->input_dim = raw_input_dim * embedding_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Adam hyperparameters.
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Create cuBLAS handle.
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Allocate and initialize host memory for FC weights.
    net->h_fc1_weight = (float*)malloc(hidden_dim * net->input_dim * sizeof(float));
    net->h_fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    float scale1 = 1.0f / sqrtf((float)net->input_dim);
    for (int i = 0; i < hidden_dim * net->input_dim; i++) {
        net->h_fc1_weight[i] = (((float)rand()/(float)RAND_MAX) * 2 - 1) * scale1;
    }
    float scale2 = 1.0f / sqrtf((float)hidden_dim);
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->h_fc2_weight[i] = (((float)rand()/(float)RAND_MAX) * 2 - 1) * scale2;
    }
    
    // Allocate device memory for FC weights and gradients.
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight_grad, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight_grad, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, net->h_fc1_weight,
                          hidden_dim * net->input_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, net->h_fc2_weight,
                          output_dim * hidden_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Allocate and initialize Adam buffers.
    CHECK_CUDA(cudaMalloc(&net->d_fc1_m, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_v, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_m, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_v, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc1_m, 0, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc1_v, 0, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    // Allocate helper arrays.
    CHECK_CUDA(cudaMalloc(&net->d_layer1_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_pre_activation, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_error_hidden, batch_size * hidden_dim * sizeof(float)));
    
    // Allocate memory for network input (after embedding) and target output.
    CHECK_CUDA(cudaMalloc(&net->d_X, batch_size * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_y, batch_size * output_dim * sizeof(float)));
    
    // Allocate and initialize the embedding table.
    size_t emb_table_size = raw_input_dim * num_bins * embedding_dim * sizeof(float);
    net->h_embedding_table = (float*)malloc(emb_table_size);
    float emb_scale = 1.0f / sqrtf((float)num_bins);
    for (int i = 0; i < raw_input_dim * num_bins * embedding_dim; i++) {
        net->h_embedding_table[i] = (((float)rand()/(float)RAND_MAX) * 2 - 1) * emb_scale;
    }
    CHECK_CUDA(cudaMalloc(&net->d_embedding_table, emb_table_size));
    CHECK_CUDA(cudaMemcpy(net->d_embedding_table, net->h_embedding_table,
                          emb_table_size, cudaMemcpyHostToDevice));
                          
    return net;
}

// ----------------------------------------------------------------------------
// GPU Kernel: swish_forward
// Applies the Swish activation function: f(x) = x/(1+exp(-x)).
// ----------------------------------------------------------------------------
__global__ void swish_forward(float *out, const float *pre, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = pre[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

// ----------------------------------------------------------------------------
// Function: forward_pass
// Given input stored in net->d_X, computes the hidden layer outputs and final logits.
// ----------------------------------------------------------------------------
void forward_pass(Net* net) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // First layer: d_layer1_output = d_fc1_weight * d_X.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             net->hidden_dim,      // rows of d_fc1_weight and d_layer1_output
                             net->batch_size,      // columns of d_X and d_layer1_output
                             net->input_dim,       // common dim
                             &alpha,
                             net->d_fc1_weight,    // A
                             net->hidden_dim,      // lda
                             net->d_X,             // B
                             net->input_dim,       // ldb
                             &beta,
                             net->d_layer1_output, // C
                             net->hidden_dim));    // ldc
                             
    // Save pre-activation.
    CHECK_CUDA(cudaMemcpy(net->d_pre_activation, net->d_layer1_output,
                          net->batch_size * net->hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Apply Swish activation.
    int total = net->batch_size * net->hidden_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    swish_forward<<<num_blocks, block_size>>>(net->d_layer1_output, net->d_pre_activation, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Second layer: d_predictions = d_fc2_weight * d_layer1_output.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             net->output_dim,       // rows of d_fc2_weight and d_predictions
                             net->batch_size,       // columns of d_layer1_output and d_predictions
                             net->hidden_dim,       // common dim
                             &alpha,
                             net->d_fc2_weight,     // A
                             net->output_dim,       // lda
                             net->d_layer1_output,  // B
                             net->hidden_dim,       // ldb
                             &beta,
                             net->d_predictions,    // C
                             net->output_dim));     // ldc
}

// ----------------------------------------------------------------------------
// GPU Kernel: cross_entropy_loss_kernel
// For each sample (row in d_predictions), this kernel computes the softmax and 
// cross‑entropy loss (with an epsilon for stability). It also sets d_error 
// to (softmax(probabilities) - one_hot(target)).
// ----------------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(const float* predictions,
                                            const int* target_labels,
                                            float* out_loss,
                                            float* d_error,
                                            int batch_size,
                                            int num_classes,
                                            float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        int offset = i * num_classes;
        float max_logit = predictions[offset];
        // Find maximum logit for numerical stability.
        for (int j = 1; j < num_classes; j++) {
            float logit = predictions[offset + j];
            if (logit > max_logit)
                max_logit = logit;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(predictions[offset + j] - max_logit);
        }
        float loss_i = 0.0f;
        int target = target_labels[i];
        for (int j = 0; j < num_classes; j++) {
            float exp_val = expf(predictions[offset + j] - max_logit);
            float prob = exp_val / sum_exp;
            // Store gradient: (softmax(prob) - one_hot)
            d_error[offset + j] = prob - ((j == target) ? 1.0f : 0.0f);
            if (j == target) {
                loss_i = -logf(prob + epsilon);
            }
        }
        out_loss[i] = loss_i;
    }
}

// ----------------------------------------------------------------------------
// Function: calculate_loss
// Computes the average cross‑entropy loss over the batch and sets d_error
// (the gradient at the output layer). The target labels (discrete class indices)
// are provided from the host as an integer array of length batch_size.
// ----------------------------------------------------------------------------
float calculate_loss(Net* net, int* target_labels_h) {
    int batch = net->batch_size;
    
    // Allocate device memory for target labels and copy from host.
    int *d_target_labels;
    CHECK_CUDA(cudaMalloc(&d_target_labels, batch * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_target_labels, target_labels_h, batch * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate device memory for per-sample loss.
    float* d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, batch * sizeof(float)));
    
    int block_size = 256;
    int num_blocks = (batch + block_size - 1) / block_size;
    cross_entropy_loss_kernel<<<num_blocks, block_size>>>(net->d_predictions, d_target_labels, d_loss, net->d_error, batch, net->output_dim, net->epsilon);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy loss values to host and compute the average.
    float* h_loss = (float*)malloc(batch * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_loss, d_loss, batch * sizeof(float), cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for (int i = 0; i < batch; i++) {
        loss += h_loss[i];
    }
    loss /= batch;
    
    free(h_loss);
    cudaFree(d_loss);
    cudaFree(d_target_labels);
    
    return loss;
}

// ----------------------------------------------------------------------------
// Function: zero_gradients
// Zeros the weight gradient arrays before performing backpropagation.
// ----------------------------------------------------------------------------
void zero_gradients(Net* net) {
    CHECK_CUDA(cudaMemset(net->d_fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float)));
}

// ----------------------------------------------------------------------------
// GPU Kernel: swish_backward_kernel
// Computes the derivative of the Swish activation function and multiplies it
// with the backpropagated error.
// The derivative is: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
// ----------------------------------------------------------------------------
__global__ void swish_backward_kernel(float* error_hidden, const float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = pre_activation[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        error_hidden[idx] *= (sigmoid + x * sigmoid * (1.0f - sigmoid));
    }
}

// ----------------------------------------------------------------------------
// Function: backward_pass
// Propagates gradients backward through the network. Assumes that d_error has
// already been computed via calculate_loss.
// ----------------------------------------------------------------------------
void backward_pass(Net* net) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Compute gradients for fc2: d_fc2_weight_grad = d_error * (d_layer1_output)^T.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             net->output_dim,
                             net->hidden_dim,
                             net->batch_size,
                             &alpha,
                             net->d_error,
                             net->output_dim,
                             net->d_layer1_output,
                             net->hidden_dim,
                             &beta,
                             net->d_fc2_weight_grad,
                             net->output_dim));
    
    // Backpropagate error to hidden layer: d_error_hidden = (d_fc2_weight)^T * d_error.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             net->hidden_dim,
                             net->batch_size,
                             net->output_dim,
                             &alpha,
                             net->d_fc2_weight,
                             net->output_dim,
                             net->d_error,
                             net->output_dim,
                             &beta,
                             net->d_error_hidden,
                             net->hidden_dim));
    
    // Apply derivative of the Swish activation.
    int total = net->batch_size * net->hidden_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(net->d_error_hidden, net->d_pre_activation, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute gradients for fc1: d_fc1_weight_grad = d_error_hidden * (d_X)^T.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             net->hidden_dim,
                             net->input_dim,
                             net->batch_size,
                             &alpha,
                             net->d_error_hidden,
                             net->hidden_dim,
                             net->d_X,
                             net->input_dim,
                             &beta,
                             net->d_fc1_weight_grad,
                             net->hidden_dim));
}

// ----------------------------------------------------------------------------
// GPU Kernel: adamw_update_kernel
// Updates the weights using the AdamW optimizer.
// ----------------------------------------------------------------------------
__global__ void adamw_update_kernel(
    float* weight,
    const float* grad,
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
        float update_val = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update_val;
    }
}

// ----------------------------------------------------------------------------
// Function: update_weights
// Updates the FC weights using the AdamW optimizer.
// ----------------------------------------------------------------------------
void update_weights(Net* net, float learning_rate) {
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update fc1 weights.
    int fc1_size = net->hidden_dim * net->input_dim;
    int fc1_blocks = (fc1_size + block_size - 1) / block_size;
    adamw_update_kernel<<<fc1_blocks, block_size>>>(
        net->d_fc1_weight,
        net->d_fc1_weight_grad,
        net->d_fc1_m,
        net->d_fc1_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        fc1_size,
        net->batch_size
    );
    
    // Update fc2 weights.
    int fc2_size = net->output_dim * net->hidden_dim;
    int fc2_blocks = (fc2_size + block_size - 1) / block_size;
    adamw_update_kernel<<<fc2_blocks, block_size>>>(
        net->d_fc2_weight,
        net->d_fc2_weight_grad,
        net->d_fc2_m,
        net->d_fc2_v,
        net->beta1,
        net->beta2,
        net->epsilon,
        learning_rate,
        net->weight_decay,
        alpha_t,
        fc2_size,
        net->batch_size
    );
}

// ----------------------------------------------------------------------------
// Function: save_model
// Saves the network (including FC weights and the embedding table) to a binary file.
// ----------------------------------------------------------------------------
void save_model(Net* net, const char* filename) {
    // Copy weights from device to host.
    CHECK_CUDA(cudaMemcpy(net->h_fc1_weight, net->d_fc1_weight,
                          net->hidden_dim * net->input_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_fc2_weight, net->d_fc2_weight,
                          net->output_dim * net->hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Also copy the embedding table from device.
    size_t emb_table_count = net->raw_input_dim * net->num_bins * net->embedding_dim;
    CHECK_CUDA(cudaMemcpy(net->h_embedding_table, net->d_embedding_table,
                          emb_table_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    // Save dimensions.
    fwrite(&net->raw_input_dim, sizeof(int), 1, file);
    fwrite(&net->num_bins, sizeof(int), 1, file);
    fwrite(&net->embedding_dim, sizeof(int), 1, file);
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    
    // Save fc1 and fc2 weights.
    fwrite(net->h_fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->h_fc2_weight, sizeof(float), net->output_dim * net->hidden_dim, file);
    
    // Save Adam state.
    fwrite(&net->t, sizeof(int), 1, file);
    // (For simplicity we do not save Adam moment vectors.)
    
    // Save embedding table.
    fwrite(net->h_embedding_table, sizeof(float), emb_table_count, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ----------------------------------------------------------------------------
// Function: load_model
// Loads the network from a binary file and copies the weights to device.
// ----------------------------------------------------------------------------
Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int raw_input_dim, num_bins, embedding_dim, input_dim, hidden_dim, output_dim, batch_size;
    fread(&raw_input_dim, sizeof(int), 1, file);
    fread(&num_bins, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize network.
    Net* net = init_net(raw_input_dim, num_bins, embedding_dim, hidden_dim, output_dim, batch_size);
    
    // Load fc1 and fc2 weights.
    fread(net->h_fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->h_fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    fread(&net->t, sizeof(int), 1, file);
    
    // Copy weights to device.
    CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, net->h_fc1_weight,
                          hidden_dim * input_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, net->h_fc2_weight,
                          output_dim * hidden_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Load embedding table.
    size_t emb_table_count = raw_input_dim * num_bins * embedding_dim;
    fread(net->h_embedding_table, sizeof(float), emb_table_count, file);
    CHECK_CUDA(cudaMemcpy(net->d_embedding_table, net->h_embedding_table,
                          emb_table_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

// ----------------------------------------------------------------------------
// Function: free_net
// Frees device and host memory allocated for the network.
// ----------------------------------------------------------------------------
void free_net(Net* net) {
    // Free device memory.
    cudaFree(net->d_fc1_weight);
    cudaFree(net->d_fc2_weight);
    cudaFree(net->d_fc1_weight_grad);
    cudaFree(net->d_fc2_weight_grad);
    cudaFree(net->d_fc1_m);
    cudaFree(net->d_fc1_v);
    cudaFree(net->d_fc2_m);
    cudaFree(net->d_fc2_v);
    cudaFree(net->d_layer1_output);
    cudaFree(net->d_predictions);
    cudaFree(net->d_error);
    cudaFree(net->d_pre_activation);
    cudaFree(net->d_error_hidden);
    cudaFree(net->d_X);
    cudaFree(net->d_y);
    cudaFree(net->d_embedding_table);
    
    // Free host memory.
    free(net->h_fc1_weight);
    free(net->h_fc2_weight);
    free(net->h_embedding_table);
    
    // Destroy cuBLAS handle.
    cublasDestroy(net->cublas_handle);
    
    free(net);
}

#endif