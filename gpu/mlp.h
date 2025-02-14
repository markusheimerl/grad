#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Define input range constants if not defined elsewhere.
#ifndef INPUT_RANGE_MIN
  #define INPUT_RANGE_MIN 0.0f
#endif
#ifndef INPUT_RANGE_MAX
  #define INPUT_RANGE_MAX 1.0f
#endif

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

typedef struct {

    // Embedding parameters.
    int raw_input_dim;   // Number of raw (continuous) input features.
    int num_bins;        // Number of bins per feature.
    int embedding_dim;   // Dimension of each embedding vector.
    // The effective input dimension is: raw_input_dim * embedding_dim.
    // This value is stored in "input_dim" below.
    
    // Device pointer for embedding table.
    // Stored as a contiguous array of size: raw_input_dim * num_bins * embedding_dim.
    // For feature j, the embeddings start at index: j*(num_bins*embedding_dim)
    float* d_embedding_table;
    // Host copy (used for model save/load).
    float* h_embedding_table;

    // Network weights and corresponding gradients.
    // Now, the input dimension is the effective embedding dimension.
    int input_dim; // = raw_input_dim * embedding_dim
    int hidden_dim;
    int output_dim;
    int batch_size;
    
    // Device pointers for fully-connected layers weights and gradients.
    float* d_fc1_weight;      // Dimensions: hidden_dim x input_dim
    float* d_fc2_weight;      // Dimensions: output_dim x hidden_dim
    float* d_fc1_weight_grad; // Same dimensions as d_fc1_weight.
    float* d_fc2_weight_grad; // Same dimensions as d_fc2_weight.

    // Host copies (for saving/loading).
    float* h_fc1_weight;
    float* h_fc2_weight;
    
    // Adam optimizer device buffers.
    float* d_fc1_m;  // First moment for fc1.
    float* d_fc1_v;  // Second moment for fc1.
    float* d_fc2_m;  // First moment for fc2.
    float* d_fc2_v;  // Second moment for fc2.
    
    // Adam hyperparameters.
    float beta1;   
    float beta2;   
    float epsilon; 
    int t;         // Time step counter.
    float weight_decay; // Weight decay (lambda) for AdamW.
    
    // Device pointers for helper arrays.
    float* d_layer1_output;   // Output of hidden layer (batch_size x hidden_dim).
    float* d_predictions;     // Output predictions (batch_size x output_dim).
    float* d_error;           // Error at output (batch_size x output_dim).
    float* d_pre_activation;  // Pre-activation values (batch_size x hidden_dim).
    float* d_error_hidden;    // Error backpropagated into hidden layer (batch_size x hidden_dim).
    
    // For passing data through the network.
    // d_X holds the network input (after embedding) of shape: batch_size x input_dim.
    // d_y holds the target output (batch_size x output_dim).
    float* d_X;
    float* d_y;
    
    // cuBLAS handle.
    cublasHandle_t cublas_handle;
    
} Net;

// ----------------------------------------------------------------------------
// Embedding lookup kernel.
// For each sample (i) and feature (j), compute the bin index from the raw input
// value, then copy the corresponding embedding vector (of length embedding_dim)
// from the embedding table. The table is laid out as:
//   [for feature 0: {bin0, bin1, ...}, for feature 1: {bin0, bin1, ...}, ...].
// The raw input is of shape (batch_size x raw_input_dim) and the embedded output
// is of shape (batch_size x (raw_input_dim * embedding_dim)).
// ----------------------------------------------------------------------------
__global__ void embed_input_kernel(
    const float* raw_input,           // shape: batch_size x raw_input_dim.
    const float* embedding_table,     // shape: raw_input_dim x (num_bins*embedding_dim).
    float* embedded_output,           // output: batch_size x (raw_input_dim*embedding_dim).
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
        // Get the raw value.
        float val = raw_input[sample * raw_input_dim + feature];
        float range = input_range_max - input_range_min;
        int bin_idx = (int)(((val - input_range_min) / range) * num_bins);
        if (bin_idx < 0) bin_idx = 0;
        if (bin_idx >= num_bins) bin_idx = num_bins - 1;
        
        // Pointer into embedding table for this feature.
        // For feature j, the embeddings start at: embedding_table + j*(num_bins*embedding_dim)
        int emb_offset = feature * num_bins * embedding_dim + bin_idx * embedding_dim;
        // Destination in embedded_output:
        // Each sample gets a concatenated vector of all features.
        int dest_offset = sample * (raw_input_dim * embedding_dim) + feature * embedding_dim;
        // Copy the embedding vector.
        for (int k = 0; k < embedding_dim; k++) {
            embedded_output[dest_offset + k] = embedding_table[emb_offset + k];
        }
    }
}

// ----------------------------------------------------------------------------
// Host function: embed raw input (host pointer) into embeddings on device.
// raw_input_h is assumed to have shape: (batch_size x raw_input_dim).
// The kernel writes the output into net->d_X, which is later used in forward_pass.
// ----------------------------------------------------------------------------
void embed_input(Net* net, float* raw_input_h) {
    // Allocate temporary device memory for the raw input.
    int raw_input_size = net->batch_size * net->raw_input_dim * sizeof(float);
    float* d_raw_input;
    CHECK_CUDA(cudaMalloc(&d_raw_input, raw_input_size));
    CHECK_CUDA(cudaMemcpy(d_raw_input, raw_input_h, raw_input_size, cudaMemcpyHostToDevice));

    // Launch the embedding lookup kernel.
    int total = net->batch_size * net->raw_input_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    embed_input_kernel<<<num_blocks, block_size>>>(
        d_raw_input,
        net->d_embedding_table,
        net->d_X,  // d_X stores the effective embedded input.
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
// Initialize the network with embeddings and FC layers.
// Now the function requires raw_input_dim, num_bins and embedding_dim.
// ----------------------------------------------------------------------------
Net* init_net(int raw_input_dim, int num_bins, int embedding_dim,
              int hidden_dim, int output_dim, int batch_size)
{
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate Net structure.\n");
        exit(EXIT_FAILURE);
    }
    
    // Store embedding parameters.
    net->raw_input_dim = raw_input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    // The effective input dimension is:
    net->input_dim = raw_input_dim * embedding_dim;
    
    // Store MLP dimensions.
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Initialize Adam hyperparameters.
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Create cuBLAS handle.
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Allocate host memory for FC weights.
    net->h_fc1_weight = (float*)malloc(hidden_dim * net->input_dim * sizeof(float));
    net->h_fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Initialize FC weights.
    float scale1 = 1.0f / sqrt((float)net->input_dim);
    for (int i = 0; i < hidden_dim * net->input_dim; i++) {
        net->h_fc1_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale1;
    }
    float scale2 = 1.0f / sqrt((float)hidden_dim);
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->h_fc2_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale2;
    }
    
    // Allocate device memory for FC weights and gradients.
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc1_weight_grad, hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&net->d_fc2_weight_grad, output_dim * hidden_dim * sizeof(float)));
    
    // Copy host-initialized weights to device.
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
    float emb_scale = 1.0f / sqrt((float)num_bins);
    for (int i = 0; i < raw_input_dim * num_bins * embedding_dim; i++) {
        net->h_embedding_table[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * emb_scale;
    }
    CHECK_CUDA(cudaMalloc(&net->d_embedding_table, emb_table_size));
    CHECK_CUDA(cudaMemcpy(net->d_embedding_table, net->h_embedding_table,
                          emb_table_size, cudaMemcpyHostToDevice));
    
    return net;
}

__global__ void swish_forward(float *out, const float *pre, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = pre[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

// ----------------------------------------------------------------------------
// Forward pass of the MLP (using an already-embedded input that is stored in net->d_X).
// In this version, we assume that embed_input has been called to fill net->d_X.
// ----------------------------------------------------------------------------
void forward_pass(Net* net) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // First layer: compute d_layer1_output = d_fc1_weight * d_X.
    // Note: Using column-major semantics in cuBLAS (with appropriate transposition) or
    // adjusting parameters to suit our row-major layout.
    // Here we call cublasSgemm with:
    //   d_layer1_output (hidden_dim x batch_size) = d_fc1_weight (hidden_dim x input_dim) * d_X (input_dim x batch_size)
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
                             
    // Save pre-activation in d_pre_activation.
    CHECK_CUDA(cudaMemcpy(net->d_pre_activation, net->d_layer1_output,
                          net->batch_size * net->hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Apply the Swish activation: f(x) = x / (1+exp(-x)).
    // Launch a kernel to do this elementwise.
    int total = net->batch_size * net->hidden_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    swish_forward<<<num_blocks, block_size>>>(net->d_layer1_output, net->d_pre_activation, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    
    // Second layer: compute predictions.
    // d_predictions = d_fc2_weight * d_layer1_output.
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
// Custom kernel for calculating error: error = predictions - y.
// ----------------------------------------------------------------------------
__global__ void calc_error_kernel(float* error,
                                  const float* predictions,
                                  const float* y,
                                  int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// ----------------------------------------------------------------------------
// Calculate loss (mean squared error) and compute gradient at output.
// Copy target output y from host to device (net->d_y is used).
// ----------------------------------------------------------------------------
float calculate_loss(Net* net, float* y_h) {
    // Copy y from host to device.
    int y_size = net->batch_size * net->output_dim * sizeof(float);
    CHECK_CUDA(cudaMemcpy(net->d_y, y_h, y_size, cudaMemcpyHostToDevice));
    
    int total = net->batch_size * net->output_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    calc_error_kernel<<<num_blocks, block_size>>>(net->d_error, net->d_predictions, net->d_y, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy error to host to compute loss.
    float* h_error = (float*)malloc(total * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_error, net->d_error, total * sizeof(float), cudaMemcpyDeviceToHost));
    float loss = 0.0f;
    for (int i = 0; i < total; i++) {
        loss += h_error[i] * h_error[i];
    }
    free(h_error);
    return loss / total;
}

// ----------------------------------------------------------------------------
// Zero the gradients (set to zero).
// ----------------------------------------------------------------------------
void zero_gradients(Net* net) {
    CHECK_CUDA(cudaMemset(net->d_fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(net->d_fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float)));
}

// ----------------------------------------------------------------------------
// Kernel for the derivative of Swish activation applied to the propagated error.
// The derivative: f'(x)= sigmoid(x)+ x * sigmoid(x)*(1-sigmoid(x))
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
// Backward pass of the network.
// ----------------------------------------------------------------------------
void backward_pass(Net* net) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Backprop for fc2 weights: d_fc2_weight_grad = d_error * (d_layer1_output)^T.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             net->output_dim,       // rows of grad matrix
                             net->hidden_dim,       // columns of grad matrix
                             net->batch_size,       // summing over samples
                             &alpha,
                             net->d_error,          // (output_dim x batch_size)
                             net->output_dim,
                             net->d_layer1_output,  // (hidden_dim x batch_size)
                             net->hidden_dim,
                             &beta,
                             net->d_fc2_weight_grad,// Result: (output_dim x hidden_dim)
                             net->output_dim));
    
    // Backpropagate error to hidden layer: d_error_hidden = (d_fc2_weight)^T * d_error.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             net->hidden_dim,       // rows of result
                             net->batch_size,       // columns of result
                             net->output_dim,       // common dimension
                             &alpha,
                             net->d_fc2_weight,      // (output_dim x hidden_dim)
                             net->output_dim,
                             net->d_error,           // (output_dim x batch_size)
                             net->output_dim,
                             &beta,
                             net->d_error_hidden,    // (hidden_dim x batch_size)
                             net->hidden_dim));
    
    // Apply derivative of Swish activation.
    int total = net->batch_size * net->hidden_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(net->d_error_hidden, net->d_pre_activation, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Backprop for fc1 weights: d_fc1_weight_grad = d_error_hidden * (d_X)^T.
    CHECK_CUBLAS(cublasSgemm(net->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             net->hidden_dim,         // rows
                             net->input_dim,          // columns
                             net->batch_size,         // common dim
                             &alpha,
                             net->d_error_hidden,     // (hidden_dim x batch_size)
                             net->hidden_dim,
                             net->d_X,                // (input_dim x batch_size)
                             net->input_dim,
                             &beta,
                             net->d_fc1_weight_grad,  // (hidden_dim x input_dim)
                             net->hidden_dim));
}

// ----------------------------------------------------------------------------
// CUDA kernel for AdamW update.
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
// Update weights using AdamW optimizer.
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
// Save model to a binary file (including embedding table).
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
    // (For simplicity, we do not save the Adam moment vectors.)
    
    // Save embedding table.
    fwrite(net->h_embedding_table, sizeof(float), emb_table_count, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ----------------------------------------------------------------------------
// Load model from a binary file (including embedding table).
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
// Free network memory.
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