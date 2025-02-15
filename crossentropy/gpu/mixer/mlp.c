#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../../data.h"     // Provides generate_synthetic_data, save_data_to_csv, and defines INPUT_RANGE_MIN/MAX.
#include "mlp.h"

// -----------------------------------------------------------------------------
// init_net: Allocates and initializes the network and its parameters.
Net* init_net(int raw_input_dim, int embedding_num_bins, int embedding_dim, int batch_size)
{
    Net* net = (Net*)malloc(sizeof(Net));
    if(!net){
        fprintf(stderr, "Failed to allocate Net.\n");
        exit(EXIT_FAILURE);
    }
    net->raw_input_dim = raw_input_dim;
    net->num_bins = embedding_num_bins;
    net->embedding_dim = embedding_dim;
    net->input_dim = raw_input_dim * embedding_dim;
    net->num_raw_targets = raw_input_dim;
    net->target_bins = embedding_dim;
    net->output_dim = net->input_dim;
    net->batch_size = batch_size;
    
    // Adam hyperparameters.
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Create cuBLAS handle (for consistency, even if not used in custom kernels).
    CHECK_CUBLAS(cublasCreate(&net->cublas_handle));
    
    // Allocate and initialize the embedding table.
    size_t emb_table_size = raw_input_dim * embedding_num_bins * embedding_dim * sizeof(float);
    net->h_embedding_table = (float*)malloc(emb_table_size);
    for(int i = 0; i < raw_input_dim * embedding_num_bins * embedding_dim; i++){
        float scale = 1.0f / sqrtf((float)embedding_num_bins);
        net->h_embedding_table[i] = (((float)rand()/RAND_MAX)*2 - 1) * scale;
    }
    CHECK_CUDA(cudaMalloc(&net->d_embedding_table, emb_table_size));
    CHECK_CUDA(cudaMemcpy(net->d_embedding_table, net->h_embedding_table, emb_table_size, cudaMemcpyHostToDevice));
    
    // Allocate gradient and Adam buffers for embedding table.
    CHECK_CUDA(cudaMalloc(&net->d_embedding_grad, emb_table_size));
    CHECK_CUDA(cudaMemset(net->d_embedding_grad, 0, emb_table_size));
    CHECK_CUDA(cudaMalloc(&net->d_embedding_m, emb_table_size));
    CHECK_CUDA(cudaMalloc(&net->d_embedding_v, emb_table_size));
    CHECK_CUDA(cudaMemset(net->d_embedding_m, 0, emb_table_size));
    CHECK_CUDA(cudaMemset(net->d_embedding_v, 0, emb_table_size));
    
    // Allocate embedding indices buffer.
    CHECK_CUDA(cudaMalloc(&net->d_embed_indices, batch_size * raw_input_dim * sizeof(int)));
    
    // Allocate and initialize projection weights (Wq, Wk, Wv). Each is [embedding_dim x embedding_dim].
    size_t proj_size = embedding_dim * embedding_dim * sizeof(float);
    net->h_Wq = (float*)malloc(proj_size);
    net->h_Wk = (float*)malloc(proj_size);
    net->h_Wv = (float*)malloc(proj_size);
    float scale_q = 1.0f / sqrtf((float)embedding_dim);
    for(int i = 0; i < embedding_dim * embedding_dim; i++){
        net->h_Wq[i] = (((float)rand()/RAND_MAX)*2 - 1) * scale_q;
        net->h_Wk[i] = (((float)rand()/RAND_MAX)*2 - 1) * scale_q;
        net->h_Wv[i] = (((float)rand()/RAND_MAX)*2 - 1) * scale_q;
    }
    CHECK_CUDA(cudaMalloc(&net->d_Wq, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wk, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wv, proj_size));
    CHECK_CUDA(cudaMemcpy(net->d_Wq, net->h_Wq, proj_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_Wk, net->h_Wk, proj_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_Wv, net->h_Wv, proj_size, cudaMemcpyHostToDevice));
    
    // Allocate gradients and Adam buffers for projections.
    CHECK_CUDA(cudaMalloc(&net->d_Wq_grad, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wk_grad, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wv_grad, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wq_grad, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wk_grad, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wv_grad, 0, proj_size));
    
    CHECK_CUDA(cudaMalloc(&net->d_Wq_m, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wq_v, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wk_m, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wk_v, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wv_m, proj_size));
    CHECK_CUDA(cudaMalloc(&net->d_Wv_v, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wq_m, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wq_v, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wk_m, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wk_v, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wv_m, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wv_v, 0, proj_size));
    
    // Allocate buffers for embedded input and projections.
    size_t X_size = batch_size * net->input_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&net->d_X, X_size));
    
    CHECK_CUDA(cudaMalloc(&net->d_Q, X_size));
    CHECK_CUDA(cudaMalloc(&net->d_K, X_size));
    CHECK_CUDA(cudaMalloc(&net->d_V, X_size));
    
    CHECK_CUDA(cudaMalloc(&net->d_Q_grad, X_size));
    CHECK_CUDA(cudaMalloc(&net->d_K_grad, X_size));
    CHECK_CUDA(cudaMalloc(&net->d_V_grad, X_size));
    
    CHECK_CUDA(cudaMalloc(&net->d_X_grad, X_size));
    
    // Allocate predictions and error buffers.
    size_t pred_size = batch_size * net->output_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&net->d_predictions, pred_size));
    CHECK_CUDA(cudaMalloc(&net->d_error, pred_size));
    
    // Allocate target buffer (if needed).
    CHECK_CUDA(cudaMalloc(&net->d_y, pred_size));
    
    // Allocate buffer for attention weights: [batch_size x (raw_input_dim * raw_input_dim)]
    size_t attn_size = batch_size * raw_input_dim * raw_input_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&net->d_attn_weights, attn_size));
    
    return net;
}

// -----------------------------------------------------------------------------
// embed_input: Copies raw input (host) to device, launches the embedding lookup kernel.
void embed_input(Net* net, float* raw_input_h)
{
    int raw_input_size = net->batch_size * net->raw_input_dim * sizeof(float);
    float* d_raw_input;
    CHECK_CUDA(cudaMalloc(&d_raw_input, raw_input_size));
    CHECK_CUDA(cudaMemcpy(d_raw_input, raw_input_h, raw_input_size, cudaMemcpyHostToDevice));
    
    int total = net->batch_size * net->raw_input_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    embed_input_kernel<<<num_blocks, block_size>>>(
        d_raw_input,
        net->d_embedding_table,
        net->d_X,  
        net->d_embed_indices,
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

// -----------------------------------------------------------------------------
// forward_pass: Runs the forward pass (projection followed by attention).
void forward_pass(Net* net)
{
    int total_tokens = net->batch_size * net->raw_input_dim;
    int block_size = 256;
    int num_blocks = (total_tokens + block_size - 1) / block_size;
    // Compute Q = X * Wq
    project_forward_kernel<<<num_blocks, block_size>>>(
        net->d_X, net->d_Wq, net->d_Q,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    // Compute K = X * Wk
    project_forward_kernel<<<num_blocks, block_size>>>(
        net->d_X, net->d_Wk, net->d_K,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    // Compute V = X * Wv
    project_forward_kernel<<<num_blocks, block_size>>>(
        net->d_X, net->d_Wv, net->d_V,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Launch attention forward kernel.
    // One block per sample; each block has raw_input_dim threads.
    attention_forward_kernel<<<net->batch_size, net->raw_input_dim>>>(
        net->d_Q, net->d_K, net->d_V,
        net->d_attn_weights, net->d_predictions,
        net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// calculate_loss: Computes average cross-entropy loss over all tokens.
float calculate_loss(Net* net, int* target_labels_h)
{
    int total_groups = net->batch_size * net->num_raw_targets;
    int* d_target_labels;
    CHECK_CUDA(cudaMalloc(&d_target_labels, total_groups * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_target_labels, target_labels_h, total_groups * sizeof(int), cudaMemcpyHostToDevice));
    
    float* d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, total_groups * sizeof(float)));
    
    int block_size = 256;
    int num_blocks = (total_groups + block_size - 1) / block_size;
    cross_entropy_loss_kernel<<<num_blocks, block_size>>>(
        net->d_predictions, d_target_labels, d_loss, net->d_error,
        net->batch_size, net->num_raw_targets, net->target_bins, net->epsilon
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float* h_loss = (float*)malloc(total_groups * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_loss, d_loss, total_groups * sizeof(float), cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for(int i = 0; i < total_groups; i++){
        loss += h_loss[i];
    }
    loss /= total_groups;
    
    free(h_loss);
    cudaFree(d_loss);
    cudaFree(d_target_labels);
    
    return loss;
}

// -----------------------------------------------------------------------------
// zero_gradients: Sets all gradient buffers to zero.
void zero_gradients(Net* net)
{
    int X_size = net->batch_size * net->input_dim * sizeof(float);
    int proj_size = net->embedding_dim * net->embedding_dim * sizeof(float);
    int attn_grad_size = net->batch_size * net->input_dim * sizeof(float);
    
    CHECK_CUDA(cudaMemset(net->d_Wq_grad, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wk_grad, 0, proj_size));
    CHECK_CUDA(cudaMemset(net->d_Wv_grad, 0, proj_size));
    int emb_table_size = net->raw_input_dim * net->num_bins * net->embedding_dim * sizeof(float);
    CHECK_CUDA(cudaMemset(net->d_embedding_grad, 0, emb_table_size));
    
    CHECK_CUDA(cudaMemset(net->d_Q_grad, 0, X_size));
    CHECK_CUDA(cudaMemset(net->d_K_grad, 0, X_size));
    CHECK_CUDA(cudaMemset(net->d_V_grad, 0, X_size));
    CHECK_CUDA(cudaMemset(net->d_X_grad, 0, X_size));
}
__global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        C[idx] += B[idx];
    }
}
// -----------------------------------------------------------------------------
// backward_pass: Backpropagates gradients through attention, projection, and embedding.
void backward_pass(Net* net)
{
    // 1. Attention backward: compute gradients dQ, dK, dV given d_error (dO).
    // Launch one block per sample.
    attention_backward_kernel<<<net->batch_size, 1>>>(
        net->d_error, net->d_Q, net->d_K, net->d_V,
        net->d_attn_weights,
        net->d_Q_grad, net->d_K_grad, net->d_V_grad,
        net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 2. Projection backward: For each projection, compute dX (from each branch) and dW.
    int total_tokens = net->batch_size * net->raw_input_dim;
    int block_size = 256;
    int num_blocks = (total_tokens + block_size - 1) / block_size;
    
    // For Wq:
    float* dX_q;
    CHECK_CUDA(cudaMalloc(&dX_q, total_tokens * net->embedding_dim * sizeof(float)));
    project_backward_dX_kernel<<<num_blocks, block_size>>>(
        net->d_Q_grad, net->d_Wq, dX_q,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    project_backward_dW_kernel<<<net->embedding_dim, net->embedding_dim>>>(
        net->d_X, net->d_Q_grad, net->d_Wq_grad,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // For Wk:
    float* dX_k;
    CHECK_CUDA(cudaMalloc(&dX_k, total_tokens * net->embedding_dim * sizeof(float)));
    project_backward_dX_kernel<<<num_blocks, block_size>>>(
        net->d_K_grad, net->d_Wk, dX_k,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    project_backward_dW_kernel<<<net->embedding_dim, net->embedding_dim>>>(
        net->d_X, net->d_K_grad, net->d_Wk_grad,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // For Wv:
    float* dX_v;
    CHECK_CUDA(cudaMalloc(&dX_v, total_tokens * net->embedding_dim * sizeof(float)));
    project_backward_dX_kernel<<<num_blocks, block_size>>>(
        net->d_V_grad, net->d_Wv, dX_v,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    project_backward_dW_kernel<<<net->embedding_dim, net->embedding_dim>>>(
        net->d_X, net->d_V_grad, net->d_Wv_grad,
        net->batch_size, net->raw_input_dim, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 3. Accumulate the gradients with respect to X from the three branches.
    // d_X_grad = dX_q + dX_k + dX_v  (element-wise addition)
    // We use a simple loop kernel (or do on host) – for simplicity use cudaMemcpy and then add on GPU.
    // Here we assume total_tokens*embedding_dim elements.
    int numel = total_tokens * net->embedding_dim;
    // First, copy dX_q to d_X_grad.
    CHECK_CUDA(cudaMemcpy(net->d_X_grad, dX_q, numel * sizeof(float), cudaMemcpyDeviceToDevice));
    // Then add dX_k to d_X_grad.
    // Use a simple kernel: (one thread per element)
    int add_block = 256, add_grid = (numel + add_block - 1) / add_block;
    // Lambda kernel (compiled as a __global__ function) for element-wise addition:
    // (For clarity, we write it as an external kernel.)

    vec_add<<<add_grid, add_block>>>(net->d_X_grad, dX_k, net->d_X_grad, numel);
    vec_add<<<add_grid, add_block>>>(net->d_X_grad, dX_v, net->d_X_grad, numel);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(dX_q);
    cudaFree(dX_k);
    cudaFree(dX_v);
    
    // 4. Embedding backward: Scatter-add d_X_grad using the stored embed_indices.
    int total_embed = net->batch_size * net->raw_input_dim;
    int embed_block = 256, embed_grid = (total_embed + embed_block - 1) / embed_block;
    embedding_backward_kernel<<<embed_grid, embed_block>>>(
        net->d_X_grad, net->d_embed_indices, net->d_embedding_grad,
        net->batch_size, net->raw_input_dim, net->num_bins, net->embedding_dim
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// update_weights: Updates projection parameters and embedding table using AdamW.
void update_weights(Net* net, float learning_rate)
{
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int proj_size = net->embedding_dim * net->embedding_dim;
    int block = 256;
    int grid_proj = (proj_size + block - 1) / block;
    // Update Wq.
    adamw_update_kernel<<<grid_proj, block>>>(
        net->d_Wq, net->d_Wq_grad,
        net->d_Wq_m, net->d_Wq_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        proj_size, net->batch_size
    );
    // Update Wk.
    adamw_update_kernel<<<grid_proj, block>>>(
        net->d_Wk, net->d_Wk_grad,
        net->d_Wk_m, net->d_Wk_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        proj_size, net->batch_size
    );
    // Update Wv.
    adamw_update_kernel<<<grid_proj, block>>>(
        net->d_Wv, net->d_Wv_grad,
        net->d_Wv_m, net->d_Wv_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        proj_size, net->batch_size
    );
    // Update embedding table.
    int emb_size = net->raw_input_dim * net->num_bins * net->embedding_dim;
    int grid_emb = (emb_size + block - 1) / block;
    adamw_update_kernel<<<grid_emb, block>>>(
        net->d_embedding_table, net->d_embedding_grad,
        net->d_embedding_m, net->d_embedding_v,
        net->beta1, net->beta2, net->epsilon,
        learning_rate, net->weight_decay, alpha_t,
        emb_size, net->batch_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// save_model: Saves network dimensions and parameters to a binary file.
void save_model(Net* net, const char* filename)
{
    // Copy parameters from device to host.
    int proj_size = net->embedding_dim * net->embedding_dim;
    CHECK_CUDA(cudaMemcpy(net->h_Wq, net->d_Wq, proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_Wk, net->d_Wk, proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(net->h_Wv, net->d_Wv, proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    size_t emb_size = net->raw_input_dim * net->num_bins * net->embedding_dim * sizeof(float);
    CHECK_CUDA(cudaMemcpy(net->h_embedding_table, net->d_embedding_table, emb_size, cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if(!file){
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions.
    fwrite(&net->raw_input_dim, sizeof(int), 1, file);
    fwrite(&net->num_bins, sizeof(int), 1, file);
    fwrite(&net->embedding_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(&net->t, sizeof(int), 1, file);
    
    // Save projection weights.
    fwrite(net->h_Wq, sizeof(float), proj_size, file);
    fwrite(net->h_Wk, sizeof(float), proj_size, file);
    fwrite(net->h_Wv, sizeof(float), proj_size, file);
    
    // Save embedding table.
    fwrite(net->h_embedding_table, sizeof(float), emb_size/sizeof(float), file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// -----------------------------------------------------------------------------
// load_model: Loads a model from a binary file.
Net* load_model(const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if(!file){
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    int raw_input_dim, num_bins, embedding_dim, batch_size, t;
    fread(&raw_input_dim, sizeof(int), 1, file);
    fread(&num_bins, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    fread(&t, sizeof(int), 1, file);
    
    Net* net = init_net(raw_input_dim, num_bins, embedding_dim, batch_size);
    net->t = t;
    
    int proj_size = embedding_dim * embedding_dim;
    fread(net->h_Wq, sizeof(float), proj_size, file);
    fread(net->h_Wk, sizeof(float), proj_size, file);
    fread(net->h_Wv, sizeof(float), proj_size, file);
    CHECK_CUDA(cudaMemcpy(net->d_Wq, net->h_Wq, proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_Wk, net->h_Wk, proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_Wv, net->h_Wv, proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    size_t emb_size = raw_input_dim * num_bins * embedding_dim * sizeof(float);
    fread(net->h_embedding_table, sizeof(float), emb_size/sizeof(float), file);
    CHECK_CUDA(cudaMemcpy(net->d_embedding_table, net->h_embedding_table, emb_size, cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

// -----------------------------------------------------------------------------
// free_net: Frees all allocated memory.
void free_net(Net* net)
{
    cudaFree(net->d_embedding_table);
    cudaFree(net->d_embedding_grad);
    cudaFree(net->d_embedding_m);
    cudaFree(net->d_embedding_v);
    cudaFree(net->d_embed_indices);
    
    cudaFree(net->d_Wq);
    cudaFree(net->d_Wq_grad);
    cudaFree(net->d_Wq_m);
    cudaFree(net->d_Wq_v);
    cudaFree(net->d_Wk);
    cudaFree(net->d_Wk_grad);
    cudaFree(net->d_Wk_m);
    cudaFree(net->d_Wk_v);
    cudaFree(net->d_Wv);
    cudaFree(net->d_Wv_grad);
    cudaFree(net->d_Wv_m);
    cudaFree(net->d_Wv_v);
    
    cudaFree(net->d_X);
    cudaFree(net->d_Q);
    cudaFree(net->d_K);
    cudaFree(net->d_V);
    cudaFree(net->d_Q_grad);
    cudaFree(net->d_K_grad);
    cudaFree(net->d_V_grad);
    cudaFree(net->d_X_grad);
    
    cudaFree(net->d_predictions);
    cudaFree(net->d_error);
    cudaFree(net->d_y);
    cudaFree(net->d_attn_weights);
    
    free(net->h_Wq);
    free(net->h_Wk);
    free(net->h_Wv);
    free(net->h_embedding_table);
    
    cublasDestroy(net->cublas_handle);
    free(net);
}


// -----------------------------------------------------------------------------
// main: trains the attention-based model on synthetic data and evaluates it.
int main()
{
    srand(time(NULL));
    
    //----------------------------------------------------------------------------
    // PARAMETERS
    //----------------------------------------------------------------------------
    const int raw_input_dim = 8;      // number of tokens per sample
    const int num_samples     = 1024;   // number of samples
    // For this attention model, we predict one output per token.
    // Embedding lookup parameters:
    const int embedding_num_bins = 16;  // number of bins per token
    const int embedding_dim     = 8;    // embedding dimension (and number of classes per token)
    const int batch_size  = num_samples; // full-batch training
    
    //----------------------------------------------------------------------------
    // DATA GENERATION
    //----------------------------------------------------------------------------
    // generate_synthetic_data generates continuous data: X and y.
    // Here, we generate as many continuous target values as tokens.
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, raw_input_dim, raw_input_dim);
    
    // Determine true min and max of targets.
    float out_min = y[0], out_max = y[0];
    int total_targets = num_samples * raw_input_dim;
    for(int i = 1; i < total_targets; i++){
        if(y[i] < out_min) out_min = y[i];
        if(y[i] > out_max) out_max = y[i];
    }
    float out_range = out_max - out_min;
    if(out_range == 0.0f) out_range = 1e-6f;
    
    //----------------------------------------------------------------------------
    // Compute discrete labels: Map each continuous target in [out_min, out_max]
    // into a discrete bin in [0, embedding_dim - 1] for each token.
    int* y_class = (int*)malloc(total_targets * sizeof(int));
    for(int i = 0; i < num_samples; i++){
        for(int j = 0; j < raw_input_dim; j++){
            int index = i * raw_input_dim + j;
            float normalized = (y[index] - out_min) / out_range;
            int bin = (int)(normalized * embedding_dim);
            if(bin < 0) bin = 0;
            if(bin >= embedding_dim) bin = embedding_dim - 1;
            y_class[index] = bin;
        }
    }
    
    //----------------------------------------------------------------------------
    // Initialize the network.
    //----------------------------------------------------------------------------
    Net* net = init_net(raw_input_dim, embedding_num_bins, embedding_dim, batch_size);
    
    //----------------------------------------------------------------------------
    // TRAINING PARAMETERS.
    //----------------------------------------------------------------------------
    const int num_epochs = 20000;
    const float learning_rate = 0.001f;
    
    //----------------------------------------------------------------------------
    // TRAINING LOOP.
    //----------------------------------------------------------------------------
    for(int epoch = 0; epoch < num_epochs; epoch++){
        // Convert raw input into embeddings.
        embed_input(net, X);
        
        // Forward pass: project embeddings then apply self-attention.
        forward_pass(net);
        
        // Compute loss (and set d_error).
        float loss = calculate_loss(net, y_class);
        
        // Zero out gradients.
        zero_gradients(net);
        
        // Backward pass.
        backward_pass(net);
        
        // Update weights.
        update_weights(net, learning_rate);
        
        if((epoch+1) % 100 == 0){
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch+1, num_epochs, loss);
        }
    }
    
    //----------------------------------------------------------------------------
    // Save the model and data.
    //----------------------------------------------------------------------------
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, raw_input_dim, raw_input_dim, data_fname);
    
    //----------------------------------------------------------------------------
    // Verify saved model.
    //----------------------------------------------------------------------------
    printf("\nVerifying saved model...\n");
    net = load_model(model_fname);
    embed_input(net, X);
    forward_pass(net);
    float verification_loss = calculate_loss(net, y_class);
    printf("Loss with loaded model: %.8f\n", verification_loss);
    
    //----------------------------------------------------------------------------
    // EVALUATION.
    // For each sample and token, determine the predicted bin (via argmax over embedding_dim logits),
    // then convert it to a continuous prediction.
    //----------------------------------------------------------------------------
    int correct = 0;
    double sum_abs_error = 0.0;
    float bin_width = out_range / embedding_dim;
    int total_predictions = num_samples * raw_input_dim;
    
    int pred_size = batch_size * net->output_dim * sizeof(float);
    float* h_predictions = (float*)malloc(pred_size);
    CHECK_CUDA(cudaMemcpy(h_predictions, net->d_predictions, pred_size, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < num_samples; i++){
        for(int j = 0; j < raw_input_dim; j++){
            int group_offset = i * (raw_input_dim * embedding_dim) + j * embedding_dim;
            int predicted_bin = 0;
            float max_logit = h_predictions[group_offset];
            for(int k = 1; k < embedding_dim; k++){
                float logit = h_predictions[group_offset + k];
                if(logit > max_logit){
                    max_logit = logit;
                    predicted_bin = k;
                }
            }
            int label = y_class[i * raw_input_dim + j];
            if(predicted_bin == label)
                correct++;
            float predicted_cont = out_min + (predicted_bin + 0.5f) * bin_width;
            float true_cont = y[i * raw_input_dim + j];
            sum_abs_error += fabs(predicted_cont - true_cont);
        }
    }
    float accuracy = 100.0f * ((float)correct)/ total_predictions;
    float mae = sum_abs_error / total_predictions;
    printf("\nClassification Accuracy: %.2f%%\n", accuracy);
    printf("Mean Absolute Error (Continuous Prediction): %.5f\n", mae);
    
    // Print sample predictions (first 15 samples).
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Sample\tTokenIdx\tPredictedBin\tPredictedCont\tTrueBin\tTrueCont\n");
    printf("--------------------------------------------------------------------------\n");
    int samples_to_print = (num_samples < 15) ? num_samples : 15;
    for(int i = 0; i < samples_to_print; i++){
        for(int j = 0; j < raw_input_dim; j++){
            int group_offset = i * (raw_input_dim * embedding_dim) + j * embedding_dim;
            int predicted_bin = 0;
            float max_logit = h_predictions[group_offset];
            for(int k = 1; k < embedding_dim; k++){
                float logit = h_predictions[group_offset + k];
                if(logit > max_logit){
                    max_logit = logit;
                    predicted_bin = k;
                }
            }
            float predicted_cont = out_min + (predicted_bin + 0.5f) * bin_width;
            int true_bin = y_class[i * raw_input_dim + j];
            float true_cont = y[i * raw_input_dim + j];
            printf("%d:\t%d\t\t%d\t\t%.5f\t\t%d\t%.5f\n", i, j, predicted_bin, predicted_cont, true_bin, true_cont);
        }
    }
    
    // Cleanup.
    free(X);
    free(y);
    free(y_class);
    free(h_predictions);
    free_net(net);
    
    return 0;
}