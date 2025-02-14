#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

#define NUM_BINS 256  // Number of bins for discretization
#define EMBED_DIM 128 // Embedding dimension

typedef struct {
    // Embedding table
    float* embedding;  // [NUM_BINS x EMBED_DIM]
    float* grad_embedding;  // Gradients for embedding
    
    // Mixing layers
    float* token_mix; // [seq_len x seq_len]
    float* grad_token_mix;
    float* chan_mix;  // [EMBED_DIM x EMBED_DIM]
    float* grad_chan_mix;
    float* output_proj; // [EMBED_DIM x NUM_BINS]
    float* grad_output_proj;
    
    // Adam optimizer state
    float* m_embedding;
    float* v_embedding;
    float* m_token_mix;
    float* v_token_mix;
    float* m_chan_mix;
    float* v_chan_mix;
    float* m_output_proj;
    float* v_output_proj;
    int t;  // timestep
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    
    // Intermediate buffers
    float* embed_out;    // [batch_size x seq_len x EMBED_DIM]
    float* token_mixed;  // [batch_size x seq_len x EMBED_DIM]
    float* chan_mixed;   // [batch_size x seq_len x EMBED_DIM]
    float* logits;      // [batch_size x seq_len x NUM_BINS]
    
    // Dimensions
    int seq_len;     // Number of input/output positions
    int batch_size;
    
    // Bin boundaries
    float* bin_edges;    // [NUM_BINS + 1]
    float* bin_centers;  // [NUM_BINS]
} Net;

// Helper function to compute bin indices
void compute_bins(float* data, int* indices, int n, float* bin_edges) {
    for (int i = 0; i < n; i++) {
        float val = data[i];
        // Binary search for bin
        int left = 0, right = NUM_BINS;
        while (left < right) {
            int mid = (left + right) / 2;
            if (bin_edges[mid] <= val)
                left = mid + 1;
            else
                right = mid;
        }
        indices[i] = left - 1;
    }
}

Net* init_net(int seq_len, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    net->seq_len = seq_len;
    net->batch_size = batch_size;
    
    // Initialize Adam parameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->weight_decay = 0.01f;
    net->t = 0;
    
    // Allocate embedding table and gradients
    net->embedding = (float*)malloc(NUM_BINS * EMBED_DIM * sizeof(float));
    net->grad_embedding = (float*)calloc(NUM_BINS * EMBED_DIM, sizeof(float));
    net->m_embedding = (float*)calloc(NUM_BINS * EMBED_DIM, sizeof(float));
    net->v_embedding = (float*)calloc(NUM_BINS * EMBED_DIM, sizeof(float));
    
    // Allocate mixing matrices and gradients
    net->token_mix = (float*)malloc(seq_len * seq_len * sizeof(float));
    net->grad_token_mix = (float*)calloc(seq_len * seq_len, sizeof(float));
    net->m_token_mix = (float*)calloc(seq_len * seq_len, sizeof(float));
    net->v_token_mix = (float*)calloc(seq_len * seq_len, sizeof(float));
    
    net->chan_mix = (float*)malloc(EMBED_DIM * EMBED_DIM * sizeof(float));
    net->grad_chan_mix = (float*)calloc(EMBED_DIM * EMBED_DIM, sizeof(float));
    net->m_chan_mix = (float*)calloc(EMBED_DIM * EMBED_DIM, sizeof(float));
    net->v_chan_mix = (float*)calloc(EMBED_DIM * EMBED_DIM, sizeof(float));
    
    net->output_proj = (float*)malloc(EMBED_DIM * NUM_BINS * sizeof(float));
    net->grad_output_proj = (float*)calloc(EMBED_DIM * NUM_BINS, sizeof(float));
    net->m_output_proj = (float*)calloc(EMBED_DIM * NUM_BINS, sizeof(float));
    net->v_output_proj = (float*)calloc(EMBED_DIM * NUM_BINS, sizeof(float));
    
    // Allocate intermediate buffers
    net->embed_out = (float*)malloc(batch_size * seq_len * EMBED_DIM * sizeof(float));
    net->token_mixed = (float*)malloc(batch_size * seq_len * EMBED_DIM * sizeof(float));
    net->chan_mixed = (float*)malloc(batch_size * seq_len * EMBED_DIM * sizeof(float));
    net->logits = (float*)malloc(batch_size * seq_len * NUM_BINS * sizeof(float));
    
    // Initialize bin boundaries
    net->bin_edges = (float*)malloc((NUM_BINS + 1) * sizeof(float));
    net->bin_centers = (float*)malloc(NUM_BINS * sizeof(float));
    
    float range = INPUT_RANGE_MAX - INPUT_RANGE_MIN;
    float step = range / NUM_BINS;
    
    for (int i = 0; i <= NUM_BINS; i++) {
        net->bin_edges[i] = INPUT_RANGE_MIN + i * step;
    }
    
    for (int i = 0; i < NUM_BINS; i++) {
        net->bin_centers[i] = (net->bin_edges[i] + net->bin_edges[i + 1]) / 2;
    }
    
    // Initialize weights with small random values
    float embed_scale = sqrtf(1.0f / EMBED_DIM);
    float mix_scale = sqrtf(1.0f / seq_len);
    float chan_scale = sqrtf(1.0f / EMBED_DIM);
    float proj_scale = sqrtf(1.0f / EMBED_DIM);
    
    for (int i = 0; i < NUM_BINS * EMBED_DIM; i++) {
        net->embedding[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * embed_scale;
    }
    
    for (int i = 0; i < seq_len * seq_len; i++) {
        net->token_mix[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * mix_scale;
    }
    
    for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++) {
        net->chan_mix[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * chan_scale;
    }
    
    for (int i = 0; i < EMBED_DIM * NUM_BINS; i++) {
        net->output_proj[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * proj_scale;
    }
    
    return net;
}

void free_net(Net* net) {
    // Free weights and gradients
    free(net->embedding);
    free(net->grad_embedding);
    free(net->token_mix);
    free(net->grad_token_mix);
    free(net->chan_mix);
    free(net->grad_chan_mix);
    free(net->output_proj);
    free(net->grad_output_proj);
    
    // Free Adam state
    free(net->m_embedding);
    free(net->v_embedding);
    free(net->m_token_mix);
    free(net->v_token_mix);
    free(net->m_chan_mix);
    free(net->v_chan_mix);
    free(net->m_output_proj);
    free(net->v_output_proj);
    
    // Free intermediate buffers
    free(net->embed_out);
    free(net->token_mixed);
    free(net->chan_mixed);
    free(net->logits);
    
    // Free bin data
    free(net->bin_edges);
    free(net->bin_centers);
    
    free(net);
}

// Swish activation in-place
void swish(float* x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

// Forward pass
void forward_pass(Net* net, int* input_indices) {
    int B = net->batch_size;
    int L = net->seq_len;
    
    // Embedding lookup
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int idx = input_indices[b * L + l];
            memcpy(&net->embed_out[b * L * EMBED_DIM + l * EMBED_DIM],
                   &net->embedding[idx * EMBED_DIM],
                   EMBED_DIM * sizeof(float));
        }
    }
    
    // Token mixing
    for (int b = 0; b < B; b++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    L, EMBED_DIM, L,
                    1.0f,
                    net->token_mix, L,
                    &net->embed_out[b * L * EMBED_DIM], EMBED_DIM,
                    0.0f,
                    &net->token_mixed[b * L * EMBED_DIM], EMBED_DIM);
    }
    
    // Swish activation
    swish(net->token_mixed, B * L * EMBED_DIM);
    
    // Channel mixing
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, EMBED_DIM, EMBED_DIM,
                        1.0f,
                        &net->token_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM,
                        net->chan_mix, EMBED_DIM,
                        0.0f,
                        &net->chan_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM);
        }
    }
    
    // Swish activation
    swish(net->chan_mixed, B * L * EMBED_DIM);
    
    // Output projection to logits
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, NUM_BINS, EMBED_DIM,
                        1.0f,
                        &net->chan_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM,
                        net->output_proj, NUM_BINS,
                        0.0f,
                        &net->logits[b * L * NUM_BINS + l * NUM_BINS], NUM_BINS);
        }
    }
}

// Cross entropy loss
float calculate_loss(Net* net, int* target_indices) {
    float loss = 0.0f;
    int B = net->batch_size;
    int L = net->seq_len;
    
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            float* logits = &net->logits[b * L * NUM_BINS + l * NUM_BINS];
            int target = target_indices[b * L + l];
            
            // Compute softmax and cross entropy
            float max_val = logits[0];
            for (int i = 1; i < NUM_BINS; i++) {
                if (logits[i] > max_val) max_val = logits[i];
            }
            
            float sum = 0.0f;
            for (int i = 0; i < NUM_BINS; i++) {
                sum += expf(logits[i] - max_val);
            }
            
            float log_sum = logf(sum) + max_val;
            loss -= logits[target] - log_sum;
        }
    }
    
    return loss / (B * L);
}

// Softmax with cross entropy gradient
void softmax_cross_entropy_grad(float* logits, int* target_indices, float* grad_out, 
                              int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        for (int l = 0; l < seq_len; l++) {
            float* cur_logits = &logits[b * seq_len * NUM_BINS + l * NUM_BINS];
            float* cur_grad = &grad_out[b * seq_len * NUM_BINS + l * NUM_BINS];
            int target = target_indices[b * seq_len + l];
            
            // Compute softmax
            float max_val = cur_logits[0];
            for (int i = 1; i < NUM_BINS; i++) {
                if (cur_logits[i] > max_val) max_val = cur_logits[i];
            }
            
            float sum = 0.0f;
            float* exp_values = (float*)malloc(NUM_BINS * sizeof(float));
            
            for (int i = 0; i < NUM_BINS; i++) {
                exp_values[i] = expf(cur_logits[i] - max_val);
                sum += exp_values[i];
            }
            
            // Gradient is softmax - one_hot_target
            for (int i = 0; i < NUM_BINS; i++) {
                cur_grad[i] = exp_values[i] / sum;
            }
            cur_grad[target] -= 1.0f;
            
            free(exp_values);
        }
    }
}

// Swish gradient
void swish_grad(float* x, float* grad_in, float* grad_out, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        grad_out[i] = grad_in[i] * (sigmoid + val * sigmoid * (1.0f - sigmoid));
    }
}

// Backward pass
void backward_pass(Net* net, int* input_indices, int* target_indices) {
    int B = net->batch_size;
    int L = net->seq_len;
    
    // Allocate gradient buffers
    float* grad_logits = (float*)malloc(B * L * NUM_BINS * sizeof(float));
    float* grad_chan_mixed = (float*)malloc(B * L * EMBED_DIM * sizeof(float));
    float* grad_token_mixed = (float*)malloc(B * L * EMBED_DIM * sizeof(float));
    float* grad_embed = (float*)malloc(B * L * EMBED_DIM * sizeof(float));
    
    // Zero out accumulated gradients
    memset(net->grad_token_mix, 0, L * L * sizeof(float));
    memset(net->grad_chan_mix, 0, EMBED_DIM * EMBED_DIM * sizeof(float));
    memset(net->grad_output_proj, 0, EMBED_DIM * NUM_BINS * sizeof(float));
    memset(net->grad_embedding, 0, NUM_BINS * EMBED_DIM * sizeof(float));
    
    // Compute initial gradients from cross entropy
    softmax_cross_entropy_grad(net->logits, target_indices, grad_logits, B, L);
    
    // Backward through output projection
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            // Gradient for chan_mixed
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, EMBED_DIM, NUM_BINS,
                       1.0f,
                       &grad_logits[b * L * NUM_BINS + l * NUM_BINS], NUM_BINS,
                       net->output_proj, NUM_BINS,
                       0.0f,
                       &grad_chan_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM);
                       
            // Accumulate gradient for output_proj
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                       EMBED_DIM, NUM_BINS, 1,
                       1.0f,
                       &net->chan_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM,
                       &grad_logits[b * L * NUM_BINS + l * NUM_BINS], NUM_BINS,
                       1.0f,
                       net->grad_output_proj, NUM_BINS);
        }
    }
    
    // Backward through second swish
    swish_grad(net->chan_mixed, grad_chan_mixed, grad_chan_mixed, B * L * EMBED_DIM);
    
    // Backward through channel mixing
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            // Gradient for token_mixed
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, EMBED_DIM, EMBED_DIM,
                       1.0f,
                       &grad_chan_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM,
                       net->chan_mix, EMBED_DIM,
                       0.0f,
                       &grad_token_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM);
                       
            // Accumulate gradient for chan_mix
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                       EMBED_DIM, EMBED_DIM, 1,
                       1.0f,
                       &net->token_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM,
                       &grad_chan_mixed[b * L * EMBED_DIM + l * EMBED_DIM], EMBED_DIM,
                       1.0f,
                       net->grad_chan_mix, EMBED_DIM);
        }
    }
    
    // Backward through first swish
    swish_grad(net->token_mixed, grad_token_mixed, grad_token_mixed, B * L * EMBED_DIM);
    
    // Backward through token mixing
    for (int b = 0; b < B; b++) {
        // Gradient for embed_out
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    L, EMBED_DIM, L,
                    1.0f,
                    net->token_mix, L,
                    &grad_token_mixed[b * L * EMBED_DIM], EMBED_DIM,
                    0.0f,
                    &grad_embed[b * L * EMBED_DIM], EMBED_DIM);
                    
        // Accumulate gradient for token_mix
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    L, L, EMBED_DIM,
                    1.0f,
                    &net->embed_out[b * L * EMBED_DIM], EMBED_DIM,
                    &grad_token_mixed[b * L * EMBED_DIM], EMBED_DIM,
                    1.0f,
                    net->grad_token_mix, L);
    }
    
    // Accumulate embedding gradients
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int idx = input_indices[b * L + l];
            cblas_saxpy(EMBED_DIM, 1.0f,
                &grad_embed[b * L * EMBED_DIM + l * EMBED_DIM], 1,
                &net->grad_embedding[idx * EMBED_DIM], 1);
        }
    }
    
    // Free temporary buffers
    free(grad_logits);
    free(grad_chan_mixed);
    free(grad_token_mixed);
    free(grad_embed);
}

// Update weights using AdamW
void update_weights(Net* net, float learning_rate) {
    net->t++;  // Increment timestep
    
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update function for each parameter
    #define UPDATE_PARAM(param, grad, m, v, size) do { \
        for (int i = 0; i < size; i++) { \
            float g = grad[i] / net->batch_size; \
            m[i] = net->beta1 * m[i] + (1.0f - net->beta1) * g; \
            v[i] = net->beta2 * v[i] + (1.0f - net->beta2) * g * g; \
            param[i] = (1.0f - learning_rate * net->weight_decay) * param[i] - \
                      alpha_t * m[i] / (sqrtf(v[i]) + net->epsilon); \
        } \
    } while(0)
    
    // Update embedding table
    UPDATE_PARAM(net->embedding, net->grad_embedding, net->m_embedding, net->v_embedding, 
                NUM_BINS * EMBED_DIM);
    
    // Update token mixing matrix
    UPDATE_PARAM(net->token_mix, net->grad_token_mix, net->m_token_mix, net->v_token_mix,
                net->seq_len * net->seq_len);
    
    // Update channel mixing matrix
    UPDATE_PARAM(net->chan_mix, net->grad_chan_mix, net->m_chan_mix, net->v_chan_mix,
                EMBED_DIM * EMBED_DIM);
    
    // Update output projection
    UPDATE_PARAM(net->output_proj, net->grad_output_proj, net->m_output_proj, net->v_output_proj,
                EMBED_DIM * NUM_BINS);
    
    #undef UPDATE_PARAM
}

// Convert continuous values to bin indices
void continuous_to_bins(Net* net, float* continuous_data, int* bin_indices, int n) {
    compute_bins(continuous_data, bin_indices, n, net->bin_edges);
}

// Convert logits to continuous predictions
void predict_continuous(Net* net, float* logits, float* predictions, int n) {
    for (int i = 0; i < n; i++) {
        float* cur_logits = &logits[i * NUM_BINS];
        
        // Find max logit (argmax)
        int max_idx = 0;
        float max_val = cur_logits[0];
        for (int j = 1; j < NUM_BINS; j++) {
            if (cur_logits[j] > max_val) {
                max_val = cur_logits[j];
                max_idx = j;
            }
        }
        
        // Use bin center as prediction
        predictions[i] = net->bin_centers[max_idx];
    }
}

// Save model to file
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions and parameters
    fwrite(&net->seq_len, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(&net->t, sizeof(int), 1, file);
    
    // Save weights
    fwrite(net->embedding, sizeof(float), NUM_BINS * EMBED_DIM, file);
    fwrite(net->token_mix, sizeof(float), net->seq_len * net->seq_len, file);
    fwrite(net->chan_mix, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fwrite(net->output_proj, sizeof(float), EMBED_DIM * NUM_BINS, file);
    
    // Save Adam state
    fwrite(net->m_embedding, sizeof(float), NUM_BINS * EMBED_DIM, file);
    fwrite(net->v_embedding, sizeof(float), NUM_BINS * EMBED_DIM, file);
    fwrite(net->m_token_mix, sizeof(float), net->seq_len * net->seq_len, file);
    fwrite(net->v_token_mix, sizeof(float), net->seq_len * net->seq_len, file);
    fwrite(net->m_chan_mix, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fwrite(net->v_chan_mix, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fwrite(net->m_output_proj, sizeof(float), EMBED_DIM * NUM_BINS, file);
    fwrite(net->v_output_proj, sizeof(float), EMBED_DIM * NUM_BINS, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model from file
Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, batch_size, t;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    fread(&t, sizeof(int), 1, file);
    
    // Initialize network
    Net* net = init_net(seq_len, batch_size);
    net->t = t;
    
    // Load weights
    fread(net->embedding, sizeof(float), NUM_BINS * EMBED_DIM, file);
    fread(net->token_mix, sizeof(float), seq_len * seq_len, file);
    fread(net->chan_mix, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fread(net->output_proj, sizeof(float), EMBED_DIM * NUM_BINS, file);
    
    // Load Adam state
    fread(net->m_embedding, sizeof(float), NUM_BINS * EMBED_DIM, file);
    fread(net->v_embedding, sizeof(float), NUM_BINS * EMBED_DIM, file);
    fread(net->m_token_mix, sizeof(float), seq_len * seq_len, file);
    fread(net->v_token_mix, sizeof(float), seq_len * seq_len, file);
    fread(net->m_chan_mix, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fread(net->v_chan_mix, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fread(net->m_output_proj, sizeof(float), EMBED_DIM * NUM_BINS, file);
    fread(net->v_output_proj, sizeof(float), EMBED_DIM * NUM_BINS, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif