#ifndef MLP_MIXER_H
#define MLP_MIXER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

// Structure for binning information
typedef struct {
    float* edges;      // Bin edges
    int n_bins;        // Number of bins
    int feature_dim;   // Dimension of the feature
} BinningInfo;

// Main MLP Mixer structure
typedef struct {
    // Dimensions
    int n_tokens;      // Number of tokens (bins)
    int n_channels;    // Number of channels (features)
    int token_dim;     // Token embedding dimension
    int batch_size;    // Batch size
    
    // Embedding weights
    float* embedding_weight;    // [n_tokens x token_dim]
    float* embedding_grad;      // Gradient for embedding
    
    // Token mixing weights
    float* token_mix_weight;    // [n_channels x n_channels]
    float* token_mix_grad;      // Gradient for token mixing
    
    // Channel mixing weights
    float* channel_mix_weight;  // [token_dim x token_dim]
    float* channel_mix_grad;    // Gradient for channel mixing
    
    // Output head weights
    float* output_weight;       // [token_dim x n_tokens]
    float* output_grad;         // Gradient for output head
    
    // Adam optimizer parameters
    float* embedding_m;         // First moment for embedding
    float* embedding_v;         // Second moment for embedding
    float* token_mix_m;        // First moment for token mixing
    float* token_mix_v;        // Second moment for token mixing
    float* channel_mix_m;      // First moment for channel mixing
    float* channel_mix_v;      // Second moment for channel mixing
    float* output_m;           // First moment for output
    float* output_v;           // Second moment for output
    
    float beta1;               // Adam beta1
    float beta2;               // Adam beta2
    float epsilon;             // Adam epsilon
    float weight_decay;        // Weight decay parameter
    int t;                     // Time step
    
    // Intermediate activations
    float* embedded;           // Embedded input
    float* token_mixed;        // After token mixing
    float* channel_mixed;      // After channel mixing
    float* output;            // Final output
    
    // Binning information
    BinningInfo* input_bins;   // Input binning info
    BinningInfo* output_bins;  // Output binning info
} MLPMixer;

// BinningInfo functions
BinningInfo* init_binning_info(int n_bins, int feature_dim) {
    BinningInfo* info = (BinningInfo*)malloc(sizeof(BinningInfo));
    info->n_bins = n_bins;
    info->feature_dim = feature_dim;
    info->edges = (float*)malloc(sizeof(float) * (n_bins + 1) * feature_dim);
    return info;
}

void free_binning_info(BinningInfo* info) {
    free(info->edges);
    free(info);
}

// Initialize the MLP Mixer
MLPMixer* init_mlp_mixer(int n_tokens, int n_channels, int token_dim, int batch_size) {
    MLPMixer* mixer = (MLPMixer*)malloc(sizeof(MLPMixer));
    
    // Store dimensions
    mixer->n_tokens = n_tokens;
    mixer->n_channels = n_channels;
    mixer->token_dim = token_dim;
    mixer->batch_size = batch_size;
    
    // Initialize Adam parameters
    mixer->beta1 = 0.9f;
    mixer->beta2 = 0.999f;
    mixer->epsilon = 1e-8f;
    mixer->weight_decay = 0.01f;
    mixer->t = 0;
    
    // Allocate weights and gradients
    size_t embedding_size = n_tokens * token_dim;
    size_t token_mix_size = n_channels * n_channels;
    size_t channel_mix_size = token_dim * token_dim;
    size_t output_size = token_dim * n_tokens;
    
    // Embedding layer
    mixer->embedding_weight = (float*)malloc(embedding_size * sizeof(float));
    mixer->embedding_grad = (float*)malloc(embedding_size * sizeof(float));
    mixer->embedding_m = (float*)calloc(embedding_size, sizeof(float));
    mixer->embedding_v = (float*)calloc(embedding_size, sizeof(float));
    
    // Token mixing layer
    mixer->token_mix_weight = (float*)malloc(token_mix_size * sizeof(float));
    mixer->token_mix_grad = (float*)malloc(token_mix_size * sizeof(float));
    mixer->token_mix_m = (float*)calloc(token_mix_size, sizeof(float));
    mixer->token_mix_v = (float*)calloc(token_mix_size, sizeof(float));
    
    // Channel mixing layer
    mixer->channel_mix_weight = (float*)malloc(channel_mix_size * sizeof(float));
    mixer->channel_mix_grad = (float*)malloc(channel_mix_size * sizeof(float));
    mixer->channel_mix_m = (float*)calloc(channel_mix_size, sizeof(float));
    mixer->channel_mix_v = (float*)calloc(channel_mix_size, sizeof(float));
    
    // Output layer
    mixer->output_weight = (float*)malloc(output_size * sizeof(float));
    mixer->output_grad = (float*)malloc(output_size * sizeof(float));
    mixer->output_m = (float*)calloc(output_size, sizeof(float));
    mixer->output_v = (float*)calloc(output_size, sizeof(float));
    
    // Allocate intermediate activations
    mixer->embedded = (float*)malloc(batch_size * n_channels * token_dim * sizeof(float));
    mixer->token_mixed = (float*)malloc(batch_size * n_channels * token_dim * sizeof(float));
    mixer->channel_mixed = (float*)malloc(batch_size * n_channels * token_dim * sizeof(float));
    mixer->output = (float*)malloc(batch_size * n_channels * n_tokens * sizeof(float));
    
    // Initialize weights using Xavier initialization
    float embedding_scale = sqrtf(2.0f / (n_tokens + token_dim));
    float token_mix_scale = sqrtf(2.0f / (n_channels + n_channels));
    float channel_mix_scale = sqrtf(2.0f / (token_dim + token_dim));
    float output_scale = sqrtf(2.0f / (token_dim + n_tokens));
    
    for (int i = 0; i < embedding_size; i++) {
        mixer->embedding_weight[i] = ((float)rand() / RAND_MAX * 2 - 1) * embedding_scale;
    }
    
    for (int i = 0; i < token_mix_size; i++) {
        mixer->token_mix_weight[i] = ((float)rand() / RAND_MAX * 2 - 1) * token_mix_scale;
    }
    
    for (int i = 0; i < channel_mix_size; i++) {
        mixer->channel_mix_weight[i] = ((float)rand() / RAND_MAX * 2 - 1) * channel_mix_scale;
    }
    
    for (int i = 0; i < output_size; i++) {
        mixer->output_weight[i] = ((float)rand() / RAND_MAX * 2 - 1) * output_scale;
    }
    
    return mixer;
}

// Free MLP Mixer memory
void free_mlp_mixer(MLPMixer* mixer) {
    // Free weights and gradients
    free(mixer->embedding_weight);
    free(mixer->embedding_grad);
    free(mixer->embedding_m);
    free(mixer->embedding_v);
    
    free(mixer->token_mix_weight);
    free(mixer->token_mix_grad);
    free(mixer->token_mix_m);
    free(mixer->token_mix_v);
    
    free(mixer->channel_mix_weight);
    free(mixer->channel_mix_grad);
    free(mixer->channel_mix_m);
    free(mixer->channel_mix_v);
    
    free(mixer->output_weight);
    free(mixer->output_grad);
    free(mixer->output_m);
    free(mixer->output_v);
    
    // Free intermediate activations
    free(mixer->embedded);
    free(mixer->token_mixed);
    free(mixer->channel_mixed);
    free(mixer->output);
    
    // Free binning info if exists
    if (mixer->input_bins) free_binning_info(mixer->input_bins);
    if (mixer->output_bins) free_binning_info(mixer->output_bins);
    
    free(mixer);
}

// Activation functions
static inline float swish(float x) {
    return x / (1.0f + expf(-x));
}

static inline float swish_derivative(float x, float sx) {
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return sx + sigmoid * (1.0f - sx);
}

// Helper function for matrix transpose
void transpose_matrix(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Forward pass functions
void embedding_forward(MLPMixer* mixer, int* input) {
    // Convert input tokens to embeddings
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            int token_idx = input[b * mixer->n_channels + c];
            float* embed_src = &mixer->embedding_weight[token_idx * mixer->token_dim];
            float* embed_dst = &mixer->embedded[b * mixer->n_channels * mixer->token_dim + 
                                              c * mixer->token_dim];
            memcpy(embed_dst, embed_src, mixer->token_dim * sizeof(float));
        }
    }
}

void token_mixing_forward(MLPMixer* mixer) {
    float* workspace = (float*)malloc(mixer->batch_size * mixer->n_channels * 
                                    mixer->token_dim * sizeof(float));
    
    // For each sample in the batch
    for (int b = 0; b < mixer->batch_size; b++) {
        float* sample_input = &mixer->embedded[b * mixer->n_channels * mixer->token_dim];
        float* sample_output = &mixer->token_mixed[b * mixer->n_channels * mixer->token_dim];
        
        // Mix tokens for each token dimension
        for (int d = 0; d < mixer->token_dim; d++) {
            // Extract token slice
            float* token_slice = &sample_input[d];
            float* output_slice = &sample_output[d];
            
            // Perform token mixing
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       mixer->n_channels, mixer->n_channels,
                       1.0f,
                       mixer->token_mix_weight,
                       mixer->n_channels,
                       token_slice,
                       mixer->token_dim,
                       0.0f,
                       output_slice,
                       mixer->token_dim);
        }
    }
    
    // Apply Swish activation
    int total_elements = mixer->batch_size * mixer->n_channels * mixer->token_dim;
    for (int i = 0; i < total_elements; i++) {
        mixer->token_mixed[i] = swish(mixer->token_mixed[i]);
    }
    
    free(workspace);
}

void channel_mixing_forward(MLPMixer* mixer) {
    float* workspace = (float*)malloc(mixer->batch_size * mixer->n_channels * 
                                    mixer->token_dim * sizeof(float));
    
    // For each sample in the batch and channel
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            float* channel_input = &mixer->token_mixed[b * mixer->n_channels * mixer->token_dim + 
                                                     c * mixer->token_dim];
            float* channel_output = &mixer->channel_mixed[b * mixer->n_channels * mixer->token_dim + 
                                                        c * mixer->token_dim];
            
            // Perform channel mixing
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       mixer->token_dim, mixer->token_dim,
                       1.0f,
                       mixer->channel_mix_weight,
                       mixer->token_dim,
                       channel_input,
                       1,
                       0.0f,
                       channel_output,
                       1);
        }
    }
    
    // Apply Swish activation
    int total_elements = mixer->batch_size * mixer->n_channels * mixer->token_dim;
    for (int i = 0; i < total_elements; i++) {
        mixer->channel_mixed[i] = swish(mixer->channel_mixed[i]);
    }
    
    free(workspace);
}

void output_forward(MLPMixer* mixer) {
    // For each sample in the batch and channel
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            float* channel_input = &mixer->channel_mixed[b * mixer->n_channels * mixer->token_dim + 
                                                       c * mixer->token_dim];
            float* channel_output = &mixer->output[b * mixer->n_channels * mixer->n_tokens + 
                                                 c * mixer->n_tokens];
            
            // Compute logits
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       mixer->n_tokens, mixer->token_dim,
                       1.0f,
                       mixer->output_weight,
                       mixer->token_dim,
                       channel_input,
                       1,
                       0.0f,
                       channel_output,
                       1);
        }
    }
    
    // Apply softmax for each channel's output
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            float* logits = &mixer->output[b * mixer->n_channels * mixer->n_tokens + 
                                         c * mixer->n_tokens];
            
            // Find max for numerical stability
            float max_val = logits[0];
            for (int i = 1; i < mixer->n_tokens; i++) {
                if (logits[i] > max_val) max_val = logits[i];
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < mixer->n_tokens; i++) {
                logits[i] = expf(logits[i] - max_val);
                sum += logits[i];
            }
            
            // Normalize
            for (int i = 0; i < mixer->n_tokens; i++) {
                logits[i] /= sum;
            }
        }
    }
}

// Complete forward pass
void forward_pass(MLPMixer* mixer, int* input) {
    embedding_forward(mixer, input);
    token_mixing_forward(mixer);
    channel_mixing_forward(mixer);
    output_forward(mixer);
}

// Loss calculation and backward pass functions
float calculate_cross_entropy_loss(MLPMixer* mixer, int* targets, float* gradients) {
    float total_loss = 0.0f;
    int total_elements = mixer->batch_size * mixer->n_channels;
    
    // Calculate cross-entropy loss and gradients
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            int target = targets[b * mixer->n_channels + c];
            float* pred = &mixer->output[b * mixer->n_channels * mixer->n_tokens + 
                                       c * mixer->n_tokens];
            float* grad = &gradients[b * mixer->n_channels * mixer->n_tokens + 
                                   c * mixer->n_tokens];
            
            // Copy predictions to gradients
            memcpy(grad, pred, mixer->n_tokens * sizeof(float));
            
            // Calculate loss and adjust gradient for target
            total_loss -= logf(pred[target] + mixer->epsilon);
            grad[target] -= 1.0f;
        }
    }
    
    return total_loss / total_elements;
}

void output_backward(MLPMixer* mixer, float* output_gradients) {
    float* workspace = (float*)malloc(mixer->token_dim * sizeof(float));
    
    // For each sample and channel
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            float* grad_output = &output_gradients[b * mixer->n_channels * mixer->n_tokens + 
                                                 c * mixer->n_tokens];
            float* grad_input = &mixer->channel_mixed[b * mixer->n_channels * mixer->token_dim + 
                                                    c * mixer->token_dim];
            
            // Compute gradient with respect to channel_mixed
            cblas_sgemv(CblasRowMajor, CblasTrans,
                       mixer->n_tokens, mixer->token_dim,
                       1.0f,
                       mixer->output_weight,
                       mixer->token_dim,
                       grad_output,
                       1,
                       0.0f,
                       workspace,
                       1);
            
            // Accumulate gradients for output weights
            cblas_sger(CblasRowMajor,
                      mixer->n_tokens, mixer->token_dim,
                      1.0f,
                      grad_output, 1,
                      grad_input, 1,
                      mixer->output_grad,
                      mixer->token_dim);
            
            // Store gradients for channel_mixed
            memcpy(grad_input, workspace, mixer->token_dim * sizeof(float));
        }
    }
    
    free(workspace);
}

void channel_mixing_backward(MLPMixer* mixer) {
    float* workspace = (float*)malloc(mixer->token_dim * sizeof(float));
    
    // For each sample and channel
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            float* grad_output = &mixer->channel_mixed[b * mixer->n_channels * mixer->token_dim + 
                                                     c * mixer->token_dim];
            float* grad_input = &mixer->token_mixed[b * mixer->n_channels * mixer->token_dim + 
                                                  c * mixer->token_dim];
            float* input = grad_input;  // Original input needed for Swish derivative
            
            // Apply Swish derivative
            for (int i = 0; i < mixer->token_dim; i++) {
                grad_output[i] *= swish_derivative(input[i], grad_output[i]);
            }
            
            // Compute gradient with respect to token_mixed
            cblas_sgemv(CblasRowMajor, CblasTrans,
                       mixer->token_dim, mixer->token_dim,
                       1.0f,
                       mixer->channel_mix_weight,
                       mixer->token_dim,
                       grad_output,
                       1,
                       0.0f,
                       workspace,
                       1);
            
            // Accumulate gradients for channel mixing weights
            cblas_sger(CblasRowMajor,
                      mixer->token_dim, mixer->token_dim,
                      1.0f,
                      grad_output, 1,
                      input, 1,
                      mixer->channel_mix_grad,
                      mixer->token_dim);
            
            // Store gradients for token_mixed
            memcpy(grad_input, workspace, mixer->token_dim * sizeof(float));
        }
    }
    
    free(workspace);
}

void token_mixing_backward(MLPMixer* mixer) {
    float* workspace = (float*)malloc(mixer->n_channels * sizeof(float));
    
    // For each sample and token dimension
    for (int b = 0; b < mixer->batch_size; b++) {
        float* sample_grad_output = &mixer->token_mixed[b * mixer->n_channels * mixer->token_dim];
        float* sample_grad_input = &mixer->embedded[b * mixer->n_channels * mixer->token_dim];
        float* sample_input = sample_grad_input;  // Original input needed for Swish derivative
        
        for (int d = 0; d < mixer->token_dim; d++) {
            float* grad_output = &sample_grad_output[d];
            float* grad_input = &sample_grad_input[d];
            float* input = &sample_input[d];
            
            // Apply Swish derivative
            for (int i = 0; i < mixer->n_channels; i++) {
                grad_output[i * mixer->token_dim] *= 
                    swish_derivative(input[i * mixer->token_dim], 
                                   grad_output[i * mixer->token_dim]);
            }
            
            // Compute gradient with respect to embedded
            cblas_sgemv(CblasRowMajor, CblasTrans,
                       mixer->n_channels, mixer->n_channels,
                       1.0f,
                       mixer->token_mix_weight,
                       mixer->n_channels,
                       grad_output,
                       mixer->token_dim,
                       0.0f,
                       workspace,
                       1);
            
            // Accumulate gradients for token mixing weights
            for (int i = 0; i < mixer->n_channels; i++) {
                cblas_saxpy(mixer->n_channels,
                           grad_output[i * mixer->token_dim],
                           input + i * mixer->token_dim,
                           1,
                           &mixer->token_mix_grad[i * mixer->n_channels],
                           1);
            }
            
            // Store gradients for embedded
            for (int i = 0; i < mixer->n_channels; i++) {
                grad_input[i * mixer->token_dim] = workspace[i];
            }
        }
    }
    
    free(workspace);
}

void embedding_backward(MLPMixer* mixer, int* input) {
    // Accumulate gradients for embedding weights
    for (int b = 0; b < mixer->batch_size; b++) {
        for (int c = 0; c < mixer->n_channels; c++) {
            int token_idx = input[b * mixer->n_channels + c];
            float* grad_output = &mixer->embedded[b * mixer->n_channels * mixer->token_dim + 
                                                c * mixer->token_dim];
            float* grad_weight = &mixer->embedding_grad[token_idx * mixer->token_dim];
            
            // Accumulate gradients
            cblas_saxpy(mixer->token_dim,
                       1.0f,
                       grad_output,
                       1,
                       grad_weight,
                       1);
        }
    }
}

// Complete backward pass
void backward_pass(MLPMixer* mixer, int* input, int* targets) {
    float* output_gradients = (float*)malloc(mixer->batch_size * mixer->n_channels * 
                                           mixer->n_tokens * sizeof(float));
    
    // Calculate initial gradients from loss
    float loss = calculate_cross_entropy_loss(mixer, targets, output_gradients);
    
    // Perform backward passes
    output_backward(mixer, output_gradients);
    channel_mixing_backward(mixer);
    token_mixing_backward(mixer);
    embedding_backward(mixer, input);
    
    free(output_gradients);
}

// Zero gradients
void zero_gradients(MLPMixer* mixer) {
    memset(mixer->embedding_grad, 0, mixer->n_tokens * mixer->token_dim * sizeof(float));
    memset(mixer->token_mix_grad, 0, mixer->n_channels * mixer->n_channels * sizeof(float));
    memset(mixer->channel_mix_grad, 0, mixer->token_dim * mixer->token_dim * sizeof(float));
    memset(mixer->output_grad, 0, mixer->n_tokens * mixer->token_dim * sizeof(float));
}

// Helper function for weight updates
static void update_layer_weights(float* weight, float* grad, float* m, float* v, 
                               int size, float alpha_t, float beta1, float beta2, 
                               float epsilon, float weight_decay, float learning_rate, 
                               int batch_size) {
    for (size_t i = 0; i < (size_t)size; i++) {
        float grad_val = grad[i] / batch_size;
        
        // Update momentum and velocity
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad_val;
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad_val * grad_val;
        
        // Calculate update
        float update = alpha_t * m[i] / (sqrtf(v[i]) + epsilon);
        
        // Apply weight decay and update
        weight[i] = weight[i] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights(MLPMixer* mixer, float learning_rate) {
    mixer->t++;  // Increment time step
    
    float beta1_t = powf(mixer->beta1, mixer->t);
    float beta2_t = powf(mixer->beta2, mixer->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update embedding weights
    update_layer_weights(
        mixer->embedding_weight, mixer->embedding_grad, 
        mixer->embedding_m, mixer->embedding_v,
        mixer->n_tokens * mixer->token_dim,
        alpha_t, mixer->beta1, mixer->beta2, mixer->epsilon,
        mixer->weight_decay, learning_rate, mixer->batch_size
    );
    
    // Update token mixing weights
    update_layer_weights(
        mixer->token_mix_weight, mixer->token_mix_grad,
        mixer->token_mix_m, mixer->token_mix_v,
        mixer->n_channels * mixer->n_channels,
        alpha_t, mixer->beta1, mixer->beta2, mixer->epsilon,
        mixer->weight_decay, learning_rate, mixer->batch_size
    );
    
    // Update channel mixing weights
    update_layer_weights(
        mixer->channel_mix_weight, mixer->channel_mix_grad,
        mixer->channel_mix_m, mixer->channel_mix_v,
        mixer->token_dim * mixer->token_dim,
        alpha_t, mixer->beta1, mixer->beta2, mixer->epsilon,
        mixer->weight_decay, learning_rate, mixer->batch_size
    );
    
    // Update output weights
    update_layer_weights(
        mixer->output_weight, mixer->output_grad,
        mixer->output_m, mixer->output_v,
        mixer->n_tokens * mixer->token_dim,
        alpha_t, mixer->beta1, mixer->beta2, mixer->epsilon,
        mixer->weight_decay, learning_rate, mixer->batch_size
    );
}

// Save model to file
void save_model(MLPMixer* mixer, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&mixer->n_tokens, sizeof(int), 1, file);
    fwrite(&mixer->n_channels, sizeof(int), 1, file);
    fwrite(&mixer->token_dim, sizeof(int), 1, file);
    fwrite(&mixer->batch_size, sizeof(int), 1, file);
    
    // Save optimizer state
    fwrite(&mixer->t, sizeof(int), 1, file);
    
    // Save weights
    fwrite(mixer->embedding_weight, sizeof(float), 
           mixer->n_tokens * mixer->token_dim, file);
    fwrite(mixer->token_mix_weight, sizeof(float), 
           mixer->n_channels * mixer->n_channels, file);
    fwrite(mixer->channel_mix_weight, sizeof(float), 
           mixer->token_dim * mixer->token_dim, file);
    fwrite(mixer->output_weight, sizeof(float), 
           mixer->n_tokens * mixer->token_dim, file);
    
    // Save Adam states
    fwrite(mixer->embedding_m, sizeof(float), 
           mixer->n_tokens * mixer->token_dim, file);
    fwrite(mixer->embedding_v, sizeof(float), 
           mixer->n_tokens * mixer->token_dim, file);
    fwrite(mixer->token_mix_m, sizeof(float), 
           mixer->n_channels * mixer->n_channels, file);
    fwrite(mixer->token_mix_v, sizeof(float), 
           mixer->n_channels * mixer->n_channels, file);
    fwrite(mixer->channel_mix_m, sizeof(float), 
           mixer->token_dim * mixer->token_dim, file);
    fwrite(mixer->channel_mix_v, sizeof(float), 
           mixer->token_dim * mixer->token_dim, file);
    fwrite(mixer->output_m, sizeof(float), 
           mixer->n_tokens * mixer->token_dim, file);
    fwrite(mixer->output_v, sizeof(float), 
           mixer->n_tokens * mixer->token_dim, file);
    
    // Save binning information if exists
    if (mixer->input_bins && mixer->output_bins) {
        fwrite(mixer->input_bins->edges, sizeof(float), 
               (mixer->input_bins->n_bins + 1) * mixer->input_bins->feature_dim, file);
        fwrite(mixer->output_bins->edges, sizeof(float), 
               (mixer->output_bins->n_bins + 1) * mixer->output_bins->feature_dim, file);
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model from file
MLPMixer* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int n_tokens, n_channels, token_dim, batch_size;
    fread(&n_tokens, sizeof(int), 1, file);
    fread(&n_channels, sizeof(int), 1, file);
    fread(&token_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize new mixer
    MLPMixer* mixer = init_mlp_mixer(n_tokens, n_channels, token_dim, batch_size);
    
    // Read optimizer state
    fread(&mixer->t, sizeof(int), 1, file);
    
    // Read weights
    fread(mixer->embedding_weight, sizeof(float), 
          n_tokens * token_dim, file);
    fread(mixer->token_mix_weight, sizeof(float), 
          n_channels * n_channels, file);
    fread(mixer->channel_mix_weight, sizeof(float), 
          token_dim * token_dim, file);
    fread(mixer->output_weight, sizeof(float), 
          n_tokens * token_dim, file);
    
    // Read Adam states
    fread(mixer->embedding_m, sizeof(float), 
          n_tokens * token_dim, file);
    fread(mixer->embedding_v, sizeof(float), 
          n_tokens * token_dim, file);
    fread(mixer->token_mix_m, sizeof(float), 
          n_channels * n_channels, file);
    fread(mixer->token_mix_v, sizeof(float), 
          n_channels * n_channels, file);
    fread(mixer->channel_mix_m, sizeof(float), 
          token_dim * token_dim, file);
    fread(mixer->channel_mix_v, sizeof(float), 
          token_dim * token_dim, file);
    fread(mixer->output_m, sizeof(float), 
          n_tokens * token_dim, file);
    fread(mixer->output_v, sizeof(float), 
          n_tokens * token_dim, file);
    
    // Try to read binning information
    size_t read_size;
    mixer->input_bins = init_binning_info(n_tokens - 1, n_channels);
    mixer->output_bins = init_binning_info(n_tokens - 1, n_channels);
    
    read_size = fread(mixer->input_bins->edges, sizeof(float), 
                      (n_tokens) * n_channels, file);
    if (read_size == (n_tokens) * n_channels) {
        read_size = fread(mixer->output_bins->edges, sizeof(float), 
                         (n_tokens) * n_channels, file);
        if (read_size != (n_tokens) * n_channels) {
            free_binning_info(mixer->input_bins);
            free_binning_info(mixer->output_bins);
            mixer->input_bins = NULL;
            mixer->output_bins = NULL;
        }
    } else {
        free_binning_info(mixer->input_bins);
        free_binning_info(mixer->output_bins);
        mixer->input_bins = NULL;
        mixer->output_bins = NULL;
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mixer;
}

// Binning utility functions
void compute_bin_edges(float* data, int n_samples, int feature_dim, int n_bins, BinningInfo* bins) {
    // Temporary storage for sorting
    float* temp_data = (float*)malloc(n_samples * sizeof(float));
    
    // Compute bin edges for each feature
    for (int feat = 0; feat < feature_dim; feat++) {
        // Copy feature data for sorting
        for (int i = 0; i < n_samples; i++) {
            temp_data[i] = data[i * feature_dim + feat];
        }
        
        // Sort the data
        for (int i = 0; i < n_samples - 1; i++) {
            for (int j = 0; j < n_samples - i - 1; j++) {
                if (temp_data[j] > temp_data[j + 1]) {
                    float temp = temp_data[j];
                    temp_data[j] = temp_data[j + 1];
                    temp_data[j + 1] = temp;
                }
            }
        }
        
        // Compute quantiles
        float* edges = &bins->edges[feat * (n_bins + 1)];
        edges[0] = temp_data[0] - 1e-8f;  // Ensure first value gets included
        
        for (int b = 1; b < n_bins; b++) {
            int idx = (int)((float)b * n_samples / n_bins);
            edges[b] = temp_data[idx];
        }
        
        edges[n_bins] = temp_data[n_samples - 1] + 1e-8f;  // Ensure last value gets included
    }
    
    free(temp_data);
}

int find_bin(float value, float* edges, int n_bins) {
    // Binary search to find the appropriate bin
    int left = 0;
    int right = n_bins;
    
    while (left < right) {
        int mid = (left + right) / 2;
        if (edges[mid] <= value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left - 1;
}

void digitize_data(float* data, int* tokens, int n_samples, int feature_dim, BinningInfo* bins) {
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < feature_dim; j++) {
            float value = data[i * feature_dim + j];
            float* edges = &bins->edges[j * (bins->n_bins + 1)];
            tokens[i * feature_dim + j] = find_bin(value, edges, bins->n_bins);
        }
    }
}

float* inverse_transform(int* tokens, int n_samples, int feature_dim, BinningInfo* bins) {
    float* result = (float*)malloc(n_samples * feature_dim * sizeof(float));
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < feature_dim; j++) {
            int bin = tokens[i * feature_dim + j];
            float* edges = &bins->edges[j * (bins->n_bins + 1)];
            result[i * feature_dim + j] = (edges[bin] + edges[bin + 1]) / 2.0f;
        }
    }
    
    return result;
}

// Evaluation metrics
float calculate_r2_score(float* y_true, float* y_pred, int n_samples, int feature_dim) {
    float* r2_scores = (float*)malloc(feature_dim * sizeof(float));
    
    for (int feat = 0; feat < feature_dim; feat++) {
        float y_mean = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            y_mean += y_true[i * feature_dim + feat];
        }
        y_mean /= n_samples;
        
        float ss_tot = 0.0f;
        float ss_res = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float true_val = y_true[i * feature_dim + feat];
            float pred_val = y_pred[i * feature_dim + feat];
            
            float diff_tot = true_val - y_mean;
            float diff_res = true_val - pred_val;
            
            ss_tot += diff_tot * diff_tot;
            ss_res += diff_res * diff_res;
        }
        
        r2_scores[feat] = 1.0f - (ss_res / (ss_tot + 1e-8f));
    }
    
    // Calculate mean R² score
    float mean_r2 = 0.0f;
    for (int feat = 0; feat < feature_dim; feat++) {
        mean_r2 += r2_scores[feat];
    }
    mean_r2 /= feature_dim;
    
    free(r2_scores);
    return mean_r2;
}

float calculate_mae(float* y_true, float* y_pred, int n_samples, int feature_dim) {
    float total_mae = 0.0f;
    
    for (int i = 0; i < n_samples * feature_dim; i++) {
        total_mae += fabsf(y_true[i] - y_pred[i]);
    }
    
    return total_mae / (n_samples * feature_dim);
}

// Training helper function
void train_batch(MLPMixer* mixer, float* X_batch, float* y_batch, float learning_rate) {
    // Allocate temporary storage for tokens
    int* X_tokens = (int*)malloc(mixer->batch_size * mixer->n_channels * sizeof(int));
    int* y_tokens = (int*)malloc(mixer->batch_size * mixer->n_channels * sizeof(int));
    
    // Convert continuous data to tokens
    digitize_data(X_batch, X_tokens, mixer->batch_size, mixer->n_channels, mixer->input_bins);
    digitize_data(y_batch, y_tokens, mixer->batch_size, mixer->n_channels, mixer->output_bins);
    
    // Forward pass
    forward_pass(mixer, X_tokens);
    
    // Backward pass
    zero_gradients(mixer);
    backward_pass(mixer, X_tokens, y_tokens);
    
    // Update weights
    update_weights(mixer, learning_rate);
    
    // Clean up
    free(X_tokens);
    free(y_tokens);
}

// Prediction helper function
float* predict(MLPMixer* mixer, float* X, int n_samples) {
    // Allocate temporary storage for tokens
    int* X_tokens = (int*)malloc(n_samples * mixer->n_channels * sizeof(int));
    
    // Convert input data to tokens
    digitize_data(X, X_tokens, n_samples, mixer->n_channels, mixer->input_bins);
    
    // Forward pass
    forward_pass(mixer, X_tokens);
    
    // Get predicted tokens (argmax of output probabilities)
    int* pred_tokens = (int*)malloc(n_samples * mixer->n_channels * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < mixer->n_channels; j++) {
            float* probs = &mixer->output[i * mixer->n_channels * mixer->n_tokens + 
                                        j * mixer->n_tokens];
            int max_idx = 0;
            float max_val = probs[0];
            
            for (int k = 1; k < mixer->n_tokens; k++) {
                if (probs[k] > max_val) {
                    max_val = probs[k];
                    max_idx = k;
                }
            }
            
            pred_tokens[i * mixer->n_channels + j] = max_idx;
        }
    }
    
    // Convert tokens back to continuous values
    float* predictions = inverse_transform(pred_tokens, n_samples, mixer->n_channels, 
                                        mixer->output_bins);
    
    // Clean up
    free(X_tokens);
    free(pred_tokens);
    
    return predictions;
}

#endif