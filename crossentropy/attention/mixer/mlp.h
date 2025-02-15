#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  In this network each raw input feature (float) is quantized into bins and then looked up
  from an embedding table. That produces “token” vectors. Instead of doing self–attention
  we now mix the tokens using a token–mixing MLP (across the tokens axis) and, then, we feed them
  to a feed–forward network (which mixes channels). Both blocks include residual connections.
  Additionally, every activation (both in token–mixing and in feed–forward) uses swish
  (swish(x)=x*sigmoid(x)) instead of ReLU.
*/

typedef struct {
    // Raw input and quantization parameters.
    int input_dim;     // Number of raw input features = number of tokens.
    int num_bins;      // Number of bins (for embedding lookup and for each output’s logits).
    int embedding_dim; // Dimension of each embedding vector (set equal to num_bins).

    // Output: output_dim equals input_dim (each token gives one output).
    int output_dim;

    // Embedding table: size = input_dim * num_bins * embedding_dim.
    float* embedding_table;

    // Token–mixing MLP parameters.
    // This MLP is applied “across” tokens (each of length = input_dim), independently for each channel.
    // First linear layer: [input_dim x token_hidden_dim] and second: [token_hidden_dim x input_dim].
    int token_hidden_dim;
    float* W_token1;       // [input_dim x token_hidden_dim]
    float* W_token1_grad;
    float* W_token1_m;
    float* W_token1_v;
    float* W_token2;       // [token_hidden_dim x input_dim]
    float* W_token2_grad;
    float* W_token2_m;
    float* W_token2_v;

    // Feed–forward network parameters (channel–mixing).
    // Two–layer network applied to each token independently.
    // First layer: [embedding_dim x ff_hidden_dim], second layer: [ff_hidden_dim x embedding_dim].
    int ff_hidden_dim;
    float* W_ff1;       // [embedding_dim x ff_hidden_dim]
    float* W_ff1_grad;
    float* W_ff1_m;
    float* W_ff1_v;
    float* W_ff2;       // [ff_hidden_dim x embedding_dim]
    float* W_ff2_grad;
    float* W_ff2_m;
    float* W_ff2_v;

    // Adam hyper–parameters.
    float beta1;
    float beta2;
    float epsilon;
    int   t;
    float weight_decay;

    // Helper arrays for forward/backward passes.
    // predictions: final output logits after channel–mixing.
    // Shape: [batch_size x (input_dim * embedding_dim)]
    float* predictions;
    // error: gradients for the output (same shape as predictions)
    float* error;

    // ff_residual: buffer to store the output of the token–mixing block (with residual).
    // It has shape: [batch_size x (input_dim * embedding_dim)].
    float* ff_residual;

    // Training batch size.
    int batch_size;  // Number of samples per batch.
} Net;

/* ---------------- Helper Activation Functions ---------------- */
static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float swishf(float x) {
    float s = sigmoidf(x);
    return x * s;
}

/* Swish derivative computed for pre–activation value z.
   Using d/dz swish(z) = sigmoid(z) + swish(z)·(1 - sigmoid(z)) */
static inline float swish_deriv(float z) {
    float s = sigmoidf(z);
    return s + swishf(z) * (1.0f - s);
}

/* ---------------- Utility Functions ---------------- */

/*
  softmax: Computes the softmax of an array of logits.
  Arguments:
    logits: input array of length num_classes.
    probs: output array of length num_classes.
    num_classes: number of classes.
*/
static inline void softmax(const float* logits, float* probs, int num_classes) {
    float max_logit = logits[0];
    for (int j = 1; j < num_classes; j++) {
        if (logits[j] > max_logit)
            max_logit = logits[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        probs[j] = expf(logits[j] - max_logit);
        sum_exp += probs[j];
    }
    for (int j = 0; j < num_classes; j++) {
        probs[j] /= sum_exp;
    }
}

/*
  bin_value: Returns the bin index for a given continuous value.
  The value is clamped into [min_value, max_value] then normalized.
*/
static inline int bin_value(float value, float min_value, float max_value, int num_bins) {
    float clamped = value;
    if (clamped < min_value) clamped = min_value;
    if (clamped > max_value) clamped = max_value;
    float normalized = (clamped - min_value) / (max_value - min_value);
    int bin = (int)(normalized * num_bins);
    if (bin < 0)
        bin = 0;
    else if (bin >= num_bins)
        bin = num_bins - 1;
    return bin;
}

/*
  unbin_value: Returns a continuous value corresponding to the center of the given bin.
*/
static inline float unbin_value(int bin, float min_value, float max_value, int num_bins) {
    float bin_width = (max_value - min_value) / num_bins;
    return min_value + (bin + 0.5f) * bin_width;
}

/*
  compute_min_max: For a data array of shape [num_samples x num_features],
  computes per–feature minimum and maximum values.
*/
static inline void compute_min_max(const float* data, int num_samples, int num_features, float* min_arr, float* max_arr) {
    for (int j = 0; j < num_features; j++) {
        min_arr[j] = data[j];
        max_arr[j] = data[j];
    }
    for (int i = 1; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            float val = data[i * num_features + j];
            if (val < min_arr[j])
                min_arr[j] = val;
            if (val > max_arr[j])
                max_arr[j] = val;
        }
    }
}

/* ---------------- Network Initialization and Freeing ---------------- */

/*
  init_net: Initializes the network with an embedding lookup, a token–mixing MLP, and a feed–forward network.
  Parameters:
    input_dim: number of raw input features (tokens).
    num_bins: number of bins.
    embedding_dim: dimension of each embedding vector (set equal to num_bins).
    output_dim: number of continuous outputs (should equal input_dim).
    batch_size: number of samples per batch.
*/
static inline Net* init_net(int input_dim, int num_bins, int embedding_dim,
                              int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate network.\n");
        exit(EXIT_FAILURE);
    }
    net->input_dim = input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;

    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;

    // Allocate and initialize token–mixing MLP parameters.
    // Token mixing is done across the "tokens" dimension.
    // Set token_hidden_dim = 4 * input_dim.
    net->token_hidden_dim = 4 * input_dim;
    int token1_size = input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * input_dim;
    net->W_token1 = (float*)malloc(token1_size * sizeof(float));
    net->W_token1_grad = (float*)malloc(token1_size * sizeof(float));
    net->W_token1_m = (float*)calloc(token1_size, sizeof(float));
    net->W_token1_v = (float*)calloc(token1_size, sizeof(float));
    net->W_token2 = (float*)malloc(token2_size * sizeof(float));
    net->W_token2_grad = (float*)malloc(token2_size * sizeof(float));
    net->W_token2_m = (float*)calloc(token2_size, sizeof(float));
    net->W_token2_v = (float*)calloc(token2_size, sizeof(float));

    float token_scale = 1.0f / sqrtf((float)input_dim);
    for (int i = 0; i < token1_size; i++) {
        net->W_token1[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * token_scale;
    }
    memset(net->W_token1_grad, 0, token1_size * sizeof(float));
    float token2_scale = 1.0f / sqrtf((float)net->token_hidden_dim);
    for (int i = 0; i < token2_size; i++) {
        net->W_token2[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * token2_scale;
    }
    memset(net->W_token2_grad, 0, token2_size * sizeof(float));

    // Allocate embedding table.
    int emb_table_size = input_dim * num_bins * embedding_dim;
    net->embedding_table = (float*)malloc(emb_table_size * sizeof(float));
    float emb_scale = 1.0f / sqrtf((float)num_bins);
    for (int i = 0; i < emb_table_size; i++) {
        net->embedding_table[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * emb_scale;
    }

    // Allocate helper arrays.
    // predictions and error: shape = [batch_size x (input_dim * embedding_dim)]
    net->predictions = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    // ff_residual: stored output of token–mixing (to be used by feed–forward block).
    net->ff_residual = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));

    // Initialize feed–forward network parameters.
    net->ff_hidden_dim = 4 * embedding_dim;
    int ff1_size = embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * embedding_dim;
    net->W_ff1 = (float*)malloc(ff1_size * sizeof(float));
    net->W_ff1_grad = (float*)malloc(ff1_size * sizeof(float));
    net->W_ff1_m = (float*)calloc(ff1_size, sizeof(float));
    net->W_ff1_v = (float*)calloc(ff1_size, sizeof(float));
    net->W_ff2 = (float*)malloc(ff2_size * sizeof(float));
    net->W_ff2_grad = (float*)malloc(ff2_size * sizeof(float));
    net->W_ff2_m = (float*)calloc(ff2_size, sizeof(float));
    net->W_ff2_v = (float*)calloc(ff2_size, sizeof(float));

    float ff1_scale = 1.0f / sqrtf((float)embedding_dim);
    for (int i = 0; i < ff1_size; i++) {
        net->W_ff1[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * ff1_scale;
    }
    memset(net->W_ff1_grad, 0, ff1_size * sizeof(float));
    float ff2_scale = 1.0f / sqrtf((float)net->ff_hidden_dim);
    for (int i = 0; i < ff2_size; i++) {
        net->W_ff2[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * ff2_scale;
    }
    memset(net->W_ff2_grad, 0, ff2_size * sizeof(float));

    return net;
}

/*
  free_net: Releases all memory allocated for the network.
*/
static inline void free_net(Net* net) {
    if(net) {
        free(net->W_token1);
        free(net->W_token1_grad);
        free(net->W_token1_m);
        free(net->W_token1_v);
        free(net->W_token2);
        free(net->W_token2_grad);
        free(net->W_token2_m);
        free(net->W_token2_v);
        free(net->embedding_table);
        free(net->predictions);
        free(net->error);
        free(net->ff_residual);
        free(net->W_ff1);
        free(net->W_ff1_grad);
        free(net->W_ff1_m);
        free(net->W_ff1_v);
        free(net->W_ff2);
        free(net->W_ff2_grad);
        free(net->W_ff2_m);
        free(net->W_ff2_v);
        free(net);
    }
}

/* ---------------- Embedding and Forward Pass Functions ---------------- */

/*
  embed_input: For each raw input (shape [batch_size x input_dim]) and given per–feature min and max,
  determine the bin using bin_value then copy the corresponding embedding vector from the embedding table
  into embedded_output.
  The output is of shape [batch_size x (input_dim * embedding_dim)].
*/
static inline void embed_input(Net* net, float* raw_input, float* embedded_output, float* in_min, float* in_max) {
    for (int i = 0; i < net->batch_size; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            float val = raw_input[i * net->input_dim + j];
            int b = bin_value(val, in_min[j], in_max[j], net->num_bins);
            float* emb = net->embedding_table +
                         j * (net->num_bins * net->embedding_dim) +
                         b * net->embedding_dim;
            float* dest = embedded_output +
                          i * (net->input_dim * net->embedding_dim) +
                          j * net->embedding_dim;
            memcpy(dest, emb, net->embedding_dim * sizeof(float));
        }
    }
}

/*
  forward_pass: Computes the forward pass.
  Steps:
    1. Token–mixing MLP: For each sample, for each channel (from 0 to embedding_dim-1),
       treat the tokens (of length input_dim) as a vector v and compute
         hidden = swish(v * W_token1)
         token_out = hidden * W_token2
         new token vector = v + token_out   (residual connection)
       The resulting matrix (shape [input_dim x embedding_dim]) is stored in ff_residual.
    2. Feed–forward network (channel mixing): For each token,
         pre_act = (token vector) * W_ff1, then apply swish,
         then f = hidden * W_ff2, and add a residual connection:
         final = ff_residual + f.
       The final result is stored in net->predictions.
*/
static inline void forward_pass(Net* net, float* X) {
    int tokens = net->input_dim;   // token count per sample
    int d_dim = net->embedding_dim; // channels
    int total_tokens = net->batch_size * tokens;
    int token_hidden = net->token_hidden_dim;
    int ff_hidden = net->ff_hidden_dim;

    // --- Token Mixing MLP ---
    // Process each sample independently
    for (int s = 0; s < net->batch_size; s++) {
        float* X_sample = X + s * tokens * d_dim;
        float* out_sample = net->ff_residual + s * tokens * d_dim;
        
        // Transpose the sample matrix to get shape [d_dim x tokens]
        float* X_transposed = (float*)malloc(tokens * d_dim * sizeof(float));
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < d_dim; j++) {
                X_transposed[j * tokens + i] = X_sample[i * d_dim + j];
            }
        }

        // Hidden = X_transposed * W_token1 (shape: [d_dim x token_hidden])
        float* hidden = (float*)malloc(d_dim * token_hidden * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d_dim, token_hidden, tokens,
                    1.0f, X_transposed, tokens,
                    net->W_token1, token_hidden,
                    0.0f, hidden, token_hidden);

        // Apply swish activation
        for (int i = 0; i < d_dim * token_hidden; i++) {
            hidden[i] = swishf(hidden[i]);
        }

        // Token_out = hidden * W_token2 (shape: [d_dim x tokens])
        float* token_out = (float*)malloc(d_dim * tokens * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d_dim, tokens, token_hidden,
                    1.0f, hidden, token_hidden,
                    net->W_token2, tokens,
                    0.0f, token_out, tokens);

        // Transpose back and add residual connection
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < d_dim; j++) {
                out_sample[i * d_dim + j] = X_sample[i * d_dim + j] + token_out[j * tokens + i];
            }
        }

        free(X_transposed);
        free(hidden);
        free(token_out);
    }

    // --- Feed-Forward Network (Channel Mixing) ---
    // Compute hidden = swish(ff_residual * W_ff1)
    float* hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    float* pre_act = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->ff_residual, d_dim,
                net->W_ff1, ff_hidden,
                0.0f, pre_act, ff_hidden);
    
    // Apply swish activation
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        hidden[i] = swishf(pre_act[i]);
    }

    // Compute f = hidden * W_ff2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, d_dim, ff_hidden,
                1.0f, hidden, ff_hidden,
                net->W_ff2, d_dim,
                0.0f, net->predictions, d_dim);

    // Final residual connection
    for (int i = 0; i < total_tokens * d_dim; i++) {
        net->predictions[i] += net->ff_residual[i];
    }

    free(hidden);
    free(pre_act);
}

/*
  calculate_loss: For each sample and for each continuous output,
  applies softmax to the corresponding group of num_bins logits and computes the cross–entropy loss.
  target_labels: array of shape [batch_size x output_dim] (flattened) with discrete bin indices.
  The average loss is returned and net->error is set accordingly.
*/
static inline float calculate_loss(Net* net, int* target_labels) {
    float loss = 0.0f;
    int batch = net->batch_size;
    int classes = net->num_bins;  // each token’s output vector has length = embedding_dim = num_bins.
    int total = net->output_dim;  // output_dim == input_dim.
    float* probs = (float*)malloc(classes * sizeof(float));
    if (!probs) {
        fprintf(stderr, "Failed to allocate memory for probabilities.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < batch; i++) {
        // Each sample has 'total' outputs (tokens).
        for (int d = 0; d < total; d++) {
            int offset = i * (classes * total) + d * classes;
            softmax(net->predictions + offset, probs, classes);
            int target = target_labels[i * total + d];
            loss -= logf(probs[target] + net->epsilon);
            for (int j = 0; j < classes; j++) {
                net->error[offset + j] = probs[j] - ((j == target) ? 1.0f : 0.0f);
            }
        }
    }
    free(probs);
    return loss / (batch * total);
}

/*
  zero_gradients: Resets the gradients stored for token–mixing and feed–forward weights.
*/
static inline void zero_gradients(Net* net) {
    int token1_size = net->input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * net->input_dim;
    memset(net->W_token1_grad, 0, token1_size * sizeof(float));
    memset(net->W_token2_grad, 0, token2_size * sizeof(float));
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    memset(net->W_ff1_grad, 0, ff1_size * sizeof(float));
    memset(net->W_ff2_grad, 0, ff2_size * sizeof(float));
}

/*
  backward_pass: Backpropagates gradients through the feed–forward block and then through the token–mixing MLP.
  X is the embedded input.
*/
static inline void backward_pass(Net* net, float* X) {
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int batch = net->batch_size;
    int total_tokens = batch * tokens;
    int ff_hidden = net->ff_hidden_dim;
    int token_hidden = net->token_hidden_dim;

    zero_gradients(net);

    // --- Backward Pass for Feed-Forward (Channel Mixing) Block ---
    float* hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    float* pre_act = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    
    // Forward computation (again) to get pre-activation values
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->ff_residual, d_dim,
                net->W_ff1, ff_hidden,
                0.0f, pre_act, ff_hidden);
    
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        hidden[i] = swishf(pre_act[i]);
    }
    
    // Compute d_hidden = error * W_ff2^T
    float* d_hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->error, d_dim,
                net->W_ff2, d_dim,
                0.0f, d_hidden, ff_hidden);
    
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        d_hidden[i] *= swish_deriv(pre_act[i]);
    }
    
    // W_ff2 gradients
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ff_hidden, d_dim, total_tokens,
                1.0f, hidden, ff_hidden,
                net->error, d_dim,
                1.0f, net->W_ff2_grad, d_dim);
    
    // W_ff1 gradients
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_dim, ff_hidden, total_tokens,
                1.0f, net->ff_residual, d_dim,
                d_hidden, ff_hidden,
                1.0f, net->W_ff1_grad, ff_hidden);
    
    // Compute d_ff_input
    float* d_ff_input = (float*)malloc(total_tokens * d_dim * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, d_dim, ff_hidden,
                1.0f, d_hidden, ff_hidden,
                net->W_ff1, ff_hidden,
                0.0f, d_ff_input, d_dim);
    
    // Update error
    for (int i = 0; i < total_tokens * d_dim; i++) {
        net->error[i] += d_ff_input[i];
    }
    
    free(hidden);
    free(pre_act);
    free(d_hidden);
    free(d_ff_input);

    // --- Backward Pass for Token-Mixing Block ---
    // Process each sample independently
    for (int s = 0; s < batch; s++) {
        float* X_sample = X + s * tokens * d_dim;
        float* d_token = net->error + s * tokens * d_dim;
        
        // Transpose input and gradient for this sample
        float* X_trans = (float*)malloc(d_dim * tokens * sizeof(float));
        float* d_token_trans = (float*)malloc(d_dim * tokens * sizeof(float));
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < d_dim; j++) {
                X_trans[j * tokens + i] = X_sample[i * d_dim + j];
                d_token_trans[j * tokens + i] = d_token[i * d_dim + j];
            }
        }

        // Forward pass (again) to get intermediates
        float* pre_token = (float*)malloc(d_dim * token_hidden * sizeof(float));
        float* hidden_token = (float*)malloc(d_dim * token_hidden * sizeof(float));
        
        // Compute pre-activation
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d_dim, token_hidden, tokens,
                    1.0f, X_trans, tokens,
                    net->W_token1, token_hidden,
                    0.0f, pre_token, token_hidden);

        // Apply swish
        for (int i = 0; i < d_dim * token_hidden; i++) {
            hidden_token[i] = swishf(pre_token[i]);
        }

        // Backward computations
        // d_hidden_token = d_token_trans * W_token2^T
        float* d_hidden_token = (float*)malloc(d_dim * token_hidden * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    d_dim, token_hidden, tokens,
                    1.0f, d_token_trans, tokens,
                    net->W_token2, tokens,
                    0.0f, d_hidden_token, token_hidden);

        // Apply swish derivative
        for (int i = 0; i < d_dim * token_hidden; i++) {
            d_hidden_token[i] *= swish_deriv(pre_token[i]);
        }

        // W_token2 gradients
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    token_hidden, tokens, d_dim,
                    1.0f, hidden_token, token_hidden,
                    d_token_trans, tokens,
                    1.0f, net->W_token2_grad, tokens);

        // W_token1 gradients
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    tokens, token_hidden, d_dim,
                    1.0f, X_trans, tokens,
                    d_hidden_token, token_hidden,
                    1.0f, net->W_token1_grad, token_hidden);

        free(X_trans);
        free(d_token_trans);
        free(pre_token);
        free(hidden_token);
        free(d_hidden_token);
    }
}

/*
  update_weights: Updates the network weights (both token–mixing and feed–forward parameters)
  using AdamW. learning_rate is the base learning rate.
*/
static inline void update_weights(Net* net, float learning_rate) {
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update token–mixing parameters.
    int token1_size = net->input_dim * net->token_hidden_dim;
    for (int i = 0; i < token1_size; i++) {
        float grad = net->W_token1_grad[i] / net->batch_size;
        net->W_token1_m[i] = net->beta1 * net->W_token1_m[i] + (1.0f - net->beta1) * grad;
        net->W_token1_v[i] = net->beta2 * net->W_token1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_token1_m[i] / (sqrtf(net->W_token1_v[i]) + net->epsilon);
        net->W_token1[i] = net->W_token1[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    int token2_size = net->token_hidden_dim * net->input_dim;
    for (int i = 0; i < token2_size; i++) {
        float grad = net->W_token2_grad[i] / net->batch_size;
        net->W_token2_m[i] = net->beta1 * net->W_token2_m[i] + (1.0f - net->beta1) * grad;
        net->W_token2_v[i] = net->beta2 * net->W_token2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_token2_m[i] / (sqrtf(net->W_token2_v[i]) + net->epsilon);
        net->W_token2[i] = net->W_token2[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    // Update Feed–Forward weights.
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    for (int i = 0; i < ff1_size; i++) {
        float grad = net->W_ff1_grad[i] / net->batch_size;
        net->W_ff1_m[i] = net->beta1 * net->W_ff1_m[i] + (1.0f - net->beta1) * grad;
        net->W_ff1_v[i] = net->beta2 * net->W_ff1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_ff1_m[i] / (sqrtf(net->W_ff1_v[i]) + net->epsilon);
        net->W_ff1[i] = net->W_ff1[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    for (int i = 0; i < ff2_size; i++) {
        float grad = net->W_ff2_grad[i] / net->batch_size;
        net->W_ff2_m[i] = net->beta1 * net->W_ff2_m[i] + (1.0f - net->beta1) * grad;
        net->W_ff2_v[i] = net->beta2 * net->W_ff2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_ff2_m[i] / (sqrtf(net->W_ff2_v[i]) + net->epsilon);
        net->W_ff2[i] = net->W_ff2[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
}

/*
  save_model: Saves network dimensions, weights, Adam state, and the embedding table (plus feed–forward and token–mixing parameters)
  to a binary file.
*/
static inline void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->num_bins, sizeof(int), 1, file);
    fwrite(&net->embedding_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);

    // Save token–mixing parameters.
    fwrite(&net->token_hidden_dim, sizeof(int), 1, file);
    int token1_size = net->input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * net->input_dim;
    fwrite(net->W_token1, sizeof(float), token1_size, file);
    fwrite(net->W_token2, sizeof(float), token2_size, file);
    fwrite(net->W_token1_m, sizeof(float), token1_size, file);
    fwrite(net->W_token1_v, sizeof(float), token1_size, file);
    fwrite(net->W_token2_m, sizeof(float), token2_size, file);
    fwrite(net->W_token2_v, sizeof(float), token2_size, file);

    // Save Adam state counter.
    fwrite(&net->t, sizeof(int), 1, file);

    // Save embedding table.
    int emb_table_size = net->input_dim * net->num_bins * net->embedding_dim;
    fwrite(net->embedding_table, sizeof(float), emb_table_size, file);

    // Save feed–forward network parameters.
    fwrite(&net->ff_hidden_dim, sizeof(int), 1, file);
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    fwrite(net->W_ff1, sizeof(float), ff1_size, file);
    fwrite(net->W_ff2, sizeof(float), ff2_size, file);
    fwrite(net->W_ff1_m, sizeof(float), ff1_size, file);
    fwrite(net->W_ff1_v, sizeof(float), ff1_size, file);
    fwrite(net->W_ff2_m, sizeof(float), ff2_size, file);
    fwrite(net->W_ff2_v, sizeof(float), ff2_size, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

/*
  load_model: Loads network dimensions, weights, Adam state, and the embedding table plus feed–forward and token–mixing parameters
  from a binary file.
*/
static inline Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    int input_dim, num_bins, embedding_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&num_bins, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);

    Net* net = init_net(input_dim, num_bins, embedding_dim, output_dim, batch_size);

    // Load token–mixing parameters.
    fread(&net->token_hidden_dim, sizeof(int), 1, file);
    int token1_size = net->input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * net->input_dim;
    fread(net->W_token1, sizeof(float), token1_size, file);
    fread(net->W_token2, sizeof(float), token2_size, file);
    fread(net->W_token1_m, sizeof(float), token1_size, file);
    fread(net->W_token1_v, sizeof(float), token1_size, file);
    fread(net->W_token2_m, sizeof(float), token2_size, file);
    fread(net->W_token2_v, sizeof(float), token2_size, file);

    fread(&net->t, sizeof(int), 1, file);

    int emb_table_size = net->input_dim * net->num_bins * net->embedding_dim;
    fread(net->embedding_table, sizeof(float), emb_table_size, file);

    // Load feed–forward network parameters.
    fread(&net->ff_hidden_dim, sizeof(int), 1, file);
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    fread(net->W_ff1, sizeof(float), ff1_size, file);
    fread(net->W_ff2, sizeof(float), ff2_size, file);
    fread(net->W_ff1_m, sizeof(float), ff1_size, file);
    fread(net->W_ff1_v, sizeof(float), ff1_size, file);
    fread(net->W_ff2_m, sizeof(float), ff2_size, file);
    fread(net->W_ff2_v, sizeof(float), ff2_size, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

#endif /* MLP_H */