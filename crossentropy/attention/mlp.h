#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  In this network, each raw input feature (of type float) is quantized into one
  of a fixed number of bins (using per–feature min/max values) and then looked up
  from an embedding table. That produces a “token” vector. Instead of concatenating
  the embeddings and sending them through a multilayer perceptron, we organize the
  tokens (one per input feature) into a sequence and process them with self–attention.
  With output_dim = input_dim and embedding_dim = num_bins, the output of the
  self–attention (of shape [batch_size x (input_dim*embedding_dim)] ) is interpreted
  directly as the logits (for each token/output, there are embedding_dim = num_bins logits).

  In this modified version we add a feed–forward network after self–attention and add
  two residual connections:
    x = x + attention(x)
    x = x + feed_forward(x)
    
  This header defines the network structure and all related utility and training functions.
*/

typedef struct {
    // Raw input and quantization parameters.
    int input_dim;     // Number of raw input features.
    int num_bins;      // Number of bins (for embedding lookup and for each output's logits).
    int embedding_dim; // Dimension of each embedding vector (set equal to num_bins).

    // Output: output_dim equals input_dim (each token gives one output).
    int output_dim;

    // Embedding table: size = input_dim * num_bins * embedding_dim.
    float* embedding_table;

    // Attention parameters (each of shape: [embedding_dim x embedding_dim]).
    float* W_Q;
    float* W_K;
    float* W_V;
    float* W_Q_grad;
    float* W_K_grad;
    float* W_V_grad;
    float* W_Q_m;
    float* W_Q_v;
    float* W_K_m;
    float* W_K_v;
    float* W_V_m;
    float* W_V_v;

    // Feed–forward network parameters.
    // We use a two–layer network: first layer: [embedding_dim x ff_hidden_dim], then
    // second layer: [ff_hidden_dim x embedding_dim]. Here we set ff_hidden_dim = 4 * embedding_dim.
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
    // predictions: final output logits after feed–forward block.
    // It has shape: [batch_size x (input_dim * embedding_dim)].
    float* predictions;
    // error: gradients for the output (same shape as predictions).
    float* error;

    // Temporary buffers for self–attention.
    // attn_Q, attn_K, attn_V: each has shape [batch_size x (input_dim * embedding_dim)]
    float* attn_Q;
    float* attn_K;
    float* attn_V;
    // attn_scores: for each sample, a matrix of shape [input_dim x input_dim].
    float* attn_scores;
    // Buffer to store the result of the first residual connection (i.e. x + attention(x))
    // used as input to the feed–forward network.
    float* ff_residual;

    // Dimensions.
    int batch_size;  // Number of samples per batch.
} Net;

/* ---------------- Utility Function Implementations ---------------- */

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
  bin_value: Returns the bin index (in [0, num_bins-1]) for a given continuous value.
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
  min_arr and max_arr must be pre–allocated arrays of length num_features.
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
  init_net: Initializes the network with an embedding lookup, self–attention and a feed–forward network.
  Parameters:
    input_dim: number of raw input features.
    num_bins: number of bins (used for embedding lookup and as number of logits per output).
    embedding_dim: dimension of each embedding vector (set equal to num_bins).
    output_dim: number of continuous outputs (should equal input_dim).
    batch_size: number of samples per training batch.
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

    // Allocate attention weight matrices.
    int attn_dim = embedding_dim * embedding_dim;
    net->W_Q = (float*)malloc(attn_dim * sizeof(float));
    net->W_K = (float*)malloc(attn_dim * sizeof(float));
    net->W_V = (float*)malloc(attn_dim * sizeof(float));
    net->W_Q_grad = (float*)malloc(attn_dim * sizeof(float));
    net->W_K_grad = (float*)malloc(attn_dim * sizeof(float));
    net->W_V_grad = (float*)malloc(attn_dim * sizeof(float));
    net->W_Q_m = (float*)calloc(attn_dim, sizeof(float));
    net->W_K_m = (float*)calloc(attn_dim, sizeof(float));
    net->W_V_m = (float*)calloc(attn_dim, sizeof(float));
    net->W_Q_v = (float*)calloc(attn_dim, sizeof(float));
    net->W_K_v = (float*)calloc(attn_dim, sizeof(float));
    net->W_V_v = (float*)calloc(attn_dim, sizeof(float));

    // Initialize attention weights with a random uniform scaled by 1/sqrt(embedding_dim).
    float scale = 1.0f / sqrtf((float)embedding_dim);
    for (int i = 0; i < attn_dim; i++) {
        net->W_Q[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * scale;
        net->W_K[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * scale;
        net->W_V[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * scale;
    }
    memset(net->W_Q_grad, 0, attn_dim * sizeof(float));
    memset(net->W_K_grad, 0, attn_dim * sizeof(float));
    memset(net->W_V_grad, 0, attn_dim * sizeof(float));

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
    // attn_Q, attn_K, attn_V: same shape as predictions.
    net->attn_Q = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    net->attn_K = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    net->attn_V = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    // attn_scores: shape = [batch_size x (input_dim * input_dim)]
    net->attn_scores = (float*)malloc(batch_size * input_dim * input_dim * sizeof(float));
    // Buffer for feed-forward residual: same shape as predictions.
    net->ff_residual = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));

    // Initialize feed-forward network parameters.
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

    // Initialize feed-forward weights.
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
        free(net->W_Q);
        free(net->W_K);
        free(net->W_V);
        free(net->W_Q_grad);
        free(net->W_K_grad);
        free(net->W_V_grad);
        free(net->W_Q_m);
        free(net->W_Q_v);
        free(net->W_K_m);
        free(net->W_K_v);
        free(net->W_V_m);
        free(net->W_V_v);
        free(net->embedding_table);
        free(net->predictions);
        free(net->error);
        free(net->attn_Q);
        free(net->attn_K);
        free(net->attn_V);
        free(net->attn_scores);
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
  embed_input: For each raw input of shape [batch_size x input_dim] and given per–feature min and max,
  determine the bin (using bin_value) for each feature, then copy the corresponding embedding vector 
  from the embedding table into embedded_output.
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
    1. Compute self–attention. For each sample:
         Q = X * W_Q, K = X * W_K, V = X * W_V.
       Then compute attention scores = softmax((Q * K^T)/sqrt(d)) and output = scores * V.
    2. Residual connection: r = X + attention_output. Save r to ff_residual.
    3. Feed–forward network: for each token, compute:
         hidden = ReLU(r * W_ff1)
         f = hidden * W_ff2
         final = r + f.
       The final result is stored in net->predictions.
*/
static inline void forward_pass(Net* net, float* X) {
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int total_tokens = net->batch_size * tokens;

    // --- Self-Attention Part ---
    // Compute Q, K, V using matrix multiplications.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, d_dim, d_dim,
                1.0f, X, d_dim,
                net->W_Q, d_dim,
                0.0f, net->attn_Q, d_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, d_dim, d_dim,
                1.0f, X, d_dim,
                net->W_K, d_dim,
                0.0f, net->attn_K, d_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, d_dim, d_dim,
                1.0f, X, d_dim,
                net->W_V, d_dim,
                0.0f, net->attn_V, d_dim);

    float scale = 1.0f / sqrtf((float)d_dim);
    for (int s = 0; s < net->batch_size; s++) {
        float* Q = net->attn_Q + s * tokens * d_dim;
        float* K = net->attn_K + s * tokens * d_dim;
        float* V = net->attn_V + s * tokens * d_dim;
        float* scores = net->attn_scores + s * tokens * tokens;
        float* attn_out = net->predictions + s * tokens * d_dim;

        // Compute scores = Q * K^T scaled.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tokens, tokens, d_dim,
                    scale, Q, d_dim,
                    K, d_dim,
                    0.0f, scores, tokens);

        // Apply softmax to each row.
        for (int i = 0; i < tokens; i++) {
            float* score_row = scores + i * tokens;
            float* softmax_row = (float*)malloc(tokens * sizeof(float));
            softmax(score_row, softmax_row, tokens);
            memcpy(score_row, softmax_row, tokens * sizeof(float));
            free(softmax_row);
        }

        // Compute attention output: attn_out = scores * V.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tokens, d_dim, tokens,
                    1.0f, scores, tokens,
                    V, d_dim,
                    0.0f, attn_out, d_dim);
    }

    // --- First Residual: r = X + attention_output ---
    for (int i = 0; i < net->batch_size * tokens * d_dim; i++) {
        net->ff_residual[i] = X[i] + net->predictions[i];
    }

    // --- Feed–Forward Network with Residual ---
    // For each sample and each token, compute:
    //   hidden = ReLU(r * W_ff1)  [W_ff1: [d_dim x ff_hidden_dim]]
    //   f = hidden * W_ff2       [W_ff2: [ff_hidden_dim x d_dim]]
    //   final_output = r + f.
    for (int s = 0; s < net->batch_size; s++) {
        for (int t = 0; t < tokens; t++) {
            int token_offset = s * tokens * d_dim + t * d_dim;
            float* r = net->ff_residual + token_offset;
            float f_output[128]; // maximum d_dim is assumed to be <= 128 (adjust if needed)
            for (int i = 0; i < d_dim; i++) {
                f_output[i] = 0.0f;
            }
            // Allocate temporary buffer for hidden layer output.
            int hidden_size = net->ff_hidden_dim;
            float* hidden = (float*)malloc(hidden_size * sizeof(float));
            // Compute hidden = r * W_ff1. (W_ff1: [d_dim x hidden_size])
            for (int j = 0; j < hidden_size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d_dim; k++) {
                    sum += r[k] * net->W_ff1[k * hidden_size + j];
                }
                // Apply ReLU.
                hidden[j] = (sum > 0.0f) ? sum : 0.0f;
            }
            // Compute feed–forward output f = hidden * W_ff2. (W_ff2: [hidden_size x d_dim])
            for (int i = 0; i < d_dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < hidden_size; j++) {
                    sum += hidden[j] * net->W_ff2[j * d_dim + i];
                }
                f_output[i] = sum;
            }
            free(hidden);
            // Final output = r + f_output, store in predictions.
            for (int i = 0; i < d_dim; i++) {
                net->predictions[token_offset + i] = r[i] + f_output[i];
            }
        }
    }
}

/*
  calculate_loss: For each sample and for each continuous output,
  applies softmax to the corresponding group of num_bins logits and computes the cross–entropy loss.
  target_labels is assumed to be an array of shape [batch_size x output_dim] (flattened)
  containing the discrete bin index for each output.
  The average loss (over all outputs and samples) is returned.
  Also sets net->error accordingly.
*/
static inline float calculate_loss(Net* net, int* target_labels) {
    float loss = 0.0f;
    int batch = net->batch_size;
    int classes = net->num_bins;  // each token's output vector has length embedding_dim == num_bins.
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
  zero_gradients: Resets the gradients stored for the attention and feed–forward weights.
*/
static inline void zero_gradients(Net* net) {
    int attn_dim = net->embedding_dim * net->embedding_dim;
    memset(net->W_Q_grad, 0, attn_dim * sizeof(float));
    memset(net->W_K_grad, 0, attn_dim * sizeof(float));
    memset(net->W_V_grad, 0, attn_dim * sizeof(float));

    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    memset(net->W_ff1_grad, 0, ff1_size * sizeof(float));
    memset(net->W_ff2_grad, 0, ff2_size * sizeof(float));
}

static inline void backward_pass(Net* net, float* X) {
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int batch = net->batch_size;
    int total_tokens = batch * tokens;
    int ff_hidden = net->ff_hidden_dim;
    float inv_sqrt = 1.0f / sqrtf((float)d_dim);

    zero_gradients(net);

    // --- Backward Pass for Feed-Forward Block ---
    // Allocate buffers for the entire batch
    float* hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    float* pre_act = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    
    // Forward computation to get hidden states (needed for gradients)
    // pre_act = ff_residual * W_ff1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->ff_residual, d_dim,
                net->W_ff1, ff_hidden,
                0.0f, pre_act, ff_hidden);
    
    // Apply ReLU to get hidden states
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        hidden[i] = (pre_act[i] > 0.0f) ? pre_act[i] : 0.0f;
    }

    // Compute d_hidden = error * W_ff2^T
    float* d_hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->error, d_dim,
                net->W_ff2, d_dim,
                0.0f, d_hidden, ff_hidden);

    // Apply ReLU gradient
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        d_hidden[i] = (pre_act[i] > 0.0f) ? d_hidden[i] : 0.0f;
    }

    // Compute W_ff2 gradients: ff2_grad += hidden^T * error
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ff_hidden, d_dim, total_tokens,
                1.0f, hidden, ff_hidden,
                net->error, d_dim,
                1.0f, net->W_ff2_grad, d_dim);

    // Compute W_ff1 gradients: ff1_grad += ff_residual^T * d_hidden
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_dim, ff_hidden, total_tokens,
                1.0f, net->ff_residual, d_dim,
                d_hidden, ff_hidden,
                1.0f, net->W_ff1_grad, ff_hidden);

    // Compute d_ff_input = d_hidden * W_ff1^T
    float* d_ff_input = (float*)malloc(total_tokens * d_dim * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, d_dim, ff_hidden,
                1.0f, d_hidden, ff_hidden,
                net->W_ff1, ff_hidden,
                0.0f, d_ff_input, d_dim);

    // Update error for attention backward pass: error += d_ff_input
    for (int i = 0; i < total_tokens * d_dim; i++) {
        net->error[i] += d_ff_input[i];
    }

    // Free temporary buffers
    free(hidden);
    free(pre_act);
    free(d_hidden);
    free(d_ff_input);

    // --- Backward Pass for Self-Attention Block ---
    for (int s = 0; s < batch; s++) {
        float* Q = net->attn_Q + s * tokens * d_dim;
        float* K = net->attn_K + s * tokens * d_dim;
        float* V = net->attn_V + s * tokens * d_dim;
        float* scores = net->attn_scores + s * tokens * tokens;
        float* out_error = net->error + s * tokens * d_dim;
        float* X_sample = X + s * tokens * d_dim;

        float* dQ = (float*)malloc(tokens * d_dim * sizeof(float));
        float* dK = (float*)malloc(tokens * d_dim * sizeof(float));
        float* dV = (float*)malloc(tokens * d_dim * sizeof(float));
        memset(dQ, 0, tokens * d_dim * sizeof(float));
        memset(dK, 0, tokens * d_dim * sizeof(float));
        memset(dV, 0, tokens * d_dim * sizeof(float));

        // Compute dV: dV = scores^T * out_error
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    tokens, d_dim, tokens,
                    1.0f, scores, tokens,
                    out_error, d_dim,
                    0.0f, dV, d_dim);

        // Compute intermediate dA = out_error * V^T
        float* dA = (float*)malloc(tokens * tokens * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tokens, tokens, d_dim,
                    1.0f, out_error, d_dim,
                    V, d_dim,
                    0.0f, dA, tokens);

        // Apply softmax derivative
        for (int i = 0; i < tokens; i++) {
            float sum_d = 0.0f;
            for (int j = 0; j < tokens; j++) {
                sum_d += scores[i * tokens + j] * dA[i * tokens + j];
            }
            for (int j = 0; j < tokens; j++) {
                dA[i * tokens + j] = scores[i * tokens + j] * (dA[i * tokens + j] - sum_d);
            }
        }

        // Compute dQ, dK, dV
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tokens, d_dim, tokens,
                    inv_sqrt, dA, tokens,
                    K, d_dim,
                    0.0f, dQ, d_dim);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    tokens, d_dim, tokens,
                    inv_sqrt, dA, tokens,
                    Q, d_dim,
                    0.0f, dK, d_dim);

        // Accumulate gradients for W_Q, W_K, W_V
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    d_dim, d_dim, tokens,
                    1.0f, X_sample, d_dim,
                    dQ, d_dim,
                    1.0f, net->W_Q_grad, d_dim);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    d_dim, d_dim, tokens,
                    1.0f, X_sample, d_dim,
                    dK, d_dim,
                    1.0f, net->W_K_grad, d_dim);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    d_dim, d_dim, tokens,
                    1.0f, X_sample, d_dim,
                    dV, d_dim,
                    1.0f, net->W_V_grad, d_dim);

        free(dQ);
        free(dK);
        free(dV);
        free(dA);
    }
}

/*
  update_weights: Updates the network weights (attention and feed–forward parameters) using AdamW.
  learning_rate: base learning rate.
*/
static inline void update_weights(Net* net, float learning_rate) {
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    int attn_dim = net->embedding_dim * net->embedding_dim;
    // Update W_Q.
    for (int i = 0; i < attn_dim; i++) {
        float grad = net->W_Q_grad[i] / net->batch_size;
        net->W_Q_m[i] = net->beta1 * net->W_Q_m[i] + (1.0f - net->beta1) * grad;
        net->W_Q_v[i] = net->beta2 * net->W_Q_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_Q_m[i] / (sqrtf(net->W_Q_v[i]) + net->epsilon);
        net->W_Q[i] = net->W_Q[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    // Update W_K.
    for (int i = 0; i < attn_dim; i++) {
        float grad = net->W_K_grad[i] / net->batch_size;
        net->W_K_m[i] = net->beta1 * net->W_K_m[i] + (1.0f - net->beta1) * grad;
        net->W_K_v[i] = net->beta2 * net->W_K_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_K_m[i] / (sqrtf(net->W_K_v[i]) + net->epsilon);
        net->W_K[i] = net->W_K[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    // Update W_V.
    for (int i = 0; i < attn_dim; i++) {
        float grad = net->W_V_grad[i] / net->batch_size;
        net->W_V_m[i] = net->beta1 * net->W_V_m[i] + (1.0f - net->beta1) * grad;
        net->W_V_v[i] = net->beta2 * net->W_V_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_V_m[i] / (sqrtf(net->W_V_v[i]) + net->epsilon);
        net->W_V[i] = net->W_V[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
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
  save_model: Saves network dimensions, weights, Adam state, and the embedding table (and feed–forward parameters)
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

    int attn_dim = net->embedding_dim * net->embedding_dim;
    fwrite(net->W_Q, sizeof(float), attn_dim, file);
    fwrite(net->W_K, sizeof(float), attn_dim, file);
    fwrite(net->W_V, sizeof(float), attn_dim, file);

    // Write Adam state for attention.
    fwrite(&net->t, sizeof(int), 1, file);
    fwrite(net->W_Q_m, sizeof(float), attn_dim, file);
    fwrite(net->W_Q_v, sizeof(float), attn_dim, file);
    fwrite(net->W_K_m, sizeof(float), attn_dim, file);
    fwrite(net->W_K_v, sizeof(float), attn_dim, file);
    fwrite(net->W_V_m, sizeof(float), attn_dim, file);
    fwrite(net->W_V_v, sizeof(float), attn_dim, file);

    // Write embedding table.
    int emb_table_size = net->input_dim * net->num_bins * net->embedding_dim;
    fwrite(net->embedding_table, sizeof(float), emb_table_size, file);

    // Write feed–forward network parameters.
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
  load_model: Loads network dimensions, weights, Adam state, and the embedding table (and feed–forward parameters)
  from a binary file. Returns a pointer to the loaded network.
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

    int attn_dim = embedding_dim * embedding_dim;
    fread(net->W_Q, sizeof(float), attn_dim, file);
    fread(net->W_K, sizeof(float), attn_dim, file);
    fread(net->W_V, sizeof(float), attn_dim, file);

    fread(&net->t, sizeof(int), 1, file);
    fread(net->W_Q_m, sizeof(float), attn_dim, file);
    fread(net->W_Q_v, sizeof(float), attn_dim, file);
    fread(net->W_K_m, sizeof(float), attn_dim, file);
    fread(net->W_K_v, sizeof(float), attn_dim, file);
    fread(net->W_V_m, sizeof(float), attn_dim, file);
    fread(net->W_V_v, sizeof(float), attn_dim, file);

    int emb_table_size = input_dim * num_bins * embedding_dim;
    fread(net->embedding_table, sizeof(float), emb_table_size, file);

    // Load feed–forward network parameters.
    fread(&net->ff_hidden_dim, sizeof(int), 1, file);
    int ff1_size = embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * embedding_dim;
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