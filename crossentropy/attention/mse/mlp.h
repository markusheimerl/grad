#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  In this multi–layer network each input feature is first embedded via a learned embedding table.
  The raw feature is “binned” (using pre–computed min/max) and the corresponding embedding is selected.
  Then the resulting tokens (each of dimension embedding_dim) are processed by a stack of transformer–style layers.
  Each layer consists of a self–attention block (with residual) and a feed–forward two–layer network (with another residual).
  Finally, a learned projection from embedding_dim down to a scalar is applied to each token and the network is trained
  using mean–squared error (MSE) loss.
*/

/* -------------- The Layer Structure -------------- */
typedef struct {
    // Self–Attention weights (each is a [embedding_dim x embedding_dim] matrix).
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
    
    // Feed–Forward Network weights. We use two layers:
    // first: [embedding_dim x ff_hidden_dim] with ff_hidden_dim = 4*embedding_dim,
    // second: [ff_hidden_dim x embedding_dim].
    int ff_hidden_dim;
    float* W_ff1;
    float* W_ff1_grad;
    float* W_ff1_m;
    float* W_ff1_v;
    float* W_ff2;
    float* W_ff2_grad;
    float* W_ff2_m;
    float* W_ff2_v;
} Layer;

/* -------------- The Network Structure -------------- */
typedef struct {
    int input_dim;      // Number of tokens (raw input features).
    int num_bins;       // Number of bins used for input embedding.
    int embedding_dim;  // Dimension of each embedding vector.
    int output_dim;     // Should equal input_dim.
    float* embedding_table; // Embedding table of shape [input_dim x num_bins x embedding_dim].
    
    int num_layers;     // Number of transformer layers.
    Layer* layers;      // Array of layers (length = num_layers).
    
    // Final projection weights from embedding_dim to scalar.
    float* W_out;
    float* W_out_grad;
    float* W_out_m;
    float* W_out_v;
    
    // Adam hyper–parameters (shared).
    float beta1;
    float beta2;
    float epsilon;
    int   t;
    float weight_decay;
    
    int batch_size;     // Batch size.
    
    // For the multi–layer forward pass we save (for each layer) the activation and some intermediates.
    // layer_input: an array of (num_layers+1) pointers; layer_input[0] is the embedded input,
    // and for l=0..num_layers–1, layer_input[l+1] is the output of layer l.
    float** layer_input; // Each buffer has shape: [batch_size x (input_dim * embedding_dim)].
    
    // Self–attention intermediates for each layer.
    float** attn_Q_layers;      // Each: [batch_size x (input_dim * embedding_dim)]
    float** attn_K_layers;      // Each: same shape.
    float** attn_V_layers;      // Each: same shape.
    float** attn_scores_layers; // Each: [batch_size x (input_dim x input_dim)]
    float** ff_residual_layers; // For residual connection: r = X + attn_out.
    
    // Feed–forward intermediates for each layer.
    // ff_preact_layers: pre–activation computed as r * W_ff1 (shape: [batch_size x (input_dim * ff_hidden_dim)]).
    // ff_hidden_layers: ReLU(ff_preact) (same shape as above).
    float** ff_preact_layers;
    float** ff_hidden_layers;
    
    // Final predictions (a scalar per token) and error (used for backprop).
    float* predictions; // Shape: [batch_size x output_dim]
    float* error;       // Same shape as predictions.
} Net;

/* ---------------- Utility Function Implementations ---------------- */

/*
  softmax: Computes the softmax of an array of logits.
  (Not used in MSE loss but kept for reference.)
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
  compute_min_max: Computes per–feature minimum and maximum values.
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

/* ---------------- Function Declarations ---------------- */
Net* init_net(int input_dim, int num_bins, int embedding_dim, int output_dim, int batch_size, int num_layers);
void free_net(Net* net);
void embed_input(Net* net, float* raw_input, float* embedded_output, float* in_min, float* in_max);
void forward_pass(Net* net);
float calculate_loss(Net* net, float* targets);
void zero_gradients(Net* net);
void backward_pass(Net* net);
void update_weights(Net* net, float learning_rate);
void save_model(Net* net, const char* filename);
Net* load_model(const char* filename);

#endif /* MLP_H */