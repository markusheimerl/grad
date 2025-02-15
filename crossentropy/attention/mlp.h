#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  In this multi–layer network each input feature is first embedded via a learned embedding table.
  Then the resulting “tokens” (each of dimension equal to num_bins) are processed by a stack
  of transformer–style layers. Each layer consists of a self–attention block (with residual)
  and a feed–forward two–layer network (with another residual). The backward pass computes
  gradients for each layer and the update uses AdamW.
*/

/* ---------------- Data Structures ---------------- */
typedef struct {
    /* Self–Attention weights and optimizer state. */
    float *W_Q;
    float *W_K;
    float *W_V;
    float *W_Q_grad;
    float *W_K_grad;
    float *W_V_grad;
    float *W_Q_m;
    float *W_Q_v;
    float *W_K_m;
    float *W_K_v;
    float *W_V_m;
    float *W_V_v;
    /* Feed–Forward Network weights.
       First layer: [embedding_dim x ff_hidden_dim]  (ff_hidden_dim = 4 * embedding_dim)
       Second layer: [ff_hidden_dim x embedding_dim]. */
    int ff_hidden_dim;
    float *W_ff1;
    float *W_ff1_grad;
    float *W_ff1_m;
    float *W_ff1_v;
    float *W_ff2;
    float *W_ff2_grad;
    float *W_ff2_m;
    float *W_ff2_v;
} Layer;

typedef struct {
    int input_dim;       // Number of raw input features (tokens).
    int num_bins;        // Number of bins (also number of logits per token).
    int embedding_dim;   // Embedding dimension (set equal to num_bins).
    int output_dim;      // Equals input_dim.
    float *embedding_table;  // [input_dim x num_bins x embedding_dim].
    
    int num_layers;      // How many transformer layers.
    Layer *layers;       // Array of layers.
    
    /* Adam hyper–parameters (shared across layers). */
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    int batch_size;      // Batch size.
    
    /* Intermediate buffers for the multi–layer forward pass. */
    float **layer_input;       // Array of activations (num_layers+1 pointers; each is [batch_size x (input_dim*embedding_dim)])
    
    /* Self–attention intermediates. */
    float **attn_Q_layers;     // Each is [batch_size x (input_dim*embedding_dim)]
    float **attn_K_layers;
    float **attn_V_layers;
    float **attn_scores_layers; // Each is [batch_size x (input_dim x input_dim)]
    float **ff_residual_layers; // For storing r = X + attn_out.
    
    /* Feed–forward intermediates. */
    float **ff_preact_layers;  // [batch_size x (input_dim*ff_hidden_dim)]
    float **ff_hidden_layers;  // After ReLU.
    
    /* Final predictions and computed error (for backprop). */
    float *predictions;  // [batch_size x (input_dim*embedding_dim)].
    float *error;        // Same shape as predictions.
} Net;

/* ---------------- Utility Functions ---------------- */

/* In–place softmax (modifies its vector argument) */
static inline void softmax_inplace(float *logits, int num_classes)
{
    float max_val = logits[0];
    for (int j = 1; j < num_classes; j++) {
        if (logits[j] > max_val)
            max_val = logits[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        logits[j] = expf(logits[j] - max_val);
        sum_exp += logits[j];
    }
    for (int j = 0; j < num_classes; j++) {
        logits[j] /= sum_exp;
    }
}

/* Returns the bin index for a given continuous value. */
static inline int bin_value(float value, float min_value, float max_value, int num_bins)
{
    float clamped = value;
    if (clamped < min_value)
        clamped = min_value;
    if (clamped > max_value)
        clamped = max_value;
    float normalized = (clamped - min_value) / (max_value - min_value);
    int bin = (int)(normalized * num_bins);
    if (bin < 0)
        bin = 0;
    else if (bin >= num_bins)
        bin = num_bins - 1;
    return bin;
}

/* Returns a continuous value corresponding to the center of the given bin. */
static inline float unbin_value(int bin, float min_value, float max_value, int num_bins)
{
    float bin_width = (max_value - min_value) / num_bins;
    return min_value + (bin + 0.5f) * bin_width;
}

/* Computes per-feature minimum and maximum values from data. */
static inline void compute_min_max(const float *data, int num_samples, int num_features, float *min_arr, float *max_arr)
{
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

/* ---------------- Model Functions ---------------- */

/* Initializes the network with the given parameters.
   The embedding table is allocated with shape [input_dim x num_bins x embedding_dim],
   and each transformer layer (attention and feed–forward) is initialized. */
static inline Net* init_net(int input_dim, int num_bins, int embedding_dim, int output_dim, int batch_size, int num_layers)
{
    Net *net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate network.\n");
        exit(EXIT_FAILURE);
    }
    net->input_dim = input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    net->num_layers = num_layers;
    
    /* Adam settings */
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    /* Allocate embedding table: shape [input_dim x num_bins x embedding_dim]. */
    int emb_table_size = input_dim * num_bins * embedding_dim;
    net->embedding_table = (float*)malloc(emb_table_size * sizeof(float));
    if (!net->embedding_table) {
        fprintf(stderr, "Failed to allocate embedding table.\n");
        exit(EXIT_FAILURE);
    }
    float emb_scale = 1.0f / sqrtf((float)num_bins);
    for (int i = 0; i < emb_table_size; i++) {
        net->embedding_table[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * emb_scale;
    }
    
    /* Allocate transformer layers. */
    net->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    if (!net->layers) {
        fprintf(stderr, "Failed to allocate layers.\n");
        exit(EXIT_FAILURE);
    }
    for (int l = 0; l < num_layers; l++) {
        Layer *layer = &net->layers[l];
        int attn_dim = embedding_dim * embedding_dim;
        /* Allocate self–attention weights and states. */
        layer->W_Q = (float*)malloc(attn_dim * sizeof(float));
        layer->W_K = (float*)malloc(attn_dim * sizeof(float));
        layer->W_V = (float*)malloc(attn_dim * sizeof(float));
        layer->W_Q_grad = (float*)calloc(attn_dim, sizeof(float));
        layer->W_K_grad = (float*)calloc(attn_dim, sizeof(float));
        layer->W_V_grad = (float*)calloc(attn_dim, sizeof(float));
        layer->W_Q_m = (float*)calloc(attn_dim, sizeof(float));
        layer->W_K_m = (float*)calloc(attn_dim, sizeof(float));
        layer->W_V_m = (float*)calloc(attn_dim, sizeof(float));
        layer->W_Q_v = (float*)calloc(attn_dim, sizeof(float));
        layer->W_K_v = (float*)calloc(attn_dim, sizeof(float));
        layer->W_V_v = (float*)calloc(attn_dim, sizeof(float));
        float scale = 1.0f / sqrtf((float)embedding_dim);
        for (int i = 0; i < attn_dim; i++) {
            layer->W_Q[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * scale;
            layer->W_K[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * scale;
            layer->W_V[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * scale;
        }
        /* Allocate feed–forward network weights.
           ff_hidden_dim is set to 4*embedding_dim. */
        layer->ff_hidden_dim = 4 * embedding_dim;
        int ff1_size = embedding_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * embedding_dim;
        layer->W_ff1 = (float*)malloc(ff1_size * sizeof(float));
        layer->W_ff1_grad = (float*)calloc(ff1_size, sizeof(float));
        layer->W_ff1_m = (float*)calloc(ff1_size, sizeof(float));
        layer->W_ff1_v = (float*)calloc(ff1_size, sizeof(float));
        layer->W_ff2 = (float*)malloc(ff2_size * sizeof(float));
        layer->W_ff2_grad = (float*)calloc(ff2_size, sizeof(float));
        layer->W_ff2_m = (float*)calloc(ff2_size, sizeof(float));
        layer->W_ff2_v = (float*)calloc(ff2_size, sizeof(float));
        float ff1_scale = 1.0f / sqrtf((float)embedding_dim);
        for (int i = 0; i < ff1_size; i++) {
            layer->W_ff1[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * ff1_scale;
        }
        float ff2_scale = 1.0f / sqrtf((float)layer->ff_hidden_dim);
        for (int i = 0; i < ff2_size; i++) {
            layer->W_ff2[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * ff2_scale;
        }
    }
    
    /* Allocate intermediate activation buffers. */
    int act_size = batch_size * input_dim * embedding_dim;
    net->layer_input = (float**)malloc((num_layers + 1) * sizeof(float*));
    for (int i = 0; i < num_layers + 1; i++) {
        net->layer_input[i] = (float*)malloc(act_size * sizeof(float));
    }
    
    /* Allocate self–attention intermediate buffers. */
    net->attn_Q_layers = (float**)malloc(num_layers * sizeof(float*));
    net->attn_K_layers = (float**)malloc(num_layers * sizeof(float*));
    net->attn_V_layers = (float**)malloc(num_layers * sizeof(float*));
    net->attn_scores_layers = (float**)malloc(num_layers * sizeof(float*));
    net->ff_residual_layers = (float**)malloc(num_layers * sizeof(float*));
    for (int i = 0; i < num_layers; i++) {
        net->attn_Q_layers[i] = (float*)malloc(act_size * sizeof(float));
        net->attn_K_layers[i] = (float*)malloc(act_size * sizeof(float));
        net->attn_V_layers[i] = (float*)malloc(act_size * sizeof(float));
        net->attn_scores_layers[i] = (float*)malloc(batch_size * input_dim * input_dim * sizeof(float));
        net->ff_residual_layers[i] = (float*)malloc(act_size * sizeof(float));
    }
    
    /* Allocate feed–forward intermediate buffers. */
    int ff_act_size = batch_size * input_dim * (4 * embedding_dim);
    net->ff_preact_layers = (float**)malloc(num_layers * sizeof(float*));
    net->ff_hidden_layers = (float**)malloc(num_layers * sizeof(float*));
    for (int i = 0; i < num_layers; i++) {
        net->ff_preact_layers[i] = (float*)malloc(ff_act_size * sizeof(float));
        net->ff_hidden_layers[i] = (float*)malloc(ff_act_size * sizeof(float));
    }
    
    /* Allocate final predictions and error buffers. */
    net->predictions = (float*)malloc(act_size * sizeof(float));
    net->error = (float*)malloc(act_size * sizeof(float));
    
    return net;
}

/* Frees all memory allocated for the network. */
static inline void free_net(Net *net)
{
    if (net) {
        free(net->embedding_table);
        for (int l = 0; l < net->num_layers; l++) {
            Layer *layer = &net->layers[l];
            free(layer->W_Q);
            free(layer->W_K);
            free(layer->W_V);
            free(layer->W_Q_grad);
            free(layer->W_K_grad);
            free(layer->W_V_grad);
            free(layer->W_Q_m);
            free(layer->W_Q_v);
            free(layer->W_K_m);
            free(layer->W_K_v);
            free(layer->W_V_m);
            free(layer->W_V_v);
            free(layer->W_ff1);
            free(layer->W_ff1_grad);
            free(layer->W_ff1_m);
            free(layer->W_ff1_v);
            free(layer->W_ff2);
            free(layer->W_ff2_grad);
            free(layer->W_ff2_m);
            free(layer->W_ff2_v);
        }
        free(net->layers);
        for (int i = 0; i < net->num_layers + 1; i++) {
            free(net->layer_input[i]);
        }
        free(net->layer_input);
        for (int i = 0; i < net->num_layers; i++) {
            free(net->attn_Q_layers[i]);
            free(net->attn_K_layers[i]);
            free(net->attn_V_layers[i]);
            free(net->attn_scores_layers[i]);
            free(net->ff_residual_layers[i]);
        }
        free(net->attn_Q_layers);
        free(net->attn_K_layers);
        free(net->attn_V_layers);
        free(net->attn_scores_layers);
        free(net->ff_residual_layers);
        for (int i = 0; i < net->num_layers; i++) {
            free(net->ff_preact_layers[i]);
            free(net->ff_hidden_layers[i]);
        }
        free(net->ff_preact_layers);
        free(net->ff_hidden_layers);
        free(net->predictions);
        free(net->error);
        free(net);
    }
}

/* Embeds raw input data into learned token embeddings.
   The raw input (of shape [batch_size x input_dim]) is quantized per feature using
   provided minimum/maximum arrays; the corresponding embedding vector is copied into layer_input[0]. */
static inline void embed_input(Net *net, float *raw_input, float *embedded_output, float *in_min, float *in_max)
{
    for (int i = 0; i < net->batch_size; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            float val = raw_input[i * net->input_dim + j];
            int b = bin_value(val, in_min[j], in_max[j], net->num_bins);
            float *emb = net->embedding_table +
                         j * (net->num_bins * net->embedding_dim) +
                         b * net->embedding_dim;
            float *dest = net->layer_input[0] +
                          i * (net->input_dim * net->embedding_dim) +
                          j * net->embedding_dim;
            memcpy(dest, emb, net->embedding_dim * sizeof(float));
        }
    }
    memcpy(embedded_output, net->layer_input[0], net->batch_size * net->input_dim * net->embedding_dim * sizeof(float));
}

/* Forward pass: processes the embedded input through each transformer layer.
   Each layer applies self–attention (with residual) followed by a feed–forward block (with residual).
   The final layer’s output is copied into net->predictions. */
static inline void forward_pass(Net *net)
{
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int total_tokens = net->batch_size * tokens;
    int ff_hidden = 4 * d_dim;
    
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        float *X = net->layer_input[l];     // Input for layer l.
        float *out = net->layer_input[l+1];   // Output for layer l.
        
        /* Self–Attention: compute Q, K, V. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, d_dim,
                    1.0f, X, d_dim,
                    layer->W_Q, d_dim,
                    0.0f, net->attn_Q_layers[l], d_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, d_dim,
                    1.0f, X, d_dim,
                    layer->W_K, d_dim,
                    0.0f, net->attn_K_layers[l], d_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, d_dim,
                    1.0f, X, d_dim,
                    layer->W_V, d_dim,
                    0.0f, net->attn_V_layers[l], d_dim);
        
        float scale = 1.0f / sqrtf((float)d_dim);
        for (int s = 0; s < net->batch_size; s++) {
            float *Q = net->attn_Q_layers[l] + s * tokens * d_dim;
            float *K = net->attn_K_layers[l] + s * tokens * d_dim;
            float *V = net->attn_V_layers[l] + s * tokens * d_dim;
            float *scores = net->attn_scores_layers[l] + s * tokens * tokens;
            /* Compute scores = (Q * K^T) * scale. */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, tokens, d_dim,
                        scale, Q, d_dim,
                        K, d_dim,
                        0.0f, scores, tokens);
            /* Apply softmax row–wise in place. */
            for (int i = 0; i < tokens; i++) {
                softmax_inplace(scores + i * tokens, tokens);
            }
            /* Compute attention output: attn_out = scores * V. */
            float *attn_out = (float*)malloc(tokens * d_dim * sizeof(float));
            if (!attn_out) {
                fprintf(stderr, "Failed to allocate memory for attention output.\n");
                exit(EXIT_FAILURE);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        1.0f, scores, tokens,
                        V, d_dim,
                        0.0f, attn_out, d_dim);
            /* Residual connection: r = X_sample + attn_out. */
            float *X_sample = X + s * tokens * d_dim;
            float *r = net->ff_residual_layers[l] + s * tokens * d_dim;
            for (int i = 0; i < tokens * d_dim; i++) {
                r[i] = X_sample[i] + attn_out[i];
            }
            free(attn_out);
        }
        
        /* Feed–Forward Block.
           1. Compute pre–activation: ff_preact = r * W_ff1.
           2. Apply ReLU.
           3. Compute f = (ReLU output) * W_ff2.
           4. Final output: add residual r. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, ff_hidden, d_dim,
                    1.0f, net->ff_residual_layers[l], d_dim,
                    layer->W_ff1, ff_hidden,
                    0.0f, net->ff_preact_layers[l], ff_hidden);
        for (int i = 0; i < total_tokens * ff_hidden; i++) {
            net->ff_hidden_layers[l][i] = (net->ff_preact_layers[l][i] > 0.0f) ? net->ff_preact_layers[l][i] : 0.0f;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, ff_hidden,
                    1.0f, net->ff_hidden_layers[l], ff_hidden,
                    layer->W_ff2, d_dim,
                    0.0f, out, d_dim);
        for (int i = 0; i < total_tokens * d_dim; i++) {
            out[i] += net->ff_residual_layers[l][i];
        }
    }
    /* Copy the final layer output into predictions. */
    memcpy(net->predictions, net->layer_input[net->num_layers], total_tokens * d_dim * sizeof(float));
}

/* Computes the average cross–entropy loss over tokens and sets net->error.
   For each token, softmax is applied to obtain probabilities and compared to the one–hot target. */
static inline float calculate_loss(Net *net, int *target_labels)
{
    float loss = 0.0f;
    int batch = net->batch_size;
    int classes = net->num_bins;  // Each token’s output length.
    int total = net->output_dim;  // Equals input_dim.
    float *probs = (float*)malloc(classes * sizeof(float));
    if (!probs) {
        fprintf(stderr, "Failed to allocate memory for probabilities.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < batch; i++) {
        for (int d = 0; d < total; d++) {
            int offset = i * (classes * total) + d * classes;
            memcpy(probs, net->predictions + offset, classes * sizeof(float));
            softmax_inplace(probs, classes);
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

/* Resets all gradients in the network to zero. */
static inline void zero_gradients(Net *net)
{
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        memset(layer->W_Q_grad, 0, attn_dim * sizeof(float));
        memset(layer->W_K_grad, 0, attn_dim * sizeof(float));
        memset(layer->W_V_grad, 0, attn_dim * sizeof(float));
        int ff1_size = d_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * d_dim;
        memset(layer->W_ff1_grad, 0, ff1_size * sizeof(float));
        memset(layer->W_ff2_grad, 0, ff2_size * sizeof(float));
    }
}

/* Backward pass: computes gradients from net->error backward through each layer.
   First, the feed–forward branch is backpropagated; then the self–attention branch.
   Gradients for all weights (attention and feed–forward) are accumulated. */
static inline void backward_pass(Net *net)
{
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int total_tokens = net->batch_size * tokens;
    int ff_hidden = 4 * d_dim;
    float inv_sqrt = 1.0f / sqrtf((float)d_dim);

    zero_gradients(net);

    /* dX holds the gradient with respect to the output of the current (or final) layer.
       We start from net->error. */
    float *dX = (float*)malloc(total_tokens * d_dim * sizeof(float));
    if (!dX) {
        fprintf(stderr, "Failed to allocate memory for dX.\n");
        exit(EXIT_FAILURE);
    }
    memcpy(dX, net->error, total_tokens * d_dim * sizeof(float));

    /* Loop backward over layers. */
    for (int l = net->num_layers - 1; l >= 0; l--) {
        Layer *layer = &net->layers[l];
        float *X = net->layer_input[l];             // Input to layer l.
        float *r = net->ff_residual_layers[l];        // r = X + attn_out.

        /* --- Backprop through Feed–Forward Block --- */
        /* f was computed as: out = r + f, where f = (ReLU(r*W_ff1))*W_ff2.
           (a) The residual connection gives d_r_add = dX. */
        float *d_r_add = (float*)malloc(total_tokens * d_dim * sizeof(float));
        memcpy(d_r_add, dX, total_tokens * d_dim * sizeof(float));
        /* (b) Propagate dX through feed–forward branch. */
        float *d_hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    total_tokens, ff_hidden, d_dim,
                    1.0f, dX, d_dim,
                    layer->W_ff2, d_dim,
                    0.0f, d_hidden, ff_hidden);
        float *d_preact = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
        float *ff_preact = net->ff_preact_layers[l];
        for (int i = 0; i < total_tokens * ff_hidden; i++) {
            d_preact[i] = (ff_preact[i] > 0.0f) ? d_hidden[i] : 0.0f;
        }
        /* Gradient for W_ff1: d(W_ff1) += r^T * d_preact. */
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    d_dim, ff_hidden, total_tokens,
                    1.0f, r, d_dim,
                    d_preact, ff_hidden,
                    1.0f, layer->W_ff1_grad, ff_hidden);
        float *d_r_ff = (float*)malloc(total_tokens * d_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    total_tokens, d_dim, ff_hidden,
                    1.0f, d_preact, ff_hidden,
                    layer->W_ff1, ff_hidden,
                    0.0f, d_r_ff, d_dim);
        /* Gradient for W_ff2: d(W_ff2) += (ReLU output)^T * dX. */
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ff_hidden, d_dim, total_tokens,
                    1.0f, net->ff_hidden_layers[l], ff_hidden,
                    dX, d_dim,
                    1.0f, layer->W_ff2_grad, d_dim);
        /* Sum gradients from residual and feed–forward branch. */
        float *d_r = (float*)malloc(total_tokens * d_dim * sizeof(float));
        for (int i = 0; i < total_tokens * d_dim; i++) {
            d_r[i] = d_r_add[i] + d_r_ff[i];
        }
        free(d_r_add);
        free(d_r_ff);
        free(d_hidden);
        free(d_preact);

        /* --- Backprop through Self–Attention Block --- */
        /* d_attn_out holds the gradient coming from the residual branch onward.
           (Since r = X + attn_out, we pass d_r to the attention branch.) */
        float *d_attn_out = (float*)malloc(total_tokens * d_dim * sizeof(float));
        memcpy(d_attn_out, d_r, total_tokens * d_dim * sizeof(float));
        for (int s = 0; s < net->batch_size; s++) {
            float *scores = net->attn_scores_layers[l] + s * tokens * tokens;
            float *d_attn_out_sample = d_attn_out + s * tokens * d_dim;
            float *V = net->attn_V_layers[l] + s * tokens * d_dim;
            float *Q = net->attn_Q_layers[l] + s * tokens * d_dim;
            float *K = net->attn_K_layers[l] + s * tokens * d_dim;

            /* Compute dV = scores^T * d_attn_out_sample. */
            float *dV = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        1.0f, scores, tokens,
                        d_attn_out_sample, d_dim,
                        0.0f, dV, d_dim);

            /* Compute intermediate dA = d_attn_out_sample * V^T. */
            float *dA = (float*)malloc(tokens * tokens * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, tokens, d_dim,
                        1.0f, d_attn_out_sample, d_dim,
                        V, d_dim,
                        0.0f, dA, tokens);
            /* Apply softmax derivative for each row. */
            for (int i = 0; i < tokens; i++) {
                float sum_d = 0.0f;
                for (int j = 0; j < tokens; j++) {
                    sum_d += scores[i * tokens + j] * dA[i * tokens + j];
                }
                for (int j = 0; j < tokens; j++) {
                    dA[i * tokens + j] = scores[i * tokens + j] * (dA[i * tokens + j] - sum_d);
                }
            }
            /* Compute dQ = (dA * K) * inv_sqrt. */
            float *dQ = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        inv_sqrt, dA, tokens,
                        K, d_dim,
                        0.0f, dQ, d_dim);
            /* Compute dK = (dA^T * Q) * inv_sqrt. */
            float *dK = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        inv_sqrt, dA, tokens,
                        Q, d_dim,
                        0.0f, dK, d_dim);
            /* Accumulate gradients for attention weights:
               d(W_Q) += X_sample^T * dQ, similarly for W_K and W_V. */
            float *X_sample = X + s * tokens * d_dim;
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d_dim, d_dim, tokens,
                        1.0f, X_sample, d_dim,
                        dQ, d_dim,
                        1.0f, layer->W_Q_grad, d_dim);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d_dim, d_dim, tokens,
                        1.0f, X_sample, d_dim,
                        dK, d_dim,
                        1.0f, layer->W_K_grad, d_dim);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d_dim, d_dim, tokens,
                        1.0f, X_sample, d_dim,
                        dV, d_dim,
                        1.0f, layer->W_V_grad, d_dim);
            /* Compute contribution to dX from the attention branch.
               This is done by multiplying dQ, dK, dV with their corresponding weight matrices and summing. */
            float *temp = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, d_dim, d_dim,
                        1.0f, dQ, d_dim,
                        layer->W_Q, d_dim,
                        0.0f, temp, d_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, d_dim, d_dim,
                        1.0f, dK, d_dim,
                        layer->W_K, d_dim,
                        1.0f, temp, d_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, d_dim, d_dim,
                        1.0f, dV, d_dim,
                        layer->W_V, d_dim,
                        1.0f, temp, d_dim);
            for (int i = 0; i < tokens * d_dim; i++) {
                dX[s * tokens * d_dim + i] += temp[i];
            }
            free(temp);
            free(dA);
            free(dQ);
            free(dK);
            free(dV);
        }
        free(d_attn_out);
        free(d_r);
    }
    free(dX);
}

/* Updates network weights via AdamW.
   For each parameter the Adam–moving averages (m and v) are updated,
   then a weight decay is applied before subtracting the calculated update. */
static inline void update_weights(Net *net, float learning_rate)
{
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        /* Update self–attention weights. */
        for (int i = 0; i < attn_dim; i++) {
            float grad = layer->W_Q_grad[i] / net->batch_size;
            layer->W_Q_m[i] = net->beta1 * layer->W_Q_m[i] + (1.0f - net->beta1) * grad;
            layer->W_Q_v[i] = net->beta2 * layer->W_Q_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_Q_m[i] / (sqrtf(layer->W_Q_v[i]) + net->epsilon);
            layer->W_Q[i] = layer->W_Q[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        for (int i = 0; i < attn_dim; i++) {
            float grad = layer->W_K_grad[i] / net->batch_size;
            layer->W_K_m[i] = net->beta1 * layer->W_K_m[i] + (1.0f - net->beta1) * grad;
            layer->W_K_v[i] = net->beta2 * layer->W_K_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_K_m[i] / (sqrtf(layer->W_K_v[i]) + net->epsilon);
            layer->W_K[i] = layer->W_K[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        for (int i = 0; i < attn_dim; i++) {
            float grad = layer->W_V_grad[i] / net->batch_size;
            layer->W_V_m[i] = net->beta1 * layer->W_V_m[i] + (1.0f - net->beta1) * grad;
            layer->W_V_v[i] = net->beta2 * layer->W_V_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_V_m[i] / (sqrtf(layer->W_V_v[i]) + net->epsilon);
            layer->W_V[i] = layer->W_V[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        /* Update feed–forward weights. */
        int ff1_size = d_dim * layer->ff_hidden_dim;
        for (int i = 0; i < ff1_size; i++) {
            float grad = layer->W_ff1_grad[i] / net->batch_size;
            layer->W_ff1_m[i] = net->beta1 * layer->W_ff1_m[i] + (1.0f - net->beta1) * grad;
            layer->W_ff1_v[i] = net->beta2 * layer->W_ff1_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_ff1_m[i] / (sqrtf(layer->W_ff1_v[i]) + net->epsilon);
            layer->W_ff1[i] = layer->W_ff1[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        int ff2_size = layer->ff_hidden_dim * d_dim;
        for (int i = 0; i < ff2_size; i++) {
            float grad = layer->W_ff2_grad[i] / net->batch_size;
            layer->W_ff2_m[i] = net->beta1 * layer->W_ff2_m[i] + (1.0f - net->beta1) * grad;
            layer->W_ff2_v[i] = net->beta2 * layer->W_ff2_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_ff2_m[i] / (sqrtf(layer->W_ff2_v[i]) + net->epsilon);
            layer->W_ff2[i] = layer->W_ff2[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
    }
}

/* Saves the network parameters to a binary file. */
static inline void save_model(Net *net, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->num_bins, sizeof(int), 1, file);
    fwrite(&net->embedding_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(&net->num_layers, sizeof(int), 1, file);
    
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        fwrite(layer->W_Q, sizeof(float), attn_dim, file);
        fwrite(layer->W_K, sizeof(float), attn_dim, file);
        fwrite(layer->W_V, sizeof(float), attn_dim, file);
    }
    fwrite(&net->t, sizeof(int), 1, file);
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        fwrite(layer->W_Q_m, sizeof(float), attn_dim, file);
        fwrite(layer->W_Q_v, sizeof(float), attn_dim, file);
        fwrite(layer->W_K_m, sizeof(float), attn_dim, file);
        fwrite(layer->W_K_v, sizeof(float), attn_dim, file);
        fwrite(layer->W_V_m, sizeof(float), attn_dim, file);
        fwrite(layer->W_V_v, sizeof(float), attn_dim, file);
    }
    int emb_table_size = net->input_dim * net->num_bins * net->embedding_dim;
    fwrite(net->embedding_table, sizeof(float), emb_table_size, file);
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        fwrite(&layer->ff_hidden_dim, sizeof(int), 1, file);
        int ff1_size = d_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * d_dim;
        fwrite(layer->W_ff1, sizeof(float), ff1_size, file);
        fwrite(layer->W_ff2, sizeof(float), ff2_size, file);
        fwrite(layer->W_ff1_m, sizeof(float), ff1_size, file);
        fwrite(layer->W_ff1_v, sizeof(float), ff1_size, file);
        fwrite(layer->W_ff2_m, sizeof(float), ff2_size, file);
        fwrite(layer->W_ff2_v, sizeof(float), ff2_size, file);
    }
    fclose(file);
    printf("Model saved to %s\n", filename);
}

/* Loads network parameters from a binary file and returns a new Net pointer. */
static inline Net* load_model(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    int input_dim, num_bins, embedding_dim, output_dim, batch_size, num_layers;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&num_bins, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
    Net *net = init_net(input_dim, num_bins, embedding_dim, output_dim, batch_size, num_layers);
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        fread(layer->W_Q, sizeof(float), attn_dim, file);
        fread(layer->W_K, sizeof(float), attn_dim, file);
        fread(layer->W_V, sizeof(float), attn_dim, file);
    }
    fread(&net->t, sizeof(int), 1, file);
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        fread(layer->W_Q_m, sizeof(float), attn_dim, file);
        fread(layer->W_Q_v, sizeof(float), attn_dim, file);
        fread(layer->W_K_m, sizeof(float), attn_dim, file);
        fread(layer->W_K_v, sizeof(float), attn_dim, file);
        fread(layer->W_V_m, sizeof(float), attn_dim, file);
        fread(layer->W_V_v, sizeof(float), attn_dim, file);
    }
    int emb_table_size = input_dim * num_bins * embedding_dim;
    fread(net->embedding_table, sizeof(float), emb_table_size, file);
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        fread(&layer->ff_hidden_dim, sizeof(int), 1, file);
        int ff1_size = d_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * d_dim;
        fread(layer->W_ff1, sizeof(float), ff1_size, file);
        fread(layer->W_ff2, sizeof(float), ff2_size, file);
        fread(layer->W_ff1_m, sizeof(float), ff1_size, file);
        fread(layer->W_ff1_v, sizeof(float), ff1_size, file);
        fread(layer->W_ff2_m, sizeof(float), ff2_size, file);
        fread(layer->W_ff2_v, sizeof(float), ff2_size, file);
    }
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

#endif /* MLP_H */