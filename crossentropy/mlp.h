#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  This header defines a multilayer perceptron (MLP) with an embedding lookup.
  Each raw input feature (of type float) is first binned into one of a fixed number
  of bins (using per‑feature min/max values) and then mapped into an embedding vector.
  The concatenated embeddings serve as the effective input to a two‑layer MLP.
  The final layer outputs logits (one per class, where number of classes = num_bins)
  which are used with softmax and cross‑entropy loss for classification.
*/

typedef struct {
    // Raw input parameters.
    int input_dim;     // Number of raw input features.
    int num_bins;      // Number of bins to quantize each feature (used for both embedding lookup and output classes).
    int embedding_dim; // Dimension of each embedding vector.

    // Embedding table: size = input_dim * num_bins * embedding_dim.
    float* embedding_table;

    // MLP weights and their gradients.
    // The effective input to the MLP is a concatenation of all embeddings,
    // whose dimension is embedded_input_dim = input_dim * embedding_dim.
    float* fc1_weight;       // Hidden layer weights: [hidden_dim x embedded_input_dim]
    float* fc2_weight;       // Output layer weights: [num_bins x hidden_dim]
    float* fc1_weight_grad;  // Gradients for fc1_weight.
    float* fc2_weight_grad;  // Gradients for fc2_weight.

    // Adam optimizer state for fc1 and fc2.
    float* fc1_m;   // First moment estimate for fc1.
    float* fc1_v;   // Second moment estimate for fc1.
    float* fc2_m;   // First moment estimate for fc2.
    float* fc2_v;   // Second moment estimate for fc2.
    float beta1;    // Decay rate for first moment.
    float beta2;    // Decay rate for second moment.
    float epsilon;  // Small constant for numerical stability.
    int   t;        // Time step counter.
    float weight_decay; // Weight decay factor for AdamW.

    // Helper arrays for forward/backward passes.
    float* hidden_output;  // Hidden layer activations (after Swish), shape: [batch_size x hidden_dim]
    float* predictions;    // Output logits; shape: [batch_size x num_bins]
    float* error;          // Output error (gradient from softmax/CE loss); shape: [batch_size x num_bins]
    float* pre_activation; // Hidden layer pre-activation values; shape: [batch_size x hidden_dim]
    float* error_hidden;   // Backpropagated error for hidden layer; shape: [batch_size x hidden_dim]

    // Dimensions.
    int embedded_input_dim; // Effective input dimension = input_dim * embedding_dim.
    int hidden_dim;         // Number of neurons in the hidden layer.
    int batch_size;         // Number of samples per training batch.
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
  The value is first clamped into [min_value, max_value] then normalized.
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
  compute_min_max: For an array “data” of shape [num_samples x num_features],
  computes per‑feature minimum and maximum. min_arr and max_arr must be pre‑allocated arrays (length num_features).
*/
static inline void compute_min_max(const float* data, int num_samples, int num_features, float* min_arr, float* max_arr) {
    for (int j = 0; j < num_features; j++) {
        min_arr[j] = data[j]; // first sample value for feature j.
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

/* ---------------- MLP and Embedding Functions ---------------- */

/*
  init_net: Initializes the network with embedding lookup and two-layer MLP.
  Parameters:
    input_dim: number of raw input features.
    num_bins: number of bins used for both embedding lookup and output classes.
    embedding_dim: dimension of each embedding vector.
    hidden_dim: number of neurons in the hidden layer.
    batch_size: number of samples per training batch.
  
  Note: The effective (embedded) input dimension is computed as (input_dim * embedding_dim).
*/
static inline Net* init_net(int input_dim, int num_bins, int embedding_dim,
                            int hidden_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate network.\n");
        exit(EXIT_FAILURE);
    }

    net->input_dim = input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    net->embedded_input_dim = input_dim * embedding_dim;

    net->hidden_dim = hidden_dim;
    // For classification, the number of output classes equals num_bins.
    // (So there is a single num_bins used for both embedding and output.)
    
    net->batch_size = batch_size;
    
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    net->fc1_weight = (float*)malloc(hidden_dim * net->embedded_input_dim * sizeof(float));
    net->fc2_weight = (float*)malloc(num_bins * hidden_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(hidden_dim * net->embedded_input_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(num_bins * hidden_dim * sizeof(float));
    
    net->fc1_m = (float*)calloc(hidden_dim * net->embedded_input_dim, sizeof(float));
    net->fc1_v = (float*)calloc(hidden_dim * net->embedded_input_dim, sizeof(float));
    net->fc2_m = (float*)calloc(num_bins * hidden_dim, sizeof(float));
    net->fc2_v = (float*)calloc(num_bins * hidden_dim, sizeof(float));
    
    net->hidden_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->predictions = (float*)malloc(batch_size * num_bins * sizeof(float));
    net->error = (float*)malloc(batch_size * num_bins * sizeof(float));
    net->pre_activation = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    float scale1 = 1.0f / sqrtf((float)net->embedded_input_dim);
    for (int i = 0; i < hidden_dim * net->embedded_input_dim; i++) {
        net->fc1_weight[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * scale1;
    }
    
    float scale2 = 1.0f / sqrtf((float)hidden_dim);
    for (int i = 0; i < num_bins * hidden_dim; i++) {
        net->fc2_weight[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * scale2;
    }
    
    net->embedding_table = (float*)malloc(input_dim * num_bins * embedding_dim * sizeof(float));
    float emb_scale = 1.0f / sqrtf((float)num_bins);
    for (int i = 0; i < input_dim * num_bins * embedding_dim; i++) {
        net->embedding_table[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * emb_scale;
    }
    
    return net;
}

/*
  free_net: Releases all memory allocated for the network.
*/
static inline void free_net(Net* net) {
    if(net) {
        free(net->fc1_weight);
        free(net->fc2_weight);
        free(net->fc1_weight_grad);
        free(net->fc2_weight_grad);
        free(net->fc1_m);
        free(net->fc1_v);
        free(net->fc2_m);
        free(net->fc2_v);
        free(net->hidden_output);
        free(net->predictions);
        free(net->error);
        free(net->pre_activation);
        free(net->error_hidden);
        free(net->embedding_table);
        free(net);
    }
}

/*
  embed_input: For each raw input (shape: [batch_size x input_dim]) and given per‑feature
  min and max (both arrays of length input_dim), determines the bin for each feature value
  and copies the corresponding embedding vector into embedded_output.
  The output shape is [batch_size x (input_dim * embedding_dim)].
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
  forward_pass: Performs a forward pass using the embedded input X (of shape: [batch_size x embedded_input_dim]).
  It computes the hidden layer activations using Swish activation and then computes the output logits.
*/
static inline void forward_pass(Net* net, float* X) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                net->batch_size, net->hidden_dim, net->embedded_input_dim,
                1.0f, X, net->embedded_input_dim,
                net->fc1_weight, net->hidden_dim,
                0.0f, net->hidden_output, net->hidden_dim);
    
    memcpy(net->pre_activation, net->hidden_output, net->batch_size * net->hidden_dim * sizeof(float));
    
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float z = net->hidden_output[i];
        net->hidden_output[i] = z / (1.0f + expf(-z));  // Swish activation: z * sigmoid(z)
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                net->batch_size, net->num_bins, net->hidden_dim,
                1.0f, net->hidden_output, net->hidden_dim,
                net->fc2_weight, net->num_bins,
                0.0f, net->predictions, net->num_bins);
}

/*
  calculate_loss: Applies softmax to the predictions, computes the cross‑entropy loss
  (averaged over the batch), and sets net->error = softmax(probabilities) - one_hot(target).
*/
static inline float calculate_loss(Net* net, int* target_labels) {
    float loss = 0.0f;
    int batch = net->batch_size;
    int classes = net->num_bins;
    float* probs = (float*)malloc(classes * sizeof(float));
    if (!probs) {
        fprintf(stderr, "Failed to allocate memory for probabilities.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < batch; i++) {
        float* logits = net->predictions + i * classes;
        softmax(logits, probs, classes);
        int target = target_labels[i];
        loss -= logf(probs[target] + net->epsilon);
        for (int j = 0; j < classes; j++) {
            net->error[i * classes + j] = probs[j] - ((j == target) ? 1.0f : 0.0f);
        }
    }
    free(probs);
    return loss / batch;
}

/*
  zero_gradients: Resets the stored gradients for fc1 and fc2.
*/
static inline void zero_gradients(Net* net) {
    memset(net->fc1_weight_grad, 0, net->hidden_dim * net->embedded_input_dim * sizeof(float));
    memset(net->fc2_weight_grad, 0, net->num_bins * net->hidden_dim * sizeof(float));
}

/*
  backward_pass: Performs backpropagation. Given the embedded input X,
  computes gradients for fc2 and fc1.
*/
static inline void backward_pass(Net* net, float* X) {
    // Compute the gradient for fc2 weights: hidden_output^T * error.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                net->hidden_dim, net->num_bins, net->batch_size,
                1.0f, net->hidden_output, net->hidden_dim,
                net->error, net->num_bins,
                0.0f, net->fc2_weight_grad, net->num_bins);
    
    // Backpropagate error to hidden layer: error_hidden = error * fc2_weight^T.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                net->batch_size, net->hidden_dim, net->num_bins,
                1.0f, net->error, net->num_bins,
                net->fc2_weight, net->num_bins,
                0.0f, net->error_hidden, net->hidden_dim);
    
    // Account for the derivative of the Swish activation.
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float z = net->pre_activation[i];
        float sigmoid = 1.0f / (1.0f + expf(-z));
        net->error_hidden[i] *= (sigmoid + z * sigmoid * (1.0f - sigmoid));
    }
    
    // Compute gradient for fc1 weights: X^T * error_hidden.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                net->embedded_input_dim, net->hidden_dim, net->batch_size,
                1.0f, X, net->embedded_input_dim,
                net->error_hidden, net->hidden_dim,
                0.0f, net->fc1_weight_grad, net->hidden_dim);
}

/*
  update_weights: Updates the network weights using the AdamW optimizer.
  learning_rate: Base learning rate.
*/
static inline void update_weights(Net* net, float learning_rate) {
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int i = 0; i < net->hidden_dim * net->embedded_input_dim; i++) {
        float grad = net->fc1_weight_grad[i] / net->batch_size;
        net->fc1_m[i] = net->beta1 * net->fc1_m[i] + (1.0f - net->beta1) * grad;
        net->fc1_v[i] = net->beta2 * net->fc1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->fc1_m[i] / (sqrtf(net->fc1_v[i]) + net->epsilon);
        net->fc1_weight[i] = net->fc1_weight[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    
    for (int i = 0; i < net->num_bins * net->hidden_dim; i++) {
        float grad = net->fc2_weight_grad[i] / net->batch_size;
        net->fc2_m[i] = net->beta1 * net->fc2_m[i] + (1.0f - net->beta1) * grad;
        net->fc2_v[i] = net->beta2 * net->fc2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->fc2_m[i] / (sqrtf(net->fc2_v[i]) + net->epsilon);
        net->fc2_weight[i] = net->fc2_weight[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
}

/*
  save_model: Saves network dimensions, weights, Adam state, and embedding table to a binary file.
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
    fwrite(&net->embedded_input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    
    fwrite(net->fc1_weight, sizeof(float), net->hidden_dim * net->embedded_input_dim, file);
    fwrite(net->fc2_weight, sizeof(float), net->num_bins * net->hidden_dim, file);
    
    fwrite(&net->t, sizeof(int), 1, file);
    fwrite(net->fc1_m, sizeof(float), net->hidden_dim * net->embedded_input_dim, file);
    fwrite(net->fc1_v, sizeof(float), net->hidden_dim * net->embedded_input_dim, file);
    fwrite(net->fc2_m, sizeof(float), net->num_bins * net->hidden_dim, file);
    fwrite(net->fc2_v, sizeof(float), net->num_bins * net->hidden_dim, file);
    
    size_t emb_table_size = (size_t)net->input_dim * net->num_bins * net->embedding_dim;
    fwrite(net->embedding_table, sizeof(float), emb_table_size, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

/*
  load_model: Loads network dimensions, weights, Adam state, and the embedding table from a binary file.
  Returns a pointer to the loaded network.
*/
static inline Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    int input_dim, num_bins, embedding_dim, embedded_input_dim, hidden_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&num_bins, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&embedded_input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    Net* net = init_net(input_dim, num_bins, embedding_dim, hidden_dim, batch_size);
    
    fread(net->fc1_weight, sizeof(float), hidden_dim * embedded_input_dim, file);
    fread(net->fc2_weight, sizeof(float), num_bins * hidden_dim, file);
    
    fread(&net->t, sizeof(int), 1, file);
    fread(net->fc1_m, sizeof(float), hidden_dim * embedded_input_dim, file);
    fread(net->fc1_v, sizeof(float), hidden_dim * embedded_input_dim, file);
    fread(net->fc2_m, sizeof(float), num_bins * hidden_dim, file);
    fread(net->fc2_v, sizeof(float), num_bins * hidden_dim, file);
    
    size_t emb_table_size = (size_t)input_dim * num_bins * embedding_dim;
    fread(net->embedding_table, sizeof(float), emb_table_size, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

#endif /* MLP_H */