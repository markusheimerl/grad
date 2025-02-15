#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

// If not already defined, set the input range for raw features.

/*
  This header defines a multilayer perceptron (MLP) with an embedding lookup.
  Each raw input feature (assumed to be a float) is first binned into one of a fixed number
  of bins and then an embedding vector is selected from an embedding table. The chosen embeddings
  (one per raw input feature) are concatenated into one 1D vector which serves as the effective input
  to the MLP.

  Additionally, the network has been modified for classification rather than direct regression.
  Instead of predicting a single continuous output value, the network’s final layer now produces
  a vector of logits (of length equal to target_bins) that represent a probability distribution over
  discrete bins (classes). The cross‑entropy loss is used for training, and during the loss computation
  the softmax function is applied to the logits.
*/

typedef struct {
    // Embedding parameters.
    int raw_input_dim;    // Number of raw (continuous) input features.
    int num_bins;         // Number of bins to discretize each input feature (for embedding lookup).
    int embedding_dim;    // Dimension of the embedding vector for each feature.
    // The embedding table is stored as a contiguous array of size:
    // raw_input_dim * num_bins * embedding_dim.
    float* embedding_table;
    
    // MLP weights and gradients.
    // Note: the effective input dimension equals raw_input_dim * embedding_dim.
    float* fc1_weight;      // Dimensions: hidden_dim x (raw_input_dim * embedding_dim)
    float* fc2_weight;      // Dimensions: target_bins x hidden_dim  (target_bins equals number of classes)
    float* fc1_weight_grad; // Same as fc1_weight dimensions.
    float* fc2_weight_grad; // Same as fc2_weight dimensions.
    
    // Adam optimizer parameters for fc1 and fc2.
    float* fc1_m;  // First moment for fc1.
    float* fc1_v;  // Second moment for fc1.
    float* fc2_m;  // First moment for fc2.
    float* fc2_v;  // Second moment for fc2.
    float beta1;   // Exponential decay rate for first moment.
    float beta2;   // Exponential decay rate for second moment.
    float epsilon; // Small constant for numerical stability.
    int   t;       // Time step counter.
    float weight_decay; // Weight decay parameter (lambda) for AdamW.
    
    // Helper arrays for forward/backward passes.
    float* layer1_output;  // Output of hidden layer (batch_size x hidden_dim)
    float* predictions;    // Output predictions (batch_size x target_bins), these are logits.
    float* error;          // Error at output (batch_size x target_bins), will store the gradient from loss.
    float* pre_activation; // Pre-activation values of hidden layer (batch_size x hidden_dim)
    float* error_hidden;   // Error backpropagated into hidden layer (batch_size x hidden_dim)
    
    // Dimensions.
    int input_dim;   // Effective input dimension = raw_input_dim * embedding_dim.
    int hidden_dim;
    int output_dim;  // For classification, this equals target_bins (the number of classes).
    int batch_size;
} Net;

///////////////////////////////////////////////////////////
// Function: softmax
// Computes the softmax of an array of logits.
//   logits: input array of length num_classes.
//   probs: output array of length num_classes (probabilities).
//   num_classes: number of classes.
///////////////////////////////////////////////////////////
static void softmax(const float* logits, float* probs, int num_classes) {
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

/////////////////////////////////////////////////////////
// Initialize the network with embeddings.
// Parameters:
//   raw_input_dim: number of raw features (before embedding).
//   num_bins: number of bins per feature for embedding lookup.
//   embedding_dim: dimension of the embedding vector for each feature.
//   hidden_dim: number of neurons in the hidden layer.
//   target_bins: number of bins (classes) for the target variable prediction.
//   batch_size: number of samples in one batch.
/////////////////////////////////////////////////////////
Net* init_net(int raw_input_dim, int num_bins, int embedding_dim,
              int hidden_dim, int target_bins, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));

    // Store embedding parameters.
    net->raw_input_dim = raw_input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    // Compute the effective input dimension.
    net->input_dim = raw_input_dim * embedding_dim;

    // Store MLP dimensions.
    net->hidden_dim = hidden_dim;
    net->output_dim = target_bins;  // For classification, output_dim equals the number of target bins.
    net->batch_size = batch_size;

    // Initialize Adam hyperparameters.
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;

    // Allocate and initialize MLP weights and gradients.
    net->fc1_weight = (float*)malloc(net->hidden_dim * net->input_dim * sizeof(float));
    net->fc2_weight = (float*)malloc(net->output_dim * net->hidden_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(net->hidden_dim * net->input_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(net->output_dim * net->hidden_dim * sizeof(float));

    // Allocate and zero Adam buffers.
    net->fc1_m = (float*)calloc(net->hidden_dim * net->input_dim, sizeof(float));
    net->fc1_v = (float*)calloc(net->hidden_dim * net->input_dim, sizeof(float));
    net->fc2_m = (float*)calloc(net->output_dim * net->hidden_dim, sizeof(float));
    net->fc2_v = (float*)calloc(net->output_dim * net->hidden_dim, sizeof(float));

    // Allocate helper arrays.
    net->layer1_output = (float*)malloc(batch_size * net->hidden_dim * sizeof(float));
    net->predictions = (float*)malloc(batch_size * net->output_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * net->output_dim * sizeof(float));
    net->pre_activation = (float*)malloc(batch_size * net->hidden_dim * sizeof(float));
    net->error_hidden = (float*)malloc(batch_size * net->hidden_dim * sizeof(float));

    // Initialize fc1 weights.
    float scale1 = 1.0f / sqrtf((float)net->input_dim);
    for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
        net->fc1_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale1;
    }

    // Initialize fc2 weights.
    float scale2 = 1.0f / sqrtf((float)net->hidden_dim);
    for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
        net->fc2_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale2;
    }

    // Allocate and initialize the embedding table.
    // Table size: raw_input_dim * num_bins * embedding_dim.
    net->embedding_table = (float*)malloc(raw_input_dim * num_bins * embedding_dim * sizeof(float));
    float emb_scale = 1.0f / sqrtf((float)num_bins);
    for (int i = 0; i < raw_input_dim * num_bins * embedding_dim; i++) {
        net->embedding_table[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * emb_scale;
    }

    return net;
}

/////////////////////////////////////////////////////////
// Free network memory.
/////////////////////////////////////////////////////////
void free_net(Net* net) {
    free(net->fc1_weight);
    free(net->fc2_weight);
    free(net->fc1_weight_grad);
    free(net->fc2_weight_grad);
    free(net->fc1_m);
    free(net->fc1_v);
    free(net->fc2_m);
    free(net->fc2_v);
    free(net->layer1_output);
    free(net->predictions);
    free(net->error);
    free(net->pre_activation);
    free(net->error_hidden);
    free(net->embedding_table);
    free(net);
}

/////////////////////////////////////////////////////////
// Embedding lookup.
// Converts raw inputs of shape (batch_size x raw_input_dim)
// into a concatenated embedding vector of shape (batch_size x (raw_input_dim * embedding_dim)).
// Assumes raw input values lie in the range [INPUT_RANGE_MIN, INPUT_RANGE_MAX].
/////////////////////////////////////////////////////////
void embed_input(Net* net, float* raw_input, float* embedded_output) {
    float min_val = INPUT_RANGE_MIN;
    float max_val = INPUT_RANGE_MAX;
    float range = max_val - min_val;

    for (int i = 0; i < net->batch_size; i++) {
        for (int j = 0; j < net->raw_input_dim; j++) {
            float val = raw_input[i * net->raw_input_dim + j];
            int bin_idx = (int)(((val - min_val) / range) * net->num_bins);
            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= net->num_bins) bin_idx = net->num_bins - 1;
            
            float* emb = net->embedding_table + j * (net->num_bins * net->embedding_dim)
                                     + bin_idx * net->embedding_dim;
            float* dest = embedded_output + i * (net->raw_input_dim * net->embedding_dim)
                                      + j * net->embedding_dim;
            memcpy(dest, emb, net->embedding_dim * sizeof(float));
        }
    }
}

/////////////////////////////////////////////////////////
// Forward pass of the MLP (using an already-embedded input).
// Expects X to be of shape (batch_size x (raw_input_dim * embedding_dim)).
// The network applies a hidden layer with Swish activation and outputs logits.
/////////////////////////////////////////////////////////
void forward_pass(Net* net, float* X) {
    // Hidden layer: Z = X * fc1_weight.
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->hidden_dim,
                net->input_dim,
                1.0f,
                X,
                net->input_dim,
                net->fc1_weight,
                net->hidden_dim,
                0.0f,
                net->layer1_output,
                net->hidden_dim);
    
    // Save pre-activation values.
    memcpy(net->pre_activation, net->layer1_output,
           net->batch_size * net->hidden_dim * sizeof(float));
    
    // Apply Swish activation: A = Z * sigmoid(Z).
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->layer1_output[i] = net->layer1_output[i] / (1.0f + expf(-net->layer1_output[i]));
    }
    
    // Output layer: logits = A * fc2_weight.
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->output_dim,
                net->hidden_dim,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->fc2_weight,
                net->output_dim,
                0.0f,
                net->predictions,
                net->output_dim);
}

/////////////////////////////////////////////////////////
// Calculate cross-entropy loss and compute the gradient at the output.
// target_labels: integer labels for each sample (values in [0, output_dim-1]).
// This function applies softmax on the logits and then computes the loss.
// It also sets net->error to the gradient for the output layer:
//     (softmax(probabilities) - one_hot(target))
/////////////////////////////////////////////////////////
float calculate_loss(Net* net, int* target_labels) {
    float loss = 0.0f;
    int batch = net->batch_size;
    int num_classes = net->output_dim;
    float* probs = (float*)malloc(num_classes * sizeof(float));
    
    for (int i = 0; i < batch; i++) {
        float* logits = net->predictions + i * num_classes;
        softmax(logits, probs, num_classes);
        
        int target = target_labels[i];
        // Avoid log(0) by adding epsilon.
        loss -= logf(probs[target] + net->epsilon);
        
        // Compute gradient: (softmax - one_hot).
        for (int j = 0; j < num_classes; j++) {
            net->error[i * num_classes + j] = probs[j] - ((j == target) ? 1.0f : 0.0f);
        }
    }
    
    free(probs);
    return loss / batch;
}

/////////////////////////////////////////////////////////
// Zero the gradients before backpropagation.
/////////////////////////////////////////////////////////
void zero_gradients(Net* net) {
    memset(net->fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float));
    memset(net->fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float));
}

/////////////////////////////////////////////////////////
// Backward pass of the network.
/////////////////////////////////////////////////////////
void backward_pass(Net* net, float* X) {
    // Backprop for fc2 weights: fc2_weight_grad = (layer1_output)^T * (error).
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->hidden_dim,
                net->output_dim,
                net->batch_size,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->error,
                net->output_dim,
                0.0f,
                net->fc2_weight_grad,
                net->output_dim);
    
    // Backpropagate error to hidden layer: error_hidden = error * (fc2_weight)^T.
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                net->batch_size,
                net->hidden_dim,
                net->output_dim,
                1.0f,
                net->error,
                net->output_dim,
                net->fc2_weight,
                net->output_dim,
                0.0f,
                net->error_hidden,
                net->hidden_dim);
    
    // Apply derivative of the Swish activation.
    // Derivative: sigmoid(Z) + Z * sigmoid(Z) * (1 - sigmoid(Z)).
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float z = net->pre_activation[i];
        float sigmoid = 1.0f / (1.0f + expf(-z));
        net->error_hidden[i] *= sigmoid + z * sigmoid * (1.0f - sigmoid);
    }
    
    // Backprop for fc1 weights: fc1_weight_grad = (X)^T * (error_hidden).
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->input_dim,
                net->hidden_dim,
                net->batch_size,
                1.0f,
                X,
                net->input_dim,
                net->error_hidden,
                net->hidden_dim,
                0.0f,
                net->fc1_weight_grad,
                net->hidden_dim);
}

/////////////////////////////////////////////////////////
// Update weights using the AdamW optimizer.
/////////////////////////////////////////////////////////
void update_weights(Net* net, float learning_rate) {
    net->t++;  // Increment time step.
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update fc1 weights.
    for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
        float grad = net->fc1_weight_grad[i] / net->batch_size;
        net->fc1_m[i] = net->beta1 * net->fc1_m[i] + (1.0f - net->beta1) * grad;
        net->fc1_v[i] = net->beta2 * net->fc1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->fc1_m[i] / (sqrtf(net->fc1_v[i]) + net->epsilon);
        net->fc1_weight[i] = net->fc1_weight[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    
    // Update fc2 weights.
    for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
        float grad = net->fc2_weight_grad[i] / net->batch_size;
        net->fc2_m[i] = net->beta1 * net->fc2_m[i] + (1.0f - net->beta1) * grad;
        net->fc2_v[i] = net->beta2 * net->fc2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->fc2_m[i] / (sqrtf(net->fc2_v[i]) + net->epsilon);
        net->fc2_weight[i] = net->fc2_weight[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
}

/////////////////////////////////////////////////////////
// Save model to a binary file.
/////////////////////////////////////////////////////////
void save_model(Net* net, const char* filename) {
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
    fwrite(net->fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_weight, sizeof(float), net->output_dim * net->hidden_dim, file);
    
    // Save Adam state.
    fwrite(&net->t, sizeof(int), 1, file);
    fwrite(net->fc1_m, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc1_v, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_m, sizeof(float), net->output_dim * net->hidden_dim, file);
    fwrite(net->fc2_v, sizeof(float), net->output_dim * net->hidden_dim, file);
    
    // Save embedding table.
    size_t emb_table_size = (size_t)net->raw_input_dim * net->num_bins * net->embedding_dim;
    fwrite(net->embedding_table, sizeof(float), emb_table_size, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

/////////////////////////////////////////////////////////
// Load model from a binary file.
/////////////////////////////////////////////////////////
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
    
    // Initialize the network.
    Net* net = init_net(raw_input_dim, num_bins, embedding_dim, hidden_dim, output_dim, batch_size);
    
    // Load fc1 and fc2 weights.
    fread(net->fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Load Adam state.
    fread(&net->t, sizeof(int), 1, file);
    fread(net->fc1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_m, sizeof(float), output_dim * hidden_dim, file);
    fread(net->fc2_v, sizeof(float), output_dim * hidden_dim, file);
    
    // Load embedding table.
    size_t emb_table_size = (size_t)raw_input_dim * num_bins * embedding_dim;
    fread(net->embedding_table, sizeof(float), emb_table_size, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

#endif