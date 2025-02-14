#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  This header defines a multilayer perceptron (MLP) with an embedding lookup.
  Each raw input feature (assumed to be a float) is first binned into one of a fixed number
  of bins and then an embedding vector is selected from an embedding table. The chosen embeddings
  (one per raw input feature) are concatenated into one 1D vector which serves as the effective input
  to the MLP.
*/

typedef struct {
    // Embedding parameters.
    int raw_input_dim;    // Number of raw (continuous) input features.
    int num_bins;         // Number of bins to discretize each feature.
    int embedding_dim;    // Dimension of the embedding vector for each feature.
    // The embedding table is stored as a contiguous array of size:
    // raw_input_dim * num_bins * embedding_dim.
    // For feature j, the embeddings start at:
    //   embedding_table + j * (num_bins * embedding_dim)
    float* embedding_table;
    
    // MLP weights and gradients.
    // Note: the effective input dimension equals raw_input_dim * embedding_dim.
    float* fc1_weight;      // Dimensions: hidden_dim x (raw_input_dim * embedding_dim)
    float* fc2_weight;      // Dimensions: output_dim x hidden_dim
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
    float* predictions;    // Output predictions (batch_size x output_dim)
    float* error;          // Error at output (batch_size x output_dim)
    float* pre_activation; // Pre-activation values of hidden layer (batch_size x hidden_dim)
    float* error_hidden;   // Error backpropagated into hidden layer (batch_size x hidden_dim)
    
    // Dimensions.
    int input_dim;   // Effective input dimension = raw_input_dim * embedding_dim.
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

/////////////////////////////////////////////////////////
// Initialize the network with embeddings.
// raw_input_dim: number of raw features (before embedding).
// num_bins: number of bins per feature.
// embedding_dim: dimension of the embedding vector for each feature.
// hidden_dim: number of neurons in the hidden layer.
// output_dim: number of output neurons.
// batch_size: number of samples in one batch.
/////////////////////////////////////////////////////////
Net* init_net(int raw_input_dim, int num_bins, int embedding_dim,
              int hidden_dim, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));

    // Store embedding parameters.
    net->raw_input_dim = raw_input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    // Compute the effective input dimension.
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
    float scale1 = 1.0f / sqrt((float)net->input_dim);
    for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
        net->fc1_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale1;
    }

    // Initialize fc2 weights.
    float scale2 = 1.0f / sqrt((float)net->hidden_dim);
    for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
        net->fc2_weight[i] = (((float)rand() / (float)RAND_MAX) * 2 - 1) * scale2;
    }

    // Allocate and initialize the embedding table.
    // The table size is: raw_input_dim * num_bins * embedding_dim.
    net->embedding_table = (float*)malloc(raw_input_dim * num_bins * embedding_dim * sizeof(float));
    float emb_scale = 1.0f / sqrt((float)num_bins);
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
            // Get the raw value for sample i and feature j.
            float val = raw_input[i * net->raw_input_dim + j];
            // Map the raw value to a bin index.
            int bin_idx = (int)(((val - min_val) / range) * net->num_bins);
            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= net->num_bins) bin_idx = net->num_bins - 1;
            
            // Get the embedding vector from the embedding table.
            // For feature j, the embeddings start at:
            //   embedding_table + j * (num_bins * embedding_dim)
            // Then offset by bin_idx * embedding_dim.
            float* emb = net->embedding_table + j * (net->num_bins * net->embedding_dim)
                                     + bin_idx * net->embedding_dim;
            // Destination in the embedded output:
            // Each sample gets a concatenated vector of size raw_input_dim * embedding_dim.
            float* dest = embedded_output + i * (net->raw_input_dim * net->embedding_dim)
                                      + j * net->embedding_dim;
            memcpy(dest, emb, net->embedding_dim * sizeof(float));
        }
    }
}

/////////////////////////////////////////////////////////
// Forward pass of the MLP (using an already-embedded input).
// Expects X to be of shape (batch_size x (raw_input_dim * embedding_dim)).
/////////////////////////////////////////////////////////
void forward_pass(Net* net, float* X) {
    // Compute Z = X * fc1_weight.
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
    
    // Apply the Swish activation: A = Z * sigmoid(Z).
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->layer1_output[i] = net->layer1_output[i] / (1.0f + expf(-net->layer1_output[i]));
    }
    
    // Compute predictions: Y = A * fc2_weight.
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
// Calculate loss (mean squared error) and compute gradient at output.
// y: target output (shape: batch_size x output_dim)
/////////////////////////////////////////////////////////
float calculate_loss(Net* net, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < net->batch_size * net->output_dim; i++) {
        net->error[i] = net->predictions[i] - y[i];
        loss += net->error[i] * net->error[i];
    }
    return loss / (net->batch_size * net->output_dim);
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
    // Backprop for fc2 weights: fc2_weight_grad = (A)^T * (dL/dY).
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
    
    // Backpropagate error to hidden layer: error_hidden = (dL/dY) * (fc2_weight)^T.
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
    // The derivative is: sigmoid(Z) + Z * sigmoid(Z) * (1-sigmoid(Z)).
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-net->pre_activation[i]));
        net->error_hidden[i] *= sigmoid + net->pre_activation[i] * sigmoid * (1.0f - sigmoid);
    }
    
    // Backprop for fc1 weights: fc1_weight_grad = (X)^T * error_hidden.
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
    
    // Initialize the network (which allocates memory for the embedding table).
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