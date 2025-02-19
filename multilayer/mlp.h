#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients for three layers:
    // fc1: maps input -> first hidden layer (dimensions: hidden_dim x input_dim)
    float* fc1_weight;      
    float* fc1_weight_grad; 

    // fc2: maps first hidden layer -> second hidden layer (dimensions: hidden_dim x hidden_dim)
    float* fc2_weight;
    float* fc2_weight_grad;

    // fc3: maps second hidden layer -> output (dimensions: output_dim x hidden_dim)
    float* fc3_weight;
    float* fc3_weight_grad;
    
    // Adam parameters for each layer
    float* fc1_m;
    float* fc1_v;
    float* fc2_m;
    float* fc2_v;
    float* fc3_m;
    float* fc3_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Helper arrays for the forward/backward pass
    // First hidden layer (from fc1)
    float* pre_activation1;  // (batch_size x hidden_dim), pre-activation values
    float* layer1_output;    // (batch_size x hidden_dim), activation output computed as: x * sigmoid(x)
    
    // Second hidden layer (from fc2 + residual connection)
    float* pre_activation2;  // (batch_size x hidden_dim), computed as (layer1_output * fc2_weight + layer1_output)
    float* layer2_output;    // (batch_size x hidden_dim), activation output of the second hidden layer
    
    // Output layer (from fc3)
    float* predictions;      // (batch_size x output_dim)
    float* error;            // (batch_size x output_dim), stores (predictions - y)
    
    // Backpropagation error buffers for hidden layers
    float* error_hidden1;    // (batch_size x hidden_dim), error propagated to first hidden layer
    float* error_hidden2;    // (batch_size x hidden_dim), error propagated to second hidden layer
    
    // Network dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

// Initialize the network with the given dimensions.
// This creates two hidden layers: the first (fc1) from input and the second (fc2) with a residual
// connection from the output of the first hidden layer. The final (fc3) maps to the output.
Net* init_net(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Initialize Adam hyperparameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Allocate weights and gradients
    // fc1 weights: dimensions hidden_dim x input_dim
    net->fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    
    // fc2 weights: dimensions hidden_dim x hidden_dim
    net->fc2_weight = (float*)malloc(hidden_dim * hidden_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(hidden_dim * hidden_dim * sizeof(float));
    
    // fc3 weights: dimensions output_dim x hidden_dim
    net->fc3_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    net->fc3_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate Adam optimizer buffers (initialized to zero)
    net->fc1_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    net->fc1_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    net->fc2_m = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    net->fc2_v = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    net->fc3_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    net->fc3_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    
    // Allocate helper arrays for forward/backward passes
    net->pre_activation1 = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->pre_activation2 = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->layer2_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->error_hidden1 = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->error_hidden2 = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    // Initialize weights with random values scaled by the input dimension
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    float scale3 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        net->fc1_weight[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale1;
    }
    
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        net->fc2_weight[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale2;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->fc3_weight[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale3;
    }
    
    return net;
}

// Free all allocated memory for the network
void free_net(Net* net) {
    free(net->fc1_weight);
    free(net->fc2_weight);
    free(net->fc3_weight);
    free(net->fc1_weight_grad);
    free(net->fc2_weight_grad);
    free(net->fc3_weight_grad);
    free(net->fc1_m);
    free(net->fc1_v);
    free(net->fc2_m);
    free(net->fc2_v);
    free(net->fc3_m);
    free(net->fc3_v);
    free(net->pre_activation1);
    free(net->layer1_output);
    free(net->pre_activation2);
    free(net->layer2_output);
    free(net->predictions);
    free(net->error);
    free(net->error_hidden1);
    free(net->error_hidden2);
    free(net);
}

// Forward pass through the network
void forward_pass(Net* net, float* X) {
    // First hidden layer (fc1): compute pre-activation = X * fc1_weight
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
                net->pre_activation1,
                net->hidden_dim);
                
    // Apply activation: A = Z * sigmoid(Z)
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float val = net->pre_activation1[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        net->layer1_output[i] = val * sigmoid;
    }
    
    // Second hidden layer (fc2): compute pre-activation = (layer1_output * fc2_weight)
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->hidden_dim,
                net->hidden_dim,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->fc2_weight,
                net->hidden_dim,
                0.0f,
                net->pre_activation2,
                net->hidden_dim);
                
    // Add the residual connection (first layer output) to the second layer pre-activation
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->pre_activation2[i] += net->layer1_output[i];
    }
    
    // Apply activation for second hidden layer
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float val = net->pre_activation2[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        net->layer2_output[i] = val * sigmoid;
    }
    
    // Output layer (fc3): compute predictions = layer2_output * fc3_weight
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->output_dim,
                net->hidden_dim,
                1.0f,
                net->layer2_output,
                net->hidden_dim,
                net->fc3_weight,
                net->output_dim,
                0.0f,
                net->predictions,
                net->output_dim);
}

// Calculate mean squared loss and compute error = predictions - y
float calculate_loss(Net* net, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < net->batch_size * net->output_dim; i++) {
        net->error[i] = net->predictions[i] - y[i];
        loss += net->error[i] * net->error[i];
    }
    return loss / (net->batch_size * net->output_dim);
}

// Zero out weight gradients
void zero_gradients(Net* net) {
    memset(net->fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float));
    memset(net->fc2_weight_grad, 0, net->hidden_dim * net->hidden_dim * sizeof(float));
    memset(net->fc3_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float));
}

// Backward pass through the network
void backward_pass(Net* net, float* X) {
    // 1. Output layer (fc3) gradients:
    // ∂L/∂W3 = (layer2_output)^T * (error)
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->hidden_dim,
                net->output_dim,
                net->batch_size,
                1.0f,
                net->layer2_output,
                net->hidden_dim,
                net->error,
                net->output_dim,
                0.0f,
                net->fc3_weight_grad,
                net->output_dim);
    
    // 2. Propagate error to second hidden layer:
    // error_hidden2 = (error) * (fc3_weight)^T
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                net->batch_size,
                net->hidden_dim,
                net->output_dim,
                1.0f,
                net->error,
                net->output_dim,
                net->fc3_weight,
                net->output_dim,
                0.0f,
                net->error_hidden2,
                net->hidden_dim);
    
    // 3. Backprop through activation of second hidden layer:
    // delta2 = error_hidden2 ⊙ f'(pre_activation2)
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float val = net->pre_activation2[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        float deriv = sigmoid + val * sigmoid * (1.0f - sigmoid);
        net->error_hidden2[i] *= deriv;
    }
    
    // 4. fc2 gradients:
    // ∂L/∂W2 = (layer1_output)^T * (error_hidden2)
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->hidden_dim,
                net->hidden_dim,
                net->batch_size,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->error_hidden2,
                net->hidden_dim,
                0.0f,
                net->fc2_weight_grad,
                net->hidden_dim);
    
    // 5. Propagate error to first hidden layer.
    // For the second layer with residual connection:
    // pre_activation2 = (layer1_output * fc2_weight) + layer1_output
    // Hence, dL/d(layer1_output) = (error_hidden2 * (fc2_weight)^T) + error_hidden2.
    float* temp_error = (float*)malloc(net->batch_size * net->hidden_dim * sizeof(float));
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                net->batch_size,
                net->hidden_dim,
                net->hidden_dim,
                1.0f,
                net->error_hidden2,
                net->hidden_dim,
                net->fc2_weight,
                net->hidden_dim,
                0.0f,
                temp_error,
                net->hidden_dim);
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->error_hidden1[i] = temp_error[i] + net->error_hidden2[i];
    }
    free(temp_error);
    
    // 6. Backprop through activation of first hidden layer:
    // delta1 = error_hidden1 ⊙ f'(pre_activation1)
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float val = net->pre_activation1[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        float deriv = sigmoid + val * sigmoid * (1.0f - sigmoid);
        net->error_hidden1[i] *= deriv;
    }
    
    // 7. fc1 gradients:
    // ∂L/∂W1 = X^T * (error_hidden1)
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->input_dim,
                net->hidden_dim,
                net->batch_size,
                1.0f,
                X,
                net->input_dim,
                net->error_hidden1,
                net->hidden_dim,
                0.0f,
                net->fc1_weight_grad,
                net->hidden_dim);
}

// Update weights using the AdamW optimizer
void update_weights(Net* net, float learning_rate) {
    net->t++;  // Increment time step
    
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int i;
    int size_fc1 = net->hidden_dim * net->input_dim;
    for (i = 0; i < size_fc1; i++) {
        float grad = net->fc1_weight_grad[i] / net->batch_size;
        net->fc1_m[i] = net->beta1 * net->fc1_m[i] + (1.0f - net->beta1) * grad;
        net->fc1_v[i] = net->beta2 * net->fc1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update = alpha_t * net->fc1_m[i] / (sqrtf(net->fc1_v[i]) + net->epsilon);
        net->fc1_weight[i] = net->fc1_weight[i] * (1.0f - learning_rate * net->weight_decay) - update;
    }
    
    int size_fc2 = net->hidden_dim * net->hidden_dim;
    for (i = 0; i < size_fc2; i++) {
        float grad = net->fc2_weight_grad[i] / net->batch_size;
        net->fc2_m[i] = net->beta1 * net->fc2_m[i] + (1.0f - net->beta1) * grad;
        net->fc2_v[i] = net->beta2 * net->fc2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update = alpha_t * net->fc2_m[i] / (sqrtf(net->fc2_v[i]) + net->epsilon);
        net->fc2_weight[i] = net->fc2_weight[i] * (1.0f - learning_rate * net->weight_decay) - update;
    }
    
    int size_fc3 = net->output_dim * net->hidden_dim;
    for (i = 0; i < size_fc3; i++) {
        float grad = net->fc3_weight_grad[i] / net->batch_size;
        net->fc3_m[i] = net->beta1 * net->fc3_m[i] + (1.0f - net->beta1) * grad;
        net->fc3_v[i] = net->beta2 * net->fc3_v[i] + (1.0f - net->beta2) * grad * grad;
        float update = alpha_t * net->fc3_m[i] / (sqrtf(net->fc3_v[i]) + net->epsilon);
        net->fc3_weight[i] = net->fc3_weight[i] * (1.0f - learning_rate * net->weight_decay) - update;
    }
}

// Save the model parameters (weights and Adam state) to a binary file.
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save network dimensions
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    
    // Save weights: fc1, fc2, and fc3
    fwrite(net->fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_weight, sizeof(float), net->hidden_dim * net->hidden_dim, file);
    fwrite(net->fc3_weight, sizeof(float), net->output_dim * net->hidden_dim, file);
    
    // Save Adam state: time step and optimizer buffers
    fwrite(&net->t, sizeof(int), 1, file);
    fwrite(net->fc1_m, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc1_v, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_m, sizeof(float), net->hidden_dim * net->hidden_dim, file);
    fwrite(net->fc2_v, sizeof(float), net->hidden_dim * net->hidden_dim, file);
    fwrite(net->fc3_m, sizeof(float), net->output_dim * net->hidden_dim, file);
    fwrite(net->fc3_v, sizeof(float), net->output_dim * net->hidden_dim, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load the model parameters from a binary file.
Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(net->fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_weight, sizeof(float), hidden_dim * hidden_dim, file);
    fread(net->fc3_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Load Adam state
    fread(&net->t, sizeof(int), 1, file);
    fread(net->fc1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_m, sizeof(float), hidden_dim * hidden_dim, file);
    fread(net->fc2_v, sizeof(float), hidden_dim * hidden_dim, file);
    fread(net->fc3_m, sizeof(float), output_dim * hidden_dim, file);
    fread(net->fc3_v, sizeof(float), output_dim * hidden_dim, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif