#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients
    float* fc1_weight;     // hidden_dim x input_dim
    float* fc2_weight;     // output_dim x hidden_dim
    float* fc1_weight_grad; // hidden_dim x input_dim
    float* fc2_weight_grad; // output_dim x hidden_dim
    
    // Adam parameters
    float* fc1_m;  // First moment for fc1
    float* fc1_v;  // Second moment for fc1
    float* fc2_m;  // First moment for fc2
    float* fc2_v;  // Second moment for fc2
    float beta1;   // Exponential decay rate for first moment
    float beta2;   // Exponential decay rate for second moment
    float epsilon; // Small constant for numerical stability
    int t;         // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Helper arrays for forward/backward pass
    float* layer1_output;   // batch_size x hidden_dim
    float* predictions;     // batch_size x output_dim
    float* error;          // batch_size x output_dim
    float* pre_activation; // batch_size x hidden_dim
    float* error_hidden;   // batch_size x hidden_dim
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

// Initialize the network with configurable dimensions
Net* init_net(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Initialize Adam parameters
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Allocate and initialize weights and gradients
    net->fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate Adam buffers
    net->fc1_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    net->fc1_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    net->fc2_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    net->fc2_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    
    // Allocate helper arrays
    net->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->pre_activation = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    // Initialize weights
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        net->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    return net;
}

// Free network memory
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
    free(net);
}

// Forward pass
void forward_pass(Net* net, float* X) {
    // Z = XW₁
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
    
    // Store Z for backward pass
    memcpy(net->pre_activation, net->layer1_output, 
           net->batch_size * net->hidden_dim * sizeof(float));
    
    // A = Zσ(Z)
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->layer1_output[i] = net->layer1_output[i] / (1.0f + expf(-net->layer1_output[i]));
    }
    
    // Y = AW₂
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

// Calculate loss
float calculate_loss(Net* net, float* y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;
    for (int i = 0; i < net->batch_size * net->output_dim; i++) {
        net->error[i] = net->predictions[i] - y[i];
        loss += net->error[i] * net->error[i];
    }
    return loss / (net->batch_size * net->output_dim);
}

// Zero gradients
void zero_gradients(Net* net) {
    memset(net->fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float));
    memset(net->fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass(Net* net, float* X) {
    // ∂L/∂W₂ = Aᵀ(∂L/∂Y)
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
    
    // ∂L/∂A = (∂L/∂Y)(W₂)ᵀ
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
    
    // ∂L/∂Z = ∂L/∂A ⊙ [σ(Z) + Zσ(Z)(1-σ(Z))]
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-net->pre_activation[i]));
        net->error_hidden[i] *= sigmoid + net->pre_activation[i] * sigmoid * (1.0f - sigmoid);
    }
    
    // ∂L/∂W₁ = Xᵀ(∂L/∂Z)
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

// Update weights using AdamW
void update_weights(Net* net, float learning_rate) {
    net->t++;  // Increment time step
    
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update fc1 weights
    for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
        float grad = net->fc1_weight_grad[i] / net->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        net->fc1_m[i] = net->beta1 * net->fc1_m[i] + (1.0f - net->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        net->fc1_v[i] = net->beta2 * net->fc1_v[i] + (1.0f - net->beta2) * grad * grad;
        
        float update = alpha_t * net->fc1_m[i] / (sqrtf(net->fc1_v[i]) + net->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        net->fc1_weight[i] = net->fc1_weight[i] * (1.0f - learning_rate * net->weight_decay) - update;
    }
    
    // Update fc2 weights
    for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
        float grad = net->fc2_weight_grad[i] / net->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        net->fc2_m[i] = net->beta1 * net->fc2_m[i] + (1.0f - net->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        net->fc2_v[i] = net->beta2 * net->fc2_v[i] + (1.0f - net->beta2) * grad * grad;
        
        float update = alpha_t * net->fc2_m[i] / (sqrtf(net->fc2_v[i]) + net->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        net->fc2_weight[i] = net->fc2_weight[i] * (1.0f - learning_rate * net->weight_decay) - update;
    }
}

// Function to save model weights to binary file
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    
    // Save weights
    fwrite(net->fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_weight, sizeof(float), net->output_dim * net->hidden_dim, file);
    
    // Save Adam state
    fwrite(&net->t, sizeof(int), 1, file);
    fwrite(net->fc1_m, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc1_v, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_m, sizeof(float), net->output_dim * net->hidden_dim, file);
    fwrite(net->fc2_v, sizeof(float), net->output_dim * net->hidden_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to load model weights from binary file
Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize network
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(net->fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Load Adam state
    fread(&net->t, sizeof(int), 1, file);
    fread(net->fc1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_m, sizeof(float), output_dim * hidden_dim, file);
    fread(net->fc2_v, sizeof(float), output_dim * hidden_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif