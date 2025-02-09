#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cblas.h>

// Neural network structure
typedef struct {
    // Weights and gradients
    float* fc1_weight;     // hidden_dim x input_dim
    float* fc2_weight;     // output_dim x hidden_dim
    float* fc1_weight_grad; // hidden_dim x input_dim
    float* fc2_weight_grad; // output_dim x hidden_dim
    
    // Helper arrays for forward/backward pass
    float* layer1_output;   // num_samples x hidden_dim
    float* predictions;     // num_samples x output_dim
    float* error;          // num_samples x output_dim
    float* pre_activation; // num_samples x hidden_dim
    float* error_hidden;   // num_samples x hidden_dim
    
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
    
    // Allocate and initialize weights and gradients
    net->fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
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
    free(net->layer1_output);
    free(net->predictions);
    free(net->error);
    free(net->pre_activation);
    free(net->error_hidden);
    free(net);
}

// Load CSV data
void load_csv(const char* filename, float** X, float** y, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    
    // Skip header
    char buffer[1024];
    fgets(buffer, sizeof(buffer), file);
    
    // Count lines
    int count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }
    *num_samples = count;
    
    // Allocate memory
    *X = (float*)malloc(count * 15 * sizeof(float));
    *y = (float*)malloc(count * 4 * sizeof(float));
    
    // Reset file pointer and skip header again
    fseek(file, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), file);
    
    // Read data
    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        char* token = strtok(buffer, ",");
        for (int i = 0; i < 15; i++) {
            (*X)[idx * 15 + i] = atof(token);
            token = strtok(NULL, ",");
        }
        for (int i = 0; i < 4; i++) {
            (*y)[idx * 4 + i] = atof(token);
            token = strtok(NULL, ",");
        }
        idx++;
    }
    
    fclose(file);
}

int main() {
    openblas_set_num_threads(4);
    // Load data
    float *X, *y;
    int num_samples;
    load_csv("20250208_163908_data.csv", &X, &y, &num_samples);
    
    // Initialize network with configurable dimensions
    const int input_dim = 15;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    Net* net = init_net(input_dim, hidden_dim, output_dim, num_samples);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        // First layer
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
        
        // Store pre-activation values
        memcpy(net->pre_activation, net->layer1_output, 
               net->batch_size * net->hidden_dim * sizeof(float));
        
        // Swish activation
        for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
            net->layer1_output[i] = net->layer1_output[i] / (1.0f + expf(-net->layer1_output[i]));
        }
        
        // Second layer
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
        
        // Calculate MSE loss and error
        float loss = 0.0f;
        for (int i = 0; i < net->batch_size * net->output_dim; i++) {
            net->error[i] = net->predictions[i] - y[i];
            loss += net->error[i] * net->error[i];
        }
        loss /= (net->batch_size * net->output_dim);
        
        // Backward pass
        // Clear gradients
        memset(net->fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float));
        memset(net->fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float));
        
        // Gradient of second layer
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
        
        // Backpropagate error through second layer
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
        
        // Swish derivative
        for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-net->pre_activation[i]));
            net->error_hidden[i] *= sigmoid + net->pre_activation[i] * sigmoid * (1.0f - sigmoid);
        }
        
        // Gradient of first layer
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
        
        // Update weights (SGD step)
        for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
            net->fc1_weight[i] -= learning_rate * net->fc1_weight_grad[i] / net->batch_size;
        }
        for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
            net->fc2_weight[i] -= learning_rate * net->fc2_weight_grad[i] / net->batch_size;
        }
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }
    
    // Cleanup
    free(X);
    free(y);
    free_net(net);
    
    return 0;
}