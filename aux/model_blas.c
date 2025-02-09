// clang -O3 -march=native model_blas.c -lopenblas -static -lm -flto -o model.out && time ./model.out

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cblas.h>

// Neural network structure
typedef struct {
    float* fc1_weight;     // 4096 x 15
    float* fc2_weight;     // 4 x 4096
    float* fc1_weight_grad; // 4096 x 15
    float* fc2_weight_grad; // 4 x 4096
} Net;

// Initialize the network
Net* init_net() {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Allocate and initialize weights and gradients
    net->fc1_weight = (float*)malloc(4096 * 15 * sizeof(float));
    net->fc2_weight = (float*)malloc(4 * 4096 * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(4096 * 15 * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(4 * 4096 * sizeof(float));
    
    // Initialize weights
    float scale1 = 1.0f / sqrt(15);
    float scale2 = 1.0f / sqrt(4096);
    
    for (int i = 0; i < 4096 * 15; i++) {
        net->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < 4 * 4096; i++) {
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
    
    // Initialize network
    Net* net = init_net();
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Allocate memory for intermediate values
    float* layer1_output = (float*)malloc(num_samples * 4096 * sizeof(float));
    float* predictions = (float*)malloc(num_samples * 4 * sizeof(float));
    float* error = (float*)malloc(num_samples * 4 * sizeof(float));
    float* pre_activation = (float*)malloc(num_samples * 4096 * sizeof(float));
    float* error_hidden = (float*)malloc(num_samples * 4096 * sizeof(float));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        // First layer
        cblas_sgemm(CblasRowMajor,    // Matrix layout
                    CblasNoTrans,      // A not transposed
                    CblasNoTrans,      // B not transposed
                    num_samples,        // M: rows of A
                    4096,               // N: columns of B
                    15,                // K: columns of A/rows of B
                    1.0f,              // alpha scaling factor
                    X,                 // matrix A
                    15,                // leading dimension of A
                    net->fc1_weight,   // matrix B
                    4096,               // leading dimension of B
                    0.0f,              // beta scaling factor
                    layer1_output,     // matrix C (output)
                    4096);              // leading dimension of C
        
        // Store pre-activation values for backward pass
        memcpy(pre_activation, layer1_output, num_samples * 4096 * sizeof(float));
        
        // Forward pass with Swish
        for (int i = 0; i < num_samples * 4096; i++) {
            layer1_output[i] = layer1_output[i] / (1.0f + expf(-layer1_output[i]));
        }
        
        // Second layer
        cblas_sgemm(CblasRowMajor,    // Matrix layout
                    CblasNoTrans,      // A not transposed
                    CblasNoTrans,      // B not transposed
                    num_samples,        // M: rows of A
                    4,                 // N: columns of B
                    4096,               // K: columns of A/rows of B
                    1.0f,              // alpha scaling factor
                    layer1_output,     // matrix A
                    4096,               // leading dimension of A
                    net->fc2_weight,   // matrix B
                    4,                 // leading dimension of B
                    0.0f,              // beta scaling factor
                    predictions,       // matrix C (output)
                    4);               // leading dimension of C
        
        // Calculate MSE loss and error
        float loss = 0.0f;
        for (int i = 0; i < num_samples * 4; i++) {
            error[i] = predictions[i] - y[i];
            loss += error[i] * error[i];
        }
        loss /= (num_samples * 4);
        
        // Backward pass
        // Clear gradients
        memset(net->fc1_weight_grad, 0, 4096 * 15 * sizeof(float));
        memset(net->fc2_weight_grad, 0, 4 * 4096 * sizeof(float));
        
        // Gradient of second layer
        cblas_sgemm(CblasRowMajor,    // Matrix layout
                    CblasTrans,        // A transposed
                    CblasNoTrans,      // B not transposed
                    4096,               // M: rows of result
                    4,                 // N: columns of result
                    num_samples,        // K: columns of A/rows of B
                    1.0f,              // alpha scaling factor
                    layer1_output,     // matrix A
                    4096,               // leading dimension of A
                    error,            // matrix B
                    4,                // leading dimension of B
                    0.0f,              // beta scaling factor
                    net->fc2_weight_grad, // matrix C (output)
                    4);               // leading dimension of C
        
        // Backpropagate error through second layer
        cblas_sgemm(CblasRowMajor,    // Matrix layout
                    CblasNoTrans,      // A not transposed
                    CblasTrans,        // B transposed
                    num_samples,        // M: rows of result
                    4096,               // N: columns of result
                    4,                 // K: columns of A/rows of B
                    1.0f,              // alpha scaling factor
                    error,            // matrix A
                    4,                // leading dimension of A
                    net->fc2_weight,   // matrix B
                    4,                // leading dimension of B
                    0.0f,              // beta scaling factor
                    error_hidden,      // matrix C (output)
                    4096);             // leading dimension of C
        
        // Swish derivative
        for (int i = 0; i < num_samples * 4096; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-pre_activation[i]));
            error_hidden[i] *= sigmoid + pre_activation[i] * sigmoid * (1.0f - sigmoid);
        }
        
        // Gradient of first layer
        cblas_sgemm(CblasRowMajor,    // Matrix layout
                    CblasTrans,        // A transposed
                    CblasNoTrans,      // B not transposed
                    15,                // M: rows of result
                    4096,               // N: columns of result
                    num_samples,        // K: columns of A/rows of B
                    1.0f,              // alpha scaling factor
                    X,                 // matrix A
                    15,                // leading dimension of A
                    error_hidden,      // matrix B
                    4096,               // leading dimension of B
                    0.0f,              // beta scaling factor
                    net->fc1_weight_grad, // matrix C (output)
                    4096);              // leading dimension of C
        
        // Update weights (SGD step)
        for (int i = 0; i < 4096 * 15; i++) {
            net->fc1_weight[i] -= learning_rate * net->fc1_weight_grad[i] / num_samples;
        }
        for (int i = 0; i < 4 * 4096; i++) {
            net->fc2_weight[i] -= learning_rate * net->fc2_weight_grad[i] / num_samples;
        }
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }
    
    // Cleanup
    free(layer1_output);
    free(predictions);
    free(error);
    free(pre_activation);
    free(error_hidden);
    free(X);
    free(y);
    free_net(net);
    
    return 0;
}