#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Matrix multiplication C = A * B
// A: m x n matrix
// B: n x k matrix
// C: m x k matrix
void matrix_multiply(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

// Matrix multiplication with transpose option
// transpose_A_or_B: 0 for normal, 1 for transpose A, 2 for transpose B
void matrix_transpose_multiply(float* A, float* B, float* C, 
                             int m, int n, int k, 
                             int transpose_A_or_B) {
    if (transpose_A_or_B == 0) {
        matrix_multiply(A, B, C, m, n, k);
    }
    else if (transpose_A_or_B == 1) {
        // C = A^T * B
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int l = 0; l < n; l++) {
                    sum += A[l * m + i] * B[l * k + j];
                }
                C[i * k + j] = sum;
            }
        }
    }
    else if (transpose_A_or_B == 2) {
        // C = A * B^T
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int l = 0; l < n; l++) {
                    sum += A[i * n + l] * B[j * n + l];
                }
                C[i * k + j] = sum;
            }
        }
    }
}

// ReLU activation function
void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

// Neural network structure
typedef struct {
    float* fc1_weight; // 512 x 15
    float* fc2_weight; // 4 x 512
    float* intermediate; // For storing intermediate results
} Net;

// Initialize the network
Net* init_net() {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Allocate and initialize weights
    net->fc1_weight = (float*)malloc(512 * 15 * sizeof(float));
    net->fc2_weight = (float*)malloc(4 * 512 * sizeof(float));
    net->intermediate = (float*)malloc(512 * sizeof(float));
    
    // Initialize weights (you'll need to load the trained weights)
    float scale1 = 1.0f / sqrt(15);
    float scale2 = 1.0f / sqrt(512);
    
    for (int i = 0; i < 512 * 15; i++) {
        net->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < 4 * 512; i++) {
        net->fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    free(net->fc1_weight);
    free(net->fc2_weight);
    free(net->intermediate);
    free(net);
}

// Forward pass through the network
void forward(Net* net, float* input, float* output, int batch_size) {
    // First layer: input -> fc1
    matrix_multiply(input, net->fc1_weight, net->intermediate, 
                   batch_size, 15, 512);
    
    // Apply ReLU
    relu(net->intermediate, batch_size * 512);
    
    // Second layer: fc1 -> fc2
    matrix_multiply(net->intermediate, net->fc2_weight, output, 
                   batch_size, 512, 4);
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

// Calculate R² score
float r2_score(float* y_true, float* y_pred, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += y_true[i];
    }
    mean /= size;
    
    float ss_res = 0.0f;
    float ss_tot = 0.0f;
    for (int i = 0; i < size; i++) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
    }
    
    return 1.0f - (ss_res / ss_tot);
}

int main() {
    // Load data
    float *X, *y;
    int num_samples;
    load_csv("20250208_163908_data.csv", &X, &y, &num_samples);
    
    // Initialize network
    Net* net = init_net();
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Allocate memory for gradients and intermediate values
    float* fc1_weight_grad = (float*)malloc(512 * 15 * sizeof(float));
    float* fc2_weight_grad = (float*)malloc(4 * 512 * sizeof(float));
    float* layer1_output = (float*)malloc(num_samples * 512 * sizeof(float));
    float* predictions = (float*)malloc(num_samples * 4 * sizeof(float));
    float* error = (float*)malloc(num_samples * 4 * sizeof(float));
    float* d_relu = (float*)malloc(num_samples * 512 * sizeof(float));
    float* error_hidden = (float*)malloc(num_samples * 512 * sizeof(float));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        // First layer
        matrix_multiply(X, net->fc1_weight, layer1_output, num_samples, 15, 512);
        
        // Store pre-activation values for backward pass
        memcpy(d_relu, layer1_output, num_samples * 512 * sizeof(float));
        
        // Apply ReLU
        relu(layer1_output, num_samples * 512);
        
        // Second layer
        matrix_multiply(layer1_output, net->fc2_weight, predictions, 
                       num_samples, 512, 4);
        
        // Calculate MSE loss and error
        float loss = 0.0f;
        for (int i = 0; i < num_samples * 4; i++) {
            error[i] = predictions[i] - y[i];
            loss += error[i] * error[i];
        }
        loss /= (num_samples * 4);
        
        // Backward pass
        // Clear gradients
        memset(fc1_weight_grad, 0, 512 * 15 * sizeof(float));
        memset(fc2_weight_grad, 0, 4 * 512 * sizeof(float));
        
        // Gradient of second layer
        matrix_transpose_multiply(layer1_output, error, fc2_weight_grad, 
                                512, num_samples, 4, 1);
        
        // Backpropagate error through second layer
        matrix_transpose_multiply(error, net->fc2_weight, error_hidden, 
                                num_samples, 4, 512, 2);
        
        // Apply ReLU gradient
        for (int i = 0; i < num_samples * 512; i++) {
            error_hidden[i] *= (d_relu[i] > 0) ? 1.0f : 0.0f;
        }
        
        // Gradient of first layer
        matrix_transpose_multiply(X, error_hidden, fc1_weight_grad, 
                                15, num_samples, 512, 1);
        
        // Update weights (SGD step)
        for (int i = 0; i < 512 * 15; i++) {
            net->fc1_weight[i] -= learning_rate * fc1_weight_grad[i] / num_samples;
        }
        for (int i = 0; i < 4 * 512; i++) {
            net->fc2_weight[i] -= learning_rate * fc2_weight_grad[i] / num_samples;
        }
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", 
                   epoch + 1, num_epochs, loss);
        }
    }
    
    // Evaluation
    float* eval_predictions = (float*)malloc(num_samples * 4 * sizeof(float));
    if (!eval_predictions) {
        printf("Failed to allocate memory for predictions\n");
        exit(1);
    }

    // Do forward pass
    float* temp_layer1 = (float*)malloc(num_samples * 512 * sizeof(float));
    if (!temp_layer1) {
        printf("Failed to allocate memory for temp_layer1\n");
        exit(1);
    }
    
    matrix_multiply(X, net->fc1_weight, temp_layer1, num_samples, 15, 512);
    relu(temp_layer1, num_samples * 512);
    matrix_multiply(temp_layer1, net->fc2_weight, eval_predictions, num_samples, 512, 4);
    
    // Calculate and print R² scores
    printf("\nR² scores:\n");
    for (int i = 0; i < 4; i++) {
        float* y_true_component = (float*)malloc(num_samples * sizeof(float));
        float* y_pred_component = (float*)malloc(num_samples * sizeof(float));
        
        if (!y_true_component || !y_pred_component) {
            printf("Failed to allocate memory for R² calculation\n");
            exit(1);
        }
        
        for (int j = 0; j < num_samples; j++) {
            y_true_component[j] = y[j * 4 + i];
            y_pred_component[j] = eval_predictions[j * 4 + i];
        }
        
        float r2 = r2_score(y_true_component, y_pred_component, num_samples);
        printf("Output y%d: %.8f\n", i, r2);
        
        printf("\nSample predictions for y%d (first 15 samples):\n", i);
        printf("Sample\tPredicted\tActual\t\tDifference\n");
        printf("------------------------------------------------\n");
        
        for (int j = 0; j < 15 && j < num_samples; j++) {
            printf("%d\t%.3f\t\t%.3f\t\t%.3f\n", 
                   j, 
                   y_pred_component[j], 
                   y_true_component[j], 
                   y_pred_component[j] - y_true_component[j]);
        }
        
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(y_pred_component[j] - y_true_component[j]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n\n", i, mae);
        
        free(y_true_component);
        free(y_pred_component);
    }
    
    // Clean up temporary arrays
    free(temp_layer1);
    free(eval_predictions);
    
    // Final cleanup
    free(fc1_weight_grad);
    free(fc2_weight_grad);
    free(layer1_output);
    free(predictions);
    free(error);
    free(d_relu);
    free(error_hidden);
    free(X);
    free(y);
    free_net(net);
    
    return 0;
}