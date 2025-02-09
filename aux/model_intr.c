#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

// Helper function to find minimum of two integers
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// Helper function to find maximum of two floats
static inline float max(float a, float b) {
    return (a > b) ? a : b;
}

// Helper function to safely load 8 floats, handling edge cases
static inline __m256 safe_load_8_floats(const float* ptr, int valid_elements) {
    if (valid_elements >= 8) {
        return _mm256_loadu_ps(ptr);
    } else {
        float temp[8] = {0};  // Zero-initialized buffer
        for (int i = 0; i < valid_elements; i++) {
            temp[i] = ptr[i];
        }
        return _mm256_loadu_ps(temp);
    }
}

// Helper function to safely store 8 floats, handling edge cases
static inline void safe_store_8_floats(float* ptr, __m256 vector, int valid_elements) {
    if (valid_elements >= 8) {
        _mm256_storeu_ps(ptr, vector);
    } else {
        float temp[8];
        _mm256_storeu_ps(temp, vector);
        for (int i = 0; i < valid_elements; i++) {
            ptr[i] = temp[i];
        }
    }
}

// Matrix multiplication C = A * B using AVX instructions
void matrix_multiply(float* A, float* B, float* C, int m, int n, int k) {
    // Initialize output matrix to zero
    memset(C, 0, m * k * sizeof(float));
    
    // Process each row of A
    for (int i = 0; i < m; i++) {
        // Process output elements in groups of 8 (AVX vector size)
        for (int j = 0; j < k; j += 8) {
            __m256 sum = _mm256_setzero_ps();  // Initialize accumulator
            int remaining_elements = k - j;     // Handle edge cases
            
            // Multiply and accumulate for each element in the row of A
            for (int l = 0; l < n; l++) {
                // Broadcast single value from A to all elements of vector
                __m256 a_vector = _mm256_set1_ps(A[i * n + l]);
                
                // Load 8 elements from B (or fewer at the edge)
                __m256 b_vector = safe_load_8_floats(&B[l * k + j], remaining_elements);
                
                // Multiply and add
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vector, b_vector));
            }
            
            // Store the results
            safe_store_8_floats(&C[i * k + j], sum, remaining_elements);
        }
    }
}

// Matrix multiplication with transpose options using AVX
void matrix_transpose_multiply(float* A, float* B, float* C, 
                             int m, int n, int k, 
                             int transpose_A_or_B) {
    if (transpose_A_or_B == 0) {
        matrix_multiply(A, B, C, m, n, k);
        return;
    }
    
    // Initialize output matrix to zero
    memset(C, 0, m * k * sizeof(float));
    
    if (transpose_A_or_B == 1) {  // C = A^T * B
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                int remaining_elements = k - j;
                
                // Main computation loop
                for (int l = 0; l < n; l++) {
                    __m256 a_vector = _mm256_set1_ps(A[l * m + i]);  // Note transposed access
                    __m256 b_vector = safe_load_8_floats(&B[l * k + j], remaining_elements);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vector, b_vector));
                }
                
                safe_store_8_floats(&C[i * k + j], sum, remaining_elements);
            }
        }
    }
    else if (transpose_A_or_B == 2) {  // C = A * B^T
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                int remaining_elements = k - j;
                
                // Main computation loop
                for (int l = 0; l < n; l++) {
                    __m256 a_vector = _mm256_set1_ps(A[i * n + l]);
                    
                    // Load from transposed B matrix
                    float temp[8] = {0};
                    for (int x = 0; x < min(8, remaining_elements); x++) {
                        temp[x] = B[(j + x) * n + l];
                    }
                    __m256 b_vector = _mm256_loadu_ps(temp);
                    
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vector, b_vector));
                }
                
                safe_store_8_floats(&C[i * k + j], sum, remaining_elements);
            }
        }
    }
}

// Optimized ReLU using AVX
void relu(float* x, int size) {
    __m256 zeros = _mm256_setzero_ps();
    
    // Process elements in groups of 8
    int vector_size = 8;
    int vector_count = size / vector_size;
    
    // Main vectorized loop
    for (int i = 0; i < vector_count * vector_size; i += vector_size) {
        __m256 values = _mm256_loadu_ps(&x[i]);
        __m256 result = _mm256_max_ps(values, zeros);
        _mm256_storeu_ps(&x[i], result);
    }
    
    // Handle remaining elements
    for (int i = vector_count * vector_size; i < size; i++) {
        x[i] = max(0.0f, x[i]);
    }
}

// Neural network structure
typedef struct {
    float* fc1_weight;     // 512 x 15
    float* fc2_weight;     // 4 x 512
    float* fc1_weight_grad; // 512 x 15
    float* fc2_weight_grad; // 4 x 512
} Net;

// Initialize the network
Net* init_net() {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Allocate and initialize weights and gradients
    net->fc1_weight = (float*)malloc(512 * 15 * sizeof(float));
    net->fc2_weight = (float*)malloc(4 * 512 * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(512 * 15 * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(4 * 512 * sizeof(float));
    
    // Initialize weights
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
        memset(net->fc1_weight_grad, 0, 512 * 15 * sizeof(float));
        memset(net->fc2_weight_grad, 0, 4 * 512 * sizeof(float));
        
        // Gradient of second layer
        matrix_transpose_multiply(layer1_output, error, net->fc2_weight_grad, 
                                512, num_samples, 4, 1);
        
        // Backpropagate error through second layer
        matrix_transpose_multiply(error, net->fc2_weight, error_hidden, 
                                num_samples, 4, 512, 2);
        
        // Apply ReLU gradient
        for (int i = 0; i < num_samples * 512; i++) {
            error_hidden[i] *= (d_relu[i] > 0) ? 1.0f : 0.0f;
        }
        
        // Gradient of first layer
        matrix_transpose_multiply(X, error_hidden, net->fc1_weight_grad, 
                                15, num_samples, 512, 1);
        
        // Update weights (SGD step)
        for (int i = 0; i < 512 * 15; i++) {
            net->fc1_weight[i] -= learning_rate * net->fc1_weight_grad[i] / num_samples;
        }
        for (int i = 0; i < 4 * 512; i++) {
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
    free(d_relu);
    free(error_hidden);
    free(X);
    free(y);
    free_net(net);
    
    return 0;
}