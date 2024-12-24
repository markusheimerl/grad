#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Matrix operations
void matrix_multiply(double* A, double* B, double* C, int m, int n, int p) {
    // A[m×n] × B[n×p] = C[m×p]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

void matrix_add(double* A, double* B, double* C, int m, int n) {
    // C[m×n] = A[m×n] + B[m×n]
    for (int i = 0; i < m*n; i++) {
        C[i] = A[i] + B[i];
    }
}

void matrix_scale(double* A, double scale, int size) {
    for (int i = 0; i < size; i++) {
        A[i] *= scale;
    }
}

// Element-wise operations
void apply_sigmoid(double* A, double* B, int size) {
    for (int i = 0; i < size; i++) {
        B[i] = 1.0 / (1.0 + exp(-A[i]));
    }
}

void apply_sigmoid_prime(double* A, double* B, int size) {
    for (int i = 0; i < size; i++) {
        double s = 1.0 / (1.0 + exp(-A[i]));
        B[i] = s * (1.0 - s);
    }
}

double random_range(double min, double max) {
    return (double)rand()/RAND_MAX * (max - min) + min;
}

int main() {
    srand(time(NULL));
    
    const int EPOCHS = 1000;
    const int BATCH_SIZE = 100;
    const double LR = 0.1;
    
    // Weight matrix: 1×3
    double weights[1*3] = {0.1, 0.1, 0};
    
    // Temporary matrices for computations
    double input_batch[100*3];    // BATCH_SIZE×3
    double labels[100*1];         // BATCH_SIZE×1
    double z_batch[100*1];        // BATCH_SIZE×1
    double pred_batch[100*1];     // BATCH_SIZE×1
    double error_batch[100*1];    // BATCH_SIZE×1
    double gradient[1*3] = {0};   // 1×3
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Generate batch
        for (int b = 0; b < BATCH_SIZE; b++) {
            double x = random_range(-1.5, 1.5);
            double y = random_range(-1.5, 1.5);
            input_batch[b*3 + 0] = x;
            input_batch[b*3 + 1] = y;
            input_batch[b*3 + 2] = 1.0;  // bias term
            labels[b] = (x*x + y*y) <= 1.0 ? 1.0 : 0.0;
        }
        
        // Forward pass
        matrix_multiply(weights, input_batch, z_batch, 1, 3, BATCH_SIZE);
        apply_sigmoid(z_batch, pred_batch, BATCH_SIZE);
        
        // Compute error
        double total_loss = 0;
        for (int b = 0; b < BATCH_SIZE; b++) {
            error_batch[b] = pred_batch[b] - labels[b];
            total_loss += error_batch[b] * error_batch[b];
        }
        
        // Backward pass
        double sigmoid_prime_vals[100*1];
        apply_sigmoid_prime(z_batch, sigmoid_prime_vals, BATCH_SIZE);
        
        // Compute gradient
        for (int i = 0; i < 3; i++) {
            gradient[i] = 0;
            for (int b = 0; b < BATCH_SIZE; b++) {
                gradient[i] += 2 * error_batch[b] * sigmoid_prime_vals[b] * input_batch[b*3 + i];
            }
        }
        
        // Update weights
        matrix_scale(gradient, LR/BATCH_SIZE, 3);
        for (int i = 0; i < 3; i++) {
            weights[i] -= gradient[i];
        }
        
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss/BATCH_SIZE);
        }
    }
    
    // Testing
    int correct = 0;
    const int TEST_SIZE = 20;
    double test_input[3];
    double test_output[1];
    
    printf("\nTesting %d points:\n", TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        double x = random_range(-1.5, 1.5);
        double y = random_range(-1.5, 1.5);
        test_input[0] = x;
        test_input[1] = y;
        test_input[2] = 1.0;
        
        int true_label = (x*x + y*y) <= 1.0;
        
        matrix_multiply(weights, test_input, test_output, 1, 3, 1);
        int pred_label = 1.0/(1.0 + exp(-test_output[0])) > 0.5;
        
        correct += (true_label == pred_label);
        printf("(%.2f, %.2f): %s, true=%s %s\n", 
               x, y, 
               pred_label ? "Inside" : "Outside",
               true_label ? "Inside" : "Outside",
               true_label == pred_label ? "✓" : "✗");
    }
    
    printf("\nAccuracy: %.1f%% (%d/%d)\n", 
           100.0 * correct / TEST_SIZE, correct, TEST_SIZE);
    
    return 0;
}