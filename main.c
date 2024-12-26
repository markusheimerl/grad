#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

// Statistical functions
double compute_mean(const double* array, int size) {
    double sum = 0;
    for(int i = 0; i < size; i++) sum += array[i];
    return sum / size;
}

double compute_std(const double* array, int size, double mean) {
    double sum_squared_diff = 0;
    for(int i = 0; i < size; i++) {
        double diff = array[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrt(sum_squared_diff / size);
}

// Data preprocessing
void normalize_data(double (*data)[13], double* y, int rows, int cols) {
    for(int j = 0; j < cols; j++) {
        double mean = 0, std = 0;
        for(int i = 0; i < rows; i++) mean += data[i][j];
        mean /= rows;
        
        for(int i = 0; i < rows; i++) {
            double diff = data[i][j] - mean;
            std += diff * diff;
        }
        std = sqrt(std / rows);
        
        for(int i = 0; i < rows; i++) 
            data[i][j] = (data[i][j] - mean) / std;
    }
    
    // Normalize target variable
    double y_mean = compute_mean(y, rows);
    double y_std = compute_std(y, rows, y_mean);
    for(int i = 0; i < rows; i++) y[i] = (y[i] - y_mean) / y_std;
}

void split_data(double** X_train, double** X_test, double* y_train, double* y_test,
                double (*X)[13], double* y, int total_samples, int train_size, int features) {
    int* indices = malloc(total_samples * sizeof(int));
    for(int i = 0; i < total_samples; i++) indices[i] = i;
    
    // Fisher-Yates shuffle
    for(int i = total_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    for(int i = 0; i < train_size; i++) {
        for(int j = 0; j < features; j++) 
            X_train[i][j] = X[indices[i]][j];
        y_train[i] = y[indices[i]];
    }
    
    int test_size = total_samples - train_size;
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < features; j++)
            X_test[i][j] = X[indices[i + train_size]][j];
        y_test[i] = y[indices[i + train_size]];
    }
    
    free(indices);
}

// Model functions
double predict(const double* weights, double bias, const double* x, int features) {
    double prediction = bias;
    for(int i = 0; i < features; i++) 
        prediction += weights[i] * x[i];
    return prediction;
}

void train(double* weights, double* bias, double** X, double* y, 
          int n_samples, int features, double learning_rate, int epochs, int batch_size) {
    double* gradients = malloc(features * sizeof(double));
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
            int actual_batch_size = fmin(batch_size, n_samples - batch_start);
            
            // Reset gradients
            for(int j = 0; j < features; j++) gradients[j] = 0;
            double bias_gradient = 0;
            
            // Compute gradients for batch
            for(int i = batch_start; i < batch_start + actual_batch_size; i++) {
                double error = predict(weights, *bias, X[i], features) - y[i];
                for(int j = 0; j < features; j++)
                    gradients[j] += error * X[i][j];
                bias_gradient += error;
            }
            
            // Update parameters
            for(int j = 0; j < features; j++)
                weights[j] -= learning_rate * (gradients[j] / actual_batch_size);
            *bias -= learning_rate * (bias_gradient / actual_batch_size);
        }
    }
    
    free(gradients);
}

double compute_mse(const double* weights, double bias, double** X, const double* y, 
                  int n_samples, int features) {
    double mse = 0;
    for(int i = 0; i < n_samples; i++) {
        double error = predict(weights, bias, X[i], features) - y[i];
        mse += error * error;
    }
    return mse / n_samples;
}

int main() {
    srand(time(NULL));
    
    const int total_samples = 506;
    const int features = 13;
    const int train_size = 0.8 * total_samples;
    const int test_size = total_samples - train_size;
    
    // Allocate memory
    double** X_train = malloc(train_size * sizeof(double*));
    double** X_test = malloc(test_size * sizeof(double*));
    for(int i = 0; i < train_size; i++) X_train[i] = malloc(features * sizeof(double));
    for(int i = 0; i < test_size; i++) X_test[i] = malloc(features * sizeof(double));
    
    double* y_train = malloc(train_size * sizeof(double));
    double* y_test = malloc(test_size * sizeof(double));
    
    // Save original y values for denormalization
    double y_mean = compute_mean(y, total_samples);
    double y_std = compute_std(y, total_samples, y_mean);
    
    // Preprocess data
    normalize_data(X, y, total_samples, features);
    split_data(X_train, X_test, y_train, y_test, X, y, total_samples, train_size, features);
    
    // Initialize model parameters
    double* weights = calloc(features, sizeof(double));
    double bias = 0;
    
    // Train model
    printf("\nTraining linear regression...\n");
    train(weights, &bias, X_train, y_train, train_size, features, 0.001, 1000, 32);
    
    // Evaluate model
    printf("\nFinal Results:\n");
    printf("Training MSE: %.6f\n", compute_mse(weights, bias, X_train, y_train, train_size, features));
    printf("Test MSE: %.6f\n", compute_mse(weights, bias, X_test, y_test, test_size, features));
    
    // Show sample predictions
    printf("\nSample predictions (showing original scale):\n");
    for(int i = 0; i < 5; i++) {
        double pred = predict(weights, bias, X_test[i], features);
        printf("True: %.2f, Predicted: %.2f\n",
               y_test[i] * y_std + y_mean,
               pred * y_std + y_mean);
    }
    
    // Cleanup
    for(int i = 0; i < train_size; i++) free(X_train[i]);
    for(int i = 0; i < test_size; i++) free(X_test[i]);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(weights);
    
    return 0;
}