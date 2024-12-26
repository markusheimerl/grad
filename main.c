#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

double compute_mean(double* array, int size) {
    double sum = 0;
    for(int i = 0; i < size; i++) sum += array[i];
    return sum / size;
}

double compute_std(double* array, int size, double mean) {
    double sum_squared_diff = 0;
    for(int i = 0; i < size; i++) {
        double diff = array[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrt(sum_squared_diff / size);
}

void normalize_feature(double (*data)[13], int rows, int col, double* mean, double* std) {
    double* column = malloc(rows * sizeof(double));
    for(int i = 0; i < rows; i++) column[i] = data[i][col];
    
    *mean = compute_mean(column, rows);
    *std = compute_std(column, rows, *mean);
    
    for(int i = 0; i < rows; i++) 
        data[i][col] = (data[i][col] - *mean) / *std;
    
    free(column);
}

void shuffle_indices(int* indices, int size) {
    for(int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

double predict(double* weights, double bias, double* x, int features) {
    double prediction = bias;
    for(int i = 0; i < features; i++) {
        prediction += weights[i] * x[i];
    }
    return prediction;
}

double compute_mse(double* weights, double bias, double** X, double* y, int n, int features) {
    double mse = 0;
    
    for(int i = 0; i < n; i++) {
        double pred = predict(weights, bias, X[i], features);
        double error = pred - y[i];
        mse += error * error;
    }
    
    return mse / n;
}

void train_batch(double* weights, double* bias, double** X, double* y, 
                int start_idx, int batch_size, double learning_rate, int features) {
    double* weight_gradients = calloc(features, sizeof(double));
    double bias_gradient = 0;
    
    for(int i = start_idx; i < start_idx + batch_size; i++) {
        double pred = predict(weights, *bias, X[i], features);
        double error = pred - y[i];
        
        for(int j = 0; j < features; j++) {
            weight_gradients[j] += error * X[i][j];
        }
        bias_gradient += error;
    }
    
    for(int j = 0; j < features; j++) {
        weights[j] -= learning_rate * (weight_gradients[j] / batch_size);
    }
    *bias -= learning_rate * (bias_gradient / batch_size);
    
    free(weight_gradients);
}

void train(double* weights, double* bias, double** X_train, double* y_train, 
          int train_size, int features, double learning_rate, int epochs, int batch_size) {
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int i = 0; i < train_size; i += batch_size) {
            int actual_batch_size = (i + batch_size > train_size) ? (train_size - i) : batch_size;
            train_batch(weights, bias, X_train, y_train, i, actual_batch_size, learning_rate, features);
        }
        
        if((epoch + 1) % 100 == 0) {
            double mse = compute_mse(weights, *bias, X_train, y_train, train_size, features);
            printf("Epoch %d: MSE = %.6f\n", epoch + 1, mse);
        }
    }
}

int main() {
    srand(time(NULL));
    
    const int total_samples = 506;
    const int features = 13;
    const int train_size = (int)(0.8 * total_samples);
    const int test_size = total_samples - train_size;
    
    double** X_train = malloc(train_size * sizeof(double*));
    double** X_test = malloc(test_size * sizeof(double*));
    double* y_train = malloc(train_size * sizeof(double));
    double* y_test = malloc(test_size * sizeof(double));
    
    for(int i = 0; i < train_size; i++) X_train[i] = malloc(features * sizeof(double));
    for(int i = 0; i < test_size; i++) X_test[i] = malloc(features * sizeof(double));
    
    for(int j = 0; j < features; j++) {
        double mean, std;
        normalize_feature(X, total_samples, j, &mean, &std);
    }
    
    double y_mean = compute_mean(y, total_samples);
    double y_std = compute_std(y, total_samples, y_mean);
    for(int i = 0; i < total_samples; i++) y[i] = (y[i] - y_mean) / y_std;
    
    int* indices = malloc(total_samples * sizeof(int));
    for(int i = 0; i < total_samples; i++) indices[i] = i;
    shuffle_indices(indices, total_samples);
    
    for(int i = 0; i < train_size; i++) {
        for(int j = 0; j < features; j++) X_train[i][j] = X[indices[i]][j];
        y_train[i] = y[indices[i]];
    }
    
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < features; j++) X_test[i][j] = X[indices[i + train_size]][j];
        y_test[i] = y[indices[i + train_size]];
    }
    
    double* weights = malloc(features * sizeof(double));
    double bias = (double)rand() / RAND_MAX * 0.1;
    
    for(int i = 0; i < features; i++) {
        weights[i] = (double)rand() / RAND_MAX * 0.1;
    }
    
    printf("\nTraining linear regression...\n");
    train(weights, &bias, X_train, y_train, train_size, features, 0.001, 1000, 32);
    
    double train_mse = compute_mse(weights, bias, X_train, y_train, train_size, features);
    double test_mse = compute_mse(weights, bias, X_test, y_test, test_size, features);
    
    printf("\nFinal Results:\n");
    printf("Training MSE: %.6f\n", train_mse);
    printf("Test MSE: %.6f\n", test_mse);
    
    printf("\nSample predictions (showing original scale):\n");
    for(int i = 0; i < 5; i++) {
        double pred = predict(weights, bias, X_test[i], features);
        printf("True: %.2f, Predicted: %.2f\n",
               y_test[i] * y_std + y_mean,
               pred * y_std + y_mean);
    }
    
    for(int i = 0; i < train_size; i++) free(X_train[i]);
    for(int i = 0; i < test_size; i++) free(X_test[i]);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(indices);
    free(weights);
    
    return 0;
}