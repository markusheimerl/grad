#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

// Utility functions
void shuffle(int* arr, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

double mean(const double* arr, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum / n;
}

double stddev(const double* arr, int n, double m) {
    double sum = 0;
    for (int i = 0; i < n; i++) 
        sum += pow(arr[i] - m, 2);
    return sqrt(sum / n);
}

// Data preprocessing
void normalize(double (*X)[13], double* y, int rows, int cols) {
    // Normalize features
    for (int j = 0; j < cols; j++) {
        double m = 0, s = 0;
        for (int i = 0; i < rows; i++) m += X[i][j];
        m /= rows;
        
        for (int i = 0; i < rows; i++) 
            s += pow(X[i][j] - m, 2);
        s = sqrt(s / rows);
        
        for (int i = 0; i < rows; i++) 
            X[i][j] = (X[i][j] - m) / s;
    }
    
    // Normalize target
    double m = mean(y, rows);
    double s = stddev(y, rows, m);
    for (int i = 0; i < rows; i++) 
        y[i] = (y[i] - m) / s;
}

// Model functions
double predict(const double* w, double b, const double* x, int n) {
    double p = b;
    for (int i = 0; i < n; i++) p += w[i] * x[i];
    return p;
}

void train(double* w, double* b, double** X, double* y, 
          int samples, int features, double lr, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0;
        
        for (int i = 0; i < samples; i++) {
            double pred = predict(w, *b, X[i], features);
            double error = pred - y[i];
            
            // Update weights and bias
            for (int j = 0; j < features; j++)
                w[j] -= lr * error * X[i][j];
            *b -= lr * error;
            
            total_error += error * error;
        }
        
        if ((epoch + 1) % 100 == 0)
            printf("Epoch %d/%d - MSE: %.6f\n", epoch + 1, epochs, total_error/samples);
    }
}

int main() {
    srand(time(NULL));
    
    const int N = 506;  // Total samples
    const int F = 13;   // Features
    const int TRAIN = 404;  // 80% for training
    const int TEST = N - TRAIN;
    
    // Allocate memory
    double** X_train = malloc(TRAIN * sizeof(double*));
    double** X_test = malloc(TEST * sizeof(double*));
    for (int i = 0; i < TRAIN; i++) X_train[i] = malloc(F * sizeof(double));
    for (int i = 0; i < TEST; i++) X_test[i] = malloc(F * sizeof(double));
    double* y_train = malloc(TRAIN * sizeof(double));
    double* y_test = malloc(TEST * sizeof(double));
    
    // Get original scale parameters for later use
    double y_mean = mean(y, N);
    double y_std = stddev(y, N, y_mean);
    
    // Normalize data
    normalize(X, y, N, F);
    
    // Split data
    int* idx = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    shuffle(idx, N);
    
    for (int i = 0; i < TRAIN; i++) {
        for (int j = 0; j < F; j++) 
            X_train[i][j] = X[idx[i]][j];
        y_train[i] = y[idx[i]];
    }
    
    for (int i = 0; i < TEST; i++) {
        for (int j = 0; j < F; j++)
            X_test[i][j] = X[idx[i + TRAIN]][j];
        y_test[i] = y[idx[i + TRAIN]];
    }
    
    // Train model
    double* weights = calloc(F, sizeof(double));
    double bias = 0;
    
    printf("\nTraining linear regression...\n");
    train(weights, &bias, X_train, y_train, TRAIN, F, 0.001, 1000);
    
    // Evaluate
    int train_correct = 0, test_correct = 0;
    for (int i = 0; i < TRAIN; i++) {
        double pred = predict(weights, bias, X_train[i], F);
        double true_value = y_train[i] * y_std + y_mean;
        double pred_value = pred * y_std + y_mean;
        if (fabs(true_value - pred_value) <= 3.0) train_correct++;
    }
    
    for (int i = 0; i < TEST; i++) {
        double pred = predict(weights, bias, X_test[i], F);
        double true_value = y_test[i] * y_std + y_mean;
        double pred_value = pred * y_std + y_mean;
        if (fabs(true_value - pred_value) <= 3.0) test_correct++;
    }
    
    printf("\nFinal Results:\n");
    printf("Training Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)train_correct/TRAIN * 100, train_correct, TRAIN);
    printf("Test Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)test_correct/TEST * 100, test_correct, TEST);
    
    printf("\nSample predictions (showing original scale):\n");
    for (int i = 0; i < 5; i++) {
        double pred = predict(weights, bias, X_test[i], F);
        double true_value = y_test[i] * y_std + y_mean;
        double pred_value = pred * y_std + y_mean;
        printf("True: %.2f, Predicted: %.2f, Difference: %.2f, Within ±3.0: %s\n",
               true_value, pred_value, 
               fabs(true_value - pred_value),
               fabs(true_value - pred_value) <= 3.0 ? "Yes" : "No");
    }
    
    // Cleanup
    for (int i = 0; i < TRAIN; i++) free(X_train[i]);
    for (int i = 0; i < TEST; i++) free(X_test[i]);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(weights);
    free(idx);
    
    return 0;
}