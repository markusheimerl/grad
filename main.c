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

int main() {
    srand(time(NULL));
    
    const int total_samples = 506;
    const int features = 13;
    const int train_size = (int)(0.8 * total_samples);
    const int test_size = total_samples - train_size;
    
    // Allocate memory
    double** X_train = malloc(train_size * sizeof(double*));
    double** X_test = malloc(test_size * sizeof(double*));
    double* y_train = malloc(train_size * sizeof(double));
    double* y_test = malloc(test_size * sizeof(double));
    
    for(int i = 0; i < train_size; i++) X_train[i] = malloc(features * sizeof(double));
    for(int i = 0; i < test_size; i++) X_test[i] = malloc(features * sizeof(double));
    
    // Normalize features
    for(int j = 0; j < features; j++) {
        double mean, std;
        normalize_feature(X, total_samples, j, &mean, &std);
    }
    
    // Normalize target
    double y_mean = compute_mean(y, total_samples);
    double y_std = compute_std(y, total_samples, y_mean);
    for(int i = 0; i < total_samples; i++) y[i] = (y[i] - y_mean) / y_std;
    
    printf("Target normalization parameters:\n");
    printf("Mean: %.3f, Std: %.3f\n", y_mean, y_std);
    
    // Create and shuffle indices
    int* indices = malloc(total_samples * sizeof(int));
    for(int i = 0; i < total_samples; i++) indices[i] = i;
    shuffle_indices(indices, total_samples);
    
    // Split data
    for(int i = 0; i < train_size; i++) {
        for(int j = 0; j < features; j++) X_train[i][j] = X[indices[i]][j];
        y_train[i] = y[indices[i]];
    }
    
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < features; j++) X_test[i][j] = X[indices[i + train_size]][j];
        y_test[i] = y[indices[i + train_size]];
    }
    
    printf("\nDataset prepared:\n");
    printf("Training samples: %d\n", train_size);
    printf("Test samples: %d\n", test_size);
    printf("Features: %d\n", features);
    
    printf("\nFirst 5 training samples:\n");
    for(int i = 0; i < 5; i++) {
        printf("Sample %d: [%.3f, %.3f, %.3f, ...] -> %.3f (original: %.3f)\n", 
               i, X_train[i][0], X_train[i][1], X_train[i][2], y_train[i], 
               y_train[i] * y_std + y_mean);
    }
    
    // Cleanup
    for(int i = 0; i < train_size; i++) free(X_train[i]);
    for(int i = 0; i < test_size; i++) free(X_test[i]);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(indices);
    
    return 0;
}