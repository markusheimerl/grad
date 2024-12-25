#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

double compute_mean(double* array, int size) {
    double sum = 0;
    for(int i = 0; i < size; i++) {
        sum += array[i];
    }
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

int main() {
    srand(time(NULL));
    
    const int total_samples = 506;
    const int features = 13;
    
    // Allocate memory for train/test splits
    const int train_size = (int)(0.8 * total_samples);
    const int test_size = total_samples - train_size;
    
    double** X_train = malloc(train_size * sizeof(double*));
    double** X_test = malloc(test_size * sizeof(double*));
    double* y_train = malloc(train_size * sizeof(double));
    double* y_test = malloc(test_size * sizeof(double));
    
    for(int i = 0; i < train_size; i++) {
        X_train[i] = malloc(features * sizeof(double));
    }
    for(int i = 0; i < test_size; i++) {
        X_test[i] = malloc(features * sizeof(double));
    }
    
    // Z-score normalization for each feature
    for(int j = 0; j < features; j++) {
        double* column = malloc(total_samples * sizeof(double));
        for(int i = 0; i < total_samples; i++) {
            column[i] = X[i][j];
        }
        
        double mean = compute_mean(column, total_samples);
        double std = compute_std(column, total_samples, mean);
        
        for(int i = 0; i < total_samples; i++) {
            X[i][j] = (X[i][j] - mean) / std;
        }
        
        free(column);
    }
    
    // Z-score normalization for target variable
    double y_mean = compute_mean(y, total_samples);
    double y_std = compute_std(y, total_samples, y_mean);
    
    for(int i = 0; i < total_samples; i++) {
        y[i] = (y[i] - y_mean) / y_std;
    }
    
    // Store normalization parameters for later use
    printf("Target normalization parameters:\n");
    printf("Mean: %.3f, Std: %.3f\n", y_mean, y_std);
    
    // Create index array for shuffling
    int* indices = malloc(total_samples * sizeof(int));
    for(int i = 0; i < total_samples; i++) {
        indices[i] = i;
    }
    
    // Shuffle indices
    for(int i = total_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Split into train and test using shuffled indices
    for(int i = 0; i < train_size; i++) {
        for(int j = 0; j < features; j++) {
            X_train[i][j] = X[indices[i]][j];
        }
        y_train[i] = y[indices[i]];
    }
    
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < features; j++) {
            X_test[i][j] = X[indices[i + train_size]][j];
        }
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
    for(int i = 0; i < train_size; i++) {
        free(X_train[i]);
    }
    for(int i = 0; i < test_size; i++) {
        free(X_test[i]);
    }
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(indices);
    
    return 0;
}