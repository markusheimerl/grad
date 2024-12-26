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

typedef struct {
    double* weights;
    double bias;
    int features;
} LinearRegression;

LinearRegression* create_model(int features) {
    LinearRegression* model = malloc(sizeof(LinearRegression));
    model->weights = malloc(features * sizeof(double));
    model->features = features;
    
    for(int i = 0; i < features; i++) {
        model->weights[i] = (double)rand() / RAND_MAX * 0.1;
    }
    model->bias = (double)rand() / RAND_MAX * 0.1;
    
    return model;
}

double predict(LinearRegression* model, double* x) {
    double prediction = model->bias;
    for(int i = 0; i < model->features; i++) {
        prediction += model->weights[i] * x[i];
    }
    return prediction;
}

double compute_mse(LinearRegression* model, double** X, double* y, int n, double lambda) {
    double mse = 0;
    double reg_term = 0;
    
    for(int i = 0; i < n; i++) {
        double pred = predict(model, X[i]);
        double error = pred - y[i];
        mse += error * error;
    }
    
    for(int j = 0; j < model->features; j++) {
        reg_term += model->weights[j] * model->weights[j];
    }
    
    return mse / n + lambda * reg_term;
}

void train_batch(LinearRegression* model, double** X, double* y, 
                int start_idx, int batch_size, double learning_rate, double lambda) {
    double* weight_gradients = calloc(model->features, sizeof(double));
    double bias_gradient = 0;
    
    for(int i = start_idx; i < start_idx + batch_size; i++) {
        double pred = predict(model, X[i]);
        double error = pred - y[i];
        
        for(int j = 0; j < model->features; j++) {
            weight_gradients[j] += error * X[i][j];
        }
        bias_gradient += error;
    }
    
    for(int j = 0; j < model->features; j++) {
        model->weights[j] = model->weights[j] * (1 - learning_rate * lambda) - 
                           learning_rate * (weight_gradients[j] / batch_size);
    }
    model->bias -= learning_rate * (bias_gradient / batch_size);
    
    free(weight_gradients);
}

void train(LinearRegression* model, double** X_train, double* y_train, int train_size,
          double learning_rate, int epochs, int batch_size, double lambda) {
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int i = 0; i < train_size; i += batch_size) {
            int actual_batch_size = (i + batch_size > train_size) ? (train_size - i) : batch_size;
            train_batch(model, X_train, y_train, i, actual_batch_size, learning_rate, lambda);
        }
        
        if((epoch + 1) % 100 == 0) {
            double mse = compute_mse(model, X_train, y_train, train_size, lambda);
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
    
    LinearRegression* model = create_model(features);
    printf("\nTraining linear regression...\n");
    train(model, X_train, y_train, train_size, 0.001, 1000, 32, 0.01);
    
    double train_mse = compute_mse(model, X_train, y_train, train_size, 0);
    double test_mse = compute_mse(model, X_test, y_test, test_size, 0);
    
    printf("\nFinal Results:\n");
    printf("Training MSE: %.6f\n", train_mse);
    printf("Test MSE: %.6f\n", test_mse);
    
    printf("\nSample predictions (showing original scale):\n");
    for(int i = 0; i < 5; i++) {
        double pred = predict(model, X_test[i]);
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
    free(model->weights);
    free(model);
    
    return 0;
}