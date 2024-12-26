#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

void shuffle(int* arr, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

void normalize(double (*X)[13], double* y, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        double m = 0, s = 0;
        for (int i = 0; i < rows; i++) m += X[i][j];
        m /= rows;
        
        for (int i = 0; i < rows; i++) s += pow(X[i][j] - m, 2);
        s = sqrt(s / rows);
        
        for (int i = 0; i < rows; i++) X[i][j] = (X[i][j] - m) / s;
    }
    
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < rows; i++) y_mean += y[i];
    y_mean /= rows;
    
    for (int i = 0; i < rows; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / rows);
    
    for (int i = 0; i < rows; i++) y[i] = (y[i] - y_mean) / y_std;
}

double predict(const double* w, double b, const double* x, int n) {
    double p = b;
    for (int i = 0; i < n; i++) p += w[i] * x[i];
    return p;
}

void train(double* w, double* b, double** X, double* y, int samples, int features, 
          double lr, int epochs, int batch_size) {
    int* indices = malloc(samples * sizeof(int));
    for (int i = 0; i < samples; i++) indices[i] = i;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0;
        shuffle(indices, samples);
        
        for (int batch = 0; batch < samples/batch_size; batch++) {
            double* w_grad = calloc(features, sizeof(double));
            double b_grad = 0;
            
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++) {
                double error = predict(w, *b, X[indices[i]], features) - y[indices[i]];
                for (int j = 0; j < features; j++) w_grad[j] += error * X[indices[i]][j];
                b_grad += error;
                total_error += error * error;
            }
            
            for (int j = 0; j < features; j++) w[j] -= lr * (w_grad[j] / batch_size);
            *b -= lr * (b_grad / batch_size);
            free(w_grad);
        }
        
        if ((epoch + 1) % 100 == 0)
            printf("Epoch %d/%d - MSE: %.6f\n", epoch + 1, epochs, total_error/samples);
    }
    free(indices);
}

int main() {
    srand(time(NULL));
    const int N = 506, F = 13, TRAIN = 404, TEST = N - TRAIN;
    
    double** X_train = malloc(TRAIN * sizeof(double*));
    double** X_test = malloc(TEST * sizeof(double*));
    for (int i = 0; i < TRAIN; i++) X_train[i] = malloc(F * sizeof(double));
    for (int i = 0; i < TEST; i++) X_test[i] = malloc(F * sizeof(double));
    double* y_train = malloc(TRAIN * sizeof(double));
    double* y_test = malloc(TEST * sizeof(double));
    
    double y_mean = 0;
    for (int i = 0; i < N; i++) y_mean += y[i];
    y_mean /= N;
    
    double y_std = 0;
    for (int i = 0; i < N; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / N);
    
    normalize(X, y, N, F);
    
    int* idx = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    shuffle(idx, N);
    
    for (int i = 0; i < TRAIN; i++) {
        for (int j = 0; j < F; j++) X_train[i][j] = X[idx[i]][j];
        y_train[i] = y[idx[i]];
    }
    
    for (int i = 0; i < TEST; i++) {
        for (int j = 0; j < F; j++) X_test[i][j] = X[idx[i + TRAIN]][j];
        y_test[i] = y[idx[i + TRAIN]];
    }
    
    double* weights = calloc(F, sizeof(double));
    double bias = 0;
    
    printf("\nTraining linear regression...\n");
    train(weights, &bias, X_train, y_train, TRAIN, F, 0.001, 1000, 32);
    
    int train_correct = 0, test_correct = 0;
    for (int i = 0; i < TRAIN; i++) {
        double pred = predict(weights, bias, X_train[i], F);
        if (fabs((y_train[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= 3.0) train_correct++;
    }
    
    for (int i = 0; i < TEST; i++) {
        double pred = predict(weights, bias, X_test[i], F);
        if (fabs((y_test[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= 3.0) test_correct++;
    }
    
    printf("\nFinal Results:\n");
    printf("Training Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)train_correct/TRAIN * 100, train_correct, TRAIN);
    printf("Test Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)test_correct/TEST * 100, test_correct, TEST);
    
    printf("\nSample predictions (showing original scale):\n");
    for (int i = 0; i < 5; i++) {
        double pred = predict(weights, bias, X_test[i], F);
        double true_val = y_test[i] * y_std + y_mean;
        double pred_val = pred * y_std + y_mean;
        printf("True: %.2f, Predicted: %.2f, Difference: %.2f, Within ±3.0: %s\n",
               true_val, pred_val, fabs(true_val - pred_val),
               fabs(true_val - pred_val) <= 3.0 ? "Yes" : "No");
    }
    
    for (int i = 0; i < TRAIN; i++) free(X_train[i]);
    for (int i = 0; i < TEST; i++) free(X_test[i]);
    free(X_train); free(X_test); free(y_train); free(y_test);
    free(weights); free(idx);
    
    return 0;
}