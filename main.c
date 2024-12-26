#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

typedef struct {
    double* weights;
    double bias;
} LinearModel;

typedef struct {
    LinearModel* base_models;
    LinearModel meta_model;
    int num_base;
    int features;
} EnsembleModel;

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

double relu(double x) {
    return x > 0 ? x : 0;
}

double predict_ensemble(const EnsembleModel* model, const double* x) {
    double* intermediate = malloc(model->num_base * sizeof(double));
    
    // Forward pass through base models
    for (int i = 0; i < model->num_base; i++) {
        double base_pred = model->base_models[i].bias;
        for (int j = 0; j < model->features; j++) {
            base_pred += model->base_models[i].weights[j] * x[j];
        }
        intermediate[i] = relu(base_pred);
    }
    
    // Forward pass through meta model
    double final_pred = model->meta_model.bias;
    for (int i = 0; i < model->num_base; i++) {
        final_pred += model->meta_model.weights[i] * intermediate[i];
    }
    
    free(intermediate);
    return final_pred;
}

void train_ensemble(EnsembleModel* model, double** X, double* y, int samples,
                   double lr, int epochs, int batch_size) {
    int* indices = malloc(samples * sizeof(int));
    for (int i = 0; i < samples; i++) indices[i] = i;
    
    // Allocate gradients for all models
    double** base_w_grads = malloc(model->num_base * sizeof(double*));
    double* base_b_grads = malloc(model->num_base * sizeof(double));
    double* meta_w_grads = malloc(model->num_base * sizeof(double));
    for (int i = 0; i < model->num_base; i++) {
        base_w_grads[i] = malloc(model->features * sizeof(double));
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0;
        shuffle(indices, samples);
        
        for (int batch = 0; batch < samples/batch_size; batch++) {
            // Reset gradients
            for (int i = 0; i < model->num_base; i++) {
                for (int j = 0; j < model->features; j++) {
                    base_w_grads[i][j] = 0;
                }
                base_b_grads[i] = 0;
                meta_w_grads[i] = 0;
            }
            double meta_b_grad = 0;
            
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++) {
                int idx = indices[i];
                double* intermediate = malloc(model->num_base * sizeof(double));
                double* base_outputs = malloc(model->num_base * sizeof(double));
                
                // Forward pass
                for (int j = 0; j < model->num_base; j++) {
                    base_outputs[j] = model->base_models[j].bias;
                    for (int k = 0; k < model->features; k++) {
                        base_outputs[j] += model->base_models[j].weights[k] * X[idx][k];
                    }
                    intermediate[j] = relu(base_outputs[j]);
                }
                
                double final_pred = model->meta_model.bias;
                for (int j = 0; j < model->num_base; j++) {
                    final_pred += model->meta_model.weights[j] * intermediate[j];
                }
                
                // Backward pass
                double error = final_pred - y[idx];
                total_error += error * error;
                
                // Meta model gradients
                for (int j = 0; j < model->num_base; j++) {
                    meta_w_grads[j] += error * intermediate[j];
                }
                meta_b_grad += error;
                
                // Base models gradients
                for (int j = 0; j < model->num_base; j++) {
                    if (base_outputs[j] <= 0) continue;  // ReLU gradient
                    
                    double base_error = error * model->meta_model.weights[j];
                    for (int k = 0; k < model->features; k++) {
                        base_w_grads[j][k] += base_error * X[idx][k];
                    }
                    base_b_grads[j] += base_error;
                }
                
                free(intermediate);
                free(base_outputs);
            }
            
            // Update weights and biases
            for (int i = 0; i < model->num_base; i++) {
                for (int j = 0; j < model->features; j++) {
                    model->base_models[i].weights[j] -= lr * base_w_grads[i][j] / batch_size;
                }
                model->base_models[i].bias -= lr * base_b_grads[i] / batch_size;
                model->meta_model.weights[i] -= lr * meta_w_grads[i] / batch_size;
            }
            model->meta_model.bias -= lr * meta_b_grad / batch_size;
        }
        
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d/%d - MSE: %.6f\n", epoch + 1, epochs, total_error/samples);
        }
    }
    
    // Cleanup
    for (int i = 0; i < model->num_base; i++) {
        free(base_w_grads[i]);
    }
    free(base_w_grads);
    free(base_b_grads);
    free(meta_w_grads);
    free(indices);
}

int main() {
    srand(time(NULL));
    const int N = 506, F = 13, TRAIN = 404, TEST = N - TRAIN;
    const int NUM_BASE = 256;
    
    // Allocate and prepare data
    double** X_train = malloc(TRAIN * sizeof(double*));
    double** X_test = malloc(TEST * sizeof(double*));
    for (int i = 0; i < TRAIN; i++) X_train[i] = malloc(F * sizeof(double));
    for (int i = 0; i < TEST; i++) X_test[i] = malloc(F * sizeof(double));
    double* y_train = malloc(TRAIN * sizeof(double));
    double* y_test = malloc(TEST * sizeof(double));
    
    // Calculate statistics and normalize
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < N; i++) y_mean += y[i];
    y_mean /= N;
    for (int i = 0; i < N; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / N);
    
    normalize(X, y, N, F);
    
    // Split data
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
    
    // Initialize ensemble model
    EnsembleModel model;
    model.num_base = NUM_BASE;
    model.features = F;
    model.base_models = malloc(NUM_BASE * sizeof(LinearModel));
    model.meta_model.weights = malloc(NUM_BASE * sizeof(double));
    
    // Initialize weights
    for (int i = 0; i < NUM_BASE; i++) {
        model.base_models[i].weights = malloc(F * sizeof(double));
        for (int j = 0; j < F; j++) {
            model.base_models[i].weights[j] = ((double)rand()/RAND_MAX * 2 - 1) * sqrt(6.0/F);
        }
        model.base_models[i].bias = ((double)rand()/RAND_MAX * 2 - 1) * sqrt(6.0/F);
        model.meta_model.weights[i] = ((double)rand()/RAND_MAX * 2 - 1) * sqrt(6.0/NUM_BASE);
    }
    model.meta_model.bias = ((double)rand()/RAND_MAX * 2 - 1) * sqrt(6.0/NUM_BASE);
    
    // Train end-to-end
    printf("Training ensemble end-to-end...\n");
    train_ensemble(&model, X_train, y_train, TRAIN, 0.001, 10000, 32);
    
    // Evaluate
    int train_correct = 0, test_correct = 0;
    for (int i = 0; i < TRAIN; i++) {
        double pred = predict_ensemble(&model, X_train[i]);
        if (fabs((y_train[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= 3.0) train_correct++;
    }
    
    for (int i = 0; i < TEST; i++) {
        double pred = predict_ensemble(&model, X_test[i]);
        if (fabs((y_test[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= 3.0) test_correct++;
    }
    
    printf("\nFinal Results:\n");
    printf("Training Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)train_correct/TRAIN * 100, train_correct, TRAIN);
    printf("Test Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)test_correct/TEST * 100, test_correct, TEST);
    
    // Cleanup
    for (int i = 0; i < NUM_BASE; i++) {
        free(model.base_models[i].weights);
    }
    free(model.base_models);
    free(model.meta_model.weights);
    
    for (int i = 0; i < TRAIN; i++) free(X_train[i]);
    for (int i = 0; i < TEST; i++) free(X_test[i]);
    free(X_train); free(X_test);
    free(y_train); free(y_test);
    free(idx);
    
    return 0;
}