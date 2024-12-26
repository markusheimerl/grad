#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

#define TRAIN_SIZE 404
#define TEST_SIZE 102
#define FEATURES 13
#define NUM_BASE_MODELS 256
#define EPOCHS 10000
#define BATCH_SIZE 32
#define LEARNING_RATE 0.001
#define TOLERANCE 3.0

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
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void normalize(double (*X)[13], double* y, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        double m = 0, s = 0;
        for (int i = 0; i < rows; i++) m += X[i][j];
        m /= rows;
        
        for (int i = 0; i < rows; i++) s += pow(X[i][j] - m, 2);
        s = sqrt(s / rows);
        
        for (int i = 0; i < rows; i++) X[i][j] = (X[i][j] - m) / (s + 1e-8);
    }
    
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < rows; i++) y_mean += y[i];
    y_mean /= rows;
    
    for (int i = 0; i < rows; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / rows);
    
    for (int i = 0; i < rows; i++) y[i] = (y[i] - y_mean) / y_std;
}

double predict_ensemble(const EnsembleModel* model, const double* x) {
    double* intermediate = malloc(model->num_base * sizeof(double));
    
    for (int i = 0; i < model->num_base; i++) {
        double base_pred = model->base_models[i].bias;
        for (int j = 0; j < model->features; j++) {
            base_pred += model->base_models[i].weights[j] * x[j];
        }
        intermediate[i] = fmax(0, base_pred);  // ReLU
    }
    
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
                memset(base_w_grads[i], 0, model->features * sizeof(double));
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
                    intermediate[j] = fmax(0, base_outputs[j]);
                }
                
                double final_pred = model->meta_model.bias;
                for (int j = 0; j < model->num_base; j++) {
                    final_pred += model->meta_model.weights[j] * intermediate[j];
                }
                
                // Backward pass
                double error = final_pred - y[idx];
                total_error += error * error;
                
                for (int j = 0; j < model->num_base; j++) {
                    meta_w_grads[j] += error * intermediate[j];
                }
                meta_b_grad += error;
                
                for (int j = 0; j < model->num_base; j++) {
                    if (base_outputs[j] <= 0) continue;
                    
                    double base_error = error * model->meta_model.weights[j];
                    for (int k = 0; k < model->features; k++) {
                        base_w_grads[j][k] += base_error * X[idx][k];
                    }
                    base_b_grads[j] += base_error;
                }
                
                free(intermediate);
                free(base_outputs);
            }
            
            // Update weights
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
    const int N = 506;
    
    // Prepare data
    double** X_train = malloc(TRAIN_SIZE * sizeof(double*));
    double** X_test = malloc(TEST_SIZE * sizeof(double*));
    for (int i = 0; i < TRAIN_SIZE; i++) X_train[i] = malloc(FEATURES * sizeof(double));
    for (int i = 0; i < TEST_SIZE; i++) X_test[i] = malloc(FEATURES * sizeof(double));
    
    double* y_train = malloc(TRAIN_SIZE * sizeof(double));
    double* y_test = malloc(TEST_SIZE * sizeof(double));
    
    // Calculate statistics and normalize
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < N; i++) y_mean += y[i];
    y_mean /= N;
    for (int i = 0; i < N; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / N);
    
    normalize(X, y, N, FEATURES);
    
    // Split data
    int* idx = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) idx[i] = i;
    shuffle(idx, N);
    
    for (int i = 0; i < TRAIN_SIZE; i++) {
        for (int j = 0; j < FEATURES; j++) X_train[i][j] = X[idx[i]][j];
        y_train[i] = y[idx[i]];
    }
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < FEATURES; j++) X_test[i][j] = X[idx[i + TRAIN_SIZE]][j];
        y_test[i] = y[idx[i + TRAIN_SIZE]];
    }
    
    // Initialize model
    EnsembleModel model;
    model.num_base = NUM_BASE_MODELS;
    model.features = FEATURES;
    model.base_models = malloc(NUM_BASE_MODELS * sizeof(LinearModel));
    model.meta_model.weights = malloc(NUM_BASE_MODELS * sizeof(double));
    
    double base_scale = sqrt(2.0/FEATURES);
    double meta_scale = sqrt(2.0/NUM_BASE_MODELS);
    
    for (int i = 0; i < NUM_BASE_MODELS; i++) {
        model.base_models[i].weights = malloc(FEATURES * sizeof(double));
        for (int j = 0; j < FEATURES; j++) {
            model.base_models[i].weights[j] = ((double)rand()/RAND_MAX * 2 - 1) * base_scale;
        }
        model.base_models[i].bias = ((double)rand()/RAND_MAX * 2 - 1) * base_scale;
        model.meta_model.weights[i] = ((double)rand()/RAND_MAX * 2 - 1) * meta_scale;
    }
    model.meta_model.bias = ((double)rand()/RAND_MAX * 2 - 1) * meta_scale;
    
    // Train
    printf("Training ensemble end-to-end...\n");
    train_ensemble(&model, X_train, y_train, TRAIN_SIZE, LEARNING_RATE, EPOCHS, BATCH_SIZE);
    
    // Evaluate
    int train_correct = 0, test_correct = 0;
    for (int i = 0; i < TRAIN_SIZE; i++) {
        double pred = predict_ensemble(&model, X_train[i]);
        if (fabs((y_train[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= TOLERANCE) 
            train_correct++;
    }
    
    for (int i = 0; i < TEST_SIZE; i++) {
        double pred = predict_ensemble(&model, X_test[i]);
        if (fabs((y_test[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= TOLERANCE) 
            test_correct++;
    }
    
    printf("\nFinal Results:\n");
    printf("Training Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)train_correct/TRAIN_SIZE * 100, train_correct, TRAIN_SIZE);
    printf("Test Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n", 
           (double)test_correct/TEST_SIZE * 100, test_correct, TEST_SIZE);
    
    // Cleanup
    for (int i = 0; i < NUM_BASE_MODELS; i++) {
        free(model.base_models[i].weights);
    }
    free(model.base_models);
    free(model.meta_model.weights);
    
    for (int i = 0; i < TRAIN_SIZE; i++) free(X_train[i]);
    for (int i = 0; i < TEST_SIZE; i++) free(X_test[i]);
    free(X_train); free(X_test);
    free(y_train); free(y_test);
    free(idx);
    
    return 0;
}