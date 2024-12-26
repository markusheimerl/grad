#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "data/boston_housing_dataset.h"

#define TRAIN_SIZE 404
#define TEST_SIZE 102
#define FEATURES 13
#define NUM_BASE_MODELS 256
#define NUM_META_MODELS 64
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
    LinearModel* meta_models;
    LinearModel meta_meta_model;
    int num_base;
    int num_meta;
    int features;
} DeepEnsembleModel;

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

double predict_deep_ensemble(const DeepEnsembleModel* model, const double* x) {
    double* base_outputs = malloc(model->num_base * sizeof(double));
    double* meta_outputs = malloc(model->num_meta * sizeof(double));
    
    for (int i = 0; i < model->num_base; i++) {
        base_outputs[i] = model->base_models[i].bias;
        for (int j = 0; j < model->features; j++) {
            base_outputs[i] += model->base_models[i].weights[j] * x[j];
        }
        base_outputs[i] = fmax(0, base_outputs[i]);
    }
    
    for (int i = 0; i < model->num_meta; i++) {
        meta_outputs[i] = model->meta_models[i].bias;
        for (int j = 0; j < model->num_base; j++) {
            meta_outputs[i] += model->meta_models[i].weights[j] * base_outputs[j];
        }
        meta_outputs[i] = fmax(0, meta_outputs[i]);
    }
    
    double final_pred = model->meta_meta_model.bias;
    for (int i = 0; i < model->num_meta; i++) {
        final_pred += model->meta_meta_model.weights[i] * meta_outputs[i];
    }
    
    free(base_outputs);
    free(meta_outputs);
    return final_pred;
}

void train_deep_ensemble(DeepEnsembleModel* model, double** X, double* y, int samples,
                        double lr, int epochs, int batch_size) {
    int* indices = malloc(samples * sizeof(int));
    for (int i = 0; i < samples; i++) indices[i] = i;
    
    double** base_w_grads = malloc(model->num_base * sizeof(double*));
    double* base_b_grads = malloc(model->num_base * sizeof(double));
    double** meta_w_grads = malloc(model->num_meta * sizeof(double*));
    double* meta_b_grads = malloc(model->num_meta * sizeof(double));
    double* meta_meta_w_grads = malloc(model->num_meta * sizeof(double));
    
    for (int i = 0; i < model->num_base; i++) {
        base_w_grads[i] = malloc(model->features * sizeof(double));
    }
    for (int i = 0; i < model->num_meta; i++) {
        meta_w_grads[i] = malloc(model->num_base * sizeof(double));
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0;
        shuffle(indices, samples);
        
        for (int batch = 0; batch < samples/batch_size; batch++) {
            memset(meta_meta_w_grads, 0, model->num_meta * sizeof(double));
            double meta_meta_b_grad = 0;
            
            for (int i = 0; i < model->num_meta; i++) {
                memset(meta_w_grads[i], 0, model->num_base * sizeof(double));
                meta_b_grads[i] = 0;
            }
            
            for (int i = 0; i < model->num_base; i++) {
                memset(base_w_grads[i], 0, model->features * sizeof(double));
                base_b_grads[i] = 0;
            }
            
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++) {
                int idx = indices[i];
                double* base_outputs = malloc(model->num_base * sizeof(double));
                double* base_raw = malloc(model->num_base * sizeof(double));
                double* meta_outputs = malloc(model->num_meta * sizeof(double));
                double* meta_raw = malloc(model->num_meta * sizeof(double));
                
                // Forward pass
                for (int j = 0; j < model->num_base; j++) {
                    base_raw[j] = model->base_models[j].bias;
                    for (int k = 0; k < model->features; k++) {
                        base_raw[j] += model->base_models[j].weights[k] * X[idx][k];
                    }
                    base_outputs[j] = fmax(0, base_raw[j]);
                }
                
                for (int j = 0; j < model->num_meta; j++) {
                    meta_raw[j] = model->meta_models[j].bias;
                    for (int k = 0; k < model->num_base; k++) {
                        meta_raw[j] += model->meta_models[j].weights[k] * base_outputs[k];
                    }
                    meta_outputs[j] = fmax(0, meta_raw[j]);
                }
                
                double final_pred = model->meta_meta_model.bias;
                for (int j = 0; j < model->num_meta; j++) {
                    final_pred += model->meta_meta_model.weights[j] * meta_outputs[j];
                }
                
                // Backward pass
                double error = final_pred - y[idx];
                total_error += error * error;
                
                for (int j = 0; j < model->num_meta; j++) {
                    meta_meta_w_grads[j] += error * meta_outputs[j];
                }
                meta_meta_b_grad += error;
                
                for (int j = 0; j < model->num_meta; j++) {
                    if (meta_raw[j] <= 0) continue;
                    double meta_error = error * model->meta_meta_model.weights[j];
                    
                    for (int k = 0; k < model->num_base; k++) {
                        meta_w_grads[j][k] += meta_error * base_outputs[k];
                    }
                    meta_b_grads[j] += meta_error;
                }
                
                for (int j = 0; j < model->num_base; j++) {
                    if (base_raw[j] <= 0) continue;
                    double base_error = 0;
                    
                    for (int k = 0; k < model->num_meta; k++) {
                        if (meta_raw[k] > 0) {
                            base_error += error * model->meta_meta_model.weights[k] *
                                        model->meta_models[k].weights[j];
                        }
                    }
                    
                    for (int k = 0; k < model->features; k++) {
                        base_w_grads[j][k] += base_error * X[idx][k];
                    }
                    base_b_grads[j] += base_error;
                }
                
                free(base_outputs);
                free(base_raw);
                free(meta_outputs);
                free(meta_raw);
            }
            
            // Update weights
            for (int i = 0; i < model->num_base; i++) {
                for (int j = 0; j < model->features; j++) {
                    model->base_models[i].weights[j] -= lr * base_w_grads[i][j] / batch_size;
                }
                model->base_models[i].bias -= lr * base_b_grads[i] / batch_size;
            }
            
            for (int i = 0; i < model->num_meta; i++) {
                for (int j = 0; j < model->num_base; j++) {
                    model->meta_models[i].weights[j] -= lr * meta_w_grads[i][j] / batch_size;
                }
                model->meta_models[i].bias -= lr * meta_b_grads[i] / batch_size;
                model->meta_meta_model.weights[i] -= lr * meta_meta_w_grads[i] / batch_size;
            }
            model->meta_meta_model.bias -= lr * meta_meta_b_grad / batch_size;
        }
        
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d/%d - MSE: %.6f\n", epoch + 1, epochs, total_error/samples);
        }
    }
    
    for (int i = 0; i < model->num_base; i++) free(base_w_grads[i]);
    for (int i = 0; i < model->num_meta; i++) free(meta_w_grads[i]);
    free(base_w_grads);
    free(meta_w_grads);
    free(base_b_grads);
    free(meta_b_grads);
    free(meta_meta_w_grads);
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
    DeepEnsembleModel model;
    model.num_base = NUM_BASE_MODELS;
    model.num_meta = NUM_META_MODELS;
    model.features = FEATURES;
    
    model.base_models = malloc(NUM_BASE_MODELS * sizeof(LinearModel));
    model.meta_models = malloc(NUM_META_MODELS * sizeof(LinearModel));
    model.meta_meta_model.weights = malloc(NUM_META_MODELS * sizeof(double));
    
    double base_scale = sqrt(2.0/FEATURES);
    double meta_scale = sqrt(2.0/NUM_BASE_MODELS);
    double meta_meta_scale = sqrt(2.0/NUM_META_MODELS);
    
    for (int i = 0; i < NUM_BASE_MODELS; i++) {
        model.base_models[i].weights = malloc(FEATURES * sizeof(double));
        for (int j = 0; j < FEATURES; j++) {
            model.base_models[i].weights[j] = ((double)rand()/RAND_MAX * 2 - 1) * base_scale;
        }
        model.base_models[i].bias = ((double)rand()/RAND_MAX * 2 - 1) * base_scale;
    }
    
    for (int i = 0; i < NUM_META_MODELS; i++) {
        model.meta_models[i].weights = malloc(NUM_BASE_MODELS * sizeof(double));
        for (int j = 0; j < NUM_BASE_MODELS; j++) {
            model.meta_models[i].weights[j] = ((double)rand()/RAND_MAX * 2 - 1) * meta_scale;
        }
        model.meta_models[i].bias = ((double)rand()/RAND_MAX * 2 - 1) * meta_scale;
        model.meta_meta_model.weights[i] = ((double)rand()/RAND_MAX * 2 - 1) * meta_meta_scale;
    }
    model.meta_meta_model.bias = ((double)rand()/RAND_MAX * 2 - 1) * meta_meta_scale;
    
    printf("Training deep ensemble...\n");
    train_deep_ensemble(&model, X_train, y_train, TRAIN_SIZE, LEARNING_RATE, EPOCHS, BATCH_SIZE);
    
    // Evaluate
    int train_correct = 0, test_correct = 0;
    for (int i = 0; i < TRAIN_SIZE; i++) {
        double pred = predict_deep_ensemble(&model, X_train[i]);
        if (fabs((y_train[i] * y_std + y_mean) - (pred * y_std + y_mean)) <= TOLERANCE) 
            train_correct++;
    }
    
    for (int i = 0; i < TEST_SIZE; i++) {
        double pred = predict_deep_ensemble(&model, X_test[i]);
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
    for (int i = 0; i < NUM_META_MODELS; i++) {
        free(model.meta_models[i].weights);
    }
    free(model.base_models);
    free(model.meta_models);
    free(model.meta_meta_model.weights);
    
    for (int i = 0; i < TRAIN_SIZE; i++) free(X_train[i]);
    for (int i = 0; i < TEST_SIZE; i++) free(X_test[i]);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(idx);
    
    return 0;
}