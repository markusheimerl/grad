#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

#define NUM_BASE_MODELS 256
#define NUM_META_MODELS 64
#define FEATURES 13
#define EPOCHS 1000
#define BATCH_SIZE 32
#define BASE_LR 0.001
#define META_LR 0.0005
#define FINAL_LR 0.0001

typedef struct {
    double* weights;
    double* m_weights;
    double* v_weights;
    double bias, m_bias, v_bias;
    size_t input_size, output_size;
} Layer;

typedef struct {
    Layer** base_layers;
    Layer** meta_layers;
    Layer* final_layer;
    size_t num_base, num_meta, features;
} EnsembleModel;

typedef struct {
    double* data;
    size_t rows, cols;
} Matrix;

typedef struct {
    double* base_outputs;
    double* base_raw;
    double* meta_outputs;
    double* meta_raw;
    double final_output;
    double final_raw;
} ForwardPass;

Layer* create_layer(size_t input_size, size_t output_size) {
    Layer* layer = malloc(sizeof(Layer));
    double scale = sqrt(2.0 / input_size);
    
    layer->weights = malloc(input_size * output_size * sizeof(double));
    layer->m_weights = calloc(input_size * output_size, sizeof(double));
    layer->v_weights = calloc(input_size * output_size, sizeof(double));
    
    for (size_t i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
    }
    layer->bias = ((double)rand()/RAND_MAX * 2 - 1) * scale;
    layer->input_size = input_size;
    layer->output_size = output_size;
    return layer;
}

EnsembleModel* create_ensemble() {
    EnsembleModel* model = malloc(sizeof(EnsembleModel));
    model->base_layers = malloc(NUM_BASE_MODELS * sizeof(Layer*));
    model->meta_layers = malloc(NUM_META_MODELS * sizeof(Layer*));
    
    for (int i = 0; i < NUM_BASE_MODELS; i++) 
        model->base_layers[i] = create_layer(FEATURES, 1);
    for (int i = 0; i < NUM_META_MODELS; i++) 
        model->meta_layers[i] = create_layer(NUM_BASE_MODELS, 1);
    
    model->final_layer = create_layer(NUM_META_MODELS, 1);
    model->num_base = NUM_BASE_MODELS;
    model->num_meta = NUM_META_MODELS;
    model->features = FEATURES;
    return model;
}

void normalize_data(Matrix* X, double* y, size_t y_size) {
    for (size_t j = 0; j < X->cols; j++) {
        double mean = 0, std = 0;
        for (size_t i = 0; i < X->rows; i++) mean += X->data[i * X->cols + j];
        mean /= X->rows;
        
        for (size_t i = 0; i < X->rows; i++) 
            std += pow(X->data[i * X->cols + j] - mean, 2);
        std = sqrt(std / X->rows);
        
        for (size_t i = 0; i < X->rows; i++)
            X->data[i * X->cols + j] = (X->data[i * X->cols + j] - mean) / (std + 1e-8);
    }
    
    double mean = 0, std = 0;
    for (size_t i = 0; i < y_size; i++) mean += y[i];
    mean /= y_size;
    
    for (size_t i = 0; i < y_size; i++) std += pow(y[i] - mean, 2);
    std = sqrt(std / y_size);
    
    for (size_t i = 0; i < y_size; i++)
        y[i] = (y[i] - mean) / (std + 1e-8);
}

ForwardPass* forward_propagate(EnsembleModel* model, const double* input) {
    ForwardPass* fp = malloc(sizeof(ForwardPass));
    fp->base_outputs = malloc(NUM_BASE_MODELS * sizeof(double));
    fp->base_raw = malloc(NUM_BASE_MODELS * sizeof(double));
    fp->meta_outputs = malloc(NUM_META_MODELS * sizeof(double));
    fp->meta_raw = malloc(NUM_META_MODELS * sizeof(double));
    
    for (size_t i = 0; i < model->num_base; i++) {
        double sum = model->base_layers[i]->bias;
        for (size_t j = 0; j < model->features; j++)
            sum += input[j] * model->base_layers[i]->weights[j];
        fp->base_raw[i] = sum;
        fp->base_outputs[i] = fmax(0, sum);
    }
    
    for (size_t i = 0; i < model->num_meta; i++) {
        double sum = model->meta_layers[i]->bias;
        for (size_t j = 0; j < model->num_base; j++)
            sum += fp->base_outputs[j] * model->meta_layers[i]->weights[j];
        fp->meta_raw[i] = sum;
        fp->meta_outputs[i] = fmax(0, sum);
    }
    
    double sum = model->final_layer->bias;
    for (size_t i = 0; i < model->num_meta; i++)
        sum += fp->meta_outputs[i] * model->final_layer->weights[i];
    
    fp->final_raw = fp->final_output = sum;
    return fp;
}

void adam_update(double* w, double* m, double* v, double g, double lr, double b1, double b2, double eps, double wd, int t) {
    *m = b1 * (*m) + (1 - b1) * g;
    *v = b2 * (*v) + (1 - b2) * g * g;
    double m_hat = *m / (1 - pow(b1, t));
    double v_hat = *v / (1 - pow(b2, t));
    *w *= (1 - lr * wd);
    *w -= lr * m_hat / (sqrt(v_hat) + eps);
}

void train_ensemble(EnsembleModel* model, Matrix* X, double* y) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
    int t = 0;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0;
        int* indices = malloc(X->rows * sizeof(int));
        for (int i = 0; i < X->rows; i++) indices[i] = i;
        
        // Shuffle indices
        for (int i = X->rows - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        for (int b = 0; b < X->rows; b += BATCH_SIZE) {
            t++;
            int batch_end = fmin(b + BATCH_SIZE, X->rows);
            int batch_size = batch_end - b;
            
            // Initialize gradients
            double** base_grads = malloc(NUM_BASE_MODELS * sizeof(double*));
            double* base_bias_grads = calloc(NUM_BASE_MODELS, sizeof(double));
            double** meta_grads = malloc(NUM_META_MODELS * sizeof(double*));
            double* meta_bias_grads = calloc(NUM_META_MODELS, sizeof(double));
            double* final_grads = calloc(NUM_META_MODELS, sizeof(double));
            double final_bias_grad = 0.0;
            
            for (int i = 0; i < NUM_BASE_MODELS; i++) 
                base_grads[i] = calloc(FEATURES, sizeof(double));
            for (int i = 0; i < NUM_META_MODELS; i++) 
                meta_grads[i] = calloc(NUM_BASE_MODELS, sizeof(double));
            
            // Process batch
            for (int i = b; i < batch_end; i++) {
                int idx = indices[i];
                ForwardPass* fp = forward_propagate(model, &X->data[idx * X->cols]);
                double error = fp->final_output - y[idx];
                loss += error * error;
                
                // Final layer gradients
                for (int j = 0; j < NUM_META_MODELS; j++) {
                    final_grads[j] += error * fp->meta_outputs[j];
                }
                final_bias_grad += error;
                
                // Meta layer gradients
                for (int j = 0; j < NUM_META_MODELS; j++) {
                    if (fp->meta_raw[j] <= 0) continue;
                    double meta_error = error * model->final_layer->weights[j];
                    
                    for (int k = 0; k < NUM_BASE_MODELS; k++) {
                        meta_grads[j][k] += meta_error * fp->base_outputs[k];
                    }
                    meta_bias_grads[j] += meta_error;
                }
                
                // Base layer gradients
                for (int j = 0; j < NUM_BASE_MODELS; j++) {
                    if (fp->base_raw[j] <= 0) continue;
                    
                    double base_error = 0;
                    for (int k = 0; k < NUM_META_MODELS; k++) {
                        if (fp->meta_raw[k] > 0) {
                            base_error += error * model->final_layer->weights[k] * 
                                        model->meta_layers[k]->weights[j];
                        }
                    }
                    
                    for (int k = 0; k < FEATURES; k++) {
                        base_grads[j][k] += base_error * X->data[idx * X->cols + k];
                    }
                    base_bias_grads[j] += base_error;
                }
                
                free(fp);
            }
            
            // Update parameters
            for (int i = 0; i < NUM_BASE_MODELS; i++) {
                Layer* layer = model->base_layers[i];
                for (int j = 0; j < FEATURES; j++) {
                    adam_update(&layer->weights[j], &layer->m_weights[j], &layer->v_weights[j],
                              base_grads[i][j] / batch_size, BASE_LR, b1, b2, eps, wd, t);
                }
                adam_update(&layer->bias, &layer->m_bias, &layer->v_bias,
                          base_bias_grads[i] / batch_size, BASE_LR, b1, b2, eps, wd, t);
            }
            
            for (int i = 0; i < NUM_META_MODELS; i++) {
                Layer* layer = model->meta_layers[i];
                for (int j = 0; j < NUM_BASE_MODELS; j++) {
                    adam_update(&layer->weights[j], &layer->m_weights[j], &layer->v_weights[j],
                              meta_grads[i][j] / batch_size, META_LR, b1, b2, eps, wd, t);
                }
                adam_update(&layer->bias, &layer->m_bias, &layer->v_bias,
                          meta_bias_grads[i] / batch_size, META_LR, b1, b2, eps, wd, t);
            }
            
            for (int i = 0; i < NUM_META_MODELS; i++) {
                adam_update(&model->final_layer->weights[i], 
                          &model->final_layer->m_weights[i],
                          &model->final_layer->v_weights[i],
                          final_grads[i] / batch_size, FINAL_LR, b1, b2, eps, wd, t);
            }
            adam_update(&model->final_layer->bias, &model->final_layer->m_bias,
                      &model->final_layer->v_bias, final_bias_grad / batch_size,
                      FINAL_LR, b1, b2, eps, wd, t);
            
            // Cleanup
            for (int i = 0; i < NUM_BASE_MODELS; i++) free(base_grads[i]);
            for (int i = 0; i < NUM_META_MODELS; i++) free(meta_grads[i]);
            free(base_grads);
            free(meta_grads);
            free(base_bias_grads);
            free(meta_bias_grads);
            free(final_grads);
        }
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %d/%d - MSE: %.6f\n", epoch + 1, EPOCHS, loss / X->rows);
        }
        
        free(indices);
    }
}

double predict(EnsembleModel* model, const double* input) {
    ForwardPass* fp = forward_propagate(model, input);
    double pred = fp->final_output;
    free(fp);
    return pred;
}

int main() {
    srand(time(NULL));
    
    Matrix X_train = {malloc(404 * 13 * sizeof(double)), 404, 13};
    Matrix X_test = {malloc(102 * 13 * sizeof(double)), 102, 13};
    double* y_train = malloc(404 * sizeof(double));
    double* y_test = malloc(102 * sizeof(double));
    
    // Create train/test split
    int* indices = malloc(506 * sizeof(int));
    for (int i = 0; i < 506; i++) indices[i] = i;
    for (int i = 505; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Split data
    for (int i = 0; i < 404; i++) {
        for (int j = 0; j < 13; j++) {
            X_train.data[i * 13 + j] = X[indices[i]][j];
        }
        y_train[i] = y[indices[i]];
    }
    
    for (int i = 0; i < 102; i++) {
        for (int j = 0; j < 13; j++) {
            X_test.data[i * 13 + j] = X[indices[i + 404]][j];
        }
        y_test[i] = y[indices[i + 404]];
    }
    
    // Calculate statistics for denormalization
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < 506; i++) y_mean += y[i];
    y_mean /= 506;
    for (int i = 0; i < 506; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / 506);
    
    normalize_data(&X_train, y_train, 404);
    normalize_data(&X_test, y_test, 102);
    
    printf("Creating and training ensemble...\n");
    EnsembleModel* model = create_ensemble();
    train_ensemble(model, &X_train, y_train);
    
    // Evaluate
    int train_correct = 0, test_correct = 0;
    
    for (int i = 0; i < 404; i++) {
        double pred = predict(model, &X_train.data[i * 13]);
        double true_val = y_train[i] * y_std + y_mean;
        double pred_val = pred * y_std + y_mean;
        if (fabs(true_val - pred_val) <= 3.0) train_correct++;
    }
    
    for (int i = 0; i < 102; i++) {
        double pred = predict(model, &X_test.data[i * 13]);
        double true_val = y_test[i] * y_std + y_mean;
        double pred_val = pred * y_std + y_mean;
        if (fabs(true_val - pred_val) <= 3.0) test_correct++;
    }
    
    printf("\nFinal Results:\n");
    printf("Training Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n",
           (double)train_correct/404 * 100, train_correct, 404);
    printf("Test Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n",
           (double)test_correct/102 * 100, test_correct, 102);
    
    // Cleanup
    free(X_train.data);
    free(X_test.data);
    free(y_train);
    free(y_test);
    free(indices);
    
    return 0;
}