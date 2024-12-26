#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "data/boston_housing_dataset.h"

// Hyperparameters
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
    double bias;
    // Adam parameters
    double* m_weights;
    double* v_weights;
    double m_bias;
    double v_bias;
    size_t input_size;
    size_t output_size;
} Layer;

typedef struct {
    Layer** base_layers;    // First level ensemble
    Layer** meta_layers;    // Second level ensemble
    Layer* final_layer;     // Final aggregation
    size_t num_base;
    size_t num_meta;
    size_t features;
} EnsembleModel;

typedef struct {
    double* data;
    size_t rows;
    size_t cols;
} Matrix;

typedef struct {
    double* data;
    size_t size;
} Vector;

// Utility functions
Matrix create_matrix(size_t rows, size_t cols) {
    Matrix m = {
        .data = calloc(rows * cols, sizeof(double)),
        .rows = rows,
        .cols = cols
    };
    return m;
}

Vector create_vector(size_t size) {
    Vector v = {
        .data = calloc(size, sizeof(double)),
        .size = size
    };
    return v;
}

void free_matrix(Matrix* m) {
    free(m->data);
    m->data = NULL;
}

void free_vector(Vector* v) {
    free(v->data);
    v->data = NULL;
}

// Layer operations
Layer* create_layer(size_t input_size, size_t output_size) {
    Layer* layer = malloc(sizeof(Layer));
    
    layer->weights = malloc(input_size * output_size * sizeof(double));
    layer->m_weights = calloc(input_size * output_size, sizeof(double));
    layer->v_weights = calloc(input_size * output_size, sizeof(double));
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->m_bias = 0.0;
    layer->v_bias = 0.0;
    
    // He initialization
    double scale = sqrt(2.0 / input_size);
    for (size_t i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
    }
    layer->bias = ((double)rand()/RAND_MAX * 2 - 1) * scale;
    
    return layer;
}

void free_layer(Layer* layer) {
    free(layer->weights);
    free(layer->m_weights);
    free(layer->v_weights);
    free(layer);
}

// Model creation and cleanup
EnsembleModel* create_ensemble() {
    EnsembleModel* model = malloc(sizeof(EnsembleModel));
    
    model->num_base = NUM_BASE_MODELS;
    model->num_meta = NUM_META_MODELS;
    model->features = FEATURES;
    
    // Create base models
    model->base_layers = malloc(NUM_BASE_MODELS * sizeof(Layer*));
    for (int i = 0; i < NUM_BASE_MODELS; i++) {
        model->base_layers[i] = create_layer(FEATURES, 1);
    }
    
    // Create meta models
    model->meta_layers = malloc(NUM_META_MODELS * sizeof(Layer*));
    for (int i = 0; i < NUM_META_MODELS; i++) {
        model->meta_layers[i] = create_layer(NUM_BASE_MODELS, 1);
    }
    
    // Create final layer
    model->final_layer = create_layer(NUM_META_MODELS, 1);
    
    return model;
}

void free_ensemble(EnsembleModel* model) {
    for (size_t i = 0; i < model->num_base; i++) {
        free_layer(model->base_layers[i]);
    }
    for (size_t i = 0; i < model->num_meta; i++) {
        free_layer(model->meta_layers[i]);
    }
    free_layer(model->final_layer);
    
    free(model->base_layers);
    free(model->meta_layers);
    free(model);
}

// Data preprocessing
void normalize_data(Matrix* X, Vector* y) {
    // Normalize features
    for (size_t j = 0; j < X->cols; j++) {
        double mean = 0, std = 0;
        
        for (size_t i = 0; i < X->rows; i++) {
            mean += X->data[i * X->cols + j];
        }
        mean /= X->rows;
        
        for (size_t i = 0; i < X->rows; i++) {
            std += pow(X->data[i * X->cols + j] - mean, 2);
        }
        std = sqrt(std / X->rows);
        
        for (size_t i = 0; i < X->rows; i++) {
            X->data[i * X->cols + j] = (X->data[i * X->cols + j] - mean) / (std + 1e-8);
        }
    }
    
    // Normalize target
    double mean = 0, std = 0;
    for (size_t i = 0; i < y->size; i++) mean += y->data[i];
    mean /= y->size;
    
    for (size_t i = 0; i < y->size; i++) std += pow(y->data[i] - mean, 2);
    std = sqrt(std / y->size);
    
    for (size_t i = 0; i < y->size; i++) {
        y->data[i] = (y->data[i] - mean) / (std + 1e-8);
    }
}

// Forward propagation structures
typedef struct {
    double* base_outputs;    // Outputs from base models
    double* base_raw;        // Pre-activation base outputs
    double* meta_outputs;    // Outputs from meta models
    double* meta_raw;        // Pre-activation meta outputs
    double final_output;     // Final prediction
    double final_raw;        // Pre-activation final output
} ForwardPass;

ForwardPass* create_forward_pass() {
    ForwardPass* fp = malloc(sizeof(ForwardPass));
    fp->base_outputs = malloc(NUM_BASE_MODELS * sizeof(double));
    fp->base_raw = malloc(NUM_BASE_MODELS * sizeof(double));
    fp->meta_outputs = malloc(NUM_META_MODELS * sizeof(double));
    fp->meta_raw = malloc(NUM_META_MODELS * sizeof(double));
    return fp;
}

void free_forward_pass(ForwardPass* fp) {
    free(fp->base_outputs);
    free(fp->base_raw);
    free(fp->meta_outputs);
    free(fp->meta_raw);
    free(fp);
}

// Activation function
double relu(double x) {
    return fmax(0, x);
}

// Forward propagation
ForwardPass* forward_propagate(EnsembleModel* model, const double* input) {
    ForwardPass* fp = create_forward_pass();
    
    // Base models forward pass
    for (size_t i = 0; i < model->num_base; i++) {
        Layer* layer = model->base_layers[i];
        double sum = layer->bias;
        
        for (size_t j = 0; j < model->features; j++) {
            sum += input[j] * layer->weights[j];
        }
        
        fp->base_raw[i] = sum;
        fp->base_outputs[i] = relu(sum);
    }
    
    // Meta models forward pass
    for (size_t i = 0; i < model->num_meta; i++) {
        Layer* layer = model->meta_layers[i];
        double sum = layer->bias;
        
        for (size_t j = 0; j < model->num_base; j++) {
            sum += fp->base_outputs[j] * layer->weights[j];
        }
        
        fp->meta_raw[i] = sum;
        fp->meta_outputs[i] = relu(sum);
    }
    
    // Final layer forward pass
    double sum = model->final_layer->bias;
    for (size_t i = 0; i < model->num_meta; i++) {
        sum += fp->meta_outputs[i] * model->final_layer->weights[i];
    }
    
    fp->final_raw = sum;
    fp->final_output = sum;  // Linear activation for final layer
    
    return fp;
}

// AdamW update function
void adam_update(double* weight, double* m, double* v, double grad, double lr, 
                double beta1, double beta2, double epsilon, double weight_decay, int t) {
    *m = beta1 * (*m) + (1 - beta1) * grad;
    *v = beta2 * (*v) + (1 - beta2) * grad * grad;
    
    double m_hat = *m / (1 - pow(beta1, t));
    double v_hat = *v / (1 - pow(beta2, t));
    
    *weight *= (1 - lr * weight_decay);
    *weight -= lr * m_hat / (sqrt(v_hat) + epsilon);
}

// Training function
void train_ensemble(EnsembleModel* model, Matrix* X, Vector* y, int epochs) {
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    const double weight_decay = 0.01;
    int t = 0;  // Adam time step
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        // Create random batch indices
        int* indices = malloc(X->rows * sizeof(int));
        for (int i = 0; i < X->rows; i++) indices[i] = i;
        for (int i = X->rows - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Mini-batch training
        for (int batch_start = 0; batch_start < X->rows; batch_start += BATCH_SIZE) {
            t++;
            int batch_end = fmin(batch_start + BATCH_SIZE, X->rows);
            int batch_size = batch_end - batch_start;
            
            // Accumulate gradients over batch
            double** base_grads = malloc(NUM_BASE_MODELS * sizeof(double*));
            double* base_bias_grads = calloc(NUM_BASE_MODELS, sizeof(double));
            double** meta_grads = malloc(NUM_META_MODELS * sizeof(double*));
            double* meta_bias_grads = calloc(NUM_META_MODELS, sizeof(double));
            double* final_grads = calloc(NUM_META_MODELS, sizeof(double));
            double final_bias_grad = 0.0;
            
            for (int i = 0; i < NUM_BASE_MODELS; i++) {
                base_grads[i] = calloc(FEATURES, sizeof(double));
            }
            for (int i = 0; i < NUM_META_MODELS; i++) {
                meta_grads[i] = calloc(NUM_BASE_MODELS, sizeof(double));
            }
            
            // Process each sample in batch
            for (int b = batch_start; b < batch_end; b++) {
                int idx = indices[b];
                ForwardPass* fp = forward_propagate(model, &X->data[idx * X->cols]);
                
                // Compute prediction error
                double error = fp->final_output - y->data[idx];
                total_loss += error * error;
                
                // Backward propagation
                // Final layer gradients
                for (int i = 0; i < NUM_META_MODELS; i++) {
                    final_grads[i] += error * fp->meta_outputs[i];
                }
                final_bias_grad += error;
                
                // Meta layer gradients
                for (int i = 0; i < NUM_META_MODELS; i++) {
                    if (fp->meta_raw[i] <= 0) continue;  // ReLU derivative
                    double meta_error = error * model->final_layer->weights[i];
                    
                    for (int j = 0; j < NUM_BASE_MODELS; j++) {
                        meta_grads[i][j] += meta_error * fp->base_outputs[j];
                    }
                    meta_bias_grads[i] += meta_error;
                }
                
                // Base layer gradients
                for (int i = 0; i < NUM_BASE_MODELS; i++) {
                    if (fp->base_raw[i] <= 0) continue;  // ReLU derivative
                    
                    double base_error = 0;
                    for (int j = 0; j < NUM_META_MODELS; j++) {
                        if (fp->meta_raw[j] > 0) {
                            base_error += error * model->final_layer->weights[j] * 
                                        model->meta_layers[j]->weights[i];
                        }
                    }
                    
                    for (int j = 0; j < FEATURES; j++) {
                        base_grads[i][j] += base_error * X->data[idx * X->cols + j];
                    }
                    base_bias_grads[i] += base_error;
                }
                
                free_forward_pass(fp);
            }
            
            // Update parameters using AdamW
            // Base layers
            for (int i = 0; i < NUM_BASE_MODELS; i++) {
                Layer* layer = model->base_layers[i];
                for (int j = 0; j < FEATURES; j++) {
                    adam_update(&layer->weights[j], &layer->m_weights[j], &layer->v_weights[j],
                              base_grads[i][j] / batch_size, BASE_LR, beta1, beta2, 
                              epsilon, weight_decay, t);
                }
                adam_update(&layer->bias, &layer->m_bias, &layer->v_bias,
                          base_bias_grads[i] / batch_size, BASE_LR, beta1, beta2,
                          epsilon, weight_decay, t);
            }
            
            // Meta layers
            for (int i = 0; i < NUM_META_MODELS; i++) {
                Layer* layer = model->meta_layers[i];
                for (int j = 0; j < NUM_BASE_MODELS; j++) {
                    adam_update(&layer->weights[j], &layer->m_weights[j], &layer->v_weights[j],
                              meta_grads[i][j] / batch_size, META_LR, beta1, beta2,
                              epsilon, weight_decay, t);
                }
                adam_update(&layer->bias, &layer->m_bias, &layer->v_bias,
                          meta_bias_grads[i] / batch_size, META_LR, beta1, beta2,
                          epsilon, weight_decay, t);
            }
            
            // Final layer
            for (int i = 0; i < NUM_META_MODELS; i++) {
                adam_update(&model->final_layer->weights[i], 
                          &model->final_layer->m_weights[i],
                          &model->final_layer->v_weights[i],
                          final_grads[i] / batch_size, FINAL_LR, beta1, beta2,
                          epsilon, weight_decay, t);
            }
            adam_update(&model->final_layer->bias, &model->final_layer->m_bias,
                      &model->final_layer->v_bias, final_bias_grad / batch_size,
                      FINAL_LR, beta1, beta2, epsilon, weight_decay, t);
            
            // Cleanup batch gradients
            for (int i = 0; i < NUM_BASE_MODELS; i++) free(base_grads[i]);
            for (int i = 0; i < NUM_META_MODELS; i++) free(meta_grads[i]);
            free(base_grads);
            free(meta_grads);
            free(base_bias_grads);
            free(meta_bias_grads);
            free(final_grads);
        }
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %d/%d - MSE: %.6f\n", 
                   epoch + 1, epochs, total_loss / X->rows);
        }
        
        free(indices);
    }
}

// Prediction function
double predict(EnsembleModel* model, const double* input) {
    ForwardPass* fp = forward_propagate(model, input);
    double prediction = fp->final_output;
    free_forward_pass(fp);
    return prediction;
}

int main() {
    srand(time(NULL));
    
    // Create and prepare data
    Matrix X_train = create_matrix(404, 13);
    Matrix X_test = create_matrix(102, 13);
    Vector y_train = create_vector(404);
    Vector y_test = create_vector(102);
    
    // Copy data and create train/test split
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
        y_train.data[i] = y[indices[i]];
    }
    
    for (int i = 0; i < 102; i++) {
        for (int j = 0; j < 13; j++) {
            X_test.data[i * 13 + j] = X[indices[i + 404]][j];
        }
        y_test.data[i] = y[indices[i + 404]];
    }
    
    // Save statistics for denormalization
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < 506; i++) y_mean += y[i];
    y_mean /= 506;
    for (int i = 0; i < 506; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / 506);
    
    // Normalize data
    normalize_data(&X_train, &y_train);
    normalize_data(&X_test, &y_test);
    
    // Create and train model
    printf("Creating and training ensemble...\n");
    EnsembleModel* model = create_ensemble();
    train_ensemble(model, &X_train, &y_train, EPOCHS);
    
    // Evaluate
    int train_correct = 0, test_correct = 0;
    
    for (int i = 0; i < 404; i++) {
        double pred = predict(model, &X_train.data[i * 13]);
        double true_val = y_train.data[i] * y_std + y_mean;
        double pred_val = pred * y_std + y_mean;
        if (fabs(true_val - pred_val) <= 3.0) train_correct++;
    }
    
    for (int i = 0; i < 102; i++) {
        double pred = predict(model, &X_test.data[i * 13]);
        double true_val = y_test.data[i] * y_std + y_mean;
        double pred_val = pred * y_std + y_mean;
        if (fabs(true_val - pred_val) <= 3.0) test_correct++;
    }
    
    printf("\nFinal Results:\n");
    printf("Training Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n",
           (double)train_correct/404 * 100, train_correct, 404);
    printf("Test Accuracy (within ±3.0): %.2f%% (%d/%d correct)\n",
           (double)test_correct/102 * 100, test_correct, 102);
    
    // Cleanup
    free_ensemble(model);
    free_matrix(&X_train);
    free_matrix(&X_test);
    free_vector(&y_train);
    free_vector(&y_test);
    free(indices);
    
    return 0;
}