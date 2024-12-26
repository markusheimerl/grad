#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data/boston_housing_dataset.h"

#define NUM_BASE 256
#define NUM_META 64
#define FEATURES 13
#define EPOCHS 1000
#define BATCH 32
#define LR_BASE 0.001
#define LR_META 0.0005
#define LR_FINAL 0.0001

typedef struct {
    double *w, *m, *v;
    double bias, m_bias, v_bias;
    size_t in_size, out_size;
} Layer;

typedef struct {
    Layer **base, **meta, *final;
    size_t n_base, n_meta, feats;
} Model;

typedef struct {
    double *data;
    size_t rows, cols;
} Matrix;

typedef struct {
    double *base_out, *base_raw;
    double *meta_out, *meta_raw;
    double final_out, final_raw;
} Forward;

Layer* create_layer(size_t in_size, size_t out_size) {
    Layer* l = malloc(sizeof(Layer));
    double scale = sqrt(2.0 / in_size);
    
    l->w = malloc(in_size * sizeof(double));
    l->m = calloc(in_size, sizeof(double));
    l->v = calloc(in_size, sizeof(double));
    
    for (size_t i = 0; i < in_size; i++)
        l->w[i] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
    l->bias = ((double)rand()/RAND_MAX * 2 - 1) * scale;
    l->m_bias = 0;
    l->v_bias = 0;
    l->in_size = in_size;
    l->out_size = out_size;
    return l;
}

Model* create_model() {
    Model* m = malloc(sizeof(Model));
    m->base = malloc(NUM_BASE * sizeof(Layer*));
    m->meta = malloc(NUM_META * sizeof(Layer*));
    
    for (int i = 0; i < NUM_BASE; i++) 
        m->base[i] = create_layer(FEATURES, 1);
    for (int i = 0; i < NUM_META; i++) 
        m->meta[i] = create_layer(NUM_BASE, 1);
    
    m->final = create_layer(NUM_META, 1);
    m->n_base = NUM_BASE;
    m->n_meta = NUM_META;
    m->feats = FEATURES;
    return m;
}

Forward* forward(Model* m, const double* input) {
    Forward* f = malloc(sizeof(Forward));
    f->base_out = malloc(NUM_BASE * sizeof(double));
    f->base_raw = malloc(NUM_BASE * sizeof(double));
    f->meta_out = malloc(NUM_META * sizeof(double));
    f->meta_raw = malloc(NUM_META * sizeof(double));
    
    // Base models
    for (size_t i = 0; i < m->n_base; i++) {
        double sum = m->base[i]->bias;
        for (size_t j = 0; j < m->feats; j++)
            sum += input[j] * m->base[i]->w[j];
        f->base_raw[i] = sum;
        f->base_out[i] = fmax(0, sum);
    }
    
    // Meta models
    for (size_t i = 0; i < m->n_meta; i++) {
        double sum = m->meta[i]->bias;
        for (size_t j = 0; j < m->n_base; j++)
            sum += f->base_out[j] * m->meta[i]->w[j];
        f->meta_raw[i] = sum;
        f->meta_out[i] = fmax(0, sum);
    }
    
    // Final layer
    f->final_raw = m->final->bias;
    for (size_t i = 0; i < m->n_meta; i++)
        f->final_raw += f->meta_out[i] * m->final->w[i];
    f->final_out = f->final_raw;
    
    return f;
}

void adam_update(double* w, double* m, double* v, double g, double lr, int t) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
    *m = b1 * (*m) + (1 - b1) * g;
    *v = b2 * (*v) + (1 - b2) * g * g;
    double m_hat = *m / (1 - pow(b1, t));
    double v_hat = *v / (1 - pow(b2, t));
    *w *= (1 - lr * wd);
    *w -= lr * m_hat / (sqrt(v_hat) + eps);
}

void train(Model* m, Matrix* X, double* y) {
    int *indices = malloc(X->rows * sizeof(int));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0;
        
        // Shuffle indices
        for (size_t i = 0; i < X->rows; i++) indices[i] = i;
        for (size_t i = X->rows - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Mini-batch training
        for (size_t b = 0; b < X->rows; b += BATCH) {
            size_t batch_end = (b + BATCH < X->rows) ? b + BATCH : X->rows;
            size_t batch_size = batch_end - b;
            
            // Accumulate gradients
            double **base_grads = malloc(NUM_BASE * sizeof(double*));
            double *base_bias_grads = calloc(NUM_BASE, sizeof(double));
            double **meta_grads = malloc(NUM_META * sizeof(double*));
            double *meta_bias_grads = calloc(NUM_META, sizeof(double));
            double *final_grads = calloc(NUM_META, sizeof(double));
            double final_bias_grad = 0;
            
            for (int i = 0; i < NUM_BASE; i++)
                base_grads[i] = calloc(FEATURES, sizeof(double));
            for (int i = 0; i < NUM_META; i++)
                meta_grads[i] = calloc(NUM_BASE, sizeof(double));
            
            for (size_t i = b; i < batch_end; i++) {
                size_t idx = indices[i];
                Forward* f = forward(m, &X->data[idx * X->cols]);
                double error = f->final_out - y[idx];
                loss += error * error;
                
                // Final layer gradients
                for (int j = 0; j < NUM_META; j++) {
                    final_grads[j] += error * f->meta_out[j];
                }
                final_bias_grad += error;
                
                // Meta layer gradients
                for (int j = 0; j < NUM_META; j++) {
                    if (f->meta_raw[j] <= 0) continue;
                    double meta_error = error * m->final->w[j];
                    
                    for (int k = 0; k < NUM_BASE; k++) {
                        meta_grads[j][k] += meta_error * f->base_out[k];
                    }
                    meta_bias_grads[j] += meta_error;
                }
                
                // Base layer gradients
                for (int j = 0; j < NUM_BASE; j++) {
                    if (f->base_raw[j] <= 0) continue;
                    double base_error = 0;
                    
                    for (int k = 0; k < NUM_META; k++) {
                        if (f->meta_raw[k] > 0) {
                            base_error += error * m->final->w[k] * m->meta[k]->w[j];
                        }
                    }
                    
                    for (int k = 0; k < FEATURES; k++) {
                        base_grads[j][k] += base_error * X->data[idx * X->cols + k];
                    }
                    base_bias_grads[j] += base_error;
                }
                
                free(f->base_out);
                free(f->base_raw);
                free(f->meta_out);
                free(f->meta_raw);
                free(f);
            }
            
            // Apply updates
            int t = epoch * X->rows + b + 1;
            
            for (int i = 0; i < NUM_BASE; i++) {
                for (int j = 0; j < FEATURES; j++) {
                    adam_update(&m->base[i]->w[j], &m->base[i]->m[j], &m->base[i]->v[j],
                              base_grads[i][j] / batch_size, LR_BASE, t);
                }
                adam_update(&m->base[i]->bias, &m->base[i]->m_bias, &m->base[i]->v_bias,
                          base_bias_grads[i] / batch_size, LR_BASE, t);
            }
            
            for (int i = 0; i < NUM_META; i++) {
                for (int j = 0; j < NUM_BASE; j++) {
                    adam_update(&m->meta[i]->w[j], &m->meta[i]->m[j], &m->meta[i]->v[j],
                              meta_grads[i][j] / batch_size, LR_META, t);
                }
                adam_update(&m->meta[i]->bias, &m->meta[i]->m_bias, &m->meta[i]->v_bias,
                          meta_bias_grads[i] / batch_size, LR_META, t);
            }
            
            for (int i = 0; i < NUM_META; i++) {
                adam_update(&m->final->w[i], &m->final->m[i], &m->final->v[i],
                          final_grads[i] / batch_size, LR_FINAL, t);
            }
            adam_update(&m->final->bias, &m->final->m_bias, &m->final->v_bias,
                      final_bias_grad / batch_size, LR_FINAL, t);
            
            // Cleanup
            for (int i = 0; i < NUM_BASE; i++) free(base_grads[i]);
            for (int i = 0; i < NUM_META; i++) free(meta_grads[i]);
            free(base_grads);
            free(meta_grads);
            free(base_bias_grads);
            free(meta_bias_grads);
            free(final_grads);
        }
        
        if ((epoch + 1) % 100 == 0)
            printf("Epoch %d/%d - MSE: %.6f\n", epoch + 1, EPOCHS, loss / X->rows);
    }
    
    free(indices);
}

double predict(Model* m, const double* input) {
    Forward* f = forward(m, input);
    double pred = f->final_out;
    free(f->base_out);
    free(f->base_raw);
    free(f->meta_out);
    free(f->meta_raw);
    free(f);
    return pred;
}

void normalize(Matrix* X, double* y, size_t y_size) {
    for (size_t j = 0; j < X->cols; j++) {
        double mean = 0, std = 0;
        for (size_t i = 0; i < X->rows; i++) 
            mean += X->data[i * X->cols + j];
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

int main() {
    srand(time(NULL));
    
    // Calculate global mean and std for y
    double y_mean = 0, y_std = 0;
    for (int i = 0; i < 506; i++) y_mean += y[i];
    y_mean /= 506;
    for (int i = 0; i < 506; i++) y_std += pow(y[i] - y_mean, 2);
    y_std = sqrt(y_std / 506);
    
    Matrix X_train = {malloc(404 * 13 * sizeof(double)), 404, 13};
    Matrix X_test = {malloc(102 * 13 * sizeof(double)), 102, 13};
    double *y_train = malloc(404 * sizeof(double));
    double *y_test = malloc(102 * sizeof(double));
    
    // Split data
    int* indices = malloc(506 * sizeof(int));
    for (int i = 0; i < 506; i++) indices[i] = i;
    for (int i = 505; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    for (int i = 0; i < 404; i++) {
        for (int j = 0; j < 13; j++)
            X_train.data[i * 13 + j] = X[indices[i]][j];
        y_train[i] = y[indices[i]];
    }
    
    for (int i = 0; i < 102; i++) {
        for (int j = 0; j < 13; j++)
            X_test.data[i * 13 + j] = X[indices[i + 404]][j];
        y_test[i] = y[indices[i + 404]];
    }
    
    normalize(&X_train, y_train, 404);
    normalize(&X_test, y_test, 102);
    
    printf("Training model...\n");
    Model* model = create_model();
    train(model, &X_train, y_train);
    
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
    
    printf("\nResults:\n");
    printf("Train Accuracy: %.2f%% (%d/%d)\n", (double)train_correct/404*100, train_correct, 404);
    printf("Test Accuracy: %.2f%% (%d/%d)\n", (double)test_correct/102*100, test_correct, 102);
    
    free(X_train.data);
    free(X_test.data);
    free(y_train);
    free(y_test);
    free(indices);
    
    return 0;
}