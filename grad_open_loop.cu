#include "data_open_loop.cuh"
#include <time.h>

#define LEARNING_RATE 0.001
#define NUM_EPOCHS 1000

typedef struct {
    double ***W;     // [output_features][window_size][input_features]
    double **V;      // [output_features][window_size] - temporal weights
    double *b;       // [output_features]
} TemporalModel;

// Initialize model parameters
TemporalModel* init_model(int window_size, int input_features, int output_features) {
    TemporalModel* model = (TemporalModel*)malloc(sizeof(TemporalModel));
    
    // Initialize weights for each timestep
    model->W = (double***)malloc(output_features * sizeof(double**));
    for(int i = 0; i < output_features; i++) {
        model->W[i] = (double**)malloc(window_size * sizeof(double*));
        for(int t = 0; t < window_size; t++) {
            model->W[i][t] = (double*)malloc(input_features * sizeof(double));
            for(int j = 0; j < input_features; j++) {
                model->W[i][t][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            }
        }
    }
    
    // Initialize temporal weights with exponential decay
    model->V = (double**)malloc(output_features * sizeof(double*));
    for(int i = 0; i < output_features; i++) {
        model->V[i] = (double*)malloc(window_size * sizeof(double));
        for(int t = 0; t < window_size; t++) {
            model->V[i][t] = exp(-0.5 * (window_size - 1 - t));
        }
    }
    
    model->b = (double*)malloc(output_features * sizeof(double));
    for(int i = 0; i < output_features; i++) {
        model->b[i] = 0.0;
    }
    
    return model;
}

// Forward pass
void forward(TemporalModel* model, double** x, double* y_pred,
            int window_size, int input_features, int output_features) {
    for(int i = 0; i < output_features; i++) {
        y_pred[i] = model->b[i];
        
        // Compute weighted sum across all timesteps
        for(int t = 0; t < window_size; t++) {
            double timestep_contrib = 0.0;
            for(int j = 0; j < input_features; j++) {
                timestep_contrib += model->W[i][t][j] * x[t][j];
            }
            y_pred[i] += model->V[i][t] * timestep_contrib;
        }
    }
}

// Mean squared error loss
double compute_loss(double* y_pred, double* y_true, int output_features) {
    double loss = 0.0;
    for(int i = 0; i < output_features; i++) {
        double diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / output_features;
}

// Update model parameters
void update_parameters(TemporalModel* model, double** x, double* y_pred, 
                      double* y_true, int window_size, int input_features, 
                      int output_features, double learning_rate) {
    for(int i = 0; i < output_features; i++) {
        double diff = y_pred[i] - y_true[i];
        
        // Update bias
        model->b[i] -= learning_rate * diff;
        
        // Update weights
        for(int t = 0; t < window_size; t++) {
            for(int j = 0; j < input_features; j++) {
                model->W[i][t][j] -= learning_rate * diff * 
                                    model->V[i][t] * x[t][j];
            }
        }
    }
}

void free_model(TemporalModel* model, int output_features, int window_size) {
    for(int i = 0; i < output_features; i++) {
        for(int t = 0; t < window_size; t++) {
            free(model->W[i][t]);
        }
        free(model->W[i]);
        free(model->V[i]);
    }
    free(model->W);
    free(model->V);
    free(model->b);
    free(model);
}

int main() {
    srand(time(NULL));
    
    // Generate synthetic data
    OpenLoopData* data = generate_open_loop_data(1000, 50, 3, 2, 0.1);

    time_t current_time = time(NULL);
    struct tm* timeinfo = localtime(&current_time);
    char data_fname[64];
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_open_loop_data.csv", timeinfo);
    
    save_open_loop_csv(data_fname, data);
    printf("Data saved to: %s\n", data_fname);

    // Initialize model
    TemporalModel* model = init_model(data->window_size, 
                                    data->input_features,
                                    data->output_features);
    
    double* y_pred = (double*)malloc(data->output_features * sizeof(double));
    
    // Training loop
    printf("Training started...\n");
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        
        for(int i = 0; i < data->n; i++) {
            forward(model, data->windows[i], y_pred,
                   data->window_size, data->input_features,
                   data->output_features);
            
            double loss = compute_loss(y_pred, data->outputs[i],
                                     data->output_features);
            epoch_loss += loss;
            
            update_parameters(model, data->windows[i], y_pred,
                            data->outputs[i], data->window_size,
                            data->input_features, data->output_features,
                            LEARNING_RATE);
        }
        
        epoch_loss /= data->n;
        if(epoch % 100 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, epoch_loss);
        }
    }
    
    // Print temporal weights for first output
    printf("\nTemporal weights for first output:\n");
    for(int t = 0; t < data->window_size; t++) {
        printf("t-%d: %.4f\n", data->window_size - 1 - t, model->V[0][t]);
    }
    
    // Print example predictions
    printf("\nExample predictions:\n");
    for(int i = 0; i < 5; i++) {
        forward(model, data->windows[i], y_pred,
               data->window_size, data->input_features,
               data->output_features);
        
        printf("Sample %d:\n", i);
        printf("  True:");
        for(int j = 0; j < data->output_features; j++) {
            printf(" %.6f", data->outputs[i][j]);
        }
        printf("\n  Pred:");
        for(int j = 0; j < data->output_features; j++) {
            printf(" %.6f", y_pred[j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(y_pred);
    free_model(model, data->output_features, data->window_size);
    free_open_loop_data(data);
    
    return 0;
}