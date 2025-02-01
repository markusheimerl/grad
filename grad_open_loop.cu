#include "data_open_loop.cuh"
#include <time.h>

#define LEARNING_RATE 0.001
#define NUM_EPOCHS 1000
#define NUM_FILTERS 16
#define KERNEL_SIZE 5

typedef struct {
    // Convolutional layer parameters
    double ***conv_filters;  // [num_filters][kernel_size][input_features]
    double *conv_bias;       // [num_filters]
    
    // Fully connected layer parameters
    double **fc_weights;     // [output_features][num_filters]
    double *fc_bias;         // [output_features]
    
    // Hyperparameters
    int num_filters;
    int kernel_size;
} ConvModel;

// Helper function for 2D array allocation
double** malloc_2d(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        arr[i] = (double*)calloc(cols, sizeof(double));
    }
    return arr;
}

// Helper function for 2D array deallocation
void free_2d(double** arr, int rows) {
    for(int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// He initialization
double he_init() {
    double rand_normal = sqrt(-2.0 * log((double)rand() / RAND_MAX)) * 
                        cos(2.0 * M_PI * (double)rand() / RAND_MAX);
    return rand_normal * sqrt(2.0 / KERNEL_SIZE);
}

// ReLU activation function
static double relu(double x) {
    return x > 0 ? x : 0;
}

// ReLU derivative
static double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Initialize model parameters
ConvModel* init_model(int input_features, int output_features) {
    ConvModel* model = (ConvModel*)malloc(sizeof(ConvModel));
    model->num_filters = NUM_FILTERS;
    model->kernel_size = KERNEL_SIZE;
    
    // Initialize conv filters
    model->conv_filters = (double***)malloc(NUM_FILTERS * sizeof(double**));
    for(int f = 0; f < NUM_FILTERS; f++) {
        model->conv_filters[f] = malloc_2d(KERNEL_SIZE, input_features);
        for(int k = 0; k < KERNEL_SIZE; k++) {
            for(int c = 0; c < input_features; c++) {
                model->conv_filters[f][k][c] = he_init();
            }
        }
    }
    
    // Initialize conv bias
    model->conv_bias = (double*)calloc(NUM_FILTERS, sizeof(double));
    
    // Initialize FC weights
    model->fc_weights = malloc_2d(output_features, NUM_FILTERS);
    for(int i = 0; i < output_features; i++) {
        for(int f = 0; f < NUM_FILTERS; f++) {
            model->fc_weights[i][f] = he_init();
        }
    }
    
    // Initialize FC bias
    model->fc_bias = (double*)calloc(output_features, sizeof(double));
    
    return model;
}

// Forward pass
void forward(ConvModel* model, double** x, double* y_pred,
            int window_size, int input_features, int output_features,
            double** conv_out) {
    int conv_out_size = window_size - model->kernel_size + 1;
    
    // Convolution
    for(int f = 0; f < model->num_filters; f++) {
        for(int t = 0; t < conv_out_size; t++) {
            conv_out[t][f] = model->conv_bias[f];
            
            for(int k = 0; k < model->kernel_size; k++) {
                for(int c = 0; c < input_features; c++) {
                    conv_out[t][f] += model->conv_filters[f][k][c] * 
                                    x[t + k][c];
                }
            }
            conv_out[t][f] = relu(conv_out[t][f]);
        }
    }
    
    // Global average pooling
    double* pooled = (double*)calloc(model->num_filters, sizeof(double));
    for(int f = 0; f < model->num_filters; f++) {
        for(int t = 0; t < conv_out_size; t++) {
            pooled[f] += conv_out[t][f];
        }
        pooled[f] /= conv_out_size;
    }
    
    // Fully connected layer
    for(int i = 0; i < output_features; i++) {
        y_pred[i] = model->fc_bias[i];
        for(int f = 0; f < model->num_filters; f++) {
            y_pred[i] += model->fc_weights[i][f] * pooled[f];
        }
    }
    
    free(pooled);
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

// Backward pass and parameter update
void update_parameters(ConvModel* model, double** x, double** conv_out,
                      double* y_pred, double* y_true,
                      int window_size, int input_features, int output_features,
                      double learning_rate) {
    int conv_out_size = window_size - model->kernel_size + 1;
    
    // Compute gradients for FC layer
    double* d_pooled = (double*)calloc(model->num_filters, sizeof(double));
    
    for(int i = 0; i < output_features; i++) {
        double d_output = 2 * (y_pred[i] - y_true[i]) / output_features;
        
        // Update FC bias
        model->fc_bias[i] -= learning_rate * d_output;
        
        // Update FC weights and compute gradients for pooled features
        for(int f = 0; f < model->num_filters; f++) {
            double pooled = 0;
            for(int t = 0; t < conv_out_size; t++) {
                pooled += conv_out[t][f];
            }
            pooled /= conv_out_size;
            
            model->fc_weights[i][f] -= learning_rate * d_output * pooled;
            d_pooled[f] += d_output * model->fc_weights[i][f];
        }
    }
    
    // Compute gradients for conv layer
    for(int f = 0; f < model->num_filters; f++) {
        double d_pool = d_pooled[f] / conv_out_size;
        
        for(int t = 0; t < conv_out_size; t++) {
            double d_relu = relu_derivative(conv_out[t][f]);
            double d_conv = d_pool * d_relu;
            
            // Update conv bias
            model->conv_bias[f] -= learning_rate * d_conv;
            
            // Update conv filters
            for(int k = 0; k < model->kernel_size; k++) {
                for(int c = 0; c < input_features; c++) {
                    model->conv_filters[f][k][c] -= learning_rate * d_conv * 
                                                   x[t + k][c];
                }
            }
        }
    }
    
    free(d_pooled);
}

void free_model(ConvModel* model, int output_features) {
    for(int f = 0; f < model->num_filters; f++) {
        free_2d(model->conv_filters[f], model->kernel_size);
    }
    free(model->conv_filters);
    free(model->conv_bias);
    free_2d(model->fc_weights, output_features);
    free(model->fc_bias);
    free(model);
}

int main() {
    srand(time(NULL));
    
    // Generate synthetic data
    OpenLoopData* data = generate_open_loop_data(1000, 50, 3, 2, 0.1);

    // Save data to CSV
    time_t current_time = time(NULL);
    struct tm* timeinfo = localtime(&current_time);
    char data_fname[64];
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_open_loop_data.csv", timeinfo);
    save_open_loop_csv(data_fname, data);
    printf("Data saved to: %s\n", data_fname);

    // Initialize model
    ConvModel* model = init_model(data->input_features, data->output_features);
    
    double* y_pred = (double*)malloc(data->output_features * sizeof(double));
    double** conv_out = malloc_2d(data->window_size - KERNEL_SIZE + 1, NUM_FILTERS);
    
    // Training loop
    printf("Training started...\n");
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        
        for(int i = 0; i < data->n; i++) {
            forward(model, data->windows[i], y_pred,
                   data->window_size, data->input_features,
                   data->output_features, conv_out);
            
            double loss = compute_loss(y_pred, data->outputs[i],
                                     data->output_features);
            epoch_loss += loss;
            
            update_parameters(model, data->windows[i], conv_out,
                            y_pred, data->outputs[i],
                            data->window_size, data->input_features,
                            data->output_features, LEARNING_RATE);
        }
        
        epoch_loss /= data->n;
        if(epoch % 100 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, epoch_loss);
        }
    }
    
    // Print example predictions
    printf("\nExample predictions:\n");
    for(int i = 0; i < 5; i++) {
        forward(model, data->windows[i], y_pred,
               data->window_size, data->input_features,
               data->output_features, conv_out);
        
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
    free_2d(conv_out, data->window_size - KERNEL_SIZE + 1);
    free_model(model, data->output_features);
    free_open_loop_data(data);
    
    return 0;
}