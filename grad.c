#include "data.h"
#include <time.h>

typedef struct {
    double *x;      // layer values
    double *dx;     // gradients
    int size;       // size of this layer
} Layer;

typedef struct {
    Layer *layers;  // array of layers
    double **W;     // weights between layers
    double **dW;    // weight gradients
    double **b;     // biases for each layer
    double **db;    // bias gradients
    double **m_W;   // momentum for weights (AdamW)
    double **v_W;   // velocity for weights (AdamW)
    double **m_b;   // momentum for biases (AdamW)
    double **v_b;   // velocity for biases (AdamW)
    int n_layers;   // number of layers
    int t;          // timestep for AdamW
} Net;

// Initialize network with given architecture
Net* init_net(int n_layers, int* sizes) {
    Net* net = malloc(sizeof(Net));
    net->n_layers = n_layers;
    net->t = 1;  // Initialize timestep
    net->layers = malloc(n_layers * sizeof(Layer));
    
    // Initialize layers
    for(int i = 0; i < n_layers; i++) {
        net->layers[i].size = sizes[i];
        net->layers[i].x = calloc(sizes[i], sizeof(double));
        net->layers[i].dx = calloc(sizes[i], sizeof(double));
    }
    
    // Initialize weights and biases
    net->W = malloc((n_layers-1) * sizeof(double*));
    net->dW = malloc((n_layers-1) * sizeof(double*));
    net->b = malloc((n_layers-1) * sizeof(double*));
    net->db = malloc((n_layers-1) * sizeof(double*));
    net->m_W = malloc((n_layers-1) * sizeof(double*));
    net->v_W = malloc((n_layers-1) * sizeof(double*));
    net->m_b = malloc((n_layers-1) * sizeof(double*));
    net->v_b = malloc((n_layers-1) * sizeof(double*));
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        net->W[i] = malloc(rows * cols * sizeof(double));
        net->dW[i] = calloc(rows * cols, sizeof(double));
        net->b[i] = malloc(rows * sizeof(double));
        net->db[i] = calloc(rows, sizeof(double));
        net->m_W[i] = calloc(rows * cols, sizeof(double));
        net->v_W[i] = calloc(rows * cols, sizeof(double));
        net->m_b[i] = calloc(rows, sizeof(double));
        net->v_b[i] = calloc(rows, sizeof(double));
        
        // Xavier initialization
        double scale = sqrt(2.0 / (sizes[i] + sizes[i+1]));
        for(int j = 0; j < rows * cols; j++)
            net->W[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        for(int j = 0; j < rows; j++)
            net->b[i][j] = 0;
    }
    return net;
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Forward pass
void forward(Net* net, double* input) {
    // Set input layer
    memcpy(net->layers[0].x, input, net->layers[0].size * sizeof(double));
    
    // Forward propagation
    for(int i = 0; i < net->n_layers-1; i++) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i+1];
        
        // Reset next layer
        memset(next->x, 0, next->size * sizeof(double));
        
        // Compute weighted sum
        for(int j = 0; j < next->size; j++) {
            for(int k = 0; k < curr->size; k++) {
                next->x[j] += curr->x[k] * net->W[i][j * curr->size + k];
            }
            next->x[j] += net->b[i][j];
            // Apply ReLU activation (except for output layer)
            if(i < net->n_layers-2) {
                next->x[j] = relu(next->x[j]);
            }
        }
    }
}

// AdamW update function
void adamw_update(Net* net, double learning_rate) {
    const double beta1 = 0.9;    // Momentum factor
    const double beta2 = 0.999;  // Velocity factor
    const double eps = 1e-8;     // Small constant for numerical stability
    const double weight_decay = 0.01; // L2 regularization factor
    
    for(int i = 0; i < net->n_layers-1; i++) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i+1];
        int w_size = next->size * curr->size;
        
        // Update weights
        for(int j = 0; j < w_size; j++) {
            // Momentum update
            net->m_W[i][j] = beta1 * net->m_W[i][j] + (1 - beta1) * net->dW[i][j];
            // Velocity update
            net->v_W[i][j] = beta2 * net->v_W[i][j] + (1 - beta2) * net->dW[i][j] * net->dW[i][j];
            
            // Bias correction
            double m_hat = net->m_W[i][j] / (1 - pow(beta1, net->t));
            double v_hat = net->v_W[i][j] / (1 - pow(beta2, net->t));
            
            // AdamW update (including weight decay)
            net->W[i][j] -= learning_rate * (m_hat / (sqrt(v_hat) + eps) + 
                                           weight_decay * net->W[i][j]);
        }
        
        // Update biases
        for(int j = 0; j < next->size; j++) {
            net->m_b[i][j] = beta1 * net->m_b[i][j] + (1 - beta1) * net->db[i][j];
            net->v_b[i][j] = beta2 * net->v_b[i][j] + (1 - beta2) * net->db[i][j] * net->db[i][j];
            
            double m_hat = net->m_b[i][j] / (1 - pow(beta1, net->t));
            double v_hat = net->v_b[i][j] / (1 - pow(beta2, net->t));
            
            net->b[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + eps);
        }
    }
    net->t++;  // Increment time step
}

// Backward pass
void backward(Net* net, double* target, double learning_rate) {
    int last = net->n_layers-1;
    
    // Compute output layer error
    for(int i = 0; i < net->layers[last].size; i++) {
        net->layers[last].dx[i] = net->layers[last].x[i] - target[i];
    }
    
    // Backward propagation
    for(int i = last-1; i >= 0; i--) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i+1];
        
        // Compute gradients for weights and biases
        for(int j = 0; j < next->size; j++) {
            for(int k = 0; k < curr->size; k++) {
                net->dW[i][j * curr->size + k] = next->dx[j] * curr->x[k];
            }
            net->db[i][j] = next->dx[j];
        }
        
        // Compute gradients for current layer
        if(i > 0) {
            memset(curr->dx, 0, curr->size * sizeof(double));
            for(int j = 0; j < curr->size; j++) {
                for(int k = 0; k < next->size; k++) {
                    curr->dx[j] += next->dx[k] * net->W[i][k * curr->size + j];
                }
                curr->dx[j] *= relu_derivative(curr->x[j]);
            }
        }
    }
    
    // Update weights and biases using AdamW
    adamw_update(net, learning_rate);
}

// Compute mean squared error
double mse(Net* net, Data* data) {
    double error = 0;
    for(int i = 0; i < data->n; i++) {
        forward(net, data->X[i]);
        for(int j = 0; j < data->fy; j++) {
            double diff = net->layers[net->n_layers-1].x[j] - data->y[i][j];
            error += diff * diff;
        }
    }
    return error / data->n;
}

void save_net(const char* f, Net* net) {
    FILE* fp = fopen(f, "wb");
    if(!fp) return;
    
    // Save network architecture
    fwrite(&net->n_layers, sizeof(int), 1, fp);
    
    // Save layer sizes
    for(int i = 0; i < net->n_layers; i++) {
        fwrite(&net->layers[i].size, sizeof(int), 1, fp);
    }
    
    // Save weights and biases
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        
        // Save weights
        fwrite(net->W[i], sizeof(double), rows * cols, fp);
        
        // Save biases
        fwrite(net->b[i], sizeof(double), rows, fp);
    }
    
    fclose(fp);
}

Net* load_net(const char* f) {
    FILE* fp = fopen(f, "rb");
    if(!fp) return NULL;
    
    // Load network architecture
    int n_layers;
    fread(&n_layers, sizeof(int), 1, fp);
    
    // Load layer sizes
    int* sizes = malloc(n_layers * sizeof(int));
    for(int i = 0; i < n_layers; i++) {
        fread(&sizes[i], sizeof(int), 1, fp);
    }
    
    // Initialize network
    Net* net = init_net(n_layers, sizes);
    free(sizes);
    
    // Load weights and biases
    for(int i = 0; i < n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        
        // Load weights
        fread(net->W[i], sizeof(double), rows * cols, fp);
        
        // Load biases
        fread(net->b[i], sizeof(double), rows, fp);
    }
    
    fclose(fp);
    return net;
}

void free_net(Net* net) {
    for(int i = 0; i < net->n_layers; i++) {
        free(net->layers[i].x);
        free(net->layers[i].dx);
    }
    for(int i = 0; i < net->n_layers-1; i++) {
        free(net->W[i]);
        free(net->dW[i]);
        free(net->b[i]);
        free(net->db[i]);
    }
    free(net->W);
    free(net->dW);
    free(net->b);
    free(net->db);
    free(net->layers);
    free(net);
}

int main() {
    srand(time(NULL));
    
    // Generate synthetic data
    Data* data = synth(1000, 15, 4, 0.1);
    
    // Initialize network
    int sz[] = {15, 256, 128, 64, 32, 4};
    int n_layers = sizeof(sz)/sizeof(sz[0]);
    Net* net = init_net(n_layers, sz);
    
    // Training parameters
    double learning_rate = 0.001;
    int epochs = 5000;
    
    // Training loop
    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int i = 0; i < data->n; i++) {
            forward(net, data->X[i]);
            backward(net, data->y[i], learning_rate);
        }
        
        if(epoch % 5 == 0 || epoch == epochs - 1) {
            printf("Epoch %d, MSE: %f\n", epoch, mse(net, data));
        }
    }
    
    // Print some example predictions
    printf("\nExample predictions (first 10 samples):\n");
    printf("Index | %-30s | %-30s\n", "Predicted", "True Values");
    printf("------+--------------------------------+--------------------------------\n");
    
    for(int i = 0; i < 10; i++) {
        forward(net, data->X[i]);
        printf("%5d | ", i);
        
        // Print predicted values
        printf("[");
        for(int j = 0; j < data->fy; j++) {
            printf("%7.3f", net->layers[net->n_layers-1].x[j]);
            if(j < data->fy-1) printf(", ");
        }
        printf("] | ");
        
        // Print true values
        printf("[");
        for(int j = 0; j < data->fy; j++) {
            printf("%7.3f", data->y[i][j]);
            if(j < data->fy-1) printf(", ");
        }
        printf("]\n");
    }
    
    // Print input features for context
    printf("\nCorresponding input features:\n");
    printf("Index | Features (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)\n");
    printf("------+-------------------------\n");
    for(int i = 0; i < 10; i++) {
        printf("%5d | [", i);
        for(int j = 0; j < data->fx; j++) {
            printf("%7.3f", data->X[i][j]);
            if(j < data->fx-1) printf(", ");
        }
        printf("]\n");
    }

    char net_fname[64], data_fname[64];
    strftime(net_fname, sizeof(net_fname), "%Y%m%d_%H%M%S_net.bin", 
             localtime(&(time_t){time(NULL)}));
    save_net(net_fname, net);
    free_net(net);
    
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&(time_t){time(NULL)}));
    save_csv(data_fname, data);
    free_data(data);

    // Verification of saved and loaded data/network
    printf("\n=== Verification of saved/loaded data and network ===\n");

    // Load the saved network and data
    Net* loaded_net = load_net(net_fname);
    if (!loaded_net) {
        printf("Failed to load network from file: %s\n", net_fname);
        return 1;
    }

    Data* loaded_data = load_csv(data_fname, 15, 4);
    if (!loaded_data) {
        printf("Failed to load data from file: %s\n", data_fname);
        free_net(loaded_net);
        return 1;
    }

    // Calculate and print MSE with loaded network and data
    printf("\nMSE with loaded network: %f\n", mse(loaded_net, loaded_data));

    // Print some predictions with loaded network
    printf("\nPredictions with loaded network (first 10 samples):\n");
    printf("Index | %-30s | %-30s\n", "Predicted", "True Values");
    printf("------+--------------------------------+--------------------------------\n");

    for(int i = 0; i < 10; i++) {
        forward(loaded_net, loaded_data->X[i]);
        printf("%5d | ", i);
        
        // Print predicted values
        printf("[");
        for(int j = 0; j < loaded_data->fy; j++) {
            printf("%7.3f", loaded_net->layers[loaded_net->n_layers-1].x[j]);
            if(j < loaded_data->fy-1) printf(", ");
        }
        printf("] | ");
        
        // Print true values
        printf("[");
        for(int j = 0; j < loaded_data->fy; j++) {
            printf("%7.3f", loaded_data->y[i][j]);
            if(j < loaded_data->fy-1) printf(", ");
        }
        printf("]\n");
    }

    // Clean up
    free_net(loaded_net);
    free_data(loaded_data);

    return 0;
}