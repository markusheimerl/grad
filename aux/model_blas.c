#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

// Neural network structure
typedef struct {
    // Weights and gradients
    float* fc1_weight;     // hidden_dim x input_dim
    float* fc2_weight;     // output_dim x hidden_dim
    float* fc1_weight_grad; // hidden_dim x input_dim
    float* fc2_weight_grad; // output_dim x hidden_dim
    
    // Helper arrays for forward/backward pass
    float* layer1_output;   // batch_size x hidden_dim
    float* predictions;     // batch_size x output_dim
    float* error;          // batch_size x output_dim
    float* pre_activation; // batch_size x hidden_dim
    float* error_hidden;   // batch_size x hidden_dim
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

// Initialize the network with configurable dimensions
Net* init_net(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Allocate and initialize weights and gradients
    net->fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate helper arrays
    net->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->pre_activation = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    // Initialize weights
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        net->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    free(net->fc1_weight);
    free(net->fc2_weight);
    free(net->fc1_weight_grad);
    free(net->fc2_weight_grad);
    free(net->layer1_output);
    free(net->predictions);
    free(net->error);
    free(net->pre_activation);
    free(net->error_hidden);
    free(net);
}

// Load CSV data
void load_csv(const char* filename, float** X, float** y, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    
    // Skip header
    char buffer[1024];
    fgets(buffer, sizeof(buffer), file);
    
    // Count lines
    int count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }
    *num_samples = count;
    
    // Allocate memory
    *X = (float*)malloc(count * 15 * sizeof(float));
    *y = (float*)malloc(count * 4 * sizeof(float));
    
    // Reset file pointer and skip header again
    fseek(file, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), file);
    
    // Read data
    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        char* token = strtok(buffer, ",");
        for (int i = 0; i < 15; i++) {
            (*X)[idx * 15 + i] = atof(token);
            token = strtok(NULL, ",");
        }
        for (int i = 0; i < 4; i++) {
            (*y)[idx * 4 + i] = atof(token);
            token = strtok(NULL, ",");
        }
        idx++;
    }
    
    fclose(file);
}

// Forward pass
void forward_pass(Net* net, float* X) {
    // First layer
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->hidden_dim,
                net->input_dim,
                1.0f,
                X,
                net->input_dim,
                net->fc1_weight,
                net->hidden_dim,
                0.0f,
                net->layer1_output,
                net->hidden_dim);
    
    // Store pre-activation values
    memcpy(net->pre_activation, net->layer1_output, 
           net->batch_size * net->hidden_dim * sizeof(float));
    
    // Swish activation
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->layer1_output[i] = net->layer1_output[i] / (1.0f + expf(-net->layer1_output[i]));
    }
    
    // Second layer
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->output_dim,
                net->hidden_dim,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->fc2_weight,
                net->output_dim,
                0.0f,
                net->predictions,
                net->output_dim);
}

// Calculate loss
float calculate_loss(Net* net, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < net->batch_size * net->output_dim; i++) {
        net->error[i] = net->predictions[i] - y[i];
        loss += net->error[i] * net->error[i];
    }
    return loss / (net->batch_size * net->output_dim);
}

// Zero gradients
void zero_gradients(Net* net) {
    memset(net->fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float));
    memset(net->fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass(Net* net, float* X) {
    // Gradient of second layer
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->hidden_dim,
                net->output_dim,
                net->batch_size,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->error,
                net->output_dim,
                0.0f,
                net->fc2_weight_grad,
                net->output_dim);
    
    // Backpropagate error through second layer
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                net->batch_size,
                net->hidden_dim,
                net->output_dim,
                1.0f,
                net->error,
                net->output_dim,
                net->fc2_weight,
                net->output_dim,
                0.0f,
                net->error_hidden,
                net->hidden_dim);
    
    // Swish derivative
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-net->pre_activation[i]));
        net->error_hidden[i] *= sigmoid + net->pre_activation[i] * sigmoid * (1.0f - sigmoid);
    }
    
    // Gradient of first layer
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->input_dim,
                net->hidden_dim,
                net->batch_size,
                1.0f,
                X,
                net->input_dim,
                net->error_hidden,
                net->hidden_dim,
                0.0f,
                net->fc1_weight_grad,
                net->hidden_dim);
}

// Update weights
void update_weights(Net* net, float learning_rate) {
    for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
        net->fc1_weight[i] -= learning_rate * net->fc1_weight_grad[i] / net->batch_size;
    }
    for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
        net->fc2_weight[i] -= learning_rate * net->fc2_weight_grad[i] / net->batch_size;
    }
}

#define MAX_SYNTHETIC_OUTPUTS 4
#define INPUT_RANGE_MIN -3.0f
#define INPUT_RANGE_MAX 3.0f

float synth_fn(const float* x, int fx, int dim) {
    switch(dim % MAX_SYNTHETIC_OUTPUTS) {
        case 0: 
            return sinf(x[0 % fx]*2)*cosf(x[1 % fx]*1.5f) + 
                   powf(x[2 % fx],2)*x[3 % fx] + 
                   expf(-powf(x[4 % fx]-x[5 % fx],2)) + 
                   0.5f*sinf(x[6 % fx]*x[7 % fx]*(float)M_PI) +
                   tanhf(x[8 % fx] + x[9 % fx]) +
                   0.3f*cosf(x[10 % fx]*x[11 % fx]) +
                   0.2f*powf(x[12 % fx], 2) +
                   x[13 % fx]*sinf(x[14 % fx]);
            
        case 1: 
            return tanhf(x[0 % fx]+x[1 % fx])*sinf(x[2 % fx]*2) + 
                   logf(fabsf(x[3 % fx])+1)*cosf(x[4 % fx]) + 
                   0.3f*powf(x[5 % fx]-x[6 % fx],3) +
                   expf(-powf(x[7 % fx],2)) +
                   sinf(x[8 % fx]*x[9 % fx]*0.5f) +
                   0.4f*cosf(x[10 % fx] + x[11 % fx]) +
                   powf(x[12 % fx]*x[13 % fx], 2) +
                   0.1f*x[14 % fx];
            
        case 2: 
            return expf(-powf(x[0 % fx]-0.5f,2))*sinf(x[1 % fx]*3) + 
                   powf(cosf(x[2 % fx]),2)*x[3 % fx] + 
                   0.2f*sinhf(x[4 % fx]*x[5 % fx]) +
                   0.5f*tanhf(x[6 % fx] + x[7 % fx]) +
                   powf(x[8 % fx], 3)*0.1f +
                   cosf(x[9 % fx]*x[10 % fx]*(float)M_PI) +
                   0.3f*expf(-powf(x[11 % fx]-x[12 % fx],2)) +
                   0.2f*(x[13 % fx] + x[14 % fx]);
            
        case 3:
            return powf(sinf(x[0 % fx]*x[1 % fx]), 2) +
                   0.4f*tanhf(x[2 % fx] + x[3 % fx]*x[4 % fx]) +
                   expf(-fabsf(x[5 % fx]-x[6 % fx])) +
                   0.3f*cosf(x[7 % fx]*x[8 % fx]*2) +
                   powf(x[9 % fx], 2)*sinf(x[10 % fx]) +
                   0.2f*logf(fabsf(x[11 % fx]*x[12 % fx])+1) +
                   0.1f*(x[13 % fx] - x[14 % fx]);
            
        default: 
            return 0.0f;
    }
}

// Replace load_csv with this synthetic data generator
void generate_synthetic_data(float** X, float** y, int num_samples, int input_dim, int output_dim) {
    // Allocate memory
    *X = (float*)malloc(num_samples * input_dim * sizeof(float));
    *y = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate random input data
    for (int i = 0; i < num_samples * input_dim; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX;
        (*X)[i] = INPUT_RANGE_MIN + rand_val * (INPUT_RANGE_MAX - INPUT_RANGE_MIN);
    }
    
    // Generate output data using synth_fn
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < output_dim; j++) {
            (*y)[i * output_dim + j] = synth_fn(&(*X)[i * input_dim], input_dim, j);
        }
    }
}

// Function to save model weights to binary file
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    
    // Save batch size
    fwrite(&net->batch_size, sizeof(int), 1, file);

    // Save weights
    fwrite(net->fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_weight, sizeof(float), net->output_dim * net->hidden_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to save data to CSV
void save_data_to_csv(float* X, float* y, int num_samples, int input_dim, int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    for (int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for (int i = 0; i < output_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", output_dim - 1);
    
    // Write data
    for (int i = 0; i < num_samples; i++) {
        // Input features
        for (int j = 0; j < input_dim; j++) {
            fprintf(file, "%.17f,", X[i * input_dim + j]);
        }
        // Output values
        for (int j = 0; j < output_dim - 1; j++) {
            fprintf(file, "%.17f,", y[i * output_dim + j]);
        }
        fprintf(file, "%.17f\n", y[i * output_dim + output_dim - 1]);
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}

Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    
    // Load batch size
    fread(&batch_size, sizeof(int), 1, file);

    // Initialize network with loaded dimensions and batch size
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(net->fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_weight, sizeof(float), output_dim * hidden_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 16;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples; // Full batch training
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);
    
    // Initialize network
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        forward_pass(net, X);
        
        // Calculate loss
        float loss = calculate_loss(net, y);
        
        // Backward pass
        zero_gradients(net);
        backward_pass(net, X);
        
        // Update weights
        update_weights(net, learning_rate);
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back
    net = load_model(model_fname);
    
    // Forward pass with loaded model
    forward_pass(net, X);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss(net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);
    
    // Cleanup
    free(X);
    free(y);
    free_net(net);
    
    return 0;
}