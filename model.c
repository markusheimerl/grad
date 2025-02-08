#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 15
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 128
#define HIDDEN3_SIZE 64
#define OUTPUT_SIZE 4
#define MAX_ROWS 10000
#define BATCH_SIZE 32
#define EPOCHS 1000

typedef struct {
    double* weights;
    double* biases;
    double* output;
} Layer;

typedef struct {
    Layer layers[4];
    int num_samples;
    double* input_data;
    double* target_data;
} Network;

typedef struct {
    double* m;  // First moment
    double* v;  // Second moment
    int size;
} AdamParam;

typedef struct {
    AdamParam weights;
    AdamParam biases;
} AdamLayer;

typedef struct {
    AdamLayer layers[4];
    double beta1;
    double beta2;
    double epsilon;
    double lr;
    int t;  // Time step
} Adam;

// Add these functions before main()
void init_adam_param(AdamParam* param, int size) {
    param->size = size;
    param->m = calloc(size, sizeof(double));
    param->v = calloc(size, sizeof(double));
}

void init_adam(Adam* adam, Network* net) {
    adam->beta1 = 0.9;
    adam->beta2 = 0.999;
    adam->epsilon = 1e-8;
    adam->lr = 0.001;
    adam->t = 0;
    
    init_adam_param(&adam->layers[0].weights, INPUT_SIZE * HIDDEN1_SIZE);
    init_adam_param(&adam->layers[0].biases, HIDDEN1_SIZE);
    
    init_adam_param(&adam->layers[1].weights, HIDDEN1_SIZE * HIDDEN2_SIZE);
    init_adam_param(&adam->layers[1].biases, HIDDEN2_SIZE);
    
    init_adam_param(&adam->layers[2].weights, HIDDEN2_SIZE * HIDDEN3_SIZE);
    init_adam_param(&adam->layers[2].biases, HIDDEN3_SIZE);
    
    init_adam_param(&adam->layers[3].weights, HIDDEN3_SIZE * OUTPUT_SIZE);
    init_adam_param(&adam->layers[3].biases, OUTPUT_SIZE);
}

void backward_and_update(Network* net, Adam* adam, int batch_idx) {
    int batch_size = (batch_idx + BATCH_SIZE <= net->num_samples) ? 
                     BATCH_SIZE : (net->num_samples - batch_idx);
    
    // Temporary storage for gradients
    double* gradients[4];
    double* input_gradients[4];
    for(int i = 0; i < 4; i++) {
        int output_size = (i == 0) ? HIDDEN1_SIZE : 
                         (i == 1) ? HIDDEN2_SIZE :
                         (i == 2) ? HIDDEN3_SIZE : OUTPUT_SIZE;
        gradients[i] = calloc(batch_size * output_size, sizeof(double));
        input_gradients[i] = calloc(batch_size * output_size, sizeof(double));
    }
    
    // Output layer gradients
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            gradients[3][i * OUTPUT_SIZE + j] = 
                (net->layers[3].output[i * OUTPUT_SIZE + j] - 
                 net->target_data[(batch_idx + i) * OUTPUT_SIZE + j]) * (2.0 / (batch_size * OUTPUT_SIZE));
        }
    }
    
    // Backward pass through layers
    for(int layer = 3; layer >= 0; layer--) {
        int input_size = (layer == 0) ? INPUT_SIZE : 
                        (layer == 1) ? HIDDEN1_SIZE :
                        (layer == 2) ? HIDDEN2_SIZE : HIDDEN3_SIZE;
        int output_size = (layer == 0) ? HIDDEN1_SIZE : 
                         (layer == 1) ? HIDDEN2_SIZE :
                         (layer == 2) ? HIDDEN3_SIZE : OUTPUT_SIZE;
        
        // Weight gradients
        double* prev_output = (layer == 0) ? &net->input_data[batch_idx * INPUT_SIZE] :
                                           net->layers[layer-1].output;
        
        // Update weights and biases using Adam
        adam->t++;
        double lr_t = adam->lr * sqrt(1.0 - pow(adam->beta2, adam->t)) / 
                                (1.0 - pow(adam->beta1, adam->t));
        
        for(int i = 0; i < input_size; i++) {
            for(int j = 0; j < output_size; j++) {
                double gradient = 0.0;
                for(int k = 0; k < batch_size; k++) {
                    gradient += prev_output[k * input_size + i] * 
                               gradients[layer][k * output_size + j];
                }
                
                int idx = i * output_size + j;
                AdamParam* p = &adam->layers[layer].weights;
                
                p->m[idx] = adam->beta1 * p->m[idx] + (1 - adam->beta1) * gradient;
                p->v[idx] = adam->beta2 * p->v[idx] + (1 - adam->beta2) * gradient * gradient;
                
                net->layers[layer].weights[idx] -= lr_t * p->m[idx] / 
                    (sqrt(p->v[idx]) + adam->epsilon);
            }
        }
        
        // Update biases
        for(int j = 0; j < output_size; j++) {
            double gradient = 0.0;
            for(int k = 0; k < batch_size; k++) {
                gradient += gradients[layer][k * output_size + j];
            }
            
            AdamParam* p = &adam->layers[layer].biases;
            p->m[j] = adam->beta1 * p->m[j] + (1 - adam->beta1) * gradient;
            p->v[j] = adam->beta2 * p->v[j] + (1 - adam->beta2) * gradient * gradient;
            
            net->layers[layer].biases[j] -= lr_t * p->m[j] / 
                (sqrt(p->v[j]) + adam->epsilon);
        }
        
        // Compute gradients for previous layer
        if(layer > 0) {
            for(int i = 0; i < batch_size; i++) {
                for(int j = 0; j < input_size; j++) {
                    double sum = 0.0;
                    for(int k = 0; k < output_size; k++) {
                        sum += gradients[layer][i * output_size + k] * 
                               net->layers[layer].weights[j * output_size + k];
                    }
                    if(layer > 0) {  // Apply ReLU gradient
                        double prev_output = net->layers[layer-1].output[i * input_size + j];
                        sum *= (prev_output > 0) ? 1.0 : 0.0;
                    }
                    gradients[layer-1][i * input_size + j] = sum;
                }
            }
        }
    }
    
    // Free temporary storage
    for(int i = 0; i < 4; i++) {
        free(gradients[i]);
        free(input_gradients[i]);
    }
}

void init_layer(Layer* layer, int inputs, int outputs) {
    layer->weights = malloc(inputs * outputs * sizeof(double));
    layer->biases = malloc(outputs * sizeof(double));
    layer->output = malloc(BATCH_SIZE * outputs * sizeof(double));
    
    // Xavier initialization
    double scale = sqrt(2.0 / inputs);
    for(int i = 0; i < inputs * outputs; i++) {
        layer->weights[i] = ((double)rand() / RAND_MAX * 2 - 1) * scale;
    }
    memset(layer->biases, 0, outputs * sizeof(double));
}

void forward(Network* net, int batch_idx) {
    int batch_size = (batch_idx + BATCH_SIZE <= net->num_samples) ? 
                     BATCH_SIZE : (net->num_samples - batch_idx);
    
    // Copy input batch
    double* current_input = &net->input_data[batch_idx * INPUT_SIZE];
    
    // Layer 1
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < HIDDEN1_SIZE; j++) {
            double sum = net->layers[0].biases[j];
            for(int k = 0; k < INPUT_SIZE; k++) {
                sum += current_input[i * INPUT_SIZE + k] * 
                       net->layers[0].weights[k * HIDDEN1_SIZE + j];
            }
            net->layers[0].output[i * HIDDEN1_SIZE + j] = sum > 0 ? sum : 0; // ReLU
        }
    }
    
    // Layer 2
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < HIDDEN2_SIZE; j++) {
            double sum = net->layers[1].biases[j];
            for(int k = 0; k < HIDDEN1_SIZE; k++) {
                sum += net->layers[0].output[i * HIDDEN1_SIZE + k] * 
                       net->layers[1].weights[k * HIDDEN2_SIZE + j];
            }
            net->layers[1].output[i * HIDDEN2_SIZE + j] = sum > 0 ? sum : 0; // ReLU
        }
    }
    
    // Layer 3
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < HIDDEN3_SIZE; j++) {
            double sum = net->layers[2].biases[j];
            for(int k = 0; k < HIDDEN2_SIZE; k++) {
                sum += net->layers[1].output[i * HIDDEN2_SIZE + k] * 
                       net->layers[2].weights[k * HIDDEN3_SIZE + j];
            }
            net->layers[2].output[i * HIDDEN3_SIZE + j] = sum > 0 ? sum : 0; // ReLU
        }
    }
    
    // Layer 4 (output)
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            double sum = net->layers[3].biases[j];
            for(int k = 0; k < HIDDEN3_SIZE; k++) {
                sum += net->layers[2].output[i * HIDDEN3_SIZE + k] * 
                       net->layers[3].weights[k * OUTPUT_SIZE + j];
            }
            net->layers[3].output[i * OUTPUT_SIZE + j] = sum;
        }
    }
}

double calculate_loss(Network* net) {
    double total_loss = 0;
    int count = 0;
    
    for(int batch = 0; batch < net->num_samples; batch += BATCH_SIZE) {
        forward(net, batch);
        int batch_size = (batch + BATCH_SIZE <= net->num_samples) ? 
                        BATCH_SIZE : (net->num_samples - batch);
        
        for(int i = 0; i < batch_size; i++) {
            for(int j = 0; j < OUTPUT_SIZE; j++) {
                double diff = net->layers[3].output[i * OUTPUT_SIZE + j] - 
                             net->target_data[(batch + i) * OUTPUT_SIZE + j];
                total_loss += diff * diff;
                count++;
            }
        }
    }
    
    return total_loss / count;
}

int main() {
    srand(42);  // Match PyTorch initialization

    Network net;
    
    // Read CSV
    FILE* f = fopen("20250208_163908_data.csv", "r");
    if (!f) {
        printf("Failed to open data file\n");
        return 1;
    }

    char line[4096];
    fgets(line, sizeof(line), f); // Skip header
    
    net.input_data = malloc(MAX_ROWS * INPUT_SIZE * sizeof(double));
    net.target_data = malloc(MAX_ROWS * OUTPUT_SIZE * sizeof(double));
    net.num_samples = 0;
    
    while(fgets(line, sizeof(line), f) && net.num_samples < MAX_ROWS) {
        char* token = strtok(line, ",");
        for(int i = 0; i < INPUT_SIZE + OUTPUT_SIZE; i++) {
            double val = atof(token);
            if(i < INPUT_SIZE)
                net.input_data[net.num_samples * INPUT_SIZE + i] = val;
            else
                net.target_data[net.num_samples * OUTPUT_SIZE + (i - INPUT_SIZE)] = val;
            token = strtok(NULL, ",");
        }
        net.num_samples++;
    }
    fclose(f);
    
    // Initialize network
    init_layer(&net.layers[0], INPUT_SIZE, HIDDEN1_SIZE);
    init_layer(&net.layers[1], HIDDEN1_SIZE, HIDDEN2_SIZE);
    init_layer(&net.layers[2], HIDDEN2_SIZE, HIDDEN3_SIZE);
    init_layer(&net.layers[3], HIDDEN3_SIZE, OUTPUT_SIZE);
    
    // Initialize Adam optimizer
    Adam adam;
    init_adam(&adam, &net);
    
    // Training loop
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        // Mini-batch training
        for(int i = 0; i < net.num_samples; i += BATCH_SIZE) {
            forward(&net, i);
            backward_and_update(&net, &adam, i);
        }
        
        // Print progress every 100 epochs
        if((epoch + 1) % 100 == 0) {
            double loss = calculate_loss(&net);
            printf("Epoch [%d/%d], Train Loss: %.4f\n", epoch + 1, EPOCHS, loss);
        }
    }

    // Calculate R² scores and show predictions
    double predictions[MAX_ROWS * OUTPUT_SIZE];
    
    // Get predictions for all samples
    for(int i = 0; i < net.num_samples; i += BATCH_SIZE) {
        forward(&net, i);
        int batch_size = (i + BATCH_SIZE <= net.num_samples) ? BATCH_SIZE : (net.num_samples - i);
        for(int j = 0; j < batch_size; j++) {
            for(int k = 0; k < OUTPUT_SIZE; k++) {
                predictions[i * OUTPUT_SIZE + j * OUTPUT_SIZE + k] = 
                    net.layers[3].output[j * OUTPUT_SIZE + k];
            }
        }
    }

    // Calculate R² scores
    for(int output = 0; output < OUTPUT_SIZE; output++) {
        double mean_actual = 0.0;
        for(int i = 0; i < net.num_samples; i++) {
            mean_actual += net.target_data[i * OUTPUT_SIZE + output];
        }
        mean_actual /= net.num_samples;

        double ss_tot = 0.0, ss_res = 0.0;
        for(int i = 0; i < net.num_samples; i++) {
            double actual = net.target_data[i * OUTPUT_SIZE + output];
            double pred = predictions[i * OUTPUT_SIZE + output];
            ss_tot += (actual - mean_actual) * (actual - mean_actual);
            ss_res += (actual - pred) * (actual - pred);
        }
        double r2 = 1.0 - (ss_res / ss_tot);
        printf("R² score for output y%d: %.4f\n", output, r2);
    }

    // Show sample predictions
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Output\t\tPredicted\tActual\t\tDifference\n");
    printf("------------------------------------------------------------\n");

    for(int output = 0; output < OUTPUT_SIZE; output++) {
        printf("\ny%d:\n", output);
        for(int sample = 0; sample < 15; sample++) {
            double pred = predictions[sample * OUTPUT_SIZE + output];
            double actual = net.target_data[sample * OUTPUT_SIZE + output];
            double diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", 
                   sample, pred, actual, diff);
        }

        // Calculate MAE for this output
        double mae = 0.0;
        for(int i = 0; i < net.num_samples; i++) {
            mae += fabs(predictions[i * OUTPUT_SIZE + output] - 
                       net.target_data[i * OUTPUT_SIZE + output]);
        }
        mae /= net.num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", output, mae);
    }
    
    // Free memory
    free(net.input_data);
    free(net.target_data);
    
    for(int i = 0; i < 4; i++) {
        free(net.layers[i].weights);
        free(net.layers[i].biases);
        free(net.layers[i].output);
        
        free(adam.layers[i].weights.m);
        free(adam.layers[i].weights.v);
        free(adam.layers[i].biases.m);
        free(adam.layers[i].biases.v);
    }
    
    return 0;
}