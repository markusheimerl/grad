#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    int rows, cols;
    double* data;
} Matrix;

typedef struct {
    Matrix weights;
    Matrix bias;
    Matrix output;
} Layer;

typedef struct {
    int num_inputs;
    int num_hidden_layers;
    int hidden_size;
    int num_outputs;
    Layer* hidden_layers;
    Layer output_layer;
} Network;

Matrix create_matrix(int rows, int cols) {
    Matrix m = {rows, cols, malloc(rows * cols * sizeof(double))};
    for(int i = 0; i < rows * cols; i++) {
        m.data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
    }
    return m;
}

void forward(Layer* layer, Matrix input) {
    for(int i = 0; i < layer->output.rows; i++) {
        double sum = layer->bias.data[i];
        for(int j = 0; j < input.rows; j++) {
            sum += layer->weights.data[i * input.rows + j] * input.data[j];
        }
        layer->output.data[i] = tanh(sum);
    }
}

Network create_network(int num_inputs, int num_hidden_layers, int hidden_size, int num_outputs) {
    Network net;
    net.num_inputs = num_inputs;
    net.num_hidden_layers = num_hidden_layers;
    net.hidden_size = hidden_size;
    net.num_outputs = num_outputs;
    
    // Create hidden layers
    net.hidden_layers = malloc(num_hidden_layers * sizeof(Layer));
    
    // First hidden layer
    net.hidden_layers[0] = (Layer){
        create_matrix(hidden_size, num_inputs),
        create_matrix(hidden_size, 1),
        create_matrix(hidden_size, 1)
    };
    
    // Remaining hidden layers
    for(int i = 1; i < num_hidden_layers; i++) {
        net.hidden_layers[i] = (Layer){
            create_matrix(hidden_size, hidden_size),
            create_matrix(hidden_size, 1),
            create_matrix(hidden_size, 1)
        };
    }
    
    // Output layer
    net.output_layer = (Layer){
        create_matrix(num_outputs, hidden_size),
        create_matrix(num_outputs, 1),
        create_matrix(num_outputs, 1)
    };
    
    return net;
}

void forward_network(Network* net, Matrix input) {
    // Forward through hidden layers
    forward(&net->hidden_layers[0], input);
    for(int i = 1; i < net->num_hidden_layers; i++) {
        forward(&net->hidden_layers[i], net->hidden_layers[i-1].output);
    }
    
    // Forward through output layer
    forward(&net->output_layer, net->hidden_layers[net->num_hidden_layers-1].output);
}

void train_network(Network* net, Matrix* inputs, double* targets, int num_samples, int epochs) {
    double learning_rate = 0.1;
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for(int sample = 0; sample < num_samples; sample++) {
            // Forward pass
            forward_network(net, inputs[sample]);
            
            // Calculate error
            double error = net->output_layer.output.data[0] - targets[sample];
            total_loss += error * error;
            
            // Backward pass
            double output_delta = error * (1 - net->output_layer.output.data[0] * net->output_layer.output.data[0]);
            
            // Update output layer
            Layer* last_hidden = &net->hidden_layers[net->num_hidden_layers-1];
            for(int j = 0; j < net->output_layer.weights.cols; j++) {
                net->output_layer.weights.data[j] -= learning_rate * output_delta * last_hidden->output.data[j];
            }
            net->output_layer.bias.data[0] -= learning_rate * output_delta;
            
            // Update hidden layers
            for(int layer = net->num_hidden_layers-1; layer >= 0; layer--) {
                Layer* current = &net->hidden_layers[layer];
                Matrix prev_output = (layer == 0) ? inputs[sample] : 
                                    net->hidden_layers[layer-1].output;
                
                for(int j = 0; j < current->output.rows; j++) {
                    double hidden_delta;
                    if(layer == net->num_hidden_layers-1) {
                        hidden_delta = output_delta * net->output_layer.weights.data[j];
                    } else {
                        hidden_delta = 0;
                        Layer* next = &net->hidden_layers[layer+1];
                        for(int k = 0; k < next->weights.rows; k++) {
                            hidden_delta += next->weights.data[k * current->output.rows + j];
                        }
                    }
                    hidden_delta *= (1 - current->output.data[j] * current->output.data[j]);
                    
                    for(int k = 0; k < current->weights.cols; k++) {
                        current->weights.data[j * current->weights.cols + k] -= 
                            learning_rate * hidden_delta * prev_output.data[k];
                    }
                    current->bias.data[j] -= learning_rate * hidden_delta;
                }
            }
        }
        
        if(epoch % 1000 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch, total_loss / num_samples);
        }
    }
}

void free_network(Network* net) {
    for(int i = 0; i < net->num_hidden_layers; i++) {
        free(net->hidden_layers[i].weights.data);
        free(net->hidden_layers[i].bias.data);
        free(net->hidden_layers[i].output.data);
    }
    free(net->hidden_layers);
    free(net->output_layer.weights.data);
    free(net->output_layer.bias.data);
    free(net->output_layer.output.data);
}

int main() {
    srand(time(NULL));
    
    // Create network with 2 inputs, 1 hidden layer of size 8, and 1 output
    Network net = create_network(2, 1, 8, 1);
    
    // Training data
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    // Convert training data to Matrix format
    Matrix inputs[4];
    for(int i = 0; i < 4; i++) {
        inputs[i] = (Matrix){2, 1, malloc(2 * sizeof(double))};
        inputs[i].data[0] = X[i][0];
        inputs[i].data[1] = X[i][1];
    }
    
    // Train network
    train_network(&net, inputs, Y, 4, 10000);
    
    // Test network
    printf("\nTesting XOR:\n");
    for(int i = 0; i < 4; i++) {
        forward_network(&net, inputs[i]);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n",
               X[i][0], X[i][1], net.output_layer.output.data[0], Y[i]);
    }
    
    // Cleanup
    for(int i = 0; i < 4; i++) {
        free(inputs[i].data);
    }
    free_network(&net);
    
    return 0;
}