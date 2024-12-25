#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    int rows, cols;
    double* data;
} Matrix;

typedef struct {
    Matrix weights, bias, output;
} Layer;

typedef struct {
    int num_inputs, num_hidden_layers, hidden_size, num_outputs;
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
    Network net = {num_inputs, num_hidden_layers, hidden_size, num_outputs, 
                  malloc(num_hidden_layers * sizeof(Layer))};
    
    net.hidden_layers[0] = (Layer){create_matrix(hidden_size, num_inputs),
                                  create_matrix(hidden_size, 1),
                                  create_matrix(hidden_size, 1)};
    
    for(int i = 1; i < num_hidden_layers; i++) {
        net.hidden_layers[i] = (Layer){create_matrix(hidden_size, hidden_size),
                                     create_matrix(hidden_size, 1),
                                     create_matrix(hidden_size, 1)};
    }
    
    net.output_layer = (Layer){create_matrix(num_outputs, hidden_size),
                              create_matrix(num_outputs, 1),
                              create_matrix(num_outputs, 1)};
    return net;
}

void forward_network(Network* net, Matrix input) {
    forward(&net->hidden_layers[0], input);
    for(int i = 1; i < net->num_hidden_layers; i++) {
        forward(&net->hidden_layers[i], net->hidden_layers[i-1].output);
    }
    forward(&net->output_layer, net->hidden_layers[net->num_hidden_layers-1].output);
}

void train_network(Network* net, Matrix* inputs, double* targets, int num_samples, int epochs) {
    const double learning_rate = 0.1;
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for(int sample = 0; sample < num_samples; sample++) {
            forward_network(net, inputs[sample]);
            double error = net->output_layer.output.data[0] - targets[sample];
            total_loss += error * error;
            
            double output_delta = error * (1 - pow(net->output_layer.output.data[0], 2));
            Layer* last_hidden = &net->hidden_layers[net->num_hidden_layers-1];
            
            for(int j = 0; j < net->output_layer.weights.cols; j++) {
                net->output_layer.weights.data[j] -= learning_rate * output_delta * last_hidden->output.data[j];
            }
            net->output_layer.bias.data[0] -= learning_rate * output_delta;
            
            for(int layer = net->num_hidden_layers-1; layer >= 0; layer--) {
                Layer* current = &net->hidden_layers[layer];
                Matrix prev_output = (layer == 0) ? inputs[sample] : net->hidden_layers[layer-1].output;
                
                for(int j = 0; j < current->output.rows; j++) {
                    double hidden_delta = (layer == net->num_hidden_layers-1) ? 
                        output_delta * net->output_layer.weights.data[j] : 0;
                    
                    if(layer < net->num_hidden_layers-1) {
                        Layer* next = &net->hidden_layers[layer+1];
                        for(int k = 0; k < next->weights.rows; k++) {
                            hidden_delta += next->weights.data[k * current->output.rows + j];
                        }
                    }
                    
                    hidden_delta *= (1 - pow(current->output.data[j], 2));
                    
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
    Network net = create_network(2, 1, 8, 1);
    
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    Matrix inputs[4];
    for(int i = 0; i < 4; i++) {
        inputs[i] = (Matrix){2, 1, malloc(2 * sizeof(double))};
        inputs[i].data[0] = X[i][0];
        inputs[i].data[1] = X[i][1];
    }
    
    train_network(&net, inputs, Y, 4, 10000);
    
    printf("\nTesting XOR:\n");
    for(int i = 0; i < 4; i++) {
        forward_network(&net, inputs[i]);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n",
               X[i][0], X[i][1], net.output_layer.output.data[0], Y[i]);
        free(inputs[i].data);
    }
    
    free_network(&net);
    return 0;
}