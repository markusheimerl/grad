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

Matrix create_matrix(int rows, int cols) {
    Matrix m = {rows, cols, malloc(rows * cols * sizeof(double))};
    for(int i = 0; i < rows * cols; i++) {
        m.data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1;
    }
    return m;
}

void forward(Layer* layer, Matrix input) {
    // Compute layer output
    for(int i = 0; i < layer->output.rows; i++) {
        double sum = layer->bias.data[i];
        for(int j = 0; j < input.rows; j++) {
            sum += layer->weights.data[i * input.rows + j] * input.data[j];
        }
        layer->output.data[i] = tanh(sum);
    }
}

int main() {
    srand(time(NULL));
    
    // Create layers
    Layer hidden = {
        create_matrix(8, 2),  // weights
        create_matrix(8, 1),  // bias
        create_matrix(8, 1)   // output
    };
    
    Layer output = {
        create_matrix(1, 8),  // weights
        create_matrix(1, 1),  // bias
        create_matrix(1, 1)   // output
    };

    // Training data
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    Matrix input = {2, 1, malloc(2 * sizeof(double))};
    
    // Training loop
    for(int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;
        
        for(int i = 0; i < 4; i++) {
            // Forward pass
            input.data[0] = X[i][0];
            input.data[1] = X[i][1];
            
            forward(&hidden, input);
            forward(&output, hidden.output);
            
            double error = output.output.data[0] - Y[i];
            total_loss += error * error;
            
            // Backward pass
            double output_delta = error * (1 - output.output.data[0] * output.output.data[0]);
            
            // Update output layer
            for(int j = 0; j < output.weights.cols; j++) {
                output.weights.data[j] -= 0.1 * output_delta * hidden.output.data[j];
            }
            output.bias.data[0] -= 0.1 * output_delta;
            
            // Update hidden layer
            for(int j = 0; j < hidden.output.rows; j++) {
                double hidden_delta = output_delta * output.weights.data[j] * 
                                    (1 - hidden.output.data[j] * hidden.output.data[j]);
                for(int k = 0; k < hidden.weights.cols; k++) {
                    hidden.weights.data[j * hidden.weights.cols + k] -= 
                        0.1 * hidden_delta * input.data[k];
                }
                hidden.bias.data[j] -= 0.1 * hidden_delta;
            }
        }
        
        if(epoch % 1000 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    // Test network
    printf("\nTesting XOR:\n");
    for(int i = 0; i < 4; i++) {
        input.data[0] = X[i][0];
        input.data[1] = X[i][1];
        forward(&hidden, input);
        forward(&output, hidden.output);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n",
               X[i][0], X[i][1], output.output.data[0], Y[i]);
    }
    
    // Cleanup
    free(input.data);
    free(hidden.weights.data);
    free(hidden.bias.data);
    free(hidden.output.data);
    free(output.weights.data);
    free(output.bias.data);
    free(output.output.data);
    
    return 0;
}