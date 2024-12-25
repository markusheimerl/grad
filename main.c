#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Matrix structure
typedef struct {
    int rows, cols;
    double* data;
} Matrix;

// Layer structure
typedef struct {
    Matrix weights;
    Matrix bias;
    Matrix output;
    Matrix gradient_w;
    Matrix gradient_b;
} Layer;

// Neural Network structure
typedef struct {
    Layer hidden;
    Layer output;
} Network;

// Matrix operations
Matrix create_matrix(int rows, int cols) {
    Matrix m = {rows, cols, malloc(rows * cols * sizeof(double))};
    return m;
}

void free_matrix(Matrix m) {
    free(m.data);
}

void random_init(Matrix m, double scale) {
    for(int i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    }
}

void matrix_multiply(Matrix a, Matrix b, Matrix c) {
    for(int i = 0; i < a.rows; i++) {
        for(int j = 0; j < b.cols; j++) {
            double sum = 0.0;
            for(int k = 0; k < a.cols; k++) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            c.data[i * c.cols + j] = sum;
        }
    }
}

void add_matrix(Matrix a, Matrix b, Matrix c) {
    for(int i = 0; i < a.rows * a.cols; i++) {
        c.data[i] = a.data[i] + b.data[i];
    }
}

void tanh_matrix(Matrix m) {
    for(int i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = tanh(m.data[i]);
    }
}

void tanh_derivative(Matrix output, Matrix gradient) {
    for(int i = 0; i < output.rows * output.cols; i++) {
        gradient.data[i] *= (1.0 - output.data[i] * output.data[i]);
    }
}

// Neural Network operations
Network create_network(int input_size, int hidden_size) {
    Network net;
    
    // Hidden layer
    net.hidden.weights = create_matrix(hidden_size, input_size);
    net.hidden.bias = create_matrix(hidden_size, 1);
    net.hidden.output = create_matrix(hidden_size, 1);
    net.hidden.gradient_w = create_matrix(hidden_size, input_size);
    net.hidden.gradient_b = create_matrix(hidden_size, 1);
    
    // Output layer
    net.output.weights = create_matrix(1, hidden_size);
    net.output.bias = create_matrix(1, 1);
    net.output.output = create_matrix(1, 1);
    net.output.gradient_w = create_matrix(1, hidden_size);
    net.output.gradient_b = create_matrix(1, 1);
    
    // Initialize weights and biases
    random_init(net.hidden.weights, 0.1);
    random_init(net.hidden.bias, 0.1);
    random_init(net.output.weights, 0.1);
    random_init(net.output.bias, 0.1);
    
    return net;
}

void forward(Network* net, Matrix input) {
    // Hidden layer
    matrix_multiply(net->hidden.weights, input, net->hidden.output);
    add_matrix(net->hidden.output, net->hidden.bias, net->hidden.output);
    tanh_matrix(net->hidden.output);
    
    // Output layer
    matrix_multiply(net->output.weights, net->hidden.output, net->output.output);
    add_matrix(net->output.output, net->output.bias, net->output.output);
    tanh_matrix(net->output.output);
}

void backward(Network* net, Matrix input, Matrix target, double learning_rate) {
    // Output layer gradients
    double error = net->output.output.data[0] - target.data[0];
    net->output.output.data[0] = error * (1 - net->output.output.data[0] * net->output.output.data[0]);
    
    // Update output layer
    for(int i = 0; i < net->output.weights.cols; i++) {
        net->output.weights.data[i] -= learning_rate * net->output.output.data[0] * net->hidden.output.data[i];
    }
    net->output.bias.data[0] -= learning_rate * net->output.output.data[0];
    
    // Hidden layer gradients
    for(int i = 0; i < net->hidden.output.rows; i++) {
        double grad = net->output.output.data[0] * net->output.weights.data[i];
        grad *= (1 - net->hidden.output.data[i] * net->hidden.output.data[i]);
        
        // Update hidden layer weights
        for(int j = 0; j < net->hidden.weights.cols; j++) {
            net->hidden.weights.data[i * net->hidden.weights.cols + j] -= 
                learning_rate * grad * input.data[j];
        }
        net->hidden.bias.data[i] -= learning_rate * grad;
    }
}

int main() {
    srand(time(NULL));
    
    // Create network
    Network net = create_network(2, 8);
    
    // Training data
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    // Training matrices
    Matrix input = create_matrix(2, 1);
    Matrix target = create_matrix(1, 1);
    
    // Training loop
    for(int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;
        
        for(int i = 0; i < 4; i++) {
            // Set input and target
            input.data[0] = X[i][0];
            input.data[1] = X[i][1];
            target.data[0] = Y[i];
            
            // Forward and backward passes
            forward(&net, input);
            total_loss += pow(net.output.output.data[0] - target.data[0], 2);
            backward(&net, input, target, 0.1);
        }
        
        if(epoch % 1000 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    // Test the network
    printf("\nTesting XOR:\n");
    for(int i = 0; i < 4; i++) {
        input.data[0] = X[i][0];
        input.data[1] = X[i][1];
        forward(&net, input);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n",
               X[i][0], X[i][1], net.output.output.data[0], Y[i]);
    }
    
    // Cleanup
    free_matrix(input);
    free_matrix(target);
    free_matrix(net.hidden.weights);
    free_matrix(net.hidden.bias);
    free_matrix(net.hidden.output);
    free_matrix(net.hidden.gradient_w);
    free_matrix(net.hidden.gradient_b);
    free_matrix(net.output.weights);
    free_matrix(net.output.bias);
    free_matrix(net.output.output);
    free_matrix(net.output.gradient_w);
    free_matrix(net.output.gradient_b);
    
    return 0;
}