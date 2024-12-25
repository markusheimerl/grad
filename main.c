#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "data/boston_housing_dataset.h"

#define BATCH_SIZE 32
#define HIDDEN_SIZE 512

typedef struct {
    double data, grad;
} Value;

typedef struct {
    int in_size, out_size;
    Value* w;
    Value* b;
    Value* out;
} Layer;

typedef struct {
    Value* inputs;
    Layer hidden;
    Layer output;
} Net;

static Net* g_net = NULL;

void cleanup() {
    if (g_net) {
        free(g_net->inputs);
        free(g_net->hidden.w);
        free(g_net->hidden.b);
        free(g_net->hidden.out);
        free(g_net->output.w);
        free(g_net->output.b);
        free(g_net->output.out);
        free(g_net);
    }
}

void signal_handler(int signum) {
    printf("\nCaught signal %d. Cleaning up...\n", signum);
    cleanup();
    exit(signum);
}

Layer create_layer(int in_size, int out_size) {
    Layer l;
    l.in_size = in_size;
    l.out_size = out_size;
    
    l.w = calloc(in_size * out_size, sizeof(Value));
    l.b = calloc(out_size, sizeof(Value));
    l.out = calloc(out_size, sizeof(Value));
    
    double scale = sqrt(2.0 / (in_size + out_size));
    for (int i = 0; i < in_size * out_size; i++) {
        l.w[i].data = ((double)rand() / RAND_MAX * 2 - 1) * scale;
    }
    return l;
}

void zero_grad(Layer* l) {
    for (int i = 0; i < l->in_size * l->out_size; i++) {
        l->w[i].grad = 0;
    }
    for (int i = 0; i < l->out_size; i++) {
        l->b[i].grad = 0;
    }
}

inline double relu(double x) {
    return x > 0 ? x : 0.01 * x;
}

inline double relu_grad(double x) {
    return x > 0 ? 1.0 : 0.01;
}

void forward_layer(Layer* l, Value* inputs) {
    for (int j = 0; j < l->out_size; j++) {
        double sum = l->b[j].data;
        for (int i = 0; i < l->in_size; i++) {
            sum += inputs[i].data * l->w[i * l->out_size + j].data;
        }
        l->out[j].data = relu(sum);
    }
}

double predict(Net* net, double* x, double* x_min, double* x_max) {
    for (int j = 0; j < 12; j++) {
        net->inputs[j].data = (x[j] - x_min[j]) / (x_max[j] - x_min[j] + 1e-8);
    }
    forward_layer(&net->hidden, net->inputs);
    forward_layer(&net->output, net->hidden.out);
    return net->output.out[0].data;
}

void backward_layer(Layer* l, Value* inputs, double grad_out) {
    for (int j = 0; j < l->out_size; j++) {
        double grad = grad_out * relu_grad(l->out[j].data);
        l->b[j].grad += grad;
        for (int i = 0; i < l->in_size; i++) {
            l->w[i * l->out_size + j].grad += inputs[i].data * grad;
        }
    }
}

void update_layer(Layer* l, double lr) {
    for (int i = 0; i < l->in_size * l->out_size; i++) {
        l->w[i].data -= lr * l->w[i].grad;
    }
    for (int i = 0; i < l->out_size; i++) {
        l->b[i].data -= lr * l->b[i].grad;
    }
}

void train_batch(Net* net, int start, int end, double* x_min, double* x_max, 
                double y_min, double y_max, double* total_loss) {
    zero_grad(&net->hidden);
    zero_grad(&net->output);
    
    double batch_loss = 0;
    for (int i = start; i < end; i++) {
        double pred = predict(net, X_train[i], x_min, x_max);
        double target = (Y_train[i] - y_min) / (y_max - y_min);
        double error = pred - target;
        batch_loss += error * error;
        
        double grad = 2 * error / (end - start);
        backward_layer(&net->output, net->hidden.out, grad);
        backward_layer(&net->hidden, net->inputs, net->output.out[0].grad);
    }
    
    *total_loss += batch_loss;
}

int main() {
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    srand(time(NULL));
    
    Net* net = malloc(sizeof(Net));
    g_net = net;
    
    net->inputs = calloc(12, sizeof(Value));
    net->hidden = create_layer(12, HIDDEN_SIZE);
    net->output = create_layer(HIDDEN_SIZE, 1);
    
    double x_min[12] = {0}, x_max[12] = {0};
    double y_min = Y_train[0], y_max = Y_train[0];
    
    for (int i = 0; i < 406; i++) {
        for (int j = 0; j < 12; j++) {
            if (i == 0 || X_train[i][j] < x_min[j]) x_min[j] = X_train[i][j];
            if (i == 0 || X_train[i][j] > x_max[j]) x_max[j] = X_train[i][j];
        }
        if (Y_train[i] < y_min) y_min = Y_train[i];
        if (Y_train[i] > y_max) y_max = Y_train[i];
    }
    
    double lr = 0.5;
    int epochs = 20000;
    double best_val_rmse = 1e9;
    int patience = 200;
    int no_improve = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0;
        
        // Shuffle training data
        for (int i = 405; i > 0; i--) {
            int j = rand() % (i + 1);
            for (int k = 0; k < 12; k++) {
                double temp = X_train[i][k];
                X_train[i][k] = X_train[j][k];
                X_train[j][k] = temp;
            }
            double temp_y = Y_train[i];
            Y_train[i] = Y_train[j];
            Y_train[j] = temp_y;
        }
        
        // Mini-batch training
        for (int i = 0; i < 406; i += BATCH_SIZE) {
            int batch_end = i + BATCH_SIZE;
            if (batch_end > 406) batch_end = 406;
            train_batch(net, i, batch_end, x_min, x_max, y_min, y_max, &total_loss);
            update_layer(&net->hidden, lr);
            update_layer(&net->output, lr);
        }
        
        // Validation
        if (epoch % 10 == 0) {
            double val_mse = 0;
            for (int i = 0; i < 100; i++) {
                double pred = predict(net, X_test[i], x_min, x_max);
                pred = pred * (y_max - y_min) + y_min;
                val_mse += pow(pred - Y_test[i], 2);
            }
            double val_rmse = sqrt(val_mse / 100);
            printf("Epoch %d: train_loss=%.4f, val_rmse=%.2f, lr=%.6f\n", 
                   epoch, total_loss/406, val_rmse, lr);
            
            if (val_rmse < best_val_rmse) {
                best_val_rmse = val_rmse;
                no_improve = 0;
            } else {
                no_improve++;
                if (no_improve >= patience) {
                    printf("Early stopping at epoch %d\n", epoch);
                    break;
                }
            }
        }
        
        lr *= 0.999;  // Learning rate decay
    }
    
    printf("\nTest Predictions:\n");
    double test_mse = 0;
    for (int i = 0; i < 10; i++) {
        double pred = predict(net, X_test[i], x_min, x_max);
        pred = pred * (y_max - y_min) + y_min;
        printf("Predicted: %.2f, Actual: %.2f\n", pred, Y_test[i]);
        test_mse += pow(pred - Y_test[i], 2);
    }
    printf("\nTest RMSE: %.2f\n", sqrt(test_mse / 10));
    
    cleanup();
    return 0;
}