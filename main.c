#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct Value {
    double data;
    double grad;
    struct Value** prev;
    int n_prev;
    void (*backward)(struct Value*);
} Value;

Value* new_value(double data) {
    Value* v = malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->prev = NULL;
    v->n_prev = 0;
    v->backward = NULL;
    return v;
}

void backward_add(Value* v) {
    v->prev[0]->grad += v->grad;
    v->prev[1]->grad += v->grad;
}

Value* add(Value* a, Value* b) {
    Value* out = new_value(a->data + b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    out->backward = backward_add;
    return out;
}

void backward_mul(Value* v) {
    v->prev[0]->grad += v->prev[1]->data * v->grad;
    v->prev[1]->grad += v->prev[0]->data * v->grad;
}

Value* mul(Value* a, Value* b) {
    Value* out = new_value(a->data * b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    out->backward = backward_mul;
    return out;
}

void backward_tanh(Value* v) {
    v->prev[0]->grad += (1.0 - (v->data * v->data)) * v->grad;
}

Value* tanh_val(Value* x) {
    Value* out = new_value(tanh(x->data));
    out->prev = malloc(sizeof(Value*));
    out->prev[0] = x;
    out->n_prev = 1;
    out->backward = backward_tanh;
    return out;
}

void backward_pass(Value* v) {
    if (v->backward) {
        v->backward(v);
    }
    for (int i = 0; i < v->n_prev; i++) {
        backward_pass(v->prev[i]);
    }
}

Value* rand_value() {
    return new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);  // Between -0.1 and 0.1
}

int main() {
    srand(time(NULL));

    // Network architecture: 2-8-1 (increased hidden layer size)
    const int HIDDEN_SIZE = 8;
    Value* w1[HIDDEN_SIZE][2];
    Value* b1[HIDDEN_SIZE];
    Value* w2[HIDDEN_SIZE];
    Value* b2;

    // Initialize parameters
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            w1[i][j] = rand_value();
        }
        b1[i] = new_value(0.0);
        w2[i] = rand_value();
    }
    b2 = new_value(0.0);

    // Training data (XOR)
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};  // Back to 0 and 1
    
    double learning_rate = 0.05;
    int epochs = 10000;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        // Shuffle training data
        if (epoch % 100 == 0) {
            for (int i = 0; i < 4; i++) {
                int j = rand() % 4;
                double temp_x0 = X[i][0];
                double temp_x1 = X[i][1];
                double temp_y = Y[i];
                X[i][0] = X[j][0];
                X[i][1] = X[j][1];
                Y[i] = Y[j];
                X[j][0] = temp_x0;
                X[j][1] = temp_x1;
                Y[j] = temp_y;
            }
        }

        for (int i = 0; i < 4; i++) {
            // Zero gradients
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) {
                    w1[j][k]->grad = 0;
                }
                b1[j]->grad = 0;
                w2[j]->grad = 0;
            }
            b2->grad = 0;

            // Forward pass
            Value* x1 = new_value(X[i][0]);
            Value* x2 = new_value(X[i][1]);

            // Hidden layer
            Value* h[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                Value* sum = add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]);
                h[j] = tanh_val(sum);
            }

            // Output layer
            Value* out = b2;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                out = add(out, mul(w2[j], h[j]));
            }
            out = tanh_val(out);

            // Loss (MSE)
            Value* diff = add(out, new_value(-Y[i]));
            Value* loss = mul(diff, diff);
            total_loss += loss->data;

            // Backward pass
            loss->grad = 1.0;
            backward_pass(loss);

            // Update parameters
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) {
                    w1[j][k]->data -= learning_rate * w1[j][k]->grad;
                }
                b1[j]->data -= learning_rate * b1[j]->grad;
                w2[j]->data -= learning_rate * w2[j]->grad;
            }
            b2->data -= learning_rate * b2->grad;
        }

        if (epoch % 1000 == 0) {
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
        }
    }

    // Test
    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        Value* x1 = new_value(X[i][0]);
        Value* x2 = new_value(X[i][1]);

        Value* h[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h[j] = tanh_val(add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]));
        }

        Value* out = b2;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            out = add(out, mul(w2[j], h[j]));
        }
        out = tanh_val(out);

        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", 
               X[i][0], X[i][1], out->data, Y[i]);
    }

    return 0;
}