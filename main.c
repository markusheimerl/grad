#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct Value {
    double data;     // The actual value
    double grad;     // Gradient for backpropagation
    struct Value** prev;  // Previous connected values
    int n_prev;      // Number of previous connections
    void (*backward)(struct Value*);  // Backward function pointer
} Value;

// Creates a new Value with initial data
Value* new_value(double data) {
    Value* v = malloc(sizeof(Value));
    *v = (Value){
        .data = data,
        .grad = 0.0,
        .prev = NULL,
        .n_prev = 0,
        .backward = NULL
    };
    return v;
}

// Adds two Values and sets up backward pass
Value* add(Value* a, Value* b) {
    Value* out = new_value(a->data + b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            v->prev[0]->grad += v->grad;
            v->prev[1]->grad += v->grad;
        }; f;
    });
    return out;
}

// Multiplies two Values and sets up backward pass
Value* mul(Value* a, Value* b) {
    Value* out = new_value(a->data * b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            v->prev[0]->grad += v->prev[1]->data * v->grad;
            v->prev[1]->grad += v->prev[0]->data * v->grad;
        }; f;
    });
    return out;
}

// Applies tanh activation and sets up backward pass
Value* tanh_val(Value* x) {
    Value* out = new_value(tanh(x->data));
    out->prev = malloc(sizeof(Value*));
    out->prev[0] = x;
    out->n_prev = 1;
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            v->prev[0]->grad += (1.0 - (v->data * v->data)) * v->grad;
        }; f;
    });
    return out;
}

// Recursively applies backward pass
void backward_pass(Value* v) {
    if (v->backward) {
        v->backward(v);
    }
    for (int i = 0; i < v->n_prev; i++) {
        backward_pass(v->prev[i]);
    }
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    // Network architecture
    const int HIDDEN_SIZE = 8;
    const double LEARNING_RATE = 0.05;
    const int EPOCHS = 10000;
    
    // Network parameters
    Value* w1[HIDDEN_SIZE][2];  // Input weights
    Value* b1[HIDDEN_SIZE];     // Hidden layer biases
    Value* w2[HIDDEN_SIZE];     // Output weights
    Value* b2;                  // Output bias
    
    // Initialize weights and biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            w1[i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
        b1[i] = new_value(0.0);
        w2[i] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
    }
    b2 = new_value(0.0);
    
    // Training data (XOR)
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        
        // Process each training example
        for (int i = 0; i < 4; i++) {
            // Reset gradients
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) w1[j][k]->grad = 0;
                b1[j]->grad = w2[j]->grad = 0;
            }
            b2->grad = 0;
            
            // Forward pass
            Value* x1 = new_value(X[i][0]);
            Value* x2 = new_value(X[i][1]);
            Value* h[HIDDEN_SIZE];
            
            // Hidden layer
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                h[j] = tanh_val(add(add(mul(w1[j][0], x1), 
                                      mul(w1[j][1], x2)), 
                                  b1[j]));
            }
            
            // Output layer
            Value* out = b2;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                out = add(out, mul(w2[j], h[j]));
            }
            out = tanh_val(out);
            
            // Compute loss
            Value* loss = mul(add(out, new_value(-Y[i])), 
                            add(out, new_value(-Y[i])));
            total_loss += loss->data;
            
            // Backward pass
            loss->grad = 1.0;
            backward_pass(loss);
            
            // Update weights and biases
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) {
                    w1[j][k]->data -= LEARNING_RATE * w1[j][k]->grad;
                }
                b1[j]->data -= LEARNING_RATE * b1[j]->grad;
                w2[j]->data -= LEARNING_RATE * w2[j]->grad;
            }
            b2->data -= LEARNING_RATE * b2->grad;
        }
        
        // Print progress
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    // Test the network
    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        Value* x1 = new_value(X[i][0]);
        Value* x2 = new_value(X[i][1]);
        Value* h[HIDDEN_SIZE];
        
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h[j] = tanh_val(add(add(mul(w1[j][0], x1), 
                                  mul(w1[j][1], x2)), 
                              b1[j]));
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