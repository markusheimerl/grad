#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct Value {
    double data;
    double grad;
    void* backward_ctx;
    void (*backward)(struct Value*);
    struct Value** prev;
    int n_prev;
} Value;

Value* new_value(double data) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->backward_ctx = NULL;
    v->backward = NULL;
    v->prev = NULL;
    v->n_prev = 0;
    return v;
}

typedef struct {
    Value* a;
    Value* b;
} OpContext;

void backward_add(Value* v) {
    OpContext* ctx = (OpContext*)v->backward_ctx;
    ctx->a->grad += v->grad;
    ctx->b->grad += v->grad;
}

Value* add(Value* a, Value* b) {
    Value* out = new_value(a->data + b->data);
    OpContext* ctx = (OpContext*)malloc(sizeof(OpContext));
    ctx->a = a;
    ctx->b = b;
    out->backward_ctx = ctx;
    out->backward = backward_add;
    out->prev = (Value**)malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    return out;
}

void backward_mul(Value* v) {
    OpContext* ctx = (OpContext*)v->backward_ctx;
    ctx->a->grad += ctx->b->data * v->grad;
    ctx->b->grad += ctx->a->data * v->grad;
}

Value* mul(Value* a, Value* b) {
    Value* out = new_value(a->data * b->data);
    OpContext* ctx = (OpContext*)malloc(sizeof(OpContext));
    ctx->a = a;
    ctx->b = b;
    out->backward_ctx = ctx;
    out->backward = backward_mul;
    out->prev = (Value**)malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    return out;
}

void backward_tanh(Value* v) {
    Value* prev = v->prev[0];
    prev->grad += (1.0 - (v->data * v->data)) * v->grad;
}

Value* tanh_val(Value* x) {
    double t = tanh(x->data);
    Value* out = new_value(t);
    out->backward = backward_tanh;
    out->prev = (Value**)malloc(sizeof(Value*));
    out->prev[0] = x;
    out->n_prev = 1;
    return out;
}

void backward_pass(Value* v) {
    if (v->backward) {
        v->backward(v);
    }
    if (v->prev) {
        for (int i = 0; i < v->n_prev; i++) {
            backward_pass(v->prev[i]);
        }
    }
}

Value* rand_value() {
    return new_value(((double)rand() / RAND_MAX) * 2.0 - 1.0);
}

int main() {
    srand(time(NULL));
    
    // Create a simple network (2-4-1)
    Value* w1[4][2];  // First layer weights
    Value* b1[4];     // First layer biases
    Value* w2[4];     // Output layer weights
    Value* b2;        // Output layer bias
    
    // Initialize parameters
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 2; j++) {
            w1[i][j] = rand_value();
        }
        b1[i] = new_value(0.0);
        w2[i] = rand_value();
    }
    b2 = new_value(0.0);
    
    // Training data: XOR
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    // Training loop
    double learning_rate = 0.1;
    
    for(int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0.0;
        
        for(int i = 0; i < 4; i++) {
            // Zero all gradients
            for(int j = 0; j < 4; j++) {
                for(int k = 0; k < 2; k++) {
                    w1[j][k]->grad = 0.0;
                }
                b1[j]->grad = 0.0;
                w2[j]->grad = 0.0;
            }
            b2->grad = 0.0;
            
            // Forward pass
            Value* x1 = new_value(X[i][0]);
            Value* x2 = new_value(X[i][1]);
            
            // Hidden layer
            Value* h[4];
            for(int j = 0; j < 4; j++) {
                h[j] = add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]);
                h[j] = tanh_val(h[j]);
            }
            
            // Output layer
            Value* out = b2;
            for(int j = 0; j < 4; j++) {
                out = add(out, mul(w2[j], h[j]));
            }
            out = tanh_val(out);
            
            // MSE Loss
            Value* diff = add(out, new_value(-Y[i]));
            Value* loss = mul(diff, diff);
            total_loss += loss->data;
            
            // Backward pass
            loss->grad = 1.0;
            backward_pass(loss);
            
            // Update parameters
            for(int j = 0; j < 4; j++) {
                for(int k = 0; k < 2; k++) {
                    w1[j][k]->data -= learning_rate * w1[j][k]->grad;
                }
                b1[j]->data -= learning_rate * b1[j]->grad;
                w2[j]->data -= learning_rate * w2[j]->grad;
            }
            b2->data -= learning_rate * b2->grad;
        }
        
        if(epoch % 100 == 0) {
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    // Test predictions
    printf("\nTesting XOR:\n");
    for(int i = 0; i < 4; i++) {
        Value* x1 = new_value(X[i][0]);
        Value* x2 = new_value(X[i][1]);
        
        Value* h[4];
        for(int j = 0; j < 4; j++) {
            h[j] = add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]);
            h[j] = tanh_val(h[j]);
        }
        
        Value* out = b2;
        for(int j = 0; j < 4; j++) {
            out = add(out, mul(w2[j], h[j]));
        }
        out = tanh_val(out);
        
        printf("Input: (%f, %f) -> Output: %f (Expected: %f)\n", 
               X[i][0], X[i][1], out->data, Y[i]);
    }
    
    return 0;
}