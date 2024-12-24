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

// Getters
double get_data(const Value* v) {
    return v->data;
}

double get_grad(const Value* v) {
    return v->grad;
}

struct Value** get_prev(const Value* v) {
    return v->prev;
}

int get_n_prev(const Value* v) {
    return v->n_prev;
}

// Setters
void set_data(Value* v, double data) {
    v->data = data;
}

void set_grad(Value* v, double grad) {
    v->grad = grad;
}

void set_prev(Value* v, struct Value** prev) {
    v->prev = prev;
}

void set_n_prev(Value* v, int n_prev) {
    v->n_prev = n_prev;
}

void set_backward(Value* v, void (*backward)(struct Value*)) {
    v->backward = backward;
}

Value* new_value(double data) {
    Value* v = malloc(sizeof(Value));
    if (v == NULL) {
        return NULL;
    }
    set_data(v, data);
    set_grad(v, 0.0);
    set_prev(v, NULL);
    set_n_prev(v, 0);
    set_backward(v, NULL);
    return v;
}

Value* add(Value* a, Value* b) {
    Value* out = new_value(get_data(a) + get_data(b));
    Value** prev = malloc(2 * sizeof(Value*));
    prev[0] = a;
    prev[1] = b;
    set_prev(out, prev);
    set_n_prev(out, 2);
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            set_grad(v->prev[0], get_grad(v->prev[0]) + get_grad(v));
            set_grad(v->prev[1], get_grad(v->prev[1]) + get_grad(v));
        }; f;
    });
    return out;
}

Value* mul(Value* a, Value* b) {
    Value* out = new_value(get_data(a) * get_data(b));
    Value** prev = malloc(2 * sizeof(Value*));
    prev[0] = a;
    prev[1] = b;
    set_prev(out, prev);
    set_n_prev(out, 2);
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            set_grad(v->prev[0], get_grad(v->prev[0]) + get_data(v->prev[1]) * get_grad(v));
            set_grad(v->prev[1], get_grad(v->prev[1]) + get_data(v->prev[0]) * get_grad(v));
        }; f;
    });
    return out;
}

Value* tanh_val(Value* x) {
    Value* out = new_value(tanh(get_data(x)));
    Value** prev = malloc(sizeof(Value*));
    prev[0] = x;
    set_prev(out, prev);
    set_n_prev(out, 1);
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            double current_grad = get_grad(v->prev[0]);
            set_grad(v->prev[0], current_grad + (1.0 - (get_data(v) * get_data(v))) * get_grad(v));
        }; f;
    });
    return out;
}

void backward_pass(Value* v) {
    if (v->backward) {
        v->backward(v);
    }
    for (int i = 0; i < get_n_prev(v); i++) {
        backward_pass(get_prev(v)[i]);
    }
}

#define HIDDEN_SIZE 8

Value* forward(Value* x1, Value* x2, Value* w1[HIDDEN_SIZE][2], 
              Value* b1[HIDDEN_SIZE], Value* w2[HIDDEN_SIZE], Value* b2) {
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
    return tanh_val(out);
}

int main() {
    srand(time(NULL));
    
    const double LEARNING_RATE = 0.05;
    const int EPOCHS = 10000;
    
    Value* w1[HIDDEN_SIZE][2];
    Value* b1[HIDDEN_SIZE];
    Value* w2[HIDDEN_SIZE];
    Value* b2;
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            w1[i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
        b1[i] = new_value(0.0);
        w2[i] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
    }
    b2 = new_value(0.0);
    
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) set_grad(w1[j][k], 0);
                set_grad(b1[j], 0);
                set_grad(w2[j], 0);
            }
            set_grad(b2, 0);
            
            Value* x1 = new_value(X[i][0]);
            Value* x2 = new_value(X[i][1]);
            Value* out = forward(x1, x2, w1, b1, w2, b2);
            
            Value* loss = mul(add(out, new_value(-Y[i])), 
                            add(out, new_value(-Y[i])));
            total_loss += get_data(loss);
            
            set_grad(loss, 1.0);
            backward_pass(loss);
            
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) {
                    set_data(w1[j][k], get_data(w1[j][k]) - LEARNING_RATE * get_grad(w1[j][k]));
                }
                set_data(b1[j], get_data(b1[j]) - LEARNING_RATE * get_grad(b1[j]));
                set_data(w2[j], get_data(w2[j]) - LEARNING_RATE * get_grad(w2[j]));
            }
            set_data(b2, get_data(b2) - LEARNING_RATE * get_grad(b2));
        }
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        Value* x1 = new_value(X[i][0]);
        Value* x2 = new_value(X[i][1]);
        Value* out = forward(x1, x2, w1, b1, w2, b2);
        
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", 
               X[i][0], X[i][1], get_data(out), Y[i]);
    }
    return 0;
}