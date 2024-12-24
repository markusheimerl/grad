#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef struct Value {
    double data;     
    double grad;     
    struct Value** prev;  
    int n_prev;      
    void (*backward)(struct Value*);
    double (*forward)(struct Value*);
} Value;

Value* new_value(double data) {
    Value* v = malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->prev = NULL;
    v->n_prev = 0;
    v->backward = NULL;
    v->forward = NULL;
    return v;
}

Value* add(Value* a, Value* b) {
    Value* out = new_value(0.0);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    
    out->forward = (double(*)(Value*))({
        double f(Value* v) {
            return v->prev[0]->data + v->prev[1]->data;
        }; f;
    });
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            v->prev[0]->grad += v->grad;
            v->prev[1]->grad += v->grad;
        }; f;
    });
    return out;
}

Value* mul(Value* a, Value* b) {
    Value* out = new_value(0.0);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    
    out->forward = (double(*)(Value*))({
        double f(Value* v) {
            return v->prev[0]->data * v->prev[1]->data;
        }; f;
    });
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            v->prev[0]->grad += v->prev[1]->data * v->grad;
            v->prev[1]->grad += v->prev[0]->data * v->grad;
        }; f;
    });
    return out;
}

Value* tanh_val(Value* x) {
    Value* out = new_value(0.0);
    out->prev = malloc(sizeof(Value*));
    out->prev[0] = x;
    out->n_prev = 1;
    
    out->forward = (double(*)(Value*))({
        double f(Value* v) {
            return tanh(v->prev[0]->data);
        }; f;
    });
    
    out->backward = (void(*)(Value*))({
        void f(Value* v) {
            v->prev[0]->grad += (1.0 - (v->data * v->data)) * v->grad;
        }; f;
    });
    return out;
}

void forward_pass(Value* v) {
    if (v->n_prev > 0) {
        for (int i = 0; i < v->n_prev; i++) {
            forward_pass(v->prev[i]);
        }
        if (v->forward) v->data = v->forward(v);
    }
}

void backward_pass(Value* v) {
    if (v->backward) v->backward(v);
    for (int i = 0; i < v->n_prev; i++) {
        backward_pass(v->prev[i]);
    }
}

void reset_node(Value* v, Value** visited, int* visited_size) {
    for (int i = 0; i < *visited_size; i++) {
        if (v == visited[i]) return;
    }
    visited[*visited_size] = v;
    (*visited_size)++;
    v->grad = 0.0;
    for (int i = 0; i < v->n_prev; i++) {
        reset_node(v->prev[i], visited, visited_size);
    }
}

#define HIDDEN_SIZE 8

int main() {
    srand(time(NULL));
    const double LEARNING_RATE = 0.05;
    const int EPOCHS = 10000;
    
    Value* w1[HIDDEN_SIZE][2], *b1[HIDDEN_SIZE], *w2[HIDDEN_SIZE], *b2;
    Value* x1 = new_value(0.0), *x2 = new_value(0.0);
    Value* visited[1000] = {NULL};
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            w1[i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
        b1[i] = new_value(0.0);
        w2[i] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
    }
    b2 = new_value(0.0);
    
    Value* h[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        h[j] = tanh_val(add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]));
    }
    
    Value* out = b2;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        out = add(out, mul(w2[j], h[j]));
    }
    out = tanh_val(out);
    
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < 4; i++) {
            int visited_size = 0;
            reset_node(out, visited, &visited_size);
            
            x1->data = X[i][0];
            x2->data = X[i][1];
            
            Value* loss = mul(add(out, new_value(-Y[i])), add(out, new_value(-Y[i])));
            forward_pass(loss);
            total_loss += loss->data;
            
            loss->grad = 1.0;
            backward_pass(loss);
            
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) {
                    w1[j][k]->data -= LEARNING_RATE * w1[j][k]->grad;
                }
                b1[j]->data -= LEARNING_RATE * b1[j]->grad;
                w2[j]->data -= LEARNING_RATE * w2[j]->grad;
            }
            b2->data -= LEARNING_RATE * b2->grad;
        }
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        int visited_size = 0;
        reset_node(out, visited, &visited_size);
        
        x1->data = X[i][0];
        x2->data = X[i][1];
        forward_pass(out);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", 
               X[i][0], X[i][1], out->data, Y[i]);
    }
    
    return 0;
}