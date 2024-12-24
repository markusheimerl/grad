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

typedef struct Net {
    Value* w1[HIDDEN_SIZE][2];  // Input -> Hidden weights
    Value* b1[HIDDEN_SIZE];     // Hidden bias
    Value* w2[HIDDEN_SIZE];     // Hidden -> Output weights
    Value* b2;                  // Output bias
    Value* x1;                  // Input 1
    Value* x2;                  // Input 2
    Value* h[HIDDEN_SIZE];      // Hidden layer nodes
    Value* out;                 // Output node
} Net;

Net* create_network() {
    Net* n = malloc(sizeof(Net));
    
    // Initialize inputs
    n->x1 = new_value(0.0);
    n->x2 = new_value(0.0);
    
    // Initialize weights and biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            n->w1[i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
        n->b1[i] = new_value(0.0);
        n->w2[i] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
    }
    n->b2 = new_value(0.0);
    
    // Build nwork structure
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        n->h[j] = tanh_val(add(add(mul(n->w1[j][0], n->x1), mul(n->w1[j][1], n->x2)), n->b1[j]));
    }
    
    n->out = n->b2;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        n->out = add(n->out, mul(n->w2[j], n->h[j]));
    }
    n->out = tanh_val(n->out);
    
    return n;
}

void update_network(Net* n, double learning_rate) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < 2; k++) {
            n->w1[j][k]->data -= learning_rate * n->w1[j][k]->grad;
        }
        n->b1[j]->data -= learning_rate * n->b1[j]->grad;
        n->w2[j]->data -= learning_rate * n->w2[j]->grad;
    }
    n->b2->data -= learning_rate * n->b2->grad;
}

void free_network(Net* n) {
    // Free all allocated memory
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) {
            free(n->w1[i][j]);
        }
        free(n->b1[i]);
        free(n->w2[i]);
        free(n->h[i]);
    }
    free(n->b2);
    free(n->x1);
    free(n->x2);
    free(n->out);
    free(n);
}

// Modified main function:
int main() {
    srand(time(NULL));
    const double LEARNING_RATE = 0.05;
    const int EPOCHS = 10000;
    
    Net* n = create_network();
    Value* visited[1000] = {NULL};
    
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < 4; i++) {
            int visited_size = 0;
            reset_node(n->out, visited, &visited_size);
            
            n->x1->data = X[i][0];
            n->x2->data = X[i][1];
            
            Value* loss = mul(add(n->out, new_value(-Y[i])), add(n->out, new_value(-Y[i])));
            forward_pass(loss);
            total_loss += loss->data;
            
            loss->grad = 1.0;
            backward_pass(loss);
            
            update_network(n, LEARNING_RATE);
        }
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
        }
    }
    
    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        int visited_size = 0;
        reset_node(n->out, visited, &visited_size);
        
        n->x1->data = X[i][0];
        n->x2->data = X[i][1];
        forward_pass(n->out);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", X[i][0], X[i][1], n->out->data, Y[i]);
    }
    
    free_network(n);
    return 0;
}