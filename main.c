#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct Value {
    double data, grad;     
    struct Value** prev;  
    int n_prev;      
    void (*backward)(struct Value*);
    double (*forward)(struct Value*);
} Value;

Value* new_value(double data) {
    Value* v = malloc(sizeof(Value));
    *v = (Value){data, 0.0, NULL, 0, NULL, NULL};
    return v;
}

Value* add(Value* a, Value* b) {
    Value* out = new_value(0.0);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    out->forward = (double(*)(Value*))({
        double f(Value* v) { return v->prev[0]->data + v->prev[1]->data; }; f;
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
        double f(Value* v) { return v->prev[0]->data * v->prev[1]->data; }; f;
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
        double f(Value* v) { return tanh(v->prev[0]->data); }; f;
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
        for (int i = 0; i < v->n_prev; i++) forward_pass(v->prev[i]);
        if (v->forward) v->data = v->forward(v);
    }
}

void backward_pass(Value* v) {
    if (v->backward) v->backward(v);
    for (int i = 0; i < v->n_prev; i++) backward_pass(v->prev[i]);
}


typedef struct Net {
    Value**** w;  // weights[layer][neuron][input]
    Value*** b;   // biases[layer][neuron]
    Value** x;    // inputs[input]
    Value*** h;   // hidden[layer][neuron]
    Value** out;  // outputs[output]
    int n_inputs;
    int n_outputs;
    int n_hidden_layers;
    int hidden_size;
} Net;

Net* create_network(int n_inputs, int n_outputs, int n_hidden_layers, int hidden_size) {
    Net* n = malloc(sizeof(Net));
    n->n_inputs = n_inputs;
    n->n_outputs = n_outputs;
    n->n_hidden_layers = n_hidden_layers;
    n->hidden_size = hidden_size;
    
    // Allocate inputs
    n->x = malloc(n_inputs * sizeof(Value*));
    for(int i = 0; i < n_inputs; i++) {
        n->x[i] = new_value(0.0);
    }
    
    // Allocate weights, biases and hidden layers
    n->w = malloc((n_hidden_layers + 1) * sizeof(Value***));
    n->b = malloc((n_hidden_layers + 1) * sizeof(Value**));
    n->h = malloc(n_hidden_layers * sizeof(Value**));
    
    // First layer (input -> first hidden)
    n->w[0] = malloc(hidden_size * sizeof(Value**));
    n->b[0] = malloc(hidden_size * sizeof(Value*));
    n->h[0] = malloc(hidden_size * sizeof(Value*));
    
    for(int i = 0; i < hidden_size; i++) {
        n->w[0][i] = malloc(n_inputs * sizeof(Value*));
        for(int j = 0; j < n_inputs; j++) {
            n->w[0][i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
        n->b[0][i] = new_value(0.0);
    }
    
    // Hidden layers
    for(int layer = 1; layer < n_hidden_layers; layer++) {
        n->w[layer] = malloc(hidden_size * sizeof(Value**));
        n->b[layer] = malloc(hidden_size * sizeof(Value*));
        n->h[layer] = malloc(hidden_size * sizeof(Value*));
        
        for(int i = 0; i < hidden_size; i++) {
            n->w[layer][i] = malloc(hidden_size * sizeof(Value*));
            for(int j = 0; j < hidden_size; j++) {
                n->w[layer][i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
            }
            n->b[layer][i] = new_value(0.0);
        }
    }
    
    // Output layer
    n->w[n_hidden_layers] = malloc(n_outputs * sizeof(Value**));
    n->b[n_hidden_layers] = malloc(n_outputs * sizeof(Value*));
    n->out = malloc(n_outputs * sizeof(Value*));
    
    for(int i = 0; i < n_outputs; i++) {
        n->w[n_hidden_layers][i] = malloc(hidden_size * sizeof(Value*));
        for(int j = 0; j < hidden_size; j++) {
            n->w[n_hidden_layers][i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
        n->b[n_hidden_layers][i] = new_value(0.0);
    }
    
    // Build network connections
    // First layer
    for(int i = 0; i < hidden_size; i++) {
        Value* sum = n->b[0][i];
        for(int j = 0; j < n_inputs; j++) {
            sum = add(sum, mul(n->w[0][i][j], n->x[j]));
        }
        n->h[0][i] = tanh_val(sum);
    }
    
    // Hidden layers
    for(int layer = 1; layer < n_hidden_layers; layer++) {
        for(int i = 0; i < hidden_size; i++) {
            Value* sum = n->b[layer][i];
            for(int j = 0; j < hidden_size; j++) {
                sum = add(sum, mul(n->w[layer][i][j], n->h[layer-1][j]));
            }
            n->h[layer][i] = tanh_val(sum);
        }
    }
    
    // Output layer
    for(int i = 0; i < n_outputs; i++) {
        Value* sum = n->b[n_hidden_layers][i];
        for(int j = 0; j < hidden_size; j++) {
            sum = add(sum, mul(n->w[n_hidden_layers][i][j], n->h[n_hidden_layers-1][j]));
        }
        n->out[i] = tanh_val(sum);
    }
    
    return n;
}

void update_network(Net* n, double lr) {
    // First layer
    for(int i = 0; i < n->hidden_size; i++) {
        for(int j = 0; j < n->n_inputs; j++) {
            n->w[0][i][j]->data -= lr * n->w[0][i][j]->grad;
        }
        n->b[0][i]->data -= lr * n->b[0][i]->grad;
    }
    
    // Hidden layers
    for(int layer = 1; layer < n->n_hidden_layers; layer++) {
        for(int i = 0; i < n->hidden_size; i++) {
            for(int j = 0; j < n->hidden_size; j++) {
                n->w[layer][i][j]->data -= lr * n->w[layer][i][j]->grad;
            }
            n->b[layer][i]->data -= lr * n->b[layer][i]->grad;
        }
    }
    
    // Output layer
    for(int i = 0; i < n->n_outputs; i++) {
        for(int j = 0; j < n->hidden_size; j++) {
            n->w[n->n_hidden_layers][i][j]->data -= lr * n->w[n->n_hidden_layers][i][j]->grad;
        }
        n->b[n->n_hidden_layers][i]->data -= lr * n->b[n->n_hidden_layers][i]->grad;
    }
}

void reset_network(Value* top) {
    Value** stack = malloc(1000 * sizeof(Value*));
    Value** visited = malloc(1000 * sizeof(Value*));
    int stack_size = 0, visited_size = 0;
    
    stack[stack_size++] = top;
    while (stack_size > 0) {
        Value* current = stack[--stack_size];
        int already_visited = 0;
        for (int i = 0; i < visited_size && !already_visited; i++)
            if (current == visited[i]) already_visited = 1;
        
        if (!already_visited) {
            visited[visited_size++] = current;
            current->grad = 0.0;
            for (int i = 0; i < current->n_prev; i++)
                stack[stack_size++] = current->prev[i];
        }
    }
    free(stack);
    free(visited);
}

void free_network(Value* top) {
    Value** stack = malloc(1000 * sizeof(Value*));
    Value** visited = malloc(1000 * sizeof(Value*));
    int stack_size = 0, visited_size = 0;
    
    stack[stack_size++] = top;
    while (stack_size > 0) {
        Value* current = stack[--stack_size];
        int already_visited = 0;
        for (int i = 0; i < visited_size; i++)
            if (current == visited[i]) {
                already_visited = 1;
                break;
            }
        
        if (!already_visited) {
            visited[visited_size++] = current;
            for (int i = 0; i < current->n_prev; i++)
                stack[stack_size++] = current->prev[i];
        }
    }
    
    for (int i = visited_size - 1; i >= 0; i--) {
        free(visited[i]->prev);
        free(visited[i]);
    }
    
    free(stack);
    free(visited);
}

int main() {
    srand(time(NULL));
    Net* n = create_network(2, 1, 1, 8);
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};

    Value* target = new_value(0.0);
    Value* loss = mul(add(n->out[0], target), add(n->out[0], target));
    
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < 4; i++) {
            reset_network(loss);
            n->x[0]->data = X[i][0];
            n->x[1]->data = X[i][1];
            target->data = -Y[i];
            forward_pass(loss);
            total_loss += loss->data;
            loss->grad = 1.0;
            backward_pass(loss);
            update_network(n, 0.05);
        }
        if (epoch % 1000 == 0)
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
    }
    
    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        n->x[0]->data = X[i][0];
        n->x[1]->data = X[i][1];
        forward_pass(n->out[0]);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", 
               X[i][0], X[i][1], n->out[0]->data, Y[i]);
    }

    free_network(loss);
    free(n);
    return 0;
}