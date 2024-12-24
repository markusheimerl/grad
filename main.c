#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define HIDDEN_SIZE 8

typedef struct Value {
    double data, grad;     
    struct Value** prev;  
    int n_prev;      
    void (*backward)(struct Value*);
    double (*forward)(struct Value*);
} Value;

typedef struct Net {
    Value* w1[HIDDEN_SIZE][2];
    Value* b1[HIDDEN_SIZE];
    Value* w2[HIDDEN_SIZE];
    Value* b2;
    Value* x1;
    Value* x2;
    Value* h[HIDDEN_SIZE];
    Value* out;
} Net;

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

Net* create_network() {
    Net* n = malloc(sizeof(Net));
    n->x1 = new_value(0.0);
    n->x2 = new_value(0.0);
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) 
            n->w1[i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        n->b1[i] = new_value(0.0);
        n->w2[i] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        n->h[i] = tanh_val(add(add(mul(n->w1[i][0], n->x1), mul(n->w1[i][1], n->x2)), n->b1[i]));
    }
    
    n->b2 = new_value(0.0);
    n->out = n->b2;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        n->out = add(n->out, mul(n->w2[j], n->h[j]));
    n->out = tanh_val(n->out);
    return n;
}

void update_network(Net* n, double lr) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < 2; k++) n->w1[j][k]->data -= lr * n->w1[j][k]->grad;
        n->b1[j]->data -= lr * n->b1[j]->grad;
        n->w2[j]->data -= lr * n->w2[j]->grad;
    }
    n->b2->data -= lr * n->b2->grad;
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
    // Create arrays to keep track of nodes to process and nodes already freed
    Value** stack = malloc(1000 * sizeof(Value*));
    Value** visited = malloc(1000 * sizeof(Value*));
    int stack_size = 0, visited_size = 0;
    
    // Start with the top node
    stack[stack_size++] = top;
    
    // Process all nodes in the graph
    while (stack_size > 0) {
        Value* current = stack[--stack_size];
        
        // Check if we've already visited this node
        int already_visited = 0;
        for (int i = 0; i < visited_size && !already_visited; i++) {
            if (current == visited[i]) {
                already_visited = 1;
            }
        }
        
        // If not visited, process it
        if (!already_visited) {
            // Add to visited list
            visited[visited_size++] = current;
            
            // Add all previous nodes to the stack
            for (int i = 0; i < current->n_prev; i++) {
                stack[stack_size++] = current->prev[i];
            }
        }
    }
    
    // Free all nodes in reverse order (to handle dependencies correctly)
    for (int i = visited_size - 1; i >= 0; i--) {
        if (visited[i]->prev) {
            free(visited[i]->prev);
        }
        free(visited[i]);
    }
    
    // Free the temporary arrays
    free(stack);
    free(visited);
}

int main() {
    srand(time(NULL));
    Net* n = create_network();
    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};

    Value* target1 = new_value(0.0);
    Value* target2 = new_value(0.0);
    Value* diff1 = add(n->out, target1);
    Value* diff2 = add(n->out, target2);
    Value* loss = mul(diff1, diff2);
    
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < 4; i++) {
            reset_network(loss);
            n->x1->data = X[i][0];
            n->x2->data = X[i][1];
            target1->data = -Y[i];
            target2->data = -Y[i];
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
        n->x1->data = X[i][0];
        n->x2->data = X[i][1];
        forward_pass(n->out);
        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", X[i][0], X[i][1], n->out->data, Y[i]);
    }

    free_network(loss);
    free(n);
    return 0;
}