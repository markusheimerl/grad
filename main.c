#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct Value {
    double data, grad;
    struct Value** prev;
    int n_prev;
    void (*backward)(struct Value*);
} Value;

Value* new_value(double data) {
    Value* v = malloc(sizeof(Value));
    *v = (Value){data, 0.0, NULL, 0, NULL};
    return v;
}

Value* add(Value* a, Value* b) {
    Value* out = new_value(a->data + b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    out->backward = (void(*)(Value*))({void f(Value* v) {
        v->prev[0]->grad += v->grad;
        v->prev[1]->grad += v->grad;
    }; f;});
    return out;
}

Value* mul(Value* a, Value* b) {
    Value* out = new_value(a->data * b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->n_prev = 2;
    out->backward = (void(*)(Value*))({void f(Value* v) {
        v->prev[0]->grad += v->prev[1]->data * v->grad;
        v->prev[1]->grad += v->prev[0]->data * v->grad;
    }; f;});
    return out;
}

Value* tanh_val(Value* x) {
    Value* out = new_value(tanh(x->data));
    out->prev = malloc(sizeof(Value*));
    out->prev[0] = x;
    out->n_prev = 1;
    out->backward = (void(*)(Value*))({void f(Value* v) {
        v->prev[0]->grad += (1.0 - (v->data * v->data)) * v->grad;
    }; f;});
    return out;
}

void backward_pass(Value* v) {
    if (v->backward) v->backward(v);
    for (int i = 0; i < v->n_prev; i++) backward_pass(v->prev[i]);
}

int main() {
    srand(time(NULL));
    const int HIDDEN_SIZE = 8;
    Value* w1[HIDDEN_SIZE][2], *b1[HIDDEN_SIZE], *w2[HIDDEN_SIZE], *b2;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < 2; j++) 
            w1[i][j] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        b1[i] = new_value(0.0);
        w2[i] = new_value(((double)rand() / RAND_MAX) * 0.2 - 0.1);
    }
    b2 = new_value(0.0);

    double X[][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[] = {0, 1, 1, 0};
    double learning_rate = 0.05;

    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++) w1[j][k]->grad = 0;
                b1[j]->grad = w2[j]->grad = 0;
            }
            b2->grad = 0;

            Value* x1 = new_value(X[i][0]);
            Value* x2 = new_value(X[i][1]);
            Value* h[HIDDEN_SIZE];

            for (int j = 0; j < HIDDEN_SIZE; j++)
                h[j] = tanh_val(add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]));

            Value* out = b2;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                out = add(out, mul(w2[j], h[j]));
            out = tanh_val(out);

            Value* loss = mul(add(out, new_value(-Y[i])), add(out, new_value(-Y[i])));
            total_loss += loss->data;

            loss->grad = 1.0;
            backward_pass(loss);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < 2; k++)
                    w1[j][k]->data -= learning_rate * w1[j][k]->grad;
                b1[j]->data -= learning_rate * b1[j]->grad;
                w2[j]->data -= learning_rate * w2[j]->grad;
            }
            b2->data -= learning_rate * b2->grad;
        }

        if (epoch % 1000 == 0)
            printf("Epoch %d: Avg Loss = %f\n", epoch, total_loss / 4);
    }

    printf("\nTesting XOR:\n");
    for (int i = 0; i < 4; i++) {
        Value* x1 = new_value(X[i][0]);
        Value* x2 = new_value(X[i][1]);
        Value* h[HIDDEN_SIZE];

        for (int j = 0; j < HIDDEN_SIZE; j++)
            h[j] = tanh_val(add(add(mul(w1[j][0], x1), mul(w1[j][1], x2)), b1[j]));

        Value* out = b2;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            out = add(out, mul(w2[j], h[j]));
        out = tanh_val(out);

        printf("Input: (%g, %g) -> Output: %g (Expected: %g)\n", 
               X[i][0], X[i][1], out->data, Y[i]);
    }
    return 0;
}