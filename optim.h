#ifndef OPTIM_H
#define OPTIM_H

#define B1 0.9
#define B2 0.999
#define EPS 1e-8
#define DECAY 0.01
#define BATCH_SIZE 128

typedef struct {
    void (*update)(double *p, double *g, double *aux, int n, int t, double lr);
    int aux_doubles_per_param;
} optimizer_t;

void adam_update(double *p, double *g, double *aux, int n, int t, double lr) {
    double *m = aux;
    double *v = aux + n;
    double lrt = lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
    for(int i = 0; i < n; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lrt * (m[i] / (sqrt(v[i]) + EPS) + DECAY * p[i]);
    }
}

void sgd_update(double *p, double *g, double *aux, int n, int t, double lr) {
    for(int i = 0; i < n; i++) {
        p[i] -= lr * g[i];
    }
}

void sga_update(double *p, double *g, double *aux, int n, int t, double lr) {
    for(int i = 0; i < n; i++) {
        p[i] += lr * g[i];
    }
}

void mbsgd_update(double *p, double *g, double *aux, int n, int t, double lr) {
    double *acc = aux;
    int batch_idx = (t - 1) % BATCH_SIZE;
    
    for(int i = 0; i < n; i++) {
        acc[i] += g[i];
    }

    if(batch_idx == BATCH_SIZE - 1) {
        for(int i = 0; i < n; i++) {
            p[i] -= (lr/BATCH_SIZE) * acc[i];
            acc[i] = 0.0;
        }
    }
}

void mbsga_update(double *p, double *g, double *aux, int n, int t, double lr) {
    double *acc = aux;
    int batch_idx = (t - 1) % BATCH_SIZE;
    
    for(int i = 0; i < n; i++) {
        acc[i] += g[i];
    }
    
    if(batch_idx == BATCH_SIZE - 1) {
        for(int i = 0; i < n; i++) {
            p[i] += (lr/BATCH_SIZE) * acc[i];
            acc[i] = 0.0;
        }
    }
}

void rmsprop_update(double *p, double *g, double *aux, int n, int t, double lr) {
    const double alpha = 0.9;
    double *v = aux;
    for(int i = 0; i < n; i++) {
        v[i] = alpha * v[i] + (1-alpha) * g[i] * g[i];
        p[i] -= lr * g[i] / (sqrt(v[i]) + EPS);
    }
}

void adagrad_update(double *p, double *g, double *aux, int n, int t, double lr) {
    double *h = aux;
    for(int i = 0; i < n; i++) {
        h[i] += g[i] * g[i];
        p[i] -= lr * g[i] / (sqrt(h[i]) + EPS);
    }
}

void lion_update(double *p, double *g, double *aux, int n, int t, double lr) {
    const double beta1 = 0.9, beta2 = 0.99;
    double *m = aux;
    for(int i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1-beta1) * g[i];
        double update = (beta2 * m[i] + (1-beta2) * g[i]);
        p[i] -= lr * (update > 0 ? 1 : (update < 0 ? -1 : 0));
    }
}

void adamw_update(double *p, double *g, double *aux, int n, int t, double lr) {
    double *m = aux;
    double *v = aux + n;
    double *acc = aux + 2*n;  // New accumulator for mini-batches
    int batch_idx = (t - 1) % BATCH_SIZE;
    
    // Accumulate gradients
    for(int i = 0; i < n; i++) {
        acc[i] += g[i];
    }

    // Only update when batch is complete
    if(batch_idx == BATCH_SIZE - 1) {
        double lrt = lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
        
        for(int i = 0; i < n; i++) {
            // Use accumulated gradients divided by batch size
            double batch_grad = acc[i] / BATCH_SIZE;
            
            m[i] = B1 * m[i] + (1-B1) * batch_grad;
            v[i] = B2 * v[i] + (1-B2) * batch_grad * batch_grad;
            p[i] -= lrt * (m[i] / (sqrt(v[i]) + EPS)) + lr * DECAY * p[i];
            
            // Reset accumulator
            acc[i] = 0.0;
        }
    }
}

void null_update(double *p, double *g, double *aux, int n, int t, double lr){}

optimizer_t adam = {.update = adam_update, .aux_doubles_per_param = 2};
optimizer_t sgd = {.update = sgd_update, .aux_doubles_per_param = 0};
optimizer_t sga = {.update = sga_update, .aux_doubles_per_param = 0};
optimizer_t mbsgd = {.update = mbsgd_update, .aux_doubles_per_param = 1};
optimizer_t mbsga = {.update = mbsga_update, .aux_doubles_per_param = 1};
optimizer_t rmsprop = {.update = rmsprop_update, .aux_doubles_per_param = 1};
optimizer_t adagrad = {.update = adagrad_update, .aux_doubles_per_param = 1};
optimizer_t lion = {.update = lion_update, .aux_doubles_per_param = 1};
optimizer_t adamw = {.update = adamw_update, .aux_doubles_per_param = 3};
optimizer_t null_opt = {.update = null_update, .aux_doubles_per_param = 0};

#endif // OPTIM_H