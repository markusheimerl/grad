#include "grad.h"
#include "data.h"

double loss(double** pred, double** tgt, int n, int m) {
    double l = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++) {
            double d = pred[i][j] - tgt[i][j];
            l += d * d;
        }
    return l / (n * m);
}

int main() {
    srand(time(NULL));
    
    // Create dataset
    int n = 1000, fx = 4, fy = 3;
    Data* data = synth(n, fx, fy, 0.1);
    save_csv("data.csv", data);
    
    // Create network
    int nl = 4, sz[] = {fx, 64, 32, fy};
    Net* net = init_net(nl, sz);
    
    // Training setup
    int epochs = 100;
    double lr = 0.001;
    double** act = malloc(nl * sizeof(double*));
    double** grad = malloc(nl * sizeof(double*));
    for(int i = 0; i < nl; i++) {
        act[i] = malloc(sz[i] * sizeof(double));
        grad[i] = malloc(sz[i] * sizeof(double));
    }
    
    // Train
    for(int e = 0; e < epochs; e++) {
        double** pred = malloc(n * sizeof(double*));
        for(int i = 0; i < n; i++) {
            pred[i] = malloc(fy * sizeof(double));
            fwd(net, data->X[i], act);
            memcpy(pred[i], act[nl-1], fy * sizeof(double));
            bwd(net, act, data->y[i], grad, lr);
        }
        printf("Epoch %d: Loss = %f\n", e+1, loss(pred, data->y, n, fy));
        for(int i = 0; i < n; i++) free(pred[i]);
        free(pred);
    }
    
    // Cleanup
    for(int i = 0; i < nl; i++) { free(act[i]); free(grad[i]); }
    free(act); free(grad);
    free_net(net);
    free_data(data);
    
    return 0;
}
