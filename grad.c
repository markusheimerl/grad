#include <time.h>
#include "data.h"
#include "grad.h"

double compute_loss(double** pred, double** tgt, int n, int m) {
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
    
    Data* data = synth(1000, 4, 3, 0.1);
    char* fname = save_csv("data.csv", data);
    free_data(data);
    
    if(!(data = load_csv(fname))) { printf("Load failed!\n"); return 1; }
    free(fname);
    
    int sz[] = {4, 128, 64, 32, 3};
    Net* net = init_net(5, sz);
    
    printf("\nTraining:\n%-6s %-12s %-8s\n", "Epoch", "Loss", "LR");
    printf("-------------------------\n");
    
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(sz[i] * sizeof(double));
        grad[i] = malloc(sz[i] * sizeof(double));
    }
    
    double best_loss = INFINITY, prev_loss = INFINITY;
    for(int e = 0; e < 10; e++) {
        double** pred = malloc(data->n * sizeof(double*));
        for(int i = 0; i < data->n; i++) {
            pred[i] = malloc(data->fy * sizeof(double));
            fwd(net, data->X[i], act);
            memcpy(pred[i], act[4], data->fy * sizeof(double));
            bwd(net, act, data->y[i], grad, prev_loss);
        }
        
        double loss = compute_loss(pred, data->y, data->n, data->fy);
        if(loss < best_loss) best_loss = loss;
        prev_loss = loss;
        
        if(e % 2 == 0 || e == 9)
            printf("%-6d %.6f%s %.6e\n", e+1, loss, loss == best_loss ? " *" : "  ", net->lr);
        
        for(int i = 0; i < data->n; i++) free(pred[i]);
        free(pred);
    }
    
    save_weights(net, "weights.bin");
    free_net(net);
    
    char* last_weights = get_timestamp_filename("weights.bin");
    net = load_weights(last_weights);
    free(last_weights);
    
    double** pred = malloc(data->n * sizeof(double*));
    for(int i = 0; i < data->n; i++) {
        pred[i] = malloc(data->fy * sizeof(double));
        fwd(net, data->X[i], act);
        memcpy(pred[i], act[4], data->fy * sizeof(double));
    }
    
    printf("\nLoaded weights loss: %.6f\n", compute_loss(pred, data->y, data->n, data->fy));
    
    for(int i = 0; i < data->n; i++) free(pred[i]);
    free(pred);
    for(int i = 0; i < 5; i++) { free(act[i]); free(grad[i]); }
    free(act); free(grad);
    free_net(net);
    free_data(data);
    
    return 0;
}