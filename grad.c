#include <time.h>
#include "grad.h"
#include "data.h"

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
    
    printf("Creating synthetic dataset...\n");
    int n = 1000, fx = 4, fy = 3;
    Data* data = synth(n, fx, fy, 0.1);
    char* filename = save_csv("data.csv", data);
    free_data(data);
    
    printf("Loading dataset...\n");
    data = load_csv(filename);
    free(filename);
    if(!data) { printf("Failed to load dataset!\n"); return 1; }
    
    printf("Initializing network...\n");
    int nl = 4, sz[] = {fx, 64, 32, fy};
    Net* net = init_net(nl, sz);
    
    printf("\nTraining:\n");
    printf("%-6s %-12s %-8s\n", "Epoch", "Loss", "LR");
    printf("-------------------------\n");
    
    int epochs = 100;
    double best_loss = INFINITY;
    double prev_loss = INFINITY;
    double** act = malloc(nl * sizeof(double*));
    double** grad = malloc(nl * sizeof(double*));
    for(int i = 0; i < nl; i++) {
        act[i] = malloc(sz[i] * sizeof(double));
        grad[i] = malloc(sz[i] * sizeof(double));
    }
    
    for(int e = 0; e < epochs; e++) {
        double** pred = malloc(n * sizeof(double*));
        for(int i = 0; i < n; i++) {
            pred[i] = malloc(fy * sizeof(double));
            fwd(net, data->X[i], act);
            memcpy(pred[i], act[nl-1], fy * sizeof(double));
            bwd(net, act, data->y[i], grad, prev_loss);
        }
        
        double total_loss = compute_loss(pred, data->y, n, fy);
        if(total_loss < best_loss) best_loss = total_loss;
        prev_loss = total_loss;
        
        if(e % 10 == 0 || e == epochs-1) {
            printf("%-6d %.6f%s %.6f\n", e+1, total_loss, 
                   total_loss == best_loss ? " *" : "  ", net->lr);
        }
        
        for(int i = 0; i < n; i++) free(pred[i]);
        free(pred);
    }
    
    printf("\nTraining complete! Best loss: %.6f\n", best_loss);
    
    for(int i = 0; i < nl; i++) { free(act[i]); free(grad[i]); }
    free(act); free(grad);
    free_net(net);
    free_data(data);
    
    return 0;
}