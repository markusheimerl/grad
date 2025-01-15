#include <time.h>
#include "grad.h"

#define N_EPOCHS 20

double compute_loss(double** pred, double** tgt, int n, int m) {
    double l = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++) {
            double d = pred[i][j] - tgt[i][j];
            l += d * d;
        }
    return l / (n * m);
}

double** get_predictions(Net* net, Data* data, double** act) {
    double** pred = malloc(data->n * sizeof(double*));
    for(int i = 0; i < data->n; i++) {
        pred[i] = malloc(data->fy * sizeof(double));
        fwd(net, data->X[i], act);
        memcpy(pred[i], act[4], data->fy * sizeof(double));
    }
    return pred;
}

void free_predictions(double** pred, int n) {
    for(int i = 0; i < n; i++) free(pred[i]);
    free(pred);
}

int main() {
    srand(time(NULL));
    
    Data* data = synth(1000, 4, 3, 0.1);
    
    char data_file[100], weights_file[100];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(data_file, sizeof(data_file), "%Y%m%d_%H%M%S_data.csv", t);
    strftime(weights_file, sizeof(weights_file), "%Y%m%d_%H%M%S_weights.bin", t);
    
    save_csv(data_file, data);

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
    for(int e = 0; e < N_EPOCHS; e++) {
        for(int i = 0; i < data->n; i++) {
            fwd(net, data->X[i], act);
            
            for(int j = 0; j < data->fy; j++) {
                double diff = act[4][j] - data->y[i][j];
                grad[4][j] = 2 * diff;
            }
            
            bwd(net, act, grad);
        }
        
        double** pred = get_predictions(net, data, act);
        double loss = compute_loss(pred, data->y, data->n, data->fy);
        free_predictions(pred, data->n);
        
        if(loss > prev_loss) net->lr *= 0.95;
        else net->lr *= 1.05;
        net->lr = fmax(1e-9, fmin(0.01, net->lr));
        
        if(loss < best_loss) best_loss = loss;
        prev_loss = loss;
        
        if(e % 2 == 0 || e == N_EPOCHS-1)
            printf("%-6d %.6f%s %.6e\n", e+1, loss, loss == best_loss ? " *" : "  ", net->lr);
    }
    
    save_weights(weights_file, net);
    free_net(net);
    free_data(data);

    printf("\nVerification:\n");
    net = load_weights(weights_file);
    data = load_csv(data_file, 4, 3);
    
    double** pred = get_predictions(net, data, act);
    printf("Loaded weights loss: %.6f\n", compute_loss(pred, data->y, data->n, data->fy));
    
    free_predictions(pred, data->n);
    for(int i = 0; i < 5; i++) { free(act[i]); free(grad[i]); }
    free(act); free(grad);
    free_net(net);
    free_data(data);
    
    return 0;
}