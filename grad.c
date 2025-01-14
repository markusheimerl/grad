#include <time.h>
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
    
    // Generate and save data
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
    for(int e = 0; e < 10; e++) {
        double** pred = malloc(data->n * sizeof(double*));
        double curr_loss = 0;
        
        for(int i = 0; i < data->n; i++) {
            pred[i] = malloc(data->fy * sizeof(double));
            fwd(net, data->X[i], act);
            memcpy(pred[i], act[4], data->fy * sizeof(double));
            
            // Compute output gradients and loss
            for(int j = 0; j < data->fy; j++) {
                double diff = act[4][j] - data->y[i][j];
                grad[4][j] = 2 * diff;
                curr_loss += diff * diff;
            }
            
            // Adjust learning rate
            if(curr_loss > prev_loss) net->lr *= 0.95;
            else net->lr *= 1.05;
            net->lr = fmax(1e-9, fmin(0.01, net->lr));
            
            // Backward pass with pre-computed gradients
            bwd(net, act, grad);
        }
        
        double loss = compute_loss(pred, data->y, data->n, data->fy);
        if(loss < best_loss) best_loss = loss;
        prev_loss = loss;
        
        if(e % 2 == 0 || e == 9)
            printf("%-6d %.6f%s %.6e\n", e+1, loss, loss == best_loss ? " *" : "  ", net->lr);
        
        for(int i = 0; i < data->n; i++) free(pred[i]);
        free(pred);
    }
    
    save_weights(weights_file, net);
    free_net(net);
    free_data(data);

    // Verification: Load weights and data from disk
    printf("\nVerification:\n");
    
    // Load weights and data
    net = load_weights(weights_file);
    if(!net) { printf("Failed to load weights!\n"); return 1; }
    
    data = load_csv(data_file);
    if(!data) { printf("Failed to load data!\n"); return 1; }

    // Compute loss with loaded weights and data
    double** pred = malloc(data->n * sizeof(double*));
    for(int i = 0; i < data->n; i++) {
        pred[i] = malloc(data->fy * sizeof(double));
        fwd(net, data->X[i], act);
        memcpy(pred[i], act[4], data->fy * sizeof(double));
    }
    
    printf("Loaded weights loss: %.6f\n", compute_loss(pred, data->y, data->n, data->fy));
    
    // Cleanup
    for(int i = 0; i < data->n; i++) free(pred[i]);
    free(pred);
    for(int i = 0; i < 5; i++) { free(act[i]); free(grad[i]); }
    free(act); free(grad);
    free_net(net);
    free_data(data);
    
    return 0;
}