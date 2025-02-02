#include "grad.cuh"
#include <time.h>

int main() {
    // Set random seed
    srand(time(NULL));
    
    // Initialize CUDA device
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    printf("Using GPU: %s\n", props.name);
    
    // Generate synthetic data
    Data* data = synth(1000, 15, 4, 0.1);
    printf("Generated synthetic dataset with %d samples\n", data->n);
    
    // Initialize network
    int sz[] = {15, 256, 128, 64, 32, 4};
    int n_layers = sizeof(sz)/sizeof(sz[0]);
    Net* net = init_net(n_layers, sz, 1e-3);
    printf("Initialized neural network with %d layers\n", n_layers);
    
    // Training loop
    int epochs = 200;
    clock_t start = clock();

    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int i = 0; i < data->n; i++) {
            forward(net, data->X[i]);
            backward(net, data->y[i], epoch, epochs);
        }
        
        if(epoch % 5 == 0 || epoch == epochs - 1) {
            double error = mse(net, data);
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            printf("Epoch %d/%d, MSE: %f, Time: %.2f s\n", 
                   epoch + 1, epochs, error, elapsed);
        }
    }
    
    double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nTraining completed in %.2f seconds\n", total_time);
    
    // Print some example predictions
    printf("\nExample predictions (first 10 samples):\n");
    printf("Index | %-30s | %-30s\n", "Predicted", "True Values");
    printf("------+--------------------------------+--------------------------------\n");
    
    for(int i = 0; i < 10; i++) {
        forward(net, data->X[i]);
        printf("%5d | ", i);
        
        // Print predicted values
        printf("[");
        for(int j = 0; j < data->fy; j++) {
            printf("%7.3f", net->layers[net->n_layers-1].x[j]);
            if(j < data->fy-1) printf(", ");
        }
        printf("] | ");
        
        // Print true values
        printf("[");
        for(int j = 0; j < data->fy; j++) {
            printf("%7.3f", data->y[i][j]);
            if(j < data->fy-1) printf(", ");
        }
        printf("]\n");
    }
    
    // Print input features for context
    printf("\nCorresponding input features:\n");
    printf("Index | Features (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)\n");
    printf("------+-------------------------\n");
    for(int i = 0; i < 10; i++) {
        printf("%5d | [", i);
        for(int j = 0; j < data->fx; j++) {
            printf("%7.3f", data->X[i][j]);
            if(j < data->fx-1) printf(", ");
        }
        printf("]\n");
    }

    // Save network and data
    char net_fname[64], data_fname[64];
    time_t current_time = time(NULL);
    struct tm* time_info = localtime(&current_time);
    
    strftime(net_fname, sizeof(net_fname), "%Y%m%d_%H%M%S_net.bin", time_info);
    save_net(net_fname, net);
    
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", time_info);
    save_csv(data_fname, data);
    
    // Verification of saved and loaded data/network
    printf("\n=== Verification of saved/loaded data and network ===\n");

    // Load the saved network and data
    Net* loaded_net = load_net(net_fname);
    if (!loaded_net) {
        printf("Failed to load network from file: %s\n", net_fname);
        free_net(net);
        free_data(data);
        return 1;
    }

    Data* loaded_data = load_csv(data_fname, 15, 4);
    if (!loaded_data) {
        printf("Failed to load data from file: %s\n", data_fname);
        free_net(net);
        free_net(loaded_net);
        free_data(data);
        return 1;
    }

    // Calculate and print MSE with loaded network
    printf("\nMSE with loaded network: %f\n", mse(loaded_net, loaded_data));

    // Print some predictions with loaded network
    printf("\nPredictions with loaded network (first 10 samples):\n");
    printf("Index | %-30s | %-30s\n", "Predicted", "True Values");
    printf("------+--------------------------------+--------------------------------\n");

    for(int i = 0; i < 10; i++) {
        forward(loaded_net, loaded_data->X[i]);
        printf("%5d | ", i);
        
        // Print predicted values
        printf("[");
        for(int j = 0; j < loaded_data->fy; j++) {
            printf("%7.3f", loaded_net->layers[loaded_net->n_layers-1].x[j]);
            if(j < loaded_data->fy-1) printf(", ");
        }
        printf("] | ");
        
        // Print true values
        printf("[");
        for(int j = 0; j < loaded_data->fy; j++) {
            printf("%7.3f", loaded_data->y[i][j]);
            if(j < loaded_data->fy-1) printf(", ");
        }
        printf("]\n");
    }

    // Clean up
    free_net(net);
    free_net(loaded_net);
    free_data(data);
    free_data(loaded_data);
    
    // Cleanup CUDA
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}