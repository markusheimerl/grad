#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "grad.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 16;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples; // Full batch training
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);
    
    // Initialize network
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        forward_pass(net, X);
        
        // Calculate loss
        float loss = calculate_loss(net, y);
        
        // Backward pass
        zero_gradients(net);
        backward_pass(net, X);
        
        // Update weights
        update_weights(net, learning_rate);
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back
    net = load_model(model_fname);
    
    // Forward pass with loaded model
    forward_pass(net, X);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss(net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);
    
    // Cleanup
    free(X);
    free(y);
    free_net(net);
    
    return 0;
}