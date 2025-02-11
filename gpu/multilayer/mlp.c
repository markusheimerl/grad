#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../data.h"
#include "mlp.h"

int main() {
    srand(time(NULL));

    // Parameters
    const int input_dim = 16;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples; // Full batch training

    // Define network architecture
    const int num_layers = 5;  // Total number of layers (including input and output)
    int layer_dims[] = {16, 1024, 512, 256, 4};  // Dimensions for each layer
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);
    
    // Initialize network
    Net* net = init_net(layer_dims, num_layers, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
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
    Net* loaded_net = load_model(model_fname);
    
    // Allocate host memory for predictions
    float* h_predictions = (float*)malloc(num_samples * output_dim * sizeof(float));

    // Forward pass with loaded model
    forward_pass(loaded_net, X);
    
    // Copy predictions from device to host
    CHECK_CUDA(cudaMemcpy(h_predictions, loaded_net->layers[loaded_net->num_layers-1]->d_output, 
                         num_samples * output_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss(loaded_net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += y[j * output_dim + i];
        }
        y_mean /= num_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = y[j * output_dim + i] - h_predictions[j * output_dim + i];
            float diff_tot = y[j * output_dim + i] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", i, r2);
    }

    // Print sample predictions
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Output\t\tPredicted\tActual\t\tDifference\n");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        printf("\ny%d:\n", i);
        for (int j = 0; j < 15; j++) {
            float pred = h_predictions[j * output_dim + i];
            float actual = y[j * output_dim + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        
        // Calculate MAE for this output
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(h_predictions[j * output_dim + i] - y[j * output_dim + i]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Print layer dimensions
    printf("\nNetwork Architecture:\n");
    printf("Input dimension: %d\n", net->input_dim);
    for (int i = 0; i < net->num_layers; i++) {
        printf("Layer %d: %d -> %d\n", i+1, 
               net->layers[i]->input_dim, 
               net->layers[i]->output_dim);
    }
    printf("Output dimension: %d\n", net->output_dim);
    
    // Calculate total number of parameters
    int total_params = 0;
    for (int i = 0; i < net->num_layers; i++) {
        total_params += net->layers[i]->input_dim * net->layers[i]->output_dim;
    }
    printf("Total trainable parameters: %d\n", total_params);
    
    // Cleanup
    free(X);
    free(y);
    free(h_predictions);
    free_net(net);
    free_net(loaded_net);
    
    return 0;
}