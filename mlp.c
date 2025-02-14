#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "mlp.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);
    
    // Parameters
    const int seq_len = 16;  // Same as before
    const int batch_size = 1024;
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, batch_size, seq_len, seq_len);
    
    // Initialize network
    Net* net = init_net(seq_len, batch_size);
    
    // Allocate bins for input and target
    int* X_bins = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* y_bins = (int*)malloc(batch_size * seq_len * sizeof(int));
    
    // Convert data to bins
    continuous_to_bins(net, X, X_bins, batch_size * seq_len);
    continuous_to_bins(net, y, y_bins, batch_size * seq_len);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        forward_pass(net, X_bins);
        
        // Calculate loss
        float loss = calculate_loss(net, y_bins);
        
        // Backward pass
        backward_pass(net, X_bins, y_bins);
        
        // Update weights
        update_weights(net, learning_rate);
        
        // Print progress
        if ((epoch + 1) % 1 == 0) {
            // Convert current predictions to continuous values
            float* predictions = (float*)malloc(batch_size * seq_len * sizeof(float));
            predict_continuous(net, net->logits, predictions, batch_size * seq_len);
            
            // Calculate MSE for interpretability
            float mse = 0.0f;
            for (int i = 0; i < batch_size * seq_len; i++) {
                float diff = predictions[i] - y[i];
                mse += diff * diff;
            }
            mse /= (batch_size * seq_len);
            
            printf("Epoch [%d/%d], CE Loss: %.8f, MSE: %.8f\n", 
                   epoch + 1, num_epochs, loss, mse);
                   
            free(predictions);
        }
    }
    
    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));
    
    // Save model and data
    save_model(net, model_fname);
    save_data_to_csv(X, y, batch_size, seq_len, seq_len, data_fname);
    
    // Verification and evaluation
    printf("\nVerifying saved model...\n");
    Net* loaded_net = load_model(model_fname);
    
    // Forward pass with loaded model
    forward_pass(loaded_net, X_bins);
    float verification_loss = calculate_loss(loaded_net, y_bins);
    
    // Convert predictions to continuous values
    float* final_predictions = (float*)malloc(batch_size * seq_len * sizeof(float));
    predict_continuous(loaded_net, loaded_net->logits, final_predictions, batch_size * seq_len);
    
    // Calculate final MSE
    float final_mse = 0.0f;
    for (int i = 0; i < batch_size * seq_len; i++) {
        float diff = final_predictions[i] - y[i];
        final_mse += diff * diff;
    }
    final_mse /= (batch_size * seq_len);
    
    printf("Loaded model - CE Loss: %.8f, MSE: %.8f\n", verification_loss, final_mse);
    
    // Calculate R² scores for each output dimension
    printf("\nR² scores:\n");
    for (int d = 0; d < seq_len; d++) {
        float y_mean = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            y_mean += y[b * seq_len + d];
        }
        y_mean /= batch_size;
        
        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float diff_res = y[b * seq_len + d] - final_predictions[b * seq_len + d];
            float diff_tot = y[b * seq_len + d] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", d, r2);
    }
    
    // Print sample predictions
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Output\t\tPredicted\tActual\t\tDifference\n");
    printf("------------------------------------------------------------\n");
    
    for (int d = 0; d < seq_len; d++) {
        printf("\ny%d:\n", d);
        for (int b = 0; b < 15; b++) {
            float pred = final_predictions[b * seq_len + d];
            float actual = y[b * seq_len + d];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", b, pred, actual, diff);
        }
        
        // Calculate MAE for this dimension
        float mae = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            mae += fabsf(final_predictions[b * seq_len + d] - y[b * seq_len + d]);
        }
        mae /= batch_size;
        printf("Mean Absolute Error for y%d: %.3f\n", d, mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_bins);
    free(y_bins);
    free(final_predictions);
    free_net(net);
    free_net(loaded_net);
    
    return 0;
}