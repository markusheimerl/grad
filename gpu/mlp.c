#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "mlp.h"

int main() {
    // Set random seed.
    srand(time(NULL));
    
    // Parameters for raw input and embeddings.
    // raw_input_dim: number of raw (continuous) features.
    // num_bins: number of bins per feature.
    // embedding_dim: dimension of each embedding vector.
    const int raw_input_dim = 8;
    const int num_bins = 10;
    const int embedding_dim = 8;
    // The effective input dimension equals raw_input_dim x embedding_dim.
    
    // MLP parameters.
    const int hidden_dim = 1024;
    const int output_dim = 8;
    const int num_samples = 1024;
    const int batch_size = num_samples;  // Full-batch training.
    
    // Generate synthetic raw data.
    // Raw inputs have shape: (num_samples x raw_input_dim) and targets (num_samples x output_dim).
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, raw_input_dim, output_dim);
    
    // Initialize the network with embeddings.
    // The init_net function now expects raw_input_dim, num_bins, and embedding_dim,
    // and internally computes the effective input dimension.
    Net* net = init_net(raw_input_dim, num_bins, embedding_dim,
                        hidden_dim, output_dim, batch_size);
    
    // Training parameters.
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Training loop.
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // The embed_input function converts the raw input (X, shape: batch_size x raw_input_dim)
        // into an embedded representation stored in net->d_X.
        embed_input(net, X);
        
        // Forward pass.
        forward_pass(net);
        
        // Calculate loss. (This copies the target y to device, computes error and returns MSE.)
        float loss = calculate_loss(net, y);
        
        // Zero gradients.
        zero_gradients(net);
        
        // Backward pass.
        backward_pass(net);
        
        // Update weights with AdamW.
        update_weights(net, learning_rate);
        
        // Print training progress every 100 epochs.
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }
    
    // Get timestamped filenames.
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    // Save model and raw data.
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, raw_input_dim, output_dim, data_fname);
    
    // Verify the saved model.
    printf("\nVerifying saved model...\n");
    // Load the model back.
    net = load_model(model_fname);
    
    // Run embedding and forward pass again.
    embed_input(net, X);
    forward_pass(net);
    
    float verification_loss = calculate_loss(net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);
    
    // Evaluate network performance.
    printf("\nEvaluating model performance...\n");
    printf("\nR² scores:\n");
    // Copy predictions from device to host.
    int pred_size = batch_size * output_dim * sizeof(float);
    float* h_predictions = (float*)malloc(pred_size);
    CHECK_CUDA(cudaMemcpy(h_predictions, net->d_predictions, pred_size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += y[j * output_dim + i];
        }
        y_mean /= num_samples;
        
        float ss_res = 0.0f, ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = y[j * output_dim + i] - h_predictions[j * output_dim + i];
            float diff_tot = y[j * output_dim + i] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", i, r2);
    }
    
    // Print sample predictions.
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
        // Compute Mean Absolute Error for this output.
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(h_predictions[j * output_dim + i] - y[j * output_dim + i]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Clean up.
    free(X);
    free(y);
    free(h_predictions);
    free_net(net);
    
    return 0;
}