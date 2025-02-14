#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "mlp_mixer.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int n_tokens = 32;  // Number of bins for discretization
    const int n_channels = 16;  // Input/output dimension
    const int token_dim = 256;  // Embedding dimension
    const int num_samples = 1024;
    const int batch_size = num_samples;  // Full batch training
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, n_channels, n_channels);
    
    // Initialize MLP Mixer
    MLPMixer* mixer = init_mlp_mixer(n_tokens, n_channels, token_dim, batch_size);
    
    // Initialize binning information
    mixer->input_bins = init_binning_info(n_tokens - 1, n_channels);
    mixer->output_bins = init_binning_info(n_tokens - 1, n_channels);
    
    // Compute bin edges for input and output data
    compute_bin_edges(X, num_samples, n_channels, n_tokens - 1, mixer->input_bins);
    compute_bin_edges(y, num_samples, n_channels, n_tokens - 1, mixer->output_bins);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Training loop
    printf("Starting training...\n");
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Train on batch
        train_batch(mixer, X, y, learning_rate);
        
        // Calculate and print predictions periodically
        if ((epoch + 1) % 10 == 0) {
            float* predictions = predict(mixer, X, num_samples);
            float mae = calculate_mae(y, predictions, num_samples, n_channels);
            float r2 = calculate_r2_score(y, predictions, num_samples, n_channels);
            printf("Epoch [%d/%d], MAE: %.8f, R²: %.8f\n", 
                   epoch + 1, num_epochs, mae, r2);
            free(predictions);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_mixer_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_mixer_data.csv", 
             localtime(&now));

    // Save model and data
    save_model(mixer, model_fname);
    save_data_to_csv(X, y, num_samples, n_channels, n_channels, data_fname);
    
    // Model verification
    printf("\nVerifying saved model...\n");
    MLPMixer* loaded_mixer = load_model(model_fname);
    
    // Generate predictions with loaded model
    float* predictions = predict(loaded_mixer, X, num_samples);
    
    // Calculate and print metrics
    float mae = calculate_mae(y, predictions, num_samples, n_channels);
    float r2 = calculate_r2_score(y, predictions, num_samples, n_channels);
    printf("Loaded Model - MAE: %.8f, R²: %.8f\n", mae, r2);

    // Print detailed R² scores for each feature
    printf("\nDetailed R² scores:\n");
    for (int i = 0; i < n_channels; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += y[j * n_channels + i];
        }
        y_mean /= num_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = y[j * n_channels + i] - predictions[j * n_channels + i];
            float diff_tot = y[j * n_channels + i] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        float feature_r2 = 1.0f - (ss_res / (ss_tot + 1e-8f));
        printf("R² score for feature %d: %.8f\n", i, feature_r2);
    }

    // Print sample predictions
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Feature\t\tPredicted\tActual\t\tDifference\n");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < n_channels; i++) {
        printf("\nFeature %d:\n", i);
        for (int j = 0; j < 15; j++) {
            float pred = predictions[j * n_channels + i];
            float actual = y[j * n_channels + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        
        // Calculate MAE for this feature
        float feature_mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            feature_mae += fabsf(predictions[j * n_channels + i] - y[j * n_channels + i]);
        }
        feature_mae /= num_samples;
        printf("Mean Absolute Error for feature %d: %.3f\n", i, feature_mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(predictions);
    free_mlp_mixer(mixer);
    free_mlp_mixer(loaded_mixer);
    
    return 0;
}