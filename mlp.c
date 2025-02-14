#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "mlp.h"

int main() {
    // Seed the random number generator and set OpenBLAS threads.
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters for raw input and embeddings.
    const int raw_input_dim = 16;  // Number of raw (continuous) features.
    const int num_bins = 10;       // Discretize each feature into 10 bins.
    const int embedding_dim = 8;   // Each embedding vector is 8-dimensional.
    // Effective input dimension of the network:
    int effective_input_dim = raw_input_dim * embedding_dim;

    // MLP parameters.
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples;  // Full-batch training.

    // Generate synthetic raw data.
    // The raw inputs have shape: num_samples x raw_input_dim.
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, raw_input_dim, output_dim);

    // Initialize the network.
    // Note: The init_net function now expects raw_input_dim, num_bins, and embedding_dim.
    Net* net = init_net(raw_input_dim, num_bins, embedding_dim,
                         hidden_dim, output_dim, batch_size);

    // Allocate memory for the embedded (transformed) input.
    float* embedded_input = (float*)malloc(batch_size * effective_input_dim * sizeof(float));

    // Training parameters.
    const int num_epochs = 2000;
    const float learning_rate = 0.001f;

    // Training loop.
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Map raw input into embeddings.
        embed_input(net, X, embedded_input);

        // Forward pass using the embedded input.
        forward_pass(net, embedded_input);

        // Compute loss and gradients.
        float loss = calculate_loss(net, y);
        zero_gradients(net);
        backward_pass(net, embedded_input);

        // Update the network weights.
        update_weights(net, learning_rate);

        // Print training progress.
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    // Get timestamp for filenames.
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save both the model and the raw data.
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, raw_input_dim, output_dim, data_fname);

    // Verify the saved model.
    printf("\nVerifying saved model...\n");
    net = load_model(model_fname);

    // For verification, embed the raw input again and run a forward pass.
    embed_input(net, X, embedded_input);
    forward_pass(net, embedded_input);
    
    float verification_loss = calculate_loss(net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    // Evaluate network performance.
    printf("\nEvaluating model performance...\n");
    printf("\nR² scores:\n");
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += y[j * output_dim + i];
        }
        y_mean /= num_samples;

        float ss_res = 0.0f, ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = y[j * output_dim + i] - net->predictions[j * output_dim + i];
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
            float pred = net->predictions[j * output_dim + i];
            float actual = y[j * output_dim + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        // Compute Mean Absolute Error (MAE) for this output.
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(net->predictions[j * output_dim + i] - y[j * output_dim + i]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }

    // Clean up resources.
    free(X);
    free(y);
    free(embedded_input);
    free_net(net);

    return 0;
}