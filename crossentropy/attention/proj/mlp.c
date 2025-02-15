#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mlp.h"
/* Assumes data.h provides generate_synthetic_data and save_data_to_csv. */
#include "../../data.h"

int main(void)
{
    srand((unsigned)time(NULL));
    /* Set OpenBLAS thread count if using OpenBLAS. */
    openblas_set_num_threads(4);

    /* PARAMETERS */
    const int input_dim = 4;          // Number of raw input features.
    const int num_samples = 1024;     // Number of samples.
    const int output_dim = input_dim; // One output scalar per input feature.
    const int num_bins = 64;          // Used only for the embedding table.
    const int embedding_dim = num_bins; // Embedding dimension (set equal to num_bins).
    const int batch_size = num_samples; // Full-batch training.

    /* DATA GENERATION */
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);

    /* Compute per–feature min and max for inputs. */
    float *input_min = (float*)malloc(input_dim * sizeof(float));
    float *input_max = (float*)malloc(input_dim * sizeof(float));
    compute_min_max(X, num_samples, input_dim, input_min, input_max);

    /* For regression we use the continuous target y directly. */

    /* Choose number of layers. */
    int num_layers = 2;  // Change to desired number.

    /* INITIALIZE NETWORK */
    Net *net = init_net(input_dim, num_bins, embedding_dim, output_dim, batch_size, num_layers);
    /* Allocate buffer for embedded input. */
    float *embedded_input = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    if (!embedded_input) {
        fprintf(stderr, "Failed to allocate memory for embedded input.\n");
        exit(EXIT_FAILURE);
    }

    /* TRAINING PARAMETERS */
    const int num_epochs = 3000;
    const float learning_rate = 0.0008f;

    /* TRAINING LOOP */
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        embed_input(net, X, embedded_input, input_min, input_max);
        forward_pass(net);
        float loss = calculate_loss(net, y);
        zero_gradients(net);
        backward_pass(net);
        update_weights(net, learning_rate);

        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    /* SAVE MODEL and DATA. */
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", tm_info);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", tm_info);
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);

    /* VERIFY SAVED MODEL */
    printf("\nVerifying saved model...\n");
    free_net(net);
    net = load_model(model_fname);
    embed_input(net, X, embedded_input, input_min, input_max);
    forward_pass(net);
    float verification_loss = calculate_loss(net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    /* EVALUATION: compute mean absolute error (MAE) and mean squared error (MSE) */
    int total_outputs = num_samples * output_dim;
    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    for (int i = 0; i < total_outputs; i++) {
        float diff = net->predictions[i] - y[i];
        sum_abs_error += fabs(diff);
        sum_sq_error += diff * diff;
    }
    float mae = sum_abs_error / total_outputs;
    float mse = sum_sq_error / total_outputs;
    printf("\nMean Absolute Error: %.5f\n", mae);
    printf("Mean Squared Error: %.5f\n", mse);

    /* Print sample predictions (first 5 samples) */
    printf("\nSample Predictions (first 5 samples):\n");
    for (int i = 0; i < 5; i++) {
        printf("Sample %d:\n", i);
        for (int d = 0; d < output_dim; d++) {
            int idx = i * output_dim + d;
            printf("  Output %d: Predicted = %.5f, True = %.5f\n",
                   d, net->predictions[idx], y[i * output_dim + d]);
        }
    }

    /* CLEANUP */
    free(X);
    free(y);
    free(input_min);
    free(input_max);
    free(embedded_input);
    free_net(net);

    return 0;
}