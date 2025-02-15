#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../../data.h"  // Assumes data.h provides generate_synthetic_data and save_data_to_csv.
#include "mlp.h"

int main() {
    srand((unsigned)time(NULL));
    openblas_set_num_threads(4);

    // PARAMETERS.
    const int input_dim = 4;            // Number of raw input features (tokens).
    const int num_samples = 1024;       // Number of samples.
    const int output_dim = input_dim;   // One output per input feature.
    const int num_bins = 64;
    const int embedding_dim = num_bins; // Embedding dimension = number of bins.

    // Token-mixing MLP operates on tokens: its input dimension equals input_dim.
    // Full–batch training.
    const int batch_size = num_samples;

    // DATA GENERATION.
    // generate_synthetic_data produces X of shape [num_samples x input_dim]
    // and y of shape [num_samples x output_dim].
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);

    // Compute per–feature min and max for input features.
    float* input_min = (float*)malloc(input_dim * sizeof(float));
    float* input_max = (float*)malloc(input_dim * sizeof(float));
    compute_min_max(X, num_samples, input_dim, input_min, input_max);

    // Compute per–output min and max.
    float* output_min_arr = (float*)malloc(output_dim * sizeof(float));
    float* output_max_arr = (float*)malloc(output_dim * sizeof(float));
    compute_min_max(y, num_samples, output_dim, output_min_arr, output_max_arr);

    // Map continuous outputs to discrete bins.
    int *y_class = (int*)malloc(num_samples * output_dim * sizeof(int));
    for (int i = 0; i < num_samples; i++) {
        for (int d = 0; d < output_dim; d++) {
            float val = y[i * output_dim + d];
            float out_min = output_min_arr[d];
            float out_max = output_max_arr[d];
            float range = out_max - out_min;
            if (range == 0.0f)
                range = 1e-6f;
            float normalized = (val - out_min) / range;
            int b = (int)(normalized * num_bins);
            if (b < 0) b = 0;
            else if (b >= num_bins) b = num_bins - 1;
            y_class[i * output_dim + d] = b;
        }
    }

    // INITIALIZE NETWORK.
    Net* net = init_net(input_dim, num_bins, embedding_dim, output_dim, batch_size);
    // embedded_input has shape [batch_size x (input_dim * embedding_dim)]
    float *embedded_input = (float*)malloc(batch_size * (input_dim * embedding_dim) * sizeof(float));
    if (!embedded_input) {
        fprintf(stderr, "Failed to allocate memory for embedded input.\n");
        exit(EXIT_FAILURE);
    }

    // TRAINING PARAMETERS.
    const int num_epochs = 3000;
    const float learning_rate = 0.0008f;

    // TRAINING LOOP.
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Embed raw input using per–feature min/max.
        embed_input(net, X, embedded_input, input_min, input_max);
        // Forward pass: first apply token–mixing, then feed–forward (both include swish activations and residuals).
        forward_pass(net, embedded_input);
        float loss = calculate_loss(net, y_class);
        // Zero gradients.
        zero_gradients(net);
        // Backward pass.
        backward_pass(net, embedded_input);
        // Update parameters.
        update_weights(net, learning_rate);
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }
    
    // SAVE MODEL and DATA.
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);

    // VERIFY SAVED MODEL.
    printf("\nVerifying saved model...\n");
    net = load_model(model_fname);
    embed_input(net, X, embedded_input, input_min, input_max);
    forward_pass(net, embedded_input);
    float verification_loss = calculate_loss(net, y_class);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    // EVALUATION.
    // For each sample and output dimension, determine the predicted bin (argmax over each token's logits),
    // then convert that bin into a continuous prediction via unbin_value.
    int correct = 0;
    double sum_abs_error = 0.0;
    for (int i = 0; i < num_samples; i++) {
        for (int d = 0; d < output_dim; d++) {
            int offset = i * (input_dim * embedding_dim) + d * embedding_dim;
            int predicted_bin = 0;
            float max_logit = net->predictions[offset];
            for (int j = 1; j < embedding_dim; j++) {
                float logit = net->predictions[offset + j];
                if (logit > max_logit) {
                    max_logit = logit;
                    predicted_bin = j;
                }
            }
            if (predicted_bin == y_class[i * output_dim + d])
                correct++;
            float out_min = output_min_arr[d];
            float out_max = output_max_arr[d];
            float predicted_cont = unbin_value(predicted_bin, out_min, out_max, num_bins);
            sum_abs_error += fabs(predicted_cont - y[i * output_dim + d]);
        }
    }
    int total_outputs = num_samples * output_dim;
    float accuracy = 100.0f * ((float)correct) / total_outputs;
    float mae = sum_abs_error / total_outputs;
    printf("\nClassification Accuracy: %.2f%%\n", accuracy);
    printf("Mean Absolute Error (Continuous Prediction): %.5f\n", mae);

    // Print sample predictions (for the first 5 samples).
    printf("\nSample Predictions (first 5 samples):\n");
    for (int i = 0; i < 5; i++) {
        printf("Sample %d:\n", i);
        for (int d = 0; d < output_dim; d++) {
            int offset = i * (input_dim * embedding_dim) + d * embedding_dim;
            int predicted_bin = 0;
            float max_logit = net->predictions[offset];
            for (int j = 1; j < embedding_dim; j++) {
                float logit = net->predictions[offset + j];
                if (logit > max_logit) {
                    max_logit = logit;
                    predicted_bin = j;
                }
            }
            float predicted_cont = unbin_value(predicted_bin, output_min_arr[d], output_max_arr[d], num_bins);
            printf("  Output %d: Predicted Bin = %d, Predicted = %.5f, True Bin = %d, True = %.5f\n",
                   d, predicted_bin, predicted_cont, y_class[i * output_dim + d], y[i * output_dim + d]);
        }
    }

    // CLEANUP.
    free(X);
    free(y);
    free(y_class);
    free(input_min);
    free(input_max);
    free(output_min_arr);
    free(output_max_arr);
    free(embedded_input);
    free_net(net);

    return 0;
}