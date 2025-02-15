#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"  // Assumes data.h provides generate_synthetic_data and save_data_to_csv.
#include "mlp.h"

int main() {
    srand((unsigned)time(NULL));
    openblas_set_num_threads(4);

    // PARAMETERS.
    const int input_dim = 8;         // Number of raw input features.
    const int num_samples = 1024;    // Number of samples.
    const int output_dim = 1;        // Continuous output dimension.

    // Single num_bins used for both input embedding and output classes.
    const int num_bins = 16;         // (Also used later for classification.)
    const int embedding_dim = 8;     // Embedding vector dimension.

    // MLP parameters.
    const int hidden_dim = 1024;
    // For output, number of classes = num_bins.
    const int batch_size = num_samples;  // Full-batch training.

    // DATA GENERATION.
    float *X, *y; // y holds continuous output values.
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);

    // Compute per‑feature min and max for the input features.
    float* input_min = (float*)malloc(input_dim * sizeof(float));
    float* input_max = (float*)malloc(input_dim * sizeof(float));
    compute_min_max(X, num_samples, input_dim, input_min, input_max);

    // Compute min and max for the output feature (since output_dim==1, these are scalars).
    float* output_min_arr = (float*)malloc(output_dim * sizeof(float));
    float* output_max_arr = (float*)malloc(output_dim * sizeof(float));
    compute_min_max(y, num_samples, output_dim, output_min_arr, output_max_arr);
    float out_min = output_min_arr[0];
    float out_max = output_max_arr[0];
    float out_range = out_max - out_min;
    if (out_range == 0.0f)
        out_range = 1e-6f;

    // Map continuous outputs to discrete bins.
    int *y_class = (int*)malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; i++) {
        float normalized = (y[i] - out_min) / out_range;
        int b = (int)(normalized * num_bins);
        if (b < 0)
            b = 0;
        else if (b >= num_bins)
            b = num_bins - 1;
        y_class[i] = b;
    }

    // INITIALIZE NETWORK.
    Net* net = init_net(input_dim, num_bins, embedding_dim, hidden_dim, batch_size);
    float *embedded_input = (float*)malloc(batch_size * (input_dim * embedding_dim) * sizeof(float));
    if (!embedded_input) {
        fprintf(stderr, "Failed to allocate memory for embedded input.\n");
        exit(EXIT_FAILURE);
    }

    // TRAINING PARAMETERS.
    const int num_epochs = 2000;
    const float learning_rate = 0.001f;

    // TRAINING LOOP.
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Embed raw input using per‑feature min/max.
        embed_input(net, X, embedded_input, input_min, input_max);
        forward_pass(net, embedded_input);
        float loss = calculate_loss(net, y_class);
        zero_gradients(net);
        backward_pass(net, embedded_input);
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

    // EVALUATION. For each sample, find the predicted bin (argmax of logits),
    // then convert that bin into a continuous prediction using unbin_value.
    int correct = 0;
    double sum_abs_error = 0.0;
    for (int i = 0; i < num_samples; i++) {
        int predicted_bin = 0;
        float max_logit = net->predictions[i * num_bins];
        for (int j = 1; j < num_bins; j++) {
            float logit = net->predictions[i * num_bins + j];
            if (logit > max_logit) {
                max_logit = logit;
                predicted_bin = j;
            }
        }
        if (predicted_bin == y_class[i])
            correct++;
        float predicted_cont = unbin_value(predicted_bin, out_min, out_max, num_bins);
        sum_abs_error += fabs(predicted_cont - y[i]);
    }
    float accuracy = 100.0f * ((float)correct) / num_samples;
    float mae = sum_abs_error / num_samples;
    printf("\nClassification Accuracy: %.2f%%\n", accuracy);
    printf("Mean Absolute Error (Continuous Prediction): %.5f\n", mae);

    // Print sample predictions (first 15 samples).
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Sample\tPredictedBin\tPredictedCont\tTrueBin\tTrueCont\n");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < 15; i++) {
        int predicted_bin = 0;
        float max_logit = net->predictions[i * num_bins];
        for (int j = 1; j < num_bins; j++) {
            float logit = net->predictions[i * num_bins + j];
            if (logit > max_logit) {
                max_logit = logit;
                predicted_bin = j;
            }
        }
        float predicted_cont = unbin_value(predicted_bin, out_min, out_max, num_bins);
        printf("%d:\t%d\t\t%.5f\t\t%d\t%.5f\n", i, predicted_bin, predicted_cont, y_class[i], y[i]);
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
