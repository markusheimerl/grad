#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../data.h"     // Provides generate_synthetic_data and save_data_to_csv, and defines INPUT_RANGE_MIN/MAX.
#include "mlp.h"          // Contains GPU network definitions and functions.

int main() {
    // Seed the random number generator.
    srand(time(NULL));

    //----------------------------------------------------------------------------
    // PARAMETERS
    //----------------------------------------------------------------------------
    const int raw_input_dim = 8;         // Number of raw input features.
    const int num_samples     = 1024;      // Number of samples.
    const int num_raw_targets = 8;         // Number of continuous target values

    // Embedding lookup parameters.
    const int embedding_num_bins = 16;     // Number of bins per feature for embedding lookup.
    const int embedding_dim     = 8;       // Dimension of each embedding vector.

    // MLP parameters.
    const int hidden_dim  = 1024;
    const int target_bins = 512;           // Number of discrete bins (classes) for each output.
    const int batch_size  = num_samples;    // Full‐batch training.

    //----------------------------------------------------------------------------
    // DATA GENERATION
    //----------------------------------------------------------------------------
    // Generate synthetic data: X (raw inputs) and y (continuous targets).
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, raw_input_dim, num_raw_targets);

    // Determine the true minimum and maximum of the continuous targets.
    float out_min = y[0], out_max = y[0];
    int total_targets = num_samples * num_raw_targets;
    for (int i = 1; i < total_targets; i++) {
        if (y[i] < out_min)
            out_min = y[i];
        if (y[i] > out_max)
            out_max = y[i];
    }
    float out_range = out_max - out_min;
    if (out_range == 0.0f)
        out_range = 1e-6f;  // Avoid division by zero.

    //----------------------------------------------------------------------------
    // Compute discrete labels: Map each continuous target in [out_min, out_max]
    // into a bin in [0, target_bins-1] for each output.
    //----------------------------------------------------------------------------
    int* y_class = (int*)malloc(total_targets * sizeof(int));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_raw_targets; j++) {
            int index = i * num_raw_targets + j;
            float normalized = (y[index] - out_min) / out_range;
            int bin = (int)(normalized * target_bins);
            if (bin < 0)
                bin = 0;
            if (bin >= target_bins)
                bin = target_bins - 1;
            y_class[index] = bin;
        }
    }

    //----------------------------------------------------------------------------
    // Initialize the GPU network.
    //----------------------------------------------------------------------------
    Net* net = init_net(raw_input_dim, embedding_num_bins, embedding_dim,
                         hidden_dim, num_raw_targets, target_bins, batch_size);

    //----------------------------------------------------------------------------
    // TRAINING PARAMETERS
    //----------------------------------------------------------------------------
    const int num_epochs = 20000;
    const float learning_rate = 0.001f;

    //----------------------------------------------------------------------------
    // TRAINING LOOP
    //----------------------------------------------------------------------------
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Convert raw inputs into embeddings.
        embed_input(net, X);

        // Forward pass: compute hidden activations and logits.
        forward_pass(net);

        // Compute cross‑entropy loss (and set the gradient d_error).
        float loss = calculate_loss(net, y_class);

        // Zero gradients.
        zero_gradients(net);

        // Backward pass.
        backward_pass(net);

        // Update weights using the AdamW optimizer.
        update_weights(net, learning_rate);

        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    //----------------------------------------------------------------------------
    // Save the model and data.
    //----------------------------------------------------------------------------
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_model(net, model_fname);
    // Note: save_data_to_csv (from data.h) saves X and y (continuous targets).
    save_data_to_csv(X, y, num_samples, raw_input_dim, num_raw_targets, data_fname);

    //----------------------------------------------------------------------------
    // Verify the saved model.
    //----------------------------------------------------------------------------
    printf("\nVerifying saved model...\n");
    net = load_model(model_fname);
    embed_input(net, X);
    forward_pass(net);
    float verification_loss = calculate_loss(net, y_class);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    //----------------------------------------------------------------------------
    // EVALUATION:
    // For each sample and for each output, determine the predicted bin (via argmax
    // over the corresponding group of logits), then convert it to a continuous
    // prediction (the bin’s center). Also, compute overall classification accuracy.
    //----------------------------------------------------------------------------
    int correct = 0;
    double sum_abs_error = 0.0;
    float bin_width = out_range / target_bins;
    int total_predictions = num_samples * num_raw_targets;

    // Copy predictions from device to host.
    int pred_size = batch_size * net->output_dim * sizeof(float);
    float* h_predictions = (float*)malloc(pred_size);
    CHECK_CUDA(cudaMemcpy(h_predictions, net->d_predictions, pred_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_raw_targets; j++) {
            int group_offset = i * (num_raw_targets * target_bins) + j * target_bins;
            int predicted_bin = 0;
            float max_logit = h_predictions[group_offset];
            for (int k = 1; k < target_bins; k++) {
                float logit = h_predictions[group_offset + k];
                if (logit > max_logit) {
                    max_logit = logit;
                    predicted_bin = k;
                }
            }
            int label = y_class[i * num_raw_targets + j];
            if (predicted_bin == label)
                correct++;
            float predicted_cont = out_min + (predicted_bin + 0.5f) * bin_width;
            float true_cont = y[i * num_raw_targets + j];
            sum_abs_error += fabs(predicted_cont - true_cont);
        }
    }
    float accuracy = 100.0f * ((float)correct) / total_predictions;
    float mae = sum_abs_error / total_predictions;
    printf("\nClassification Accuracy: %.2f%%\n", accuracy);
    printf("Mean Absolute Error (Continuous Prediction): %.5f\n", mae);

    //----------------------------------------------------------------------------
    // Print sample predictions (first 15 samples).
    //----------------------------------------------------------------------------
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Sample\tOutputIdx\tPredictedBin\tPredictedCont\tTrueBin\tTrueCont\n");
    printf("--------------------------------------------------------------------------\n");
    int samples_to_print = (num_samples < 15) ? num_samples : 15;
    for (int i = 0; i < samples_to_print; i++) {
        for (int j = 0; j < num_raw_targets; j++) {
            int group_offset = i * (num_raw_targets * target_bins) + j * target_bins;
            int predicted_bin = 0;
            float max_logit = h_predictions[group_offset];
            for (int k = 1; k < target_bins; k++) {
                float logit = h_predictions[group_offset + k];
                if (logit > max_logit) {
                    max_logit = logit;
                    predicted_bin = k;
                }
            }
            float predicted_cont = out_min + (predicted_bin + 0.5f) * bin_width;
            int true_bin = y_class[i * num_raw_targets + j];
            float true_cont = y[i * num_raw_targets + j];
            printf("%d:\t%d\t\t%d\t\t%.5f\t\t%d\t%.5f\n", i, j, predicted_bin, predicted_cont, true_bin, true_cont);
        }
    }

    //----------------------------------------------------------------------------
    // Cleanup.
    //----------------------------------------------------------------------------
    free(X);
    free(y);
    free(y_class);
    free(h_predictions);
    free_net(net);

    return 0;
}