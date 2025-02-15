#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"   // Provides generate_synthetic_data, save_data_to_csv, INPUT_RANGE_MIN/MAX.
#include "mlp.h"       // Contains Net definition and various network functions.

int main() {
    srand((unsigned)time(NULL));
    openblas_set_num_threads(4);

    // PARAMETERS
    const int raw_input_dim = 8;         // Number of raw continuous features.
    const int num_samples = 1024;        // Number of samples.
    const int output_dim = 1;            // Continuous target dimension.

    // Embedding lookup parameters.
    const int embedding_num_bins = 10;   // Number of bins per feature.
    const int embedding_dim = 8;         // Dimension of each embedding vector.

    // MLP parameters.
    const int hidden_dim = 1024;
    const int target_bins = 256;         // Number of discrete target classes.
    const int batch_size = num_samples;  // Full-batch training.

    // DATA GENERATION
    float *X, *y;  // y holds continuous target values.
    generate_synthetic_data(&X, &y, num_samples, raw_input_dim, output_dim);

    // Determine true minimum and maximum of targets.
    float out_min = y[0], out_max = y[0];
    for (int i = 1; i < num_samples * output_dim; i++) {
        if (y[i] < out_min)
            out_min = y[i];
        if (y[i] > out_max)
            out_max = y[i];
    }
    float out_range = out_max - out_min;
    if (out_range == 0.0f)
        out_range = 1e-6f;

    // Map continuous targets to discrete classes.
    int *y_class = (int*)malloc(num_samples * sizeof(int));
    if(!y_class){
        fprintf(stderr, "Failed to allocate memory for y_class.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_samples; i++) {
        float normalized = (y[i] - out_min) / out_range;
        int bin = (int)(normalized * target_bins);
        if (bin < 0)
            bin = 0;
        else if (bin >= target_bins)
            bin = target_bins - 1;
        y_class[i] = bin;
    }

    // INITIALIZE NETWORK
    Net* net = init_net(raw_input_dim, embedding_num_bins, embedding_dim,
                         hidden_dim, target_bins, batch_size);
    int effective_input_dim = raw_input_dim * embedding_dim;
    float *embedded_input = (float*)malloc(batch_size * effective_input_dim * sizeof(float));
    if (!embedded_input) {
        fprintf(stderr, "Failed to allocate memory for embedded input.\n");
        exit(EXIT_FAILURE);
    }

    // TRAINING PARAMETERS
    const int num_epochs = 2000;
    const float learning_rate = 0.001f;

    // TRAINING LOOP
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Convert raw inputs into embeddings.
        embed_input(net, X, embedded_input);
        forward_pass(net, embedded_input);
        float loss = calculate_loss(net, y_class);
        zero_gradients(net);
        backward_pass(net, embedded_input);
        update_weights(net, learning_rate);

        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    // SAVE MODEL and DATA
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_model(net, model_fname);
    save_data_to_csv(X, y, num_samples, raw_input_dim, output_dim, data_fname);

    // VERIFY SAVED MODEL
    printf("\nVerifying saved model...\n");
    net = load_model(model_fname);
    embed_input(net, X, embedded_input);
    forward_pass(net, embedded_input);
    float verification_loss = calculate_loss(net, y_class);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    // EVALUATION: compute discrete accuracy and mean absolute error.
    int correct = 0;
    double sum_abs_error = 0.0;
    float bin_width = out_range / target_bins;
    for (int i = 0; i < num_samples; i++) {
        int predicted_bin = 0;
        float max_logit = net->predictions[i * target_bins];
        for (int j = 1; j < target_bins; j++) {
            float logit = net->predictions[i * target_bins + j];
            if (logit > max_logit) {
                max_logit = logit;
                predicted_bin = j;
            }
        }
        if (predicted_bin == y_class[i])
            correct++;

        float predicted_cont = out_min + (predicted_bin + 0.5f) * bin_width;
        sum_abs_error += fabs(predicted_cont - y[i]);
    }
    float accuracy = 100.0f * ((float)correct) / num_samples;
    float mae = sum_abs_error / num_samples;
    printf("\nClassification Accuracy: %.2f%%\n", accuracy);
    printf("Mean Absolute Error (Continuous Prediction): %.5f\n", mae);

    // Print sample predictions for the first 15 samples.
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Sample\tPredictedBin\tPredictedCont\tTrueBin\tTrueCont\n");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < 15; i++) {
        int predicted_bin = 0;
        float max_logit = net->predictions[i * target_bins];
        for (int j = 1; j < target_bins; j++) {
            float logit = net->predictions[i * target_bins + j];
            if (logit > max_logit) {
                max_logit = logit;
                predicted_bin = j;
            }
        }
        float predicted_cont = out_min + (predicted_bin + 0.5f) * bin_width;
        printf("%d:\t%d\t\t%.5f\t\t%d\t%.5f\n", i, predicted_bin, predicted_cont, y_class[i], y[i]);
    }

    // CLEANUP
    free(X);
    free(y);
    free(y_class);
    free(embedded_input);
    free_net(net);

    return 0;
}