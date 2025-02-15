#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../../../data.h"  // Assumes data.h provides generate_synthetic_data and save_data_to_csv.
#include "mlp.h"

int main() {
    srand((unsigned)time(NULL));
    openblas_set_num_threads(4);

    // PARAMETERS.
    const int input_dim = 4;            // Number of raw input features (tokens).
    const int num_samples = 1024;       // Number of samples.
    const int output_dim = input_dim;   // One output per input feature.
    const int embedding_dim = 64;       // Dimension of the learned embedding.
    // Full–batch training.
    const int batch_size = num_samples;

    // DATA GENERATION.
    // generate_synthetic_data produces X of shape [num_samples x input_dim]
    // and y of shape [num_samples x output_dim] with continuous values.
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);

    // INITIALIZE NETWORK.
    Net* net = init_net(input_dim, embedding_dim, output_dim, batch_size);
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
        // Project raw input using the learned input projection.
        project_input(net, X, embedded_input);
        // Forward pass: token–mixing MLP, feed–forward (both include swish activations and residuals),
        // then output projection.
        forward_pass(net, embedded_input);
        float loss = calculate_loss(net, y);
        // Zero gradients.
        zero_gradients(net);
        // Backward pass.
        backward_pass(net, embedded_input, y);
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
    project_input(net, X, embedded_input);
    forward_pass(net, embedded_input);
    float verification_loss = calculate_loss(net, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    // EVALUATION.
    // For each sample and each token, we now compare the final scalar output to the true continuous value.
    double sum_abs_error = 0.0;
    for (int i = 0; i < num_samples; i++) {
        for (int d = 0; d < output_dim; d++) {
            float pred = net->final_output[i * output_dim + d];
            sum_abs_error += fabs(pred - y[i * output_dim + d]);
        }
    }
    int total_outputs = num_samples * output_dim;
    float mae = sum_abs_error / total_outputs;
    printf("\nMean Absolute Error (Continuous Prediction): %.5f\n", mae);

    // Print sample predictions (for the first 5 samples).
    printf("\nSample Predictions (first 5 samples):\n");
    for (int i = 0; i < 5; i++) {
        printf("Sample %d:\n", i);
        for (int d = 0; d < output_dim; d++) {
            float pred = net->final_output[i * output_dim + d];
            printf("  Output %d: Predicted = %.5f, True = %.5f\n", d, pred, y[i * output_dim + d]);
        }
    }

    // CLEANUP.
    free(X);
    free(y);
    free(embedded_input);
    free_net(net);

    return 0;
}