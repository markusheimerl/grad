#ifndef GRAD_H
#define GRAD_H

typedef struct {
    // Weights and gradients
    float* fc1_weight;     // hidden_dim x input_dim
    float* fc2_weight;     // output_dim x hidden_dim
    float* fc1_weight_grad; // hidden_dim x input_dim
    float* fc2_weight_grad; // output_dim x hidden_dim
    
    // Helper arrays for forward/backward pass
    float* layer1_output;   // batch_size x hidden_dim
    float* predictions;     // batch_size x output_dim
    float* error;          // batch_size x output_dim
    float* pre_activation; // batch_size x hidden_dim
    float* error_hidden;   // batch_size x hidden_dim
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} Net;

// Initialize the network with configurable dimensions
Net* init_net(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    
    // Store dimensions
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    
    // Allocate and initialize weights and gradients
    net->fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    net->fc1_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    net->fc2_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate helper arrays
    net->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->pre_activation = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    net->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    // Initialize weights
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        net->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        net->fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    return net;
}

// Free network memory
void free_net(Net* net) {
    free(net->fc1_weight);
    free(net->fc2_weight);
    free(net->fc1_weight_grad);
    free(net->fc2_weight_grad);
    free(net->layer1_output);
    free(net->predictions);
    free(net->error);
    free(net->pre_activation);
    free(net->error_hidden);
    free(net);
}

// Forward pass
void forward_pass(Net* net, float* X) {
    // First layer
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->hidden_dim,
                net->input_dim,
                1.0f,
                X,
                net->input_dim,
                net->fc1_weight,
                net->hidden_dim,
                0.0f,
                net->layer1_output,
                net->hidden_dim);
    
    // Store pre-activation values
    memcpy(net->pre_activation, net->layer1_output, 
           net->batch_size * net->hidden_dim * sizeof(float));
    
    // Swish activation
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        net->layer1_output[i] = net->layer1_output[i] / (1.0f + expf(-net->layer1_output[i]));
    }
    
    // Second layer
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                net->batch_size,
                net->output_dim,
                net->hidden_dim,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->fc2_weight,
                net->output_dim,
                0.0f,
                net->predictions,
                net->output_dim);
}

// Calculate loss
float calculate_loss(Net* net, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < net->batch_size * net->output_dim; i++) {
        net->error[i] = net->predictions[i] - y[i];
        loss += net->error[i] * net->error[i];
    }
    return loss / (net->batch_size * net->output_dim);
}

// Zero gradients
void zero_gradients(Net* net) {
    memset(net->fc1_weight_grad, 0, net->hidden_dim * net->input_dim * sizeof(float));
    memset(net->fc2_weight_grad, 0, net->output_dim * net->hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass(Net* net, float* X) {
    // Gradient of second layer
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->hidden_dim,
                net->output_dim,
                net->batch_size,
                1.0f,
                net->layer1_output,
                net->hidden_dim,
                net->error,
                net->output_dim,
                0.0f,
                net->fc2_weight_grad,
                net->output_dim);
    
    // Backpropagate error through second layer
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                net->batch_size,
                net->hidden_dim,
                net->output_dim,
                1.0f,
                net->error,
                net->output_dim,
                net->fc2_weight,
                net->output_dim,
                0.0f,
                net->error_hidden,
                net->hidden_dim);
    
    // Swish derivative
    for (int i = 0; i < net->batch_size * net->hidden_dim; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-net->pre_activation[i]));
        net->error_hidden[i] *= sigmoid + net->pre_activation[i] * sigmoid * (1.0f - sigmoid);
    }
    
    // Gradient of first layer
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                net->input_dim,
                net->hidden_dim,
                net->batch_size,
                1.0f,
                X,
                net->input_dim,
                net->error_hidden,
                net->hidden_dim,
                0.0f,
                net->fc1_weight_grad,
                net->hidden_dim);
}

// Update weights
void update_weights(Net* net, float learning_rate) {
    for (int i = 0; i < net->hidden_dim * net->input_dim; i++) {
        net->fc1_weight[i] -= learning_rate * net->fc1_weight_grad[i] / net->batch_size;
    }
    for (int i = 0; i < net->output_dim * net->hidden_dim; i++) {
        net->fc2_weight[i] -= learning_rate * net->fc2_weight_grad[i] / net->batch_size;
    }
}

// Function to save model weights to binary file
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    
    // Save batch size
    fwrite(&net->batch_size, sizeof(int), 1, file);

    // Save weights
    fwrite(net->fc1_weight, sizeof(float), net->hidden_dim * net->input_dim, file);
    fwrite(net->fc2_weight, sizeof(float), net->output_dim * net->hidden_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to load model weights from binary file
Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    
    // Load batch size
    fread(&batch_size, sizeof(int), 1, file);

    // Initialize network with loaded dimensions and batch size
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(net->fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(net->fc2_weight, sizeof(float), output_dim * hidden_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return net;
}

#endif