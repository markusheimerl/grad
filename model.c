#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "dnnl.h"

// Model Architecture
#define INPUT_SIZE 15
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 128
#define HIDDEN3_SIZE 64
#define OUTPUT_SIZE 4
#define BATCH_SIZE 32
#define NUM_EPOCHS 1000
#define LEARNING_RATE 0.001
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// Error checking macro
#define CHECK(f) do { \
    dnnl_status_t s = f; \
    if (s != dnnl_success) { \
        printf("Error: %s\n", dnnl_status2str(s)); \
        exit(1); \
    } \
} while(0)

// Memory allocation macro
#define MALLOC_CHECK(ptr) do { \
    if (ptr == NULL) { \
        printf("Memory allocation failed\n"); \
        exit(1); \
    } \
} while(0)

// Layer structure
typedef struct {
    dnnl_memory_desc_t src_md;
    dnnl_memory_desc_t weights_md;
    dnnl_memory_desc_t bias_md;
    dnnl_memory_desc_t dst_md;
    dnnl_primitive_desc_t fwd_pd;
    dnnl_primitive_desc_t bwd_pd;
    dnnl_primitive_t forward;
    dnnl_primitive_t backward;
    float* weights;
    float* bias;
    float* weights_grad;
    float* bias_grad;
    int input_size;
    int output_size;
} Layer;

// Adam optimizer structure
typedef struct {
    float* m;  // First moment
    float* v;  // Second moment
    float beta1_t;  // beta1^t
    float beta2_t;  // beta2^t
    int size;
} AdamOptimizer;

// Model structure
typedef struct {
    Layer layer1;
    Layer layer2;
    Layer layer3;
    Layer layer4;
    dnnl_engine_t engine;
    dnnl_stream_t stream;
    AdamOptimizer* optimizers[4];
} Model;

// Function declarations
float* allocate_float_array(size_t size);
void xavier_init(float* array, int fan_in, int fan_out);
void create_layer(Layer* layer, int input_size, int output_size, 
                 dnnl_engine_t engine, dnnl_primitive_attr_t attr);
AdamOptimizer* create_optimizer(int size);
void update_weights_adam(AdamOptimizer* opt, float* weights, float* gradients, 
                        int size, float lr);
float calculate_mse(float* pred, float* target, int size);
float calculate_r2_score(float* y_true, float* y_pred, int size);
void read_csv(const char* filename, float** X, float** y, int* num_samples);
void save_model(Model* model, const char* filename);
void load_model(Model* model, const char* filename);

// Utility function implementations
float* allocate_float_array(size_t size) {
    float* array = (float*)malloc(size * sizeof(float));
    MALLOC_CHECK(array);
    return array;
}

void xavier_init(float* array, int fan_in, int fan_out) {
    float scale = sqrt(2.0f / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        array[i] = scale * ((float)rand() / RAND_MAX * 2 - 1);
    }
}

void read_csv(const char* filename, float** X, float** y, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }

    // Count lines first
    char line[4096];
    *num_samples = 0;
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    (*num_samples)--; // Remove header line

    // Allocate memory
    *X = allocate_float_array(*num_samples * INPUT_SIZE);
    *y = allocate_float_array(*num_samples * OUTPUT_SIZE);

    // Reset file pointer and skip header
    rewind(file);
    fgets(line, sizeof(line), file);

    // Read data
    int row = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (int i = 0; i < INPUT_SIZE; i++) {
            (*X)[row * INPUT_SIZE + i] = atof(token);
            token = strtok(NULL, ",");
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            (*y)[row * OUTPUT_SIZE + i] = atof(token);
            token = strtok(NULL, ",");
        }
        row++;
    }
    fclose(file);
}

float calculate_mse(float* pred, float* target, int size) {
    float mse = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = pred[i] - target[i];
        mse += diff * diff;
    }
    return mse / size;
}

float calculate_r2_score(float* y_true, float* y_pred, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += y_true[i];
    }
    mean /= size;

    float ss_tot = 0.0f;
    float ss_res = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff_tot = y_true[i] - mean;
        float diff_res = y_true[i] - y_pred[i];
        ss_tot += diff_tot * diff_tot;
        ss_res += diff_res * diff_res;
    }

    return 1.0f - (ss_res / ss_tot);
}

// Adam optimizer implementation
AdamOptimizer* create_optimizer(int size) {
    AdamOptimizer* opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    MALLOC_CHECK(opt);
    
    opt->m = allocate_float_array(size);
    opt->v = allocate_float_array(size);
    memset(opt->m, 0, size * sizeof(float));
    memset(opt->v, 0, size * sizeof(float));
    opt->beta1_t = BETA1;
    opt->beta2_t = BETA2;
    opt->size = size;
    
    return opt;
}

void update_weights_adam(AdamOptimizer* opt, float* weights, float* gradients, 
                        int size, float lr) {
    float beta1_correction = 1.0f / (1.0f - opt->beta1_t);
    float beta2_correction = 1.0f / (1.0f - opt->beta2_t);
    
    for (int i = 0; i < size; i++) {
        // Update biased first moment estimate
        opt->m[i] = BETA1 * opt->m[i] + (1.0f - BETA1) * gradients[i];
        // Update biased second raw moment estimate
        opt->v[i] = BETA2 * opt->v[i] + (1.0f - BETA2) * gradients[i] * gradients[i];
        
        // Compute bias-corrected first moment estimate
        float m_hat = opt->m[i] * beta1_correction;
        // Compute bias-corrected second raw moment estimate
        float v_hat = opt->v[i] * beta2_correction;
        
        // Update weights
        weights[i] -= lr * m_hat / (sqrt(v_hat) + EPSILON);
    }
    
    // Update beta powers
    opt->beta1_t *= BETA1;
    opt->beta2_t *= BETA2;
}

// Layer creation and initialization
void create_layer(Layer* layer, int input_size, int output_size, 
                 dnnl_engine_t engine, dnnl_primitive_attr_t attr) {
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // Allocate memory for weights and biases
    layer->weights = allocate_float_array(input_size * output_size);
    layer->bias = allocate_float_array(output_size);
    layer->weights_grad = allocate_float_array(input_size * output_size);
    layer->bias_grad = allocate_float_array(output_size);
    
    // Initialize weights using Xavier initialization
    xavier_init(layer->weights, input_size, output_size);
    memset(layer->bias, 0, output_size * sizeof(float));
    
    // Create memory descriptors
    CHECK(dnnl_memory_desc_create_with_tag(&layer->src_md, 2, 
        (int64_t[]){BATCH_SIZE, input_size}, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_desc_create_with_tag(&layer->weights_md, 2,
        (int64_t[]){output_size, input_size}, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_desc_create_with_tag(&layer->bias_md, 1,
        (int64_t[]){output_size}, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_desc_create_with_tag(&layer->dst_md, 2,
        (int64_t[]){BATCH_SIZE, output_size}, dnnl_f32, dnnl_nc));
    
    // Create forward primitive descriptor
    CHECK(dnnl_inner_product_forward_primitive_desc_create(
        &layer->fwd_pd, engine, dnnl_forward_training,
        layer->src_md, layer->weights_md, layer->bias_md,
        layer->dst_md, attr));
    
    // Create backward primitive descriptor
    dnnl_memory_desc_t diff_src_md, diff_weights_md, diff_bias_md, diff_dst_md;
    CHECK(dnnl_memory_desc_clone(&diff_src_md, layer->src_md));
    CHECK(dnnl_memory_desc_clone(&diff_weights_md, layer->weights_md));
    CHECK(dnnl_memory_desc_clone(&diff_bias_md, layer->bias_md));
    CHECK(dnnl_memory_desc_clone(&diff_dst_md, layer->dst_md));
    
    CHECK(dnnl_inner_product_backward_primitive_desc_create(
        &layer->bwd_pd, engine, dnnl_backward,
        diff_src_md, diff_weights_md, diff_bias_md,
        diff_dst_md, layer->weights_md, attr, layer->fwd_pd));
    
    // Create primitives
    CHECK(dnnl_primitive_create(&layer->forward, layer->fwd_pd));
    CHECK(dnnl_primitive_create(&layer->backward, layer->bwd_pd));
}

// ReLU activation implementation
void apply_relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

void apply_relu_gradient(float* grad, float* data, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] = data[i] > 0 ? grad[i] : 0;
    }
}

// Model creation
Model* create_model(void) {
    Model* model = (Model*)malloc(sizeof(Model));
    MALLOC_CHECK(model);
    
    // Create DNNL engine and stream
    CHECK(dnnl_engine_create(&model->engine, dnnl_cpu, 0));
    CHECK(dnnl_stream_create(&model->stream, model->engine, dnnl_stream_default_flags));
    
    // Create primitive attribute (used for layer creation)
    dnnl_primitive_attr_t attr;
    CHECK(dnnl_primitive_attr_create(&attr));
    
    // Create layers
    create_layer(&model->layer1, INPUT_SIZE, HIDDEN1_SIZE, model->engine, attr);
    create_layer(&model->layer2, HIDDEN1_SIZE, HIDDEN2_SIZE, model->engine, attr);
    create_layer(&model->layer3, HIDDEN2_SIZE, HIDDEN3_SIZE, model->engine, attr);
    create_layer(&model->layer4, HIDDEN3_SIZE, OUTPUT_SIZE, model->engine, attr);
    
    // Create optimizers
    model->optimizers[0] = create_optimizer(INPUT_SIZE * HIDDEN1_SIZE + HIDDEN1_SIZE);
    model->optimizers[1] = create_optimizer(HIDDEN1_SIZE * HIDDEN2_SIZE + HIDDEN2_SIZE);
    model->optimizers[2] = create_optimizer(HIDDEN2_SIZE * HIDDEN3_SIZE + HIDDEN3_SIZE);
    model->optimizers[3] = create_optimizer(HIDDEN3_SIZE * OUTPUT_SIZE + OUTPUT_SIZE);
    
    // Cleanup
    CHECK(dnnl_primitive_attr_destroy(attr));
    
    return model;
}

// Model destruction
void destroy_model(Model* model) {
    // Destroy layers
    dnnl_primitive_destroy(model->layer1.forward);
    dnnl_primitive_destroy(model->layer1.backward);
    dnnl_primitive_destroy(model->layer2.forward);
    dnnl_primitive_destroy(model->layer2.backward);
    dnnl_primitive_destroy(model->layer3.forward);
    dnnl_primitive_destroy(model->layer3.backward);
    dnnl_primitive_destroy(model->layer4.forward);
    dnnl_primitive_destroy(model->layer4.backward);
    
    // Free memory
    free(model->layer1.weights);
    free(model->layer1.bias);
    free(model->layer1.weights_grad);
    free(model->layer1.bias_grad);
    free(model->layer2.weights);
    free(model->layer2.bias);
    free(model->layer2.weights_grad);
    free(model->layer2.bias_grad);
    free(model->layer3.weights);
    free(model->layer3.bias);
    free(model->layer3.weights_grad);
    free(model->layer3.bias_grad);
    free(model->layer4.weights);
    free(model->layer4.bias);
    free(model->layer4.weights_grad);
    free(model->layer4.bias_grad);
    
    // Free optimizers
    for (int i = 0; i < 4; i++) {
        free(model->optimizers[i]->m);
        free(model->optimizers[i]->v);
        free(model->optimizers[i]);
    }
    
    // Destroy DNNL objects
    dnnl_stream_destroy(model->stream);
    dnnl_engine_destroy(model->engine);
    
    free(model);
}

// Forward pass implementation
void forward_pass(Model* model, float* input, float* output, float** layer_outputs, int training) {
    dnnl_memory_t src_memory, weights_memory, bias_memory, dst_memory;
    const void* fwd_args[4];
    
    // Layer 1
    CHECK(dnnl_memory_create(&src_memory, model->layer1.src_md, model->engine, input));
    CHECK(dnnl_memory_create(&weights_memory, model->layer1.weights_md, model->engine, model->layer1.weights));
    CHECK(dnnl_memory_create(&bias_memory, model->layer1.bias_md, model->engine, model->layer1.bias));
    CHECK(dnnl_memory_create(&dst_memory, model->layer1.dst_md, model->engine, layer_outputs[0]));
    
    fwd_args[0] = src_memory;
    fwd_args[1] = weights_memory;
    fwd_args[2] = bias_memory;
    fwd_args[3] = dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer1.forward, model->stream, 4, fwd_args));
    apply_relu(layer_outputs[0], BATCH_SIZE * HIDDEN1_SIZE);
    
    // Layer 2
    CHECK(dnnl_memory_create(&src_memory, model->layer2.src_md, model->engine, layer_outputs[0]));
    CHECK(dnnl_memory_create(&weights_memory, model->layer2.weights_md, model->engine, model->layer2.weights));
    CHECK(dnnl_memory_create(&bias_memory, model->layer2.bias_md, model->engine, model->layer2.bias));
    CHECK(dnnl_memory_create(&dst_memory, model->layer2.dst_md, model->engine, layer_outputs[1]));
    
    fwd_args[0] = src_memory;
    fwd_args[1] = weights_memory;
    fwd_args[2] = bias_memory;
    fwd_args[3] = dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer2.forward, model->stream, 4, fwd_args));
    apply_relu(layer_outputs[1], BATCH_SIZE * HIDDEN2_SIZE);
    
    // Layer 3
    CHECK(dnnl_memory_create(&src_memory, model->layer3.src_md, model->engine, layer_outputs[1]));
    CHECK(dnnl_memory_create(&weights_memory, model->layer3.weights_md, model->engine, model->layer3.weights));
    CHECK(dnnl_memory_create(&bias_memory, model->layer3.bias_md, model->engine, model->layer3.bias));
    CHECK(dnnl_memory_create(&dst_memory, model->layer3.dst_md, model->engine, layer_outputs[2]));
    
    fwd_args[0] = src_memory;
    fwd_args[1] = weights_memory;
    fwd_args[2] = bias_memory;
    fwd_args[3] = dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer3.forward, model->stream, 4, fwd_args));
    apply_relu(layer_outputs[2], BATCH_SIZE * HIDDEN3_SIZE);
    
    // Layer 4 (output layer)
    CHECK(dnnl_memory_create(&src_memory, model->layer4.src_md, model->engine, layer_outputs[2]));
    CHECK(dnnl_memory_create(&weights_memory, model->layer4.weights_md, model->engine, model->layer4.weights));
    CHECK(dnnl_memory_create(&bias_memory, model->layer4.bias_md, model->engine, model->layer4.bias));
    CHECK(dnnl_memory_create(&dst_memory, model->layer4.dst_md, model->engine, output));
    
    fwd_args[0] = src_memory;
    fwd_args[1] = weights_memory;
    fwd_args[2] = bias_memory;
    fwd_args[3] = dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer4.forward, model->stream, 4, fwd_args));
}

void backward_pass(Model* model, float* input, float* output, float* target, 
                  float** layer_outputs, float** layer_gradients) {
    // Calculate output gradient (MSE derivative)
    for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; i++) {
        layer_gradients[3][i] = 2.0f * (output[i] - target[i]) / BATCH_SIZE;
    }
    
    dnnl_memory_t src_memory, weights_memory, dst_memory;
    dnnl_memory_t diff_src_memory, diff_weights_memory, diff_bias_memory, diff_dst_memory;
    const void* bwd_args[6];

    // Layer 4 backward (output layer)
    CHECK(dnnl_memory_create(&diff_dst_memory, model->layer4.dst_md, model->engine, layer_gradients[3]));
    CHECK(dnnl_memory_create(&src_memory, model->layer4.src_md, model->engine, layer_outputs[2]));
    CHECK(dnnl_memory_create(&weights_memory, model->layer4.weights_md, model->engine, model->layer4.weights));
    CHECK(dnnl_memory_create(&diff_src_memory, model->layer4.src_md, model->engine, layer_gradients[2]));
    CHECK(dnnl_memory_create(&diff_weights_memory, model->layer4.weights_md, model->engine, model->layer4.weights_grad));
    CHECK(dnnl_memory_create(&diff_bias_memory, model->layer4.bias_md, model->engine, model->layer4.bias_grad));
    
    bwd_args[0] = src_memory;
    bwd_args[1] = diff_dst_memory;
    bwd_args[2] = weights_memory;
    bwd_args[3] = diff_src_memory;
    bwd_args[4] = diff_weights_memory;
    bwd_args[5] = diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer4.backward, model->stream, 6, bwd_args));
    
    // Clean up layer 4 memories
    CHECK(dnnl_memory_destroy(src_memory));
    CHECK(dnnl_memory_destroy(weights_memory));
    CHECK(dnnl_memory_destroy(diff_dst_memory));
    CHECK(dnnl_memory_destroy(diff_src_memory));
    CHECK(dnnl_memory_destroy(diff_weights_memory));
    CHECK(dnnl_memory_destroy(diff_bias_memory));

    // Layer 3 backward
    apply_relu_gradient(layer_gradients[2], layer_outputs[2], BATCH_SIZE * HIDDEN3_SIZE);
    
    CHECK(dnnl_memory_create(&diff_dst_memory, model->layer3.dst_md, model->engine, layer_gradients[2]));
    CHECK(dnnl_memory_create(&src_memory, model->layer3.src_md, model->engine, layer_outputs[1]));
    CHECK(dnnl_memory_create(&weights_memory, model->layer3.weights_md, model->engine, model->layer3.weights));
    CHECK(dnnl_memory_create(&diff_src_memory, model->layer3.src_md, model->engine, layer_gradients[1]));
    CHECK(dnnl_memory_create(&diff_weights_memory, model->layer3.weights_md, model->engine, model->layer3.weights_grad));
    CHECK(dnnl_memory_create(&diff_bias_memory, model->layer3.bias_md, model->engine, model->layer3.bias_grad));
    
    bwd_args[0] = src_memory;
    bwd_args[1] = diff_dst_memory;
    bwd_args[2] = weights_memory;
    bwd_args[3] = diff_src_memory;
    bwd_args[4] = diff_weights_memory;
    bwd_args[5] = diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer3.backward, model->stream, 6, bwd_args));
    
    // Clean up layer 3 memories
    CHECK(dnnl_memory_destroy(src_memory));
    CHECK(dnnl_memory_destroy(weights_memory));
    CHECK(dnnl_memory_destroy(diff_dst_memory));
    CHECK(dnnl_memory_destroy(diff_src_memory));
    CHECK(dnnl_memory_destroy(diff_weights_memory));
    CHECK(dnnl_memory_destroy(diff_bias_memory));

    // Layer 2 backward
    apply_relu_gradient(layer_gradients[1], layer_outputs[1], BATCH_SIZE * HIDDEN2_SIZE);
    
    CHECK(dnnl_memory_create(&diff_dst_memory, model->layer2.dst_md, model->engine, layer_gradients[1]));
    CHECK(dnnl_memory_create(&src_memory, model->layer2.src_md, model->engine, layer_outputs[0]));
    CHECK(dnnl_memory_create(&weights_memory, model->layer2.weights_md, model->engine, model->layer2.weights));
    CHECK(dnnl_memory_create(&diff_src_memory, model->layer2.src_md, model->engine, layer_gradients[0]));
    CHECK(dnnl_memory_create(&diff_weights_memory, model->layer2.weights_md, model->engine, model->layer2.weights_grad));
    CHECK(dnnl_memory_create(&diff_bias_memory, model->layer2.bias_md, model->engine, model->layer2.bias_grad));
    
    bwd_args[0] = src_memory;
    bwd_args[1] = diff_dst_memory;
    bwd_args[2] = weights_memory;
    bwd_args[3] = diff_src_memory;
    bwd_args[4] = diff_weights_memory;
    bwd_args[5] = diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer2.backward, model->stream, 6, bwd_args));
    
    // Clean up layer 2 memories
    CHECK(dnnl_memory_destroy(src_memory));
    CHECK(dnnl_memory_destroy(weights_memory));
    CHECK(dnnl_memory_destroy(diff_dst_memory));
    CHECK(dnnl_memory_destroy(diff_src_memory));
    CHECK(dnnl_memory_destroy(diff_weights_memory));
    CHECK(dnnl_memory_destroy(diff_bias_memory));

    // Layer 1 backward
    apply_relu_gradient(layer_gradients[0], layer_outputs[0], BATCH_SIZE * HIDDEN1_SIZE);
    
    CHECK(dnnl_memory_create(&diff_dst_memory, model->layer1.dst_md, model->engine, layer_gradients[0]));
    CHECK(dnnl_memory_create(&src_memory, model->layer1.src_md, model->engine, input));
    CHECK(dnnl_memory_create(&weights_memory, model->layer1.weights_md, model->engine, model->layer1.weights));
    CHECK(dnnl_memory_create(&diff_src_memory, model->layer1.src_md, model->engine, layer_gradients[0]));
    CHECK(dnnl_memory_create(&diff_weights_memory, model->layer1.weights_md, model->engine, model->layer1.weights_grad));
    CHECK(dnnl_memory_create(&diff_bias_memory, model->layer1.bias_md, model->engine, model->layer1.bias_grad));
    
    bwd_args[0] = src_memory;
    bwd_args[1] = diff_dst_memory;
    bwd_args[2] = weights_memory;
    bwd_args[3] = diff_src_memory;
    bwd_args[4] = diff_weights_memory;
    bwd_args[5] = diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer1.backward, model->stream, 6, bwd_args));
    
    // Clean up layer 1 memories
    CHECK(dnnl_memory_destroy(src_memory));
    CHECK(dnnl_memory_destroy(weights_memory));
    CHECK(dnnl_memory_destroy(diff_dst_memory));
    CHECK(dnnl_memory_destroy(diff_src_memory));
    CHECK(dnnl_memory_destroy(diff_weights_memory));
    CHECK(dnnl_memory_destroy(diff_bias_memory));

    // Ensure all operations are complete
    CHECK(dnnl_stream_wait(model->stream));
}

int main() {
    srand(time(NULL));
    
    // Load data
    float *X, *y;
    int num_samples;
    read_csv("20250208_163908_data.csv", &X, &y, &num_samples);
    
    // Create model
    Model* model = create_model();
    
    // Allocate memory for intermediate results
    float** layer_outputs = (float**)malloc(4 * sizeof(float*));
    float** layer_gradients = (float**)malloc(4 * sizeof(float*));
    
    layer_outputs[0] = allocate_float_array(BATCH_SIZE * HIDDEN1_SIZE);
    layer_outputs[1] = allocate_float_array(BATCH_SIZE * HIDDEN2_SIZE);
    layer_outputs[2] = allocate_float_array(BATCH_SIZE * HIDDEN3_SIZE);
    layer_outputs[3] = allocate_float_array(BATCH_SIZE * OUTPUT_SIZE);
    
    layer_gradients[0] = allocate_float_array(BATCH_SIZE * HIDDEN1_SIZE);
    layer_gradients[1] = allocate_float_array(BATCH_SIZE * HIDDEN2_SIZE);
    layer_gradients[2] = allocate_float_array(BATCH_SIZE * HIDDEN3_SIZE);
    layer_gradients[3] = allocate_float_array(BATCH_SIZE * OUTPUT_SIZE);
    
    // Training loop
    float* batch_X = allocate_float_array(BATCH_SIZE * INPUT_SIZE);
    float* batch_y = allocate_float_array(BATCH_SIZE * OUTPUT_SIZE);
    float* output = allocate_float_array(BATCH_SIZE * OUTPUT_SIZE);
    
    printf("Starting training...\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_samples / BATCH_SIZE; batch++) {
            // Prepare batch
            int batch_start = batch * BATCH_SIZE;
            memcpy(batch_X, &X[batch_start * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float));
            memcpy(batch_y, &y[batch_start * OUTPUT_SIZE], BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            
            // Forward pass
            forward_pass(model, batch_X, output, layer_outputs, 1);
            
            // Calculate loss
            float batch_loss = calculate_mse(output, batch_y, BATCH_SIZE * OUTPUT_SIZE);
            epoch_loss += batch_loss;
            
            // Backward pass
            backward_pass(model, batch_X, output, batch_y, layer_outputs, layer_gradients);
            
            // Update weights using Adam
            update_weights_adam(model->optimizers[0], model->layer1.weights, 
                              model->layer1.weights_grad, 
                              INPUT_SIZE * HIDDEN1_SIZE, LEARNING_RATE);
            // ... Similar updates for other layers
        }
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.4f\n", 
                   epoch + 1, NUM_EPOCHS, epoch_loss / (num_samples / BATCH_SIZE));
        }
    }
    
    // Calculate and print R² scores
    forward_pass(model, X, output, layer_outputs, 0);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float r2 = calculate_r2_score(&y[i], &output[i], num_samples);
        printf("R² score for output y%d: %.4f\n", i, r2);
    }
    
    // Save model
    save_model(model, "model.bin");
    
    // Cleanup
    free(X);
    free(y);
    free(batch_X);
    free(batch_y);
    free(output);
    for (int i = 0; i < 4; i++) {
        free(layer_outputs[i]);
        free(layer_gradients[i]);
    }
    free(layer_outputs);
    free(layer_gradients);
    destroy_model(model);
    
    return 0;
}