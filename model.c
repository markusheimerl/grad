// gcc -o model model.c -ldnnl -lm -O3 -march=native -ffast-math

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
    // Descriptors
    dnnl_memory_desc_t src_md;
    dnnl_memory_desc_t weights_md;
    dnnl_memory_desc_t bias_md;
    dnnl_memory_desc_t dst_md;
    
    // Primitive descriptors
    dnnl_primitive_desc_t fwd_pd;
    dnnl_primitive_desc_t bwd_pd;
    dnnl_primitive_desc_t relu_fwd_pd;
    dnnl_primitive_desc_t relu_bwd_pd;
    
    // Primitives
    dnnl_primitive_t forward;
    dnnl_primitive_t backward;
    dnnl_primitive_t relu_forward;
    dnnl_primitive_t relu_backward;
    
    // Memory objects
    dnnl_memory_t src_memory;
    dnnl_memory_t weights_memory;
    dnnl_memory_t bias_memory;
    dnnl_memory_t dst_memory;
    dnnl_memory_t relu_dst_memory;
    dnnl_memory_t diff_src_memory;
    dnnl_memory_t diff_weights_memory;
    dnnl_memory_t diff_bias_memory;
    dnnl_memory_t diff_dst_memory;
    dnnl_memory_t relu_diff_dst_memory;
    
    // Data buffers
    float* weights;
    float* bias;
    float* weights_grad;
    float* bias_grad;
    float* output;
    float* relu_output;
    float* input_grad;
    
    int input_size;
    int output_size;
} Layer;

// Adam optimizer structure
typedef struct {
    float* m_weights;  // First moment for weights
    float* v_weights;  // Second moment for weights
    float* m_bias;     // First moment for bias
    float* v_bias;     // Second moment for bias
    float beta1_t;     // beta1^t
    float beta2_t;     // beta2^t
    int weights_size;
    int bias_size;
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

AdamOptimizer* create_optimizer(int weights_size, int bias_size) {
    AdamOptimizer* opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    MALLOC_CHECK(opt);
    
    opt->m_weights = allocate_float_array(weights_size);
    opt->v_weights = allocate_float_array(weights_size);
    opt->m_bias = allocate_float_array(bias_size);
    opt->v_bias = allocate_float_array(bias_size);
    
    memset(opt->m_weights, 0, weights_size * sizeof(float));
    memset(opt->v_weights, 0, weights_size * sizeof(float));
    memset(opt->m_bias, 0, bias_size * sizeof(float));
    memset(opt->v_bias, 0, bias_size * sizeof(float));
    
    opt->beta1_t = BETA1;
    opt->beta2_t = BETA2;
    opt->weights_size = weights_size;
    opt->bias_size = bias_size;
    
    return opt;
}

void read_csv(const char* filename, float** X, float** y, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }

    char line[4096];
    *num_samples = 0;
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    (*num_samples)--; // Remove header line

    *X = allocate_float_array(*num_samples * INPUT_SIZE);
    *y = allocate_float_array(*num_samples * OUTPUT_SIZE);

    rewind(file);
    fgets(line, sizeof(line), file); // Skip header

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

void create_layer(Layer* layer, int input_size, int output_size, 
                 dnnl_engine_t engine, dnnl_primitive_attr_t attr) {
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // Allocate memory for weights, biases, and gradients
    layer->weights = allocate_float_array(input_size * output_size);
    layer->bias = allocate_float_array(output_size);
    layer->weights_grad = allocate_float_array(input_size * output_size);
    layer->bias_grad = allocate_float_array(output_size);
    layer->output = allocate_float_array(BATCH_SIZE * output_size);
    layer->relu_output = allocate_float_array(BATCH_SIZE * output_size);
    layer->input_grad = allocate_float_array(BATCH_SIZE * input_size);
    
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

    // Create forward primitive descriptor for linear layer
    CHECK(dnnl_inner_product_forward_primitive_desc_create(
        &layer->fwd_pd, engine, dnnl_forward_training,
        layer->src_md, layer->weights_md, layer->bias_md,
        layer->dst_md, attr));

    // Create ReLU forward descriptor and primitive descriptor
    dnnl_eltwise_desc_t relu_desc;
    CHECK(dnnl_eltwise_forward_desc_init(&relu_desc,
        dnnl_forward_training, dnnl_eltwise_relu,
        layer->dst_md, 0.0f, 0.0f));
    
    CHECK(dnnl_primitive_desc_create(&layer->relu_fwd_pd,
        &relu_desc, attr, engine, NULL));

    // Create backward primitive descriptors
    dnnl_memory_desc_t diff_src_md, diff_weights_md, diff_bias_md, diff_dst_md;
    CHECK(dnnl_memory_desc_clone(&diff_src_md, layer->src_md));
    CHECK(dnnl_memory_desc_clone(&diff_weights_md, layer->weights_md));
    CHECK(dnnl_memory_desc_clone(&diff_bias_md, layer->bias_md));
    CHECK(dnnl_memory_desc_clone(&diff_dst_md, layer->dst_md));

    // Create backward primitive descriptor for linear layer
    CHECK(dnnl_inner_product_backward_primitive_desc_create(
        &layer->bwd_pd, engine, dnnl_backward,
        diff_src_md, diff_weights_md, diff_bias_md,
        diff_dst_md, layer->weights_md, attr, layer->fwd_pd));

    // Create ReLU backward descriptor and primitive descriptor
    dnnl_eltwise_desc_t relu_bwd_desc;
    CHECK(dnnl_eltwise_backward_desc_init(&relu_bwd_desc,
        dnnl_eltwise_relu, diff_dst_md, diff_dst_md, 0.0f, 0.0f));
    
    CHECK(dnnl_primitive_desc_create(&layer->relu_bwd_pd,
        &relu_bwd_desc, attr, engine, layer->relu_fwd_pd));

    // Create primitives
    CHECK(dnnl_primitive_create(&layer->forward, layer->fwd_pd));
    CHECK(dnnl_primitive_create(&layer->backward, layer->bwd_pd));
    CHECK(dnnl_primitive_create(&layer->relu_forward, layer->relu_fwd_pd));
    CHECK(dnnl_primitive_create(&layer->relu_backward, layer->relu_bwd_pd));

    // Create memory objects
    CHECK(dnnl_memory_create(&layer->src_memory, layer->src_md, engine, NULL));
    CHECK(dnnl_memory_create(&layer->weights_memory, layer->weights_md, engine, layer->weights));
    CHECK(dnnl_memory_create(&layer->bias_memory, layer->bias_md, engine, layer->bias));
    CHECK(dnnl_memory_create(&layer->dst_memory, layer->dst_md, engine, layer->output));
    CHECK(dnnl_memory_create(&layer->relu_dst_memory, layer->dst_md, engine, layer->relu_output));
    
    CHECK(dnnl_memory_create(&layer->diff_src_memory, diff_src_md, engine, layer->input_grad));
    CHECK(dnnl_memory_create(&layer->diff_weights_memory, diff_weights_md, engine, layer->weights_grad));
    CHECK(dnnl_memory_create(&layer->diff_bias_memory, diff_bias_md, engine, layer->bias_grad));
    CHECK(dnnl_memory_create(&layer->diff_dst_memory, diff_dst_md, engine, NULL));
    CHECK(dnnl_memory_create(&layer->relu_diff_dst_memory, diff_dst_md, engine, NULL));
}

void destroy_layer(Layer* layer) {
    // Destroy primitives
    dnnl_primitive_destroy(layer->forward);
    dnnl_primitive_destroy(layer->backward);
    dnnl_primitive_destroy(layer->relu_forward);
    dnnl_primitive_destroy(layer->relu_backward);

    // Destroy primitive descriptors
    dnnl_primitive_desc_destroy(layer->fwd_pd);
    dnnl_primitive_desc_destroy(layer->bwd_pd);
    dnnl_primitive_desc_destroy(layer->relu_fwd_pd);
    dnnl_primitive_desc_destroy(layer->relu_bwd_pd);

    // Destroy memory objects
    dnnl_memory_destroy(layer->src_memory);
    dnnl_memory_destroy(layer->weights_memory);
    dnnl_memory_destroy(layer->bias_memory);
    dnnl_memory_destroy(layer->dst_memory);
    dnnl_memory_destroy(layer->relu_dst_memory);
    dnnl_memory_destroy(layer->diff_src_memory);
    dnnl_memory_destroy(layer->diff_weights_memory);
    dnnl_memory_destroy(layer->diff_bias_memory);
    dnnl_memory_destroy(layer->diff_dst_memory);
    dnnl_memory_destroy(layer->relu_diff_dst_memory);

    // Free data buffers
    free(layer->weights);
    free(layer->bias);
    free(layer->weights_grad);
    free(layer->bias_grad);
    free(layer->output);
    free(layer->relu_output);
    free(layer->input_grad);
}

void update_weights_adam(AdamOptimizer* opt, float* weights, float* bias,
                        float* weights_grad, float* bias_grad, float lr) {
    float beta1_correction = 1.0f / (1.0f - opt->beta1_t);
    float beta2_correction = 1.0f / (1.0f - opt->beta2_t);

    // Update weights
    for (int i = 0; i < opt->weights_size; i++) {
        opt->m_weights[i] = BETA1 * opt->m_weights[i] + (1.0f - BETA1) * weights_grad[i];
        opt->v_weights[i] = BETA2 * opt->v_weights[i] + (1.0f - BETA2) * weights_grad[i] * weights_grad[i];
        
        float m_hat = opt->m_weights[i] * beta1_correction;
        float v_hat = opt->v_weights[i] * beta2_correction;
        
        weights[i] -= lr * m_hat / (sqrt(v_hat) + EPSILON);
    }

    // Update bias
    for (int i = 0; i < opt->bias_size; i++) {
        opt->m_bias[i] = BETA1 * opt->m_bias[i] + (1.0f - BETA1) * bias_grad[i];
        opt->v_bias[i] = BETA2 * opt->v_bias[i] + (1.0f - BETA2) * bias_grad[i] * bias_grad[i];
        
        float m_hat = opt->m_bias[i] * beta1_correction;
        float v_hat = opt->v_bias[i] * beta2_correction;
        
        bias[i] -= lr * m_hat / (sqrt(v_hat) + EPSILON);
    }

    opt->beta1_t *= BETA1;
    opt->beta2_t *= BETA2;
}

Model* create_model(void) {
    Model* model = (Model*)malloc(sizeof(Model));
    MALLOC_CHECK(model);
    
    // Create DNNL engine and stream
    CHECK(dnnl_engine_create(&model->engine, dnnl_cpu, 0));
    CHECK(dnnl_stream_create(&model->stream, model->engine, dnnl_stream_default_flags));
    
    // Create primitive attribute
    dnnl_primitive_attr_t attr;
    CHECK(dnnl_primitive_attr_create(&attr));
    
    // Create layers
    create_layer(&model->layer1, INPUT_SIZE, HIDDEN1_SIZE, model->engine, attr);
    create_layer(&model->layer2, HIDDEN1_SIZE, HIDDEN2_SIZE, model->engine, attr);
    create_layer(&model->layer3, HIDDEN2_SIZE, HIDDEN3_SIZE, model->engine, attr);
    create_layer(&model->layer4, HIDDEN3_SIZE, OUTPUT_SIZE, model->engine, attr);
    
    // Create optimizers
    model->optimizers[0] = create_optimizer(INPUT_SIZE * HIDDEN1_SIZE, HIDDEN1_SIZE);
    model->optimizers[1] = create_optimizer(HIDDEN1_SIZE * HIDDEN2_SIZE, HIDDEN2_SIZE);
    model->optimizers[2] = create_optimizer(HIDDEN2_SIZE * HIDDEN3_SIZE, HIDDEN3_SIZE);
    model->optimizers[3] = create_optimizer(HIDDEN3_SIZE * OUTPUT_SIZE, OUTPUT_SIZE);
    
    // Cleanup
    CHECK(dnnl_primitive_attr_destroy(attr));
    
    return model;
}

void destroy_model(Model* model) {
    destroy_layer(&model->layer1);
    destroy_layer(&model->layer2);
    destroy_layer(&model->layer3);
    destroy_layer(&model->layer4);
    
    for (int i = 0; i < 4; i++) {
        free(model->optimizers[i]->m_weights);
        free(model->optimizers[i]->v_weights);
        free(model->optimizers[i]->m_bias);
        free(model->optimizers[i]->v_bias);
        free(model->optimizers[i]);
    }
    
    dnnl_stream_destroy(model->stream);
    dnnl_engine_destroy(model->engine);
    
    free(model);
}

void forward_pass(Model* model, float* input, float* output, int training) {
    const void* fwd_args[4];
    const void* relu_args[2];
    
    // Layer 1
    CHECK(dnnl_memory_set_data_handle(model->layer1.src_memory, input));
    
    fwd_args[0] = model->layer1.src_memory;
    fwd_args[1] = model->layer1.weights_memory;
    fwd_args[2] = model->layer1.bias_memory;
    fwd_args[3] = model->layer1.dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer1.forward, model->stream, 4, fwd_args));
    
    relu_args[0] = model->layer1.dst_memory;
    relu_args[1] = model->layer1.relu_dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer1.relu_forward, model->stream, 2, relu_args));
    
    // Layer 2
    CHECK(dnnl_memory_set_data_handle(model->layer2.src_memory, model->layer1.relu_output));
    
    fwd_args[0] = model->layer2.src_memory;
    fwd_args[1] = model->layer2.weights_memory;
    fwd_args[2] = model->layer2.bias_memory;
    fwd_args[3] = model->layer2.dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer2.forward, model->stream, 4, fwd_args));
    
    relu_args[0] = model->layer2.dst_memory;
    relu_args[1] = model->layer2.relu_dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer2.relu_forward, model->stream, 2, relu_args));
    
    // Layer 3
    CHECK(dnnl_memory_set_data_handle(model->layer3.src_memory, model->layer2.relu_output));
    
    fwd_args[0] = model->layer3.src_memory;
    fwd_args[1] = model->layer3.weights_memory;
    fwd_args[2] = model->layer3.bias_memory;
    fwd_args[3] = model->layer3.dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer3.forward, model->stream, 4, fwd_args));
    
    relu_args[0] = model->layer3.dst_memory;
    relu_args[1] = model->layer3.relu_dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer3.relu_forward, model->stream, 2, relu_args));
    
    // Layer 4 (output layer)
    CHECK(dnnl_memory_set_data_handle(model->layer4.src_memory, model->layer3.relu_output));
    CHECK(dnnl_memory_set_data_handle(model->layer4.dst_memory, output));
    
    fwd_args[0] = model->layer4.src_memory;
    fwd_args[1] = model->layer4.weights_memory;
    fwd_args[2] = model->layer4.bias_memory;
    fwd_args[3] = model->layer4.dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer4.forward, model->stream, 4, fwd_args));
    
    CHECK(dnnl_stream_wait(model->stream));
}

void backward_pass(Model* model, float* input, float* output, float* target) {
    const void* bwd_args[6];
    const void* relu_bwd_args[3];
    
    // Calculate output gradient (MSE derivative)
    for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; i++) {
        model->layer4.input_grad[i] = 2.0f * (output[i] - target[i]) / BATCH_SIZE;
    }
    
    // Layer 4 backward
    CHECK(dnnl_memory_set_data_handle(model->layer4.diff_dst_memory, model->layer4.input_grad));
    
    bwd_args[0] = model->layer4.src_memory;
    bwd_args[1] = model->layer4.diff_dst_memory;
    bwd_args[2] = model->layer4.weights_memory;
    bwd_args[3] = model->layer4.diff_src_memory;
    bwd_args[4] = model->layer4.diff_weights_memory;
    bwd_args[5] = model->layer4.diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer4.backward, model->stream, 6, bwd_args));
    
    // Layer 3 backward
    CHECK(dnnl_memory_set_data_handle(model->layer3.diff_dst_memory, model->layer4.input_grad));
    
    relu_bwd_args[0] = model->layer3.dst_memory;
    relu_bwd_args[1] = model->layer3.diff_dst_memory;
    relu_bwd_args[2] = model->layer3.relu_diff_dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer3.relu_backward, model->stream, 3, relu_bwd_args));
    
    bwd_args[0] = model->layer3.src_memory;
    bwd_args[1] = model->layer3.relu_diff_dst_memory;
    bwd_args[2] = model->layer3.weights_memory;
    bwd_args[3] = model->layer3.diff_src_memory;
    bwd_args[4] = model->layer3.diff_weights_memory;
    bwd_args[5] = model->layer3.diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer3.backward, model->stream, 6, bwd_args));
    
    // Layer 2 backward
    CHECK(dnnl_memory_set_data_handle(model->layer2.diff_dst_memory, model->layer3.input_grad));
    
    relu_bwd_args[0] = model->layer2.dst_memory;
    relu_bwd_args[1] = model->layer2.diff_dst_memory;
    relu_bwd_args[2] = model->layer2.relu_diff_dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer2.relu_backward, model->stream, 3, relu_bwd_args));
    
    bwd_args[0] = model->layer2.src_memory;
    bwd_args[1] = model->layer2.relu_diff_dst_memory;
    bwd_args[2] = model->layer2.weights_memory;
    bwd_args[3] = model->layer2.diff_src_memory;
    bwd_args[4] = model->layer2.diff_weights_memory;
    bwd_args[5] = model->layer2.diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer2.backward, model->stream, 6, bwd_args));
    
    // Layer 1 backward
    CHECK(dnnl_memory_set_data_handle(model->layer1.diff_dst_memory, model->layer2.input_grad));
    
    relu_bwd_args[0] = model->layer1.dst_memory;
    relu_bwd_args[1] = model->layer1.diff_dst_memory;
    relu_bwd_args[2] = model->layer1.relu_diff_dst_memory;
    
    CHECK(dnnl_primitive_execute(model->layer1.relu_backward, model->stream, 3, relu_bwd_args));
    
    bwd_args[0] = model->layer1.src_memory;
    bwd_args[1] = model->layer1.relu_diff_dst_memory;
    bwd_args[2] = model->layer1.weights_memory;
    bwd_args[3] = model->layer1.diff_src_memory;
    bwd_args[4] = model->layer1.diff_weights_memory;
    bwd_args[5] = model->layer1.diff_bias_memory;
    
    CHECK(dnnl_primitive_execute(model->layer1.backward, model->stream, 6, bwd_args));
    
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
            memcpy(batch_X, &X[batch_start * INPUT_SIZE], 
                   BATCH_SIZE * INPUT_SIZE * sizeof(float));
            memcpy(batch_y, &y[batch_start * OUTPUT_SIZE], 
                   BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            
            // Forward pass
            forward_pass(model, batch_X, output, 1);
            
            // Calculate loss
            float batch_loss = calculate_mse(output, batch_y, 
                                           BATCH_SIZE * OUTPUT_SIZE);
            epoch_loss += batch_loss;
            
            // Backward pass
            backward_pass(model, batch_X, output, batch_y);
            
            // Update weights using Adam
            update_weights_adam(model->optimizers[0], 
                              model->layer1.weights, model->layer1.bias,
                              model->layer1.weights_grad, model->layer1.bias_grad,
                              LEARNING_RATE);
            
            update_weights_adam(model->optimizers[1], 
                              model->layer2.weights, model->layer2.bias,
                              model->layer2.weights_grad, model->layer2.bias_grad,
                              LEARNING_RATE);
            
            update_weights_adam(model->optimizers[2], 
                              model->layer3.weights, model->layer3.bias,
                              model->layer3.weights_grad, model->layer3.bias_grad,
                              LEARNING_RATE);
            
            update_weights_adam(model->optimizers[3], 
                              model->layer4.weights, model->layer4.bias,
                              model->layer4.weights_grad, model->layer4.bias_grad,
                              LEARNING_RATE);
        }
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.4f\n", 
                   epoch + 1, NUM_EPOCHS, 
                   epoch_loss / (num_samples / BATCH_SIZE));
        }
    }
    
    // Calculate and print R² scores
    forward_pass(model, X, output, 0);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float r2 = calculate_r2_score(&y[i], &output[i], num_samples);
        printf("R² score for output y%d: %.4f\n", i, r2);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(batch_X);
    free(batch_y);
    free(output);
    destroy_model(model);
    
    return 0;
}