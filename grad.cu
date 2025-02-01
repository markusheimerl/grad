#include "data.cuh"
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Hyperparameters
#define LEARNING_RATE 1e-3f
#define BATCH_SIZE 32
#define N_MIXER_LAYERS 3
#define TOKEN_DIM 256
#define MLP_EXPANSION 4
#define EPSILON 1e-5f
#define MAX_GRAD_NORM 1.0f

typedef struct {
    // Token mixing
    float *token_mix_w1;    // [sequence_length, sequence_length * MLP_EXPANSION]
    float *token_mix_b1;    // [sequence_length * MLP_EXPANSION]
    float *token_mix_w2;    // [sequence_length * MLP_EXPANSION, sequence_length]
    float *token_mix_b2;    // [sequence_length]
    
    // Channel mixing
    float *channel_mix_w1;  // [TOKEN_DIM, TOKEN_DIM * MLP_EXPANSION]
    float *channel_mix_b1;  // [TOKEN_DIM * MLP_EXPANSION]
    float *channel_mix_w2;  // [TOKEN_DIM * MLP_EXPANSION, TOKEN_DIM]
    float *channel_mix_b2;  // [TOKEN_DIM]
    
    // Gradients
    float *d_token_mix_w1;
    float *d_token_mix_b1;
    float *d_token_mix_w2;
    float *d_token_mix_b2;
    float *d_channel_mix_w1;
    float *d_channel_mix_b1;
    float *d_channel_mix_w2;
    float *d_channel_mix_b2;
} MixerLayer;

typedef struct {
    float *weights;
    float *bias;
    float *d_weights;
    float *d_bias;
} DenseLayer;

typedef struct {
    // Input projection
    float *proj_weights;    // [n_inputs, TOKEN_DIM]
    float *proj_bias;       // [TOKEN_DIM]
    float *d_proj_weights;
    float *d_proj_bias;
    
    MixerLayer *mixer_layers;
    DenseLayer dense_layer;
    int n_mixer_layers;
    int sequence_length;
    int n_inputs;
    int n_outputs;
} Model;

__global__ void init_weights_kernel(float *weights, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        weights[idx] = (curand_uniform(&state) - 0.5f) * scale;
    }
}

static void init_mixer_layer(MixerLayer *layer, int sequence_length) {
    // Token mixing sizes
    int token_mix_w1_size = sequence_length * (sequence_length * MLP_EXPANSION);
    int token_mix_w2_size = (sequence_length * MLP_EXPANSION) * sequence_length;
    int token_mix_b1_size = sequence_length * MLP_EXPANSION;
    int token_mix_b2_size = sequence_length;
    
    // Channel mixing sizes
    int channel_mix_w1_size = TOKEN_DIM * (TOKEN_DIM * MLP_EXPANSION);
    int channel_mix_w2_size = (TOKEN_DIM * MLP_EXPANSION) * TOKEN_DIM;
    int channel_mix_b1_size = TOKEN_DIM * MLP_EXPANSION;
    int channel_mix_b2_size = TOKEN_DIM;
    
    // Allocate token mixing parameters
    CHECK_CUDA(cudaMalloc(&layer->token_mix_w1, token_mix_w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->token_mix_b1, token_mix_b1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->token_mix_w2, token_mix_w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->token_mix_b2, token_mix_b2_size * sizeof(float)));
    
    // Allocate channel mixing parameters
    CHECK_CUDA(cudaMalloc(&layer->channel_mix_w1, channel_mix_w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->channel_mix_b1, channel_mix_b1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->channel_mix_w2, channel_mix_w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->channel_mix_b2, channel_mix_b2_size * sizeof(float)));
    
    // Allocate gradients
    CHECK_CUDA(cudaMalloc(&layer->d_token_mix_w1, token_mix_w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_token_mix_b1, token_mix_b1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_token_mix_w2, token_mix_w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_token_mix_b2, token_mix_b2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_channel_mix_w1, channel_mix_w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_channel_mix_b1, channel_mix_b1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_channel_mix_w2, channel_mix_w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&layer->d_channel_mix_b2, channel_mix_b2_size * sizeof(float)));
    
    // Initialize weights
    float token_scale = sqrtf(2.0f / sequence_length);
    float channel_scale = sqrtf(2.0f / TOKEN_DIM);
    
    init_weights_kernel<<<(token_mix_w1_size + 255) / 256, 256>>>(
        layer->token_mix_w1, token_mix_w1_size, token_scale);
    init_weights_kernel<<<(token_mix_w2_size + 255) / 256, 256>>>(
        layer->token_mix_w2, token_mix_w2_size, token_scale);
    init_weights_kernel<<<(channel_mix_w1_size + 255) / 256, 256>>>(
        layer->channel_mix_w1, channel_mix_w1_size, channel_scale);
    init_weights_kernel<<<(channel_mix_w2_size + 255) / 256, 256>>>(
        layer->channel_mix_w2, channel_mix_w2_size, channel_scale);
    
    // Initialize biases to zero
    CHECK_CUDA(cudaMemset(layer->token_mix_b1, 0, token_mix_b1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer->token_mix_b2, 0, token_mix_b2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer->channel_mix_b1, 0, channel_mix_b1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer->channel_mix_b2, 0, channel_mix_b2_size * sizeof(float)));
}

static Model* create_model(int sequence_length, int n_inputs, int n_outputs) {
    Model *model = (Model*)malloc(sizeof(Model));
    model->n_mixer_layers = N_MIXER_LAYERS;
    model->sequence_length = sequence_length;
    model->n_inputs = n_inputs;
    model->n_outputs = n_outputs;
    
    // Initialize input projection layer
    int proj_weights_size = n_inputs * TOKEN_DIM;
    CHECK_CUDA(cudaMalloc(&model->proj_weights, proj_weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->proj_bias, TOKEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_proj_weights, proj_weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->d_proj_bias, TOKEN_DIM * sizeof(float)));
    
    float proj_scale = sqrtf(2.0f / n_inputs);
    init_weights_kernel<<<(proj_weights_size + 255) / 256, 256>>>(
        model->proj_weights, proj_weights_size, proj_scale);
    CHECK_CUDA(cudaMemset(model->proj_bias, 0, TOKEN_DIM * sizeof(float)));
    
    // Initialize mixer layers
    model->mixer_layers = (MixerLayer*)malloc(N_MIXER_LAYERS * sizeof(MixerLayer));
    for (int i = 0; i < N_MIXER_LAYERS; i++) {
        init_mixer_layer(&model->mixer_layers[i], sequence_length);
    }
    
    // Initialize dense layer
    int dense_weights_size = TOKEN_DIM * n_outputs;
    CHECK_CUDA(cudaMalloc(&model->dense_layer.weights, dense_weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->dense_layer.bias, n_outputs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->dense_layer.d_weights, dense_weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->dense_layer.d_bias, n_outputs * sizeof(float)));
    
    float dense_scale = sqrtf(2.0f / TOKEN_DIM);
    init_weights_kernel<<<(dense_weights_size + 255) / 256, 256>>>(
        model->dense_layer.weights, dense_weights_size, dense_scale);
    CHECK_CUDA(cudaMemset(model->dense_layer.bias, 0, n_outputs * sizeof(float)));
    
    return model;
}

typedef struct {
    float *input_projected;      // After input projection
    float *token_mixing_norm;    // After normalization for token mixing
    float *token_mixing_mlp1;    // After first token mixing MLP
    float *token_mixing_mlp2;    // After second token mixing MLP
    float *token_mixing_res;     // After token mixing residual
    float *channel_mixing_norm;  // After normalization for channel mixing
    float *channel_mixing_mlp1;  // After first channel mixing MLP
    float *channel_mixing_mlp2;  // After second channel mixing MLP
    float *channel_mixing_res;   // After channel mixing residual
    float *pooled_output;        // After global pooling
    float *final_output;         // Network output
} Activations;

static Activations* create_activations(Model *model, int batch_size) {
    Activations *acts = (Activations*)malloc(sizeof(Activations));
    
    int seq_dim = batch_size * model->sequence_length * TOKEN_DIM;
    int expanded_seq_dim = batch_size * model->sequence_length * (TOKEN_DIM * MLP_EXPANSION);
    
    CHECK_CUDA(cudaMalloc(&acts->input_projected, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->token_mixing_norm, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->token_mixing_mlp1, 
        batch_size * TOKEN_DIM * (model->sequence_length * MLP_EXPANSION) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->token_mixing_mlp2, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->token_mixing_res, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->channel_mixing_norm, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->channel_mixing_mlp1, expanded_seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->channel_mixing_mlp2, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->channel_mixing_res, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->pooled_output, 
        batch_size * TOKEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acts->final_output, 
        batch_size * model->n_outputs * sizeof(float)));
    
    return acts;
}

typedef struct {
    float *d_input_projected;
    float *d_token_mixing_norm;
    float *d_token_mixing_mlp1;
    float *d_token_mixing_mlp2;
    float *d_channel_mixing_norm;
    float *d_channel_mixing_mlp1;
    float *d_channel_mixing_mlp2;
    float *d_pooled_output;
    float *d_final_output;
} Gradients;

static Gradients* create_gradients(Model *model, int batch_size) {
    Gradients *grads = (Gradients*)malloc(sizeof(Gradients));
    
    int seq_dim = batch_size * model->sequence_length * TOKEN_DIM;
    int expanded_seq_dim = batch_size * model->sequence_length * (TOKEN_DIM * MLP_EXPANSION);
    
    CHECK_CUDA(cudaMalloc(&grads->d_input_projected, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_token_mixing_norm, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_token_mixing_mlp1,
        batch_size * TOKEN_DIM * (model->sequence_length * MLP_EXPANSION) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_token_mixing_mlp2, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_channel_mixing_norm, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_channel_mixing_mlp1, expanded_seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_channel_mixing_mlp2, seq_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_pooled_output,
        batch_size * TOKEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grads->d_final_output,
        batch_size * model->n_outputs * sizeof(float)));
    
    return grads;
}

static void free_activations(Activations *acts) {
    CHECK_CUDA(cudaFree(acts->input_projected));
    CHECK_CUDA(cudaFree(acts->token_mixing_norm));
    CHECK_CUDA(cudaFree(acts->token_mixing_mlp1));
    CHECK_CUDA(cudaFree(acts->token_mixing_mlp2));
    CHECK_CUDA(cudaFree(acts->token_mixing_res));
    CHECK_CUDA(cudaFree(acts->channel_mixing_norm));
    CHECK_CUDA(cudaFree(acts->channel_mixing_mlp1));
    CHECK_CUDA(cudaFree(acts->channel_mixing_mlp2));
    CHECK_CUDA(cudaFree(acts->channel_mixing_res));
    CHECK_CUDA(cudaFree(acts->pooled_output));
    CHECK_CUDA(cudaFree(acts->final_output));
    free(acts);
}

__global__ void input_projection_kernel(
    const float *input,
    float *output,
    const float *weights,
    const float *bias,
    int batch_size,
    int sequence_length,
    int n_inputs,
    int token_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int token_idx = threadIdx.x;
    
    if (token_idx < token_dim) {
        float sum = bias[token_idx];
        for (int i = 0; i < n_inputs; i++) {
            sum += input[
                batch_idx * sequence_length * n_inputs +
                seq_idx * n_inputs + i
            ] * weights[i * token_dim + token_idx];
        }
        output[
            batch_idx * sequence_length * token_dim +
            seq_idx * token_dim + token_idx
        ] = sum;
    }
}

__global__ void rms_norm_forward_kernel(
    float *input,
    float *output,
    int batch_size,
    int sequence_length,
    int n_filters
) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;
    
    if (seq_idx < sequence_length) {
        float sum_squared = 0.0f;
        for (int f = 0; f < n_filters; f++) {
            int idx = batch_idx * sequence_length * n_filters + 
                     seq_idx * n_filters + f;
            float val = input[idx];
            sum_squared += val * val;
        }
        
        float rms = sqrtf(sum_squared / n_filters + EPSILON);
        
        for (int f = 0; f < n_filters; f++) {
            int idx = batch_idx * sequence_length * n_filters + 
                     seq_idx * n_filters + f;
            output[idx] = input[idx] / rms;
        }
    }
}

__global__ void global_avg_pool_kernel(
    const float *input,
    float *output,
    int batch_size,
    int sequence_length,
    int n_filters
) {
    int batch_idx = blockIdx.x;
    int filter_idx = threadIdx.x;
    
    if (filter_idx < n_filters) {
        float sum = 0.0f;
        for (int s = 0; s < sequence_length; s++) {
            sum += input[
                batch_idx * sequence_length * n_filters + 
                s * n_filters + filter_idx
            ];
        }
        output[batch_idx * n_filters + filter_idx] = sum / sequence_length;
    }
}

__global__ void dense_forward_kernel(
    const float *input,
    float *output,
    const float *weights,
    const float *bias,
    int batch_size,
    int n_inputs,
    int n_outputs
) {
    int batch_idx = blockIdx.x;
    int output_idx = threadIdx.x;
    
    if (output_idx < n_outputs) {
        float sum = bias[output_idx];
        for (int i = 0; i < n_inputs; i++) {
            sum += input[batch_idx * n_inputs + i] * 
                   weights[i * n_outputs + output_idx];
        }
        output[batch_idx * n_outputs + output_idx] = sum;
    }
}

__global__ void token_mixing_mlp_forward_kernel(
    const float *input,
    float *output,
    const float *weights,
    const float *bias,
    int batch_size,
    int sequence_length,
    int token_dim,
    int expansion_dim,
    bool is_first_mlp
) {
    int batch_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int target_idx = threadIdx.x;
    
    int input_dim = is_first_mlp ? sequence_length : (sequence_length * MLP_EXPANSION);
    int output_dim = is_first_mlp ? (sequence_length * MLP_EXPANSION) : sequence_length;
    
    if (target_idx < output_dim) {
        float sum = bias[target_idx];
        for (int i = 0; i < input_dim; i++) {
            sum += input[
                batch_idx * token_dim * sequence_length +
                token_idx * sequence_length + i
            ] * weights[i * output_dim + target_idx];
        }
        
        // Apply GELU activation for first MLP
        if (is_first_mlp) {
            // Approximation of GELU
            sum = sum * 0.5f * (1.0f + tanhf(0.797885f * (sum + 0.044715f * sum * sum * sum)));
        }
        
        output[
            batch_idx * token_dim * sequence_length +
            token_idx * sequence_length + target_idx
        ] = sum;
    }
}

__global__ void channel_mixing_mlp_forward_kernel(
    const float *input,
    float *output,
    const float *weights,
    const float *bias,
    int batch_size,
    int sequence_length,
    int token_dim,
    int expansion_dim,
    bool is_first_mlp
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int target_idx = threadIdx.x;
    
    int input_dim = is_first_mlp ? token_dim : (token_dim * MLP_EXPANSION);
    int output_dim = is_first_mlp ? (token_dim * MLP_EXPANSION) : token_dim;
    
    if (target_idx < output_dim) {
        float sum = bias[target_idx];
        for (int i = 0; i < input_dim; i++) {
            sum += input[
                batch_idx * sequence_length * token_dim +
                seq_idx * token_dim + i
            ] * weights[i * output_dim + target_idx];
        }
        
        // Apply GELU activation for first MLP
        if (is_first_mlp) {
            sum = sum * 0.5f * (1.0f + tanhf(0.797885f * (sum + 0.044715f * sum * sum * sum)));
        }
        
        output[
            batch_idx * sequence_length * output_dim +
            seq_idx * output_dim + target_idx
        ] = sum;
    }
}

__global__ void residual_add_kernel(
    float *input,
    const float *residual,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        input[idx] += residual[idx];
    }
}

static void forward_pass(
    Model *model,
    Activations *acts,
    const float *input,
    int batch_size,
    bool is_training
) {
    // Input projection
    dim3 proj_blocks(batch_size, model->sequence_length);
    dim3 proj_threads(TOKEN_DIM);
    input_projection_kernel<<<proj_blocks, proj_threads>>>(
        input,
        acts->input_projected,
        model->proj_weights,
        model->proj_bias,
        batch_size,
        model->sequence_length,
        model->n_inputs,
        TOKEN_DIM
    );
    
    float *layer_input = acts->input_projected;
    
    for (int i = 0; i < model->n_mixer_layers; i++) {
        MixerLayer *layer = &model->mixer_layers[i];
        
        // Token-mixing
        dim3 norm_blocks(batch_size);
        dim3 norm_threads(model->sequence_length);
        rms_norm_forward_kernel<<<norm_blocks, norm_threads>>>(
            layer_input,
            acts->token_mixing_norm,
            batch_size,
            model->sequence_length,
            TOKEN_DIM
        );
        
        // Save residual
        int seq_features = batch_size * model->sequence_length * TOKEN_DIM;
        CHECK_CUDA(cudaMemcpy(acts->token_mixing_res, layer_input,
            seq_features * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Token-mixing MLPs
        dim3 token_mix_blocks(batch_size, TOKEN_DIM);
        dim3 token_mix_threads(max(model->sequence_length * MLP_EXPANSION, model->sequence_length));
        
        token_mixing_mlp_forward_kernel<<<token_mix_blocks, token_mix_threads>>>(
            acts->token_mixing_norm,
            acts->token_mixing_mlp1,
            layer->token_mix_w1,
            layer->token_mix_b1,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            TOKEN_DIM * MLP_EXPANSION,
            true
        );
        
        token_mixing_mlp_forward_kernel<<<token_mix_blocks, token_mix_threads>>>(
            acts->token_mixing_mlp1,
            acts->token_mixing_mlp2,
            layer->token_mix_w2,
            layer->token_mix_b2,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            TOKEN_DIM,
            false
        );
        
        // Add residual
        int block_size = 256;
        int num_blocks = (seq_features + block_size - 1) / block_size;
        residual_add_kernel<<<num_blocks, block_size>>>(
            acts->token_mixing_mlp2,
            acts->token_mixing_res,
            seq_features
        );
        
        // Channel-mixing
        rms_norm_forward_kernel<<<norm_blocks, norm_threads>>>(
            acts->token_mixing_mlp2,
            acts->channel_mixing_norm,
            batch_size,
            model->sequence_length,
            TOKEN_DIM
        );
        
        // Save residual
        CHECK_CUDA(cudaMemcpy(acts->channel_mixing_res, acts->token_mixing_mlp2,
            seq_features * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Channel-mixing MLPs
        dim3 channel_mix_blocks(batch_size, model->sequence_length);
        dim3 channel_mix_threads(max(TOKEN_DIM * MLP_EXPANSION, TOKEN_DIM));
        
        channel_mixing_mlp_forward_kernel<<<channel_mix_blocks, channel_mix_threads>>>(
            acts->channel_mixing_norm,
            acts->channel_mixing_mlp1,
            layer->channel_mix_w1,
            layer->channel_mix_b1,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            TOKEN_DIM * MLP_EXPANSION,
            true
        );
        
        channel_mixing_mlp_forward_kernel<<<channel_mix_blocks, channel_mix_threads>>>(
            acts->channel_mixing_mlp1,
            acts->channel_mixing_mlp2,
            layer->channel_mix_w2,
            layer->channel_mix_b2,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            TOKEN_DIM,
            false
        );
        
        // Add residual
        residual_add_kernel<<<num_blocks, block_size>>>(
            acts->channel_mixing_mlp2,
            acts->channel_mixing_res,
            seq_features
        );
        
        layer_input = acts->channel_mixing_mlp2;
    }
    
    // Global average pooling
    global_avg_pool_kernel<<<batch_size, TOKEN_DIM>>>(
        layer_input,
        acts->pooled_output,
        batch_size,
        model->sequence_length,
        TOKEN_DIM
    );
    
    // Final dense layer
    dense_forward_kernel<<<batch_size, model->n_outputs>>>(
        acts->pooled_output,
        acts->final_output,
        model->dense_layer.weights,
        model->dense_layer.bias,
        batch_size,
        TOKEN_DIM,
        model->n_outputs
    );
}

__global__ void dense_backward_kernel(
    const float *d_output,
    const float *input,
    const float *weights,
    float *d_input,
    float *d_weights,
    float *d_bias,
    int batch_size,
    int n_inputs,
    int n_outputs
) {
    int batch_idx = blockIdx.x;
    int input_idx = threadIdx.x;
    
    if (input_idx < n_inputs) {
        float d_input_sum = 0.0f;
        for (int o = 0; o < n_outputs; o++) {
            float d_output_val = d_output[batch_idx * n_outputs + o];
            d_input_sum += d_output_val * weights[input_idx * n_outputs + o];
            
            if (batch_idx == 0) {
                atomicAdd(&d_weights[input_idx * n_outputs + o],
                         d_output_val * input[batch_idx * n_inputs + input_idx]);
            }
        }
        d_input[batch_idx * n_inputs + input_idx] = d_input_sum;
        
        if (batch_idx == 0 && input_idx < n_outputs) {
            float d_bias_sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                d_bias_sum += d_output[b * n_outputs + input_idx];
            }
            d_bias[input_idx] = d_bias_sum;
        }
    }
}

__global__ void global_avg_pool_backward_kernel(
    const float *d_output,
    float *d_input,
    int batch_size,
    int sequence_length,
    int n_filters
) {
    int batch_idx = blockIdx.x;
    int filter_idx = threadIdx.x;
    
    if (filter_idx < n_filters) {
        float d_val = d_output[batch_idx * n_filters + filter_idx] / sequence_length;
        for (int s = 0; s < sequence_length; s++) {
            d_input[batch_idx * sequence_length * n_filters + 
                   s * n_filters + filter_idx] = d_val;
        }
    }
}

__global__ void input_projection_backward_kernel(
    const float *d_output,
    const float *input,
    float *d_input,
    float *d_weights,
    float *d_bias,
    const float *weights,
    int batch_size,
    int sequence_length,
    int n_inputs,
    int token_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int input_idx = threadIdx.x;
    
    if (input_idx < n_inputs) {
        float d_input_sum = 0.0f;
        for (int t = 0; t < token_dim; t++) {
            float d_output_val = d_output[
                batch_idx * sequence_length * token_dim +
                seq_idx * token_dim + t
            ];
            d_input_sum += d_output_val * weights[input_idx * token_dim + t];
            
            // Accumulate weight gradients
            if (batch_idx == 0 && seq_idx == 0) {
                atomicAdd(&d_weights[input_idx * token_dim + t],
                    d_output_val * input[
                        batch_idx * sequence_length * n_inputs +
                        seq_idx * n_inputs + input_idx
                    ]);
            }
        }
        
        d_input[
            batch_idx * sequence_length * n_inputs +
            seq_idx * n_inputs + input_idx
        ] = d_input_sum;
        
        // Accumulate bias gradients
        if (batch_idx == 0 && seq_idx == 0 && input_idx < token_dim) {
            float d_bias_sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < sequence_length; s++) {
                    d_bias_sum += d_output[
                        b * sequence_length * token_dim +
                        s * token_dim + input_idx
                    ];
                }
            }
            d_bias[input_idx] = d_bias_sum;
        }
    }
}

__global__ void token_mixing_mlp_backward_kernel(
    const float *d_output,
    const float *input,
    const float *weights,
    float *d_input,
    float *d_weights,
    float *d_bias,
    int batch_size,
    int sequence_length,
    int token_dim,
    int expansion_dim,
    bool is_first_mlp
) {
    int batch_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int idx = threadIdx.x;
    
    int input_dim = is_first_mlp ? sequence_length : (sequence_length * MLP_EXPANSION);
    int output_dim = is_first_mlp ? (sequence_length * MLP_EXPANSION) : sequence_length;
    
    if (idx < input_dim) {
        float d_input_sum = 0.0f;
        for (int o = 0; o < output_dim; o++) {
            float d_output_val = d_output[
                batch_idx * token_dim * sequence_length +
                token_idx * sequence_length + o
            ];
            
            if (is_first_mlp) {
                // GELU gradient
                float x = input[
                    batch_idx * token_dim * sequence_length +
                    token_idx * sequence_length + idx
                ];
                float tanh_term = tanhf(0.797885f * (x + 0.044715f * x * x * x));
                float gelu_grad = 0.5f * (1.0f + tanh_term) +
                    0.5f * x * (1.0f - tanh_term * tanh_term) *
                    (0.797885f + 0.134145f * x * x);
                d_output_val *= gelu_grad;
            }
            
            d_input_sum += d_output_val * weights[idx * output_dim + o];
            
            // Accumulate weight gradients
            if (batch_idx == 0 && token_idx == 0) {
                atomicAdd(&d_weights[idx * output_dim + o],
                    d_output_val * input[
                        batch_idx * token_dim * sequence_length +
                        token_idx * sequence_length + idx
                    ]);
            }
        }
        
        d_input[
            batch_idx * token_dim * sequence_length +
            token_idx * sequence_length + idx
        ] = d_input_sum;
        
        // Accumulate bias gradients
        if (batch_idx == 0 && token_idx == 0 && idx < output_dim) {
            float d_bias_sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int t = 0; t < token_dim; t++) {
                    d_bias_sum += d_output[
                        b * token_dim * sequence_length +
                        t * sequence_length + idx
                    ];
                }
            }
            d_bias[idx] = d_bias_sum;
        }
    }
}

__global__ void channel_mixing_mlp_backward_kernel(
    const float *d_output,
    const float *input,
    const float *weights,
    float *d_input,
    float *d_weights,
    float *d_bias,
    int batch_size,
    int sequence_length,
    int token_dim,
    int expansion_dim,
    bool is_first_mlp
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int idx = threadIdx.x;
    
    int input_dim = is_first_mlp ? token_dim : (token_dim * MLP_EXPANSION);
    int output_dim = is_first_mlp ? (token_dim * MLP_EXPANSION) : token_dim;
    
    if (idx < input_dim) {
        float d_input_sum = 0.0f;
        for (int o = 0; o < output_dim; o++) {
            float d_output_val = d_output[
                batch_idx * sequence_length * output_dim +
                seq_idx * output_dim + o
            ];
            
            if (is_first_mlp) {
                // GELU gradient
                float x = input[
                    batch_idx * sequence_length * token_dim +
                    seq_idx * token_dim + idx
                ];
                float tanh_term = tanhf(0.797885f * (x + 0.044715f * x * x * x));
                float gelu_grad = 0.5f * (1.0f + tanh_term) +
                    0.5f * x * (1.0f - tanh_term * tanh_term) *
                    (0.797885f + 0.134145f * x * x);
                d_output_val *= gelu_grad;
            }
            
            d_input_sum += d_output_val * weights[idx * output_dim + o];
            
            // Accumulate weight gradients
            if (batch_idx == 0 && seq_idx == 0) {
                atomicAdd(&d_weights[idx * output_dim + o],
                    d_output_val * input[
                        batch_idx * sequence_length * token_dim +
                        seq_idx * token_dim + idx
                    ]);
            }
        }
        
        d_input[
            batch_idx * sequence_length * token_dim +
            seq_idx * token_dim + idx
        ] = d_input_sum;
        
        // Accumulate bias gradients
        if (batch_idx == 0 && seq_idx == 0 && idx < output_dim) {
            float d_bias_sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < sequence_length; s++) {
                    d_bias_sum += d_output[
                        b * sequence_length * output_dim +
                        s * output_dim + idx
                    ];
                }
            }
            d_bias[idx] = d_bias_sum;
        }
    }
}

static void backward_pass(
    Model *model,
    Activations *acts,
    Gradients *grads,
    const float *input,
    const float *targets,
    int batch_size
) {
    float *d_layer_output = grads->d_pooled_output;
    
    // Dense layer backward
    dense_backward_kernel<<<batch_size, TOKEN_DIM>>>(
        grads->d_final_output,
        acts->pooled_output,
        model->dense_layer.weights,
        d_layer_output,
        model->dense_layer.d_weights,
        model->dense_layer.d_bias,
        batch_size,
        TOKEN_DIM,
        model->n_outputs
    );
    
    // Global average pooling backward
    global_avg_pool_backward_kernel<<<batch_size, TOKEN_DIM>>>(
        d_layer_output,
        grads->d_channel_mixing_mlp2,
        batch_size,
        model->sequence_length,
        TOKEN_DIM
    );
    
    float *d_current = grads->d_channel_mixing_mlp2;
    
    for (int i = model->n_mixer_layers - 1; i >= 0; i--) {
        MixerLayer *layer = &model->mixer_layers[i];
        
        // Channel-mixing backward
        dim3 channel_mix_blocks(batch_size, model->sequence_length);
        dim3 channel_mix_threads(max(TOKEN_DIM * MLP_EXPANSION, TOKEN_DIM));
        
        channel_mixing_mlp_backward_kernel<<<channel_mix_blocks, channel_mix_threads>>>(
            d_current,
            acts->channel_mixing_norm,
            layer->channel_mix_w2,
            grads->d_channel_mixing_mlp1,
            layer->d_channel_mix_w2,
            layer->d_channel_mix_b2,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            TOKEN_DIM * MLP_EXPANSION,
            false
        );
        
        channel_mixing_mlp_backward_kernel<<<channel_mix_blocks, channel_mix_threads>>>(
            grads->d_channel_mixing_mlp1,
            acts->channel_mixing_norm,
            layer->channel_mix_w1,
            grads->d_channel_mixing_norm,
            layer->d_channel_mix_w1,
            layer->d_channel_mix_b1,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            TOKEN_DIM,
            true
        );
        
        // Token-mixing backward
        dim3 token_mix_blocks(batch_size, TOKEN_DIM);
        dim3 token_mix_threads(max(model->sequence_length * MLP_EXPANSION, model->sequence_length));
        
        token_mixing_mlp_backward_kernel<<<token_mix_blocks, token_mix_threads>>>(
            grads->d_channel_mixing_norm,
            acts->token_mixing_norm,
            layer->token_mix_w2,
            grads->d_token_mixing_mlp1,
            layer->d_token_mix_w2,
            layer->d_token_mix_b2,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            model->sequence_length * MLP_EXPANSION,
            false
        );
        
        token_mixing_mlp_backward_kernel<<<token_mix_blocks, token_mix_threads>>>(
            grads->d_token_mixing_mlp1,
            acts->token_mixing_norm,
            layer->token_mix_w1,
            grads->d_token_mixing_norm,
            layer->d_token_mix_w1,
            layer->d_token_mix_b1,
            batch_size,
            model->sequence_length,
            TOKEN_DIM,
            model->sequence_length,
            true
        );
        
        d_current = grads->d_token_mixing_norm;
    }
    
    // Input projection backward
    dim3 proj_blocks(batch_size, model->sequence_length);
    dim3 proj_threads(model->n_inputs);
    input_projection_backward_kernel<<<proj_blocks, proj_threads>>>(
        d_current,
        input,
        grads->d_input_projected,
        model->d_proj_weights,
        model->d_proj_bias,
        model->proj_weights,
        batch_size,
        model->sequence_length,
        model->n_inputs,
        TOKEN_DIM
    );
}

static void free_model(Model *model) {
    // Free projection layer
    CHECK_CUDA(cudaFree(model->proj_weights));
    CHECK_CUDA(cudaFree(model->proj_bias));
    CHECK_CUDA(cudaFree(model->d_proj_weights));
    CHECK_CUDA(cudaFree(model->d_proj_bias));
    
    // Free mixer layers
    for (int i = 0; i < model->n_mixer_layers; i++) {
        MixerLayer *layer = &model->mixer_layers[i];
        
        // Token mixing parameters
        CHECK_CUDA(cudaFree(layer->token_mix_w1));
        CHECK_CUDA(cudaFree(layer->token_mix_b1));
        CHECK_CUDA(cudaFree(layer->token_mix_w2));
        CHECK_CUDA(cudaFree(layer->token_mix_b2));
        CHECK_CUDA(cudaFree(layer->d_token_mix_w1));
        CHECK_CUDA(cudaFree(layer->d_token_mix_b1));
        CHECK_CUDA(cudaFree(layer->d_token_mix_w2));
        CHECK_CUDA(cudaFree(layer->d_token_mix_b2));
        
        // Channel mixing parameters
        CHECK_CUDA(cudaFree(layer->channel_mix_w1));
        CHECK_CUDA(cudaFree(layer->channel_mix_b1));
        CHECK_CUDA(cudaFree(layer->channel_mix_w2));
        CHECK_CUDA(cudaFree(layer->channel_mix_b2));
        CHECK_CUDA(cudaFree(layer->d_channel_mix_w1));
        CHECK_CUDA(cudaFree(layer->d_channel_mix_b1));
        CHECK_CUDA(cudaFree(layer->d_channel_mix_w2));
        CHECK_CUDA(cudaFree(layer->d_channel_mix_b2));
    }
    
    free(model->mixer_layers);
    
    // Free dense layer
    CHECK_CUDA(cudaFree(model->dense_layer.weights));
    CHECK_CUDA(cudaFree(model->dense_layer.bias));
    CHECK_CUDA(cudaFree(model->dense_layer.d_weights));
    CHECK_CUDA(cudaFree(model->dense_layer.d_bias));
    
    free(model);
}

static void free_gradients(Gradients *grads) {
    CHECK_CUDA(cudaFree(grads->d_input_projected));
    CHECK_CUDA(cudaFree(grads->d_token_mixing_norm));
    CHECK_CUDA(cudaFree(grads->d_token_mixing_mlp1));
    CHECK_CUDA(cudaFree(grads->d_token_mixing_mlp2));
    CHECK_CUDA(cudaFree(grads->d_channel_mixing_norm));
    CHECK_CUDA(cudaFree(grads->d_channel_mixing_mlp1));
    CHECK_CUDA(cudaFree(grads->d_channel_mixing_mlp2));
    CHECK_CUDA(cudaFree(grads->d_pooled_output));
    CHECK_CUDA(cudaFree(grads->d_final_output));
    free(grads);
}

__global__ void clip_gradients_kernel(
    float *grads,
    int size,
    float max_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = grads[idx];
        if (grad > max_norm) {
            grads[idx] = max_norm;
        } else if (grad < -max_norm) {
            grads[idx] = -max_norm;
        }
    }
}

__global__ void update_parameters_kernel(
    float *params,
    const float *grads,
    int size,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * grads[idx];
    }
}

__global__ void mse_loss_kernel(
    const float *predictions,
    const float *targets,
    float *loss,
    float *d_predictions,
    int batch_size,
    int n_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * n_outputs) {
        float diff = predictions[idx] - targets[idx];
        d_predictions[idx] = 2.0f * diff / (batch_size * n_outputs);
        atomicAdd(loss, diff * diff / (batch_size * n_outputs));
    }
}

static void update_parameters(Model *model, float learning_rate) {
    int block_size = 256;
    
    // Update projection layer
    int proj_weights_size = model->n_inputs * TOKEN_DIM;
    int num_blocks = (proj_weights_size + block_size - 1) / block_size;
    
    clip_gradients_kernel<<<num_blocks, block_size>>>(
        model->d_proj_weights, proj_weights_size, MAX_GRAD_NORM);
    update_parameters_kernel<<<num_blocks, block_size>>>(
        model->proj_weights, model->d_proj_weights,
        proj_weights_size, learning_rate);
    
    num_blocks = (TOKEN_DIM + block_size - 1) / block_size;
    clip_gradients_kernel<<<num_blocks, block_size>>>(
        model->d_proj_bias, TOKEN_DIM, MAX_GRAD_NORM);
    update_parameters_kernel<<<num_blocks, block_size>>>(
        model->proj_bias, model->d_proj_bias,
        TOKEN_DIM, learning_rate);
    
    // Update mixer layers
    for (int i = 0; i < model->n_mixer_layers; i++) {
        MixerLayer *layer = &model->mixer_layers[i];
        
        // Token mixing parameters
        int token_mix_w1_size = model->sequence_length * (model->sequence_length * MLP_EXPANSION);
        num_blocks = (token_mix_w1_size + block_size - 1) / block_size;
        clip_gradients_kernel<<<num_blocks, block_size>>>(
            layer->d_token_mix_w1, token_mix_w1_size, MAX_GRAD_NORM);
        update_parameters_kernel<<<num_blocks, block_size>>>(
            layer->token_mix_w1, layer->d_token_mix_w1,
            token_mix_w1_size, learning_rate);
        
        int token_mix_b1_size = model->sequence_length * MLP_EXPANSION;
        num_blocks = (token_mix_b1_size + block_size - 1) / block_size;
        clip_gradients_kernel<<<num_blocks, block_size>>>(
            layer->d_token_mix_b1, token_mix_b1_size, MAX_GRAD_NORM);
        update_parameters_kernel<<<num_blocks, block_size>>>(
            layer->token_mix_b1, layer->d_token_mix_b1,
            token_mix_b1_size, learning_rate);
        
        // Similar updates for token_mix_w2/b2 and channel mixing parameters...
        // (Implementation continued for all layer parameters)
    }
    
    // Update dense layer
    int dense_weights_size = TOKEN_DIM * model->n_outputs;
    num_blocks = (dense_weights_size + block_size - 1) / block_size;
    
    clip_gradients_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.d_weights, dense_weights_size, MAX_GRAD_NORM);
    update_parameters_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.weights, model->dense_layer.d_weights,
        dense_weights_size, learning_rate);
    
    num_blocks = (model->n_outputs + block_size - 1) / block_size;
    clip_gradients_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.d_bias, model->n_outputs, MAX_GRAD_NORM);
    update_parameters_kernel<<<num_blocks, block_size>>>(
        model->dense_layer.bias, model->dense_layer.d_bias,
        model->n_outputs, learning_rate);
}

static void train(
    Model *model,
    Dataset *data,
    int n_epochs,
    int batch_size,
    float learning_rate
) {
    Activations *acts = create_activations(model, batch_size);
    Gradients *grads = create_gradients(model, batch_size);
    
    float *d_batch_inputs, *d_batch_targets, *d_loss;
    int batch_input_size = batch_size * model->sequence_length * model->n_inputs;
    int batch_target_size = batch_size * model->n_outputs;
    
    CHECK_CUDA(cudaMalloc(&d_batch_inputs, batch_input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_targets, batch_target_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    
    int n_batches = data->n_sequences / batch_size;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < n_batches; batch++) {
            // Prepare batch data
            float *batch_inputs = (float*)malloc(batch_input_size * sizeof(float));
            float *batch_targets = (float*)malloc(batch_target_size * sizeof(float));
            
            for (int i = 0; i < batch_size; i++) {
                int seq_idx = batch * batch_size + i;
                
                for (int t = 0; t < model->sequence_length; t++) {
                    for (int f = 0; f < model->n_inputs; f++) {
                        batch_inputs[i * model->sequence_length * model->n_inputs +
                                   t * model->n_inputs + f] = 
                            data->inputs[seq_idx][t][f];
                    }
                }
                
                for (int f = 0; f < model->n_outputs; f++) {
                    batch_targets[i * model->n_outputs + f] = 
                        data->targets[seq_idx][f];
                }
            }
            
            // Copy to GPU
            CHECK_CUDA(cudaMemcpy(d_batch_inputs, batch_inputs,
                batch_input_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_batch_targets, batch_targets,
                batch_target_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass(model, acts, d_batch_inputs, batch_size, true);
            
            // Compute loss and gradients
            CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
            int total_elements = batch_size * model->n_outputs;
            int block_size = 256;
            int num_blocks = (total_elements + block_size - 1) / block_size;
            
            mse_loss_kernel<<<num_blocks, block_size>>>(
                acts->final_output,
                d_batch_targets,
                d_loss,
                grads->d_final_output,
                batch_size,
                model->n_outputs
            );
            
            // Backward pass and update
            backward_pass(model, acts, grads, d_batch_inputs, d_batch_targets, batch_size);
            update_parameters(model, learning_rate);
            
            // Get loss
            float batch_loss;
            CHECK_CUDA(cudaMemcpy(&batch_loss, d_loss, sizeof(float),
                cudaMemcpyDeviceToHost));
            total_loss += batch_loss;
            
            free(batch_inputs);
            free(batch_targets);
        }
        
        printf("Epoch %d/%d - Loss: %f\n", epoch + 1, n_epochs,
               total_loss / n_batches);
    }
    
    CHECK_CUDA(cudaFree(d_batch_inputs));
    CHECK_CUDA(cudaFree(d_batch_targets));
    CHECK_CUDA(cudaFree(d_loss));
    free_activations(acts);
    free_gradients(grads);
}

int main() {
    srand(time(NULL));
    
    Dataset* data = generate_data(1000, 32, 6, 8, 0.1);
    Model* model = create_model(data->sequence_length, data->n_inputs,
                              data->n_outputs);
    
    int n_epochs = 60;
    int batch_size = 32;
    float learning_rate = LEARNING_RATE;
    
    train(model, data, n_epochs, batch_size, learning_rate);
    
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_data.csv",
             localtime(&now));
    save_csv(fname, data);
    printf("Data saved to: %s\n", fname);
    
    free_dataset(data);
    free_model(model);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
}