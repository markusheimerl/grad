#include "data_open_loop.cuh"
#include <time.h>

#define LR 0.001
#define EPOCHS 1000
#define ATTN_DIM 32
#define B_SIZE 256
#define BATCH_SIZE 32

typedef struct {
    double *d_q_w, *d_k_w, *d_v_w;  // Query, Key, Value weights
    double *d_q_b, *d_k_b, *d_v_b;  // Query, Key, Value biases
    double *d_fc_w, *d_fc_b;        // FC weights and bias
    int attn_dim;                   // Attention dimension
} Model;

__global__ void attention_forward_kernel(
    const double* x,         // [batch_size, seq_len, in_feat]
    const double* q_w,       // [in_feat, attn_dim]
    const double* k_w,       // [in_feat, attn_dim]
    const double* v_w,       // [in_feat, attn_dim]
    const double* q_b,       // [attn_dim]
    const double* k_b,       // [attn_dim]
    const double* v_b,       // [attn_dim]
    double* q,              // [batch_size, seq_len, attn_dim]
    double* k,              // [batch_size, seq_len, attn_dim]
    double* v,              // [batch_size, seq_len, attn_dim]
    double* attn_scores,    // [batch_size, seq_len, seq_len]
    double* attn_output,    // [batch_size, seq_len, attn_dim]
    int seq_len, int in_feat, int attn_dim, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate Q, K, V
    if (idx < batch_size * seq_len * attn_dim) {
        int b = idx / (seq_len * attn_dim);
        int s = (idx % (seq_len * attn_dim)) / attn_dim;
        int d = idx % attn_dim;
        
        double q_sum = q_b[d];
        double k_sum = k_b[d];
        double v_sum = v_b[d];
        
        for(int f = 0; f < in_feat; f++) {
            double x_val = x[b * seq_len * in_feat + s * in_feat + f];
            q_sum += x_val * q_w[f * attn_dim + d];
            k_sum += x_val * k_w[f * attn_dim + d];
            v_sum += x_val * v_w[f * attn_dim + d];
        }
        
        q[idx] = q_sum;
        k[idx] = k_sum;
        v[idx] = v_sum;
    }
    
    __syncthreads();
    
    // Calculate attention scores
    if (idx < batch_size * seq_len * seq_len) {
        int b = idx / (seq_len * seq_len);
        int i = (idx % (seq_len * seq_len)) / seq_len;
        int j = idx % seq_len;
        
        double score = 0;
        for(int d = 0; d < attn_dim; d++) {
            score += q[b * seq_len * attn_dim + i * attn_dim + d] *
                    k[b * seq_len * attn_dim + j * attn_dim + d];
        }
        
        attn_scores[idx] = score / sqrt((double)attn_dim);
    }
    
    __syncthreads();
    
    // Apply softmax
    if (idx < batch_size * seq_len) {
        int b = idx / seq_len;
        int i = idx % seq_len;
        
        double max_val = -INFINITY;
        for(int j = 0; j < seq_len; j++) {
            max_val = max(max_val, 
                         attn_scores[b * seq_len * seq_len + i * seq_len + j]);
        }
        
        double sum = 0;
        for(int j = 0; j < seq_len; j++) {
            double val = exp(attn_scores[b * seq_len * seq_len + i * seq_len + j] - max_val);
            attn_scores[b * seq_len * seq_len + i * seq_len + j] = val;
            sum += val;
        }
        
        for(int j = 0; j < seq_len; j++) {
            attn_scores[b * seq_len * seq_len + i * seq_len + j] /= sum;
        }
    }
    
    __syncthreads();
    
    // Calculate attention output
    if (idx < batch_size * seq_len * attn_dim) {
        int b = idx / (seq_len * attn_dim);
        int i = (idx % (seq_len * attn_dim)) / attn_dim;
        int d = idx % attn_dim;
        
        double sum = 0;
        for(int j = 0; j < seq_len; j++) {
            sum += attn_scores[b * seq_len * seq_len + i * seq_len + j] *
                   v[b * seq_len * attn_dim + j * attn_dim + d];
        }
        
        attn_output[idx] = sum;
    }
}

__global__ void attention_backward_kernel(
    const double* x,
    const double* q, const double* k, const double* v,
    const double* attn_scores,
    const double* attn_output,
    const double* grad_output,
    double* grad_q_w, double* grad_k_w, double* grad_v_w,
    double* grad_q_b, double* grad_k_b, double* grad_v_b,
    int seq_len, int in_feat, int attn_dim, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * seq_len * attn_dim) {
        int b = idx / (seq_len * attn_dim);
        int i = (idx % (seq_len * attn_dim)) / attn_dim;
        int d = idx % attn_dim;
        
        double grad_attn = grad_output[b * seq_len * attn_dim + i * attn_dim + d];
        
        // Gradient for V
        for(int j = 0; j < seq_len; j++) {
            atomicAdd(&grad_v_w[j * attn_dim + d],
                     grad_attn * attn_scores[b * seq_len * seq_len + i * seq_len + j] *
                     x[b * seq_len * in_feat + i * in_feat + j]);
        }
        
        // Gradient for Q and K (simplified version)
        double grad_qk = grad_attn / sqrt((double)attn_dim);
        for(int f = 0; f < in_feat; f++) {
            atomicAdd(&grad_q_w[f * attn_dim + d],
                     grad_qk * x[b * seq_len * in_feat + i * in_feat + f]);
            atomicAdd(&grad_k_w[f * attn_dim + d],
                     grad_qk * x[b * seq_len * in_feat + i * in_feat + f]);
        }
        
        // Gradient for biases
        atomicAdd(&grad_q_b[d], grad_qk);
        atomicAdd(&grad_k_b[d], grad_qk);
        atomicAdd(&grad_v_b[d], grad_attn);
    }
}

__global__ void fc_forward_kernel(
    const double* attn_output,  // [batch_size, seq_len, attn_dim]
    const double* fc_w,         // [attn_dim, out_feat]
    const double* fc_b,         // [out_feat]
    double* output,             // [batch_size, out_feat]
    int seq_len, int attn_dim, int out_feat, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * out_feat) {
        int b = idx / out_feat;
        int o = idx % out_feat;
        
        // Just use the last timestep's output
        double sum = fc_b[o];
        const double* last_hidden = &attn_output[b * seq_len * attn_dim + (seq_len-1) * attn_dim];
        for(int d = 0; d < attn_dim; d++) {
            sum += last_hidden[d] * fc_w[d * out_feat + o];
        }
        output[idx] = sum;
    }
}

__global__ void update_params_kernel(double* param, const double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) param[idx] -= LR * grad[idx];
}

Model* init_model(int in_feat, int out_feat) {
    Model* m = (Model*)malloc(sizeof(Model));
    m->attn_dim = ATTN_DIM;
    
    // Calculate sizes
    int qkv_size = in_feat * ATTN_DIM;
    int fc_size = ATTN_DIM * out_feat;
    
    // Initialize with Xavier/Glorot initialization
    double qkv_scale = sqrt(2.0 / (in_feat + ATTN_DIM));
    double fc_scale = sqrt(2.0 / (ATTN_DIM + out_feat));
    
    double *h_qkv_w = (double*)malloc(qkv_size * sizeof(double));
    double *h_fc_w = (double*)malloc(fc_size * sizeof(double));
    
    for(int i = 0; i < qkv_size; i++)
        h_qkv_w[i] = ((double)rand()/RAND_MAX * 2 - 1) * qkv_scale;
    for(int i = 0; i < fc_size; i++)
        h_fc_w[i] = ((double)rand()/RAND_MAX * 2 - 1) * fc_scale;
    
    // Allocate and copy to device
    cudaMalloc(&m->d_q_w, qkv_size * sizeof(double));
    cudaMalloc(&m->d_k_w, qkv_size * sizeof(double));
    cudaMalloc(&m->d_v_w, qkv_size * sizeof(double));
    cudaMalloc(&m->d_q_b, ATTN_DIM * sizeof(double));
    cudaMalloc(&m->d_k_b, ATTN_DIM * sizeof(double));
    cudaMalloc(&m->d_v_b, ATTN_DIM * sizeof(double));
    cudaMalloc(&m->d_fc_w, fc_size * sizeof(double));
    cudaMalloc(&m->d_fc_b, out_feat * sizeof(double));
    
    cudaMemcpy(m->d_q_w, h_qkv_w, qkv_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m->d_k_w, h_qkv_w, qkv_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m->d_v_w, h_qkv_w, qkv_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(m->d_q_b, 0, ATTN_DIM * sizeof(double));
    cudaMemset(m->d_k_b, 0, ATTN_DIM * sizeof(double));
    cudaMemset(m->d_v_b, 0, ATTN_DIM * sizeof(double));
    cudaMemcpy(m->d_fc_w, h_fc_w, fc_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(m->d_fc_b, 0, out_feat * sizeof(double));
    
    free(h_qkv_w);
    free(h_fc_w);
    return m;
}

void free_model(Model* m) {
    cudaFree(m->d_q_w);
    cudaFree(m->d_k_w);
    cudaFree(m->d_v_w);
    cudaFree(m->d_q_b);
    cudaFree(m->d_k_b);
    cudaFree(m->d_v_b);
    cudaFree(m->d_fc_w);
    cudaFree(m->d_fc_b);
    free(m);
}

__global__ void fc_backward_kernel(
    const double* attn_output,  // [batch_size, seq_len, attn_dim]
    const double* fc_w,         // [attn_dim, out_feat]
    const double* y_pred,       // [batch_size, out_feat]
    const double* y_true,       // [batch_size, out_feat]
    double* grad_fc_w,         // [attn_dim, out_feat]
    double* grad_fc_b,         // [out_feat]
    double* grad_attn_output,  // [batch_size, seq_len, attn_dim]
    int seq_len, int attn_dim, int out_feat, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * out_feat) {
        int b = idx / out_feat;
        int o = idx % out_feat;
        
        // Compute output gradient
        double d_out = 2.0 * (y_pred[idx] - y_true[idx]) / (out_feat * batch_size);
        
        // Gradient for bias
        atomicAdd(&grad_fc_b[o], d_out);
        
        // Get pointer to last timestep's output
        const double* last_hidden = &attn_output[b * seq_len * attn_dim + (seq_len-1) * attn_dim];
        double* last_grad = &grad_attn_output[b * seq_len * attn_dim + (seq_len-1) * attn_dim];
        
        // Gradients for weights and attention output
        for(int d = 0; d < attn_dim; d++) {
            atomicAdd(&grad_fc_w[d * out_feat + o], d_out * last_hidden[d]);
            atomicAdd(&last_grad[d], d_out * fc_w[d * out_feat + o]);
        }
    }
}

int main() {
    srand(time(NULL));
    OpenLoopData* data = generate_open_loop_data(1000, 32, 3, 2, 0.1);
    
    // Save data
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_open_loop_csv(fname, data);
    printf("Data saved to: %s\n", fname);

    Model* model = init_model(data->input_features, data->output_features);
    
    // Allocate device memory
    double *d_x, *d_y_true;
    double *d_q, *d_k, *d_v, *d_attn_scores, *d_attn_output, *d_y_pred;
    double *d_grad_q_w, *d_grad_k_w, *d_grad_v_w;
    double *d_grad_q_b, *d_grad_k_b, *d_grad_v_b;
    double *d_grad_fc_w, *d_grad_fc_b;
    
    int qkv_size = data->input_features * ATTN_DIM;
    int fc_size = ATTN_DIM * data->output_features;
    
    cudaMalloc(&d_x, BATCH_SIZE * data->window_size * data->input_features * sizeof(double));
    cudaMalloc(&d_y_true, BATCH_SIZE * data->output_features * sizeof(double));
    cudaMalloc(&d_q, BATCH_SIZE * data->window_size * ATTN_DIM * sizeof(double));
    cudaMalloc(&d_k, BATCH_SIZE * data->window_size * ATTN_DIM * sizeof(double));
    cudaMalloc(&d_v, BATCH_SIZE * data->window_size * ATTN_DIM * sizeof(double));
    cudaMalloc(&d_attn_scores, BATCH_SIZE * data->window_size * data->window_size * sizeof(double));
    cudaMalloc(&d_attn_output, BATCH_SIZE * data->window_size * ATTN_DIM * sizeof(double));
    cudaMalloc(&d_y_pred, BATCH_SIZE * data->output_features * sizeof(double));
    
    cudaMalloc(&d_grad_q_w, qkv_size * sizeof(double));
    cudaMalloc(&d_grad_k_w, qkv_size * sizeof(double));
    cudaMalloc(&d_grad_v_w, qkv_size * sizeof(double));
    cudaMalloc(&d_grad_q_b, ATTN_DIM * sizeof(double));
    cudaMalloc(&d_grad_k_b, ATTN_DIM * sizeof(double));
    cudaMalloc(&d_grad_v_b, ATTN_DIM * sizeof(double));
    cudaMalloc(&d_grad_fc_w, fc_size * sizeof(double));
    cudaMalloc(&d_grad_fc_b, data->output_features * sizeof(double));
    
    printf("Training started...\n");
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0;
        
        for(int b = 0; b < data->n; b += BATCH_SIZE) {
            int batch_size = min(BATCH_SIZE, data->n - b);
            
            // Prepare batch data
            double* batch_x = (double*)malloc(
                batch_size * data->window_size * data->input_features * sizeof(double));
            double* batch_y = (double*)malloc(
                batch_size * data->output_features * sizeof(double));
            
            for(int i = 0; i < batch_size; i++) {
                for(int t = 0; t < data->window_size; t++)
                    for(int f = 0; f < data->input_features; f++)
                        batch_x[i * data->window_size * data->input_features + 
                               t * data->input_features + f] = data->windows[b + i][t][f];
                
                memcpy(&batch_y[i * data->output_features], data->outputs[b + i],
                       data->output_features * sizeof(double));
            }
            
            cudaMemcpy(d_x, batch_x, 
                      batch_size * data->window_size * data->input_features * sizeof(double),
                      cudaMemcpyHostToDevice);
            cudaMemcpy(d_y_true, batch_y,
                      batch_size * data->output_features * sizeof(double),
                      cudaMemcpyHostToDevice);
            
            // Forward pass
            attention_forward_kernel<<<(batch_size * data->window_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                d_x, model->d_q_w, model->d_k_w, model->d_v_w,
                model->d_q_b, model->d_k_b, model->d_v_b,
                d_q, d_k, d_v, d_attn_scores, d_attn_output,
                data->window_size, data->input_features, ATTN_DIM, batch_size);
            
            fc_forward_kernel<<<(batch_size * data->output_features + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                d_attn_output, model->d_fc_w, model->d_fc_b, d_y_pred,
                data->window_size, ATTN_DIM, data->output_features, batch_size);
            
            // Backward pass and parameter updates
            // Zero gradients
            cudaMemset(d_grad_q_w, 0, qkv_size * sizeof(double));
            cudaMemset(d_grad_k_w, 0, qkv_size * sizeof(double));
            cudaMemset(d_grad_v_w, 0, qkv_size * sizeof(double));
            cudaMemset(d_grad_q_b, 0, ATTN_DIM * sizeof(double));
            cudaMemset(d_grad_k_b, 0, ATTN_DIM * sizeof(double));
            cudaMemset(d_grad_v_b, 0, ATTN_DIM * sizeof(double));
            cudaMemset(d_grad_fc_w, 0, fc_size * sizeof(double));
            cudaMemset(d_grad_fc_b, 0, data->output_features * sizeof(double));

            // Backward pass
            double* d_grad_attn_output;
            cudaMalloc(&d_grad_attn_output, 
                      batch_size * data->window_size * ATTN_DIM * sizeof(double));
            cudaMemset(d_grad_attn_output, 0, 
                      batch_size * data->window_size * ATTN_DIM * sizeof(double));

            fc_backward_kernel<<<(batch_size * data->output_features + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                d_attn_output, model->d_fc_w, d_y_pred, d_y_true,
                d_grad_fc_w, d_grad_fc_b, d_grad_attn_output,
                data->window_size, ATTN_DIM, data->output_features, batch_size);

            attention_backward_kernel<<<(batch_size * data->window_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                d_x, d_q, d_k, d_v, d_attn_scores, d_attn_output, d_grad_attn_output,
                d_grad_q_w, d_grad_k_w, d_grad_v_w,
                d_grad_q_b, d_grad_k_b, d_grad_v_b,
                data->window_size, data->input_features, ATTN_DIM, batch_size);

            // Update parameters
            update_params_kernel<<<(qkv_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_q_w, d_grad_q_w, qkv_size);
            update_params_kernel<<<(qkv_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_k_w, d_grad_k_w, qkv_size);
            update_params_kernel<<<(qkv_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_v_w, d_grad_v_w, qkv_size);
            update_params_kernel<<<(ATTN_DIM + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_q_b, d_grad_q_b, ATTN_DIM);
            update_params_kernel<<<(ATTN_DIM + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_k_b, d_grad_k_b, ATTN_DIM);
            update_params_kernel<<<(ATTN_DIM + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_v_b, d_grad_v_b, ATTN_DIM);
            update_params_kernel<<<(fc_size + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_fc_w, d_grad_fc_w, fc_size);
            update_params_kernel<<<(data->output_features + B_SIZE - 1)/B_SIZE, B_SIZE>>>(
                model->d_fc_b, d_grad_fc_b, data->output_features);

            cudaFree(d_grad_attn_output);
            
            // Compute loss
            double* h_y_pred = (double*)malloc(batch_size * data->output_features * sizeof(double));
            cudaMemcpy(h_y_pred, d_y_pred,
                      batch_size * data->output_features * sizeof(double),
                      cudaMemcpyDeviceToHost);
            
            for(int i = 0; i < batch_size; i++) {
                for(int j = 0; j < data->output_features; j++) {
                    double diff = h_y_pred[i * data->output_features + j] - batch_y[i * data->output_features + j];
                    loss += diff * diff;
                }
            }
            
            free(batch_x);
            free(batch_y);
            free(h_y_pred);
        }
        
        if(epoch % 10 == 0)
            printf("Epoch %d, Loss: %.6f\n", epoch, loss/data->n);
    }
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y_true);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_attn_scores);
    cudaFree(d_attn_output);
    cudaFree(d_y_pred);
    cudaFree(d_grad_q_w);
    cudaFree(d_grad_k_w);
    cudaFree(d_grad_v_w);
    cudaFree(d_grad_q_b);
    cudaFree(d_grad_k_b);
    cudaFree(d_grad_v_b);
    cudaFree(d_grad_fc_w);
    cudaFree(d_grad_fc_b);
    
    free_model(model);
    free_open_loop_data(data);
    
    return 0;
}