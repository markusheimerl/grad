#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mlp.h"
// Assumes data.h provides generate_synthetic_data and save_data_to_csv.
#include "../../../data.h"

////////////////////////////////////////////////////////////////////////////////
// Initialization and Freeing
////////////////////////////////////////////////////////////////////////////////
Net* init_net(int input_dim, int num_bins, int embedding_dim, int output_dim, int batch_size, int num_layers) {
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate network.\n");
        exit(EXIT_FAILURE);
    }
    net->input_dim = input_dim;
    net->num_bins = num_bins;
    net->embedding_dim = embedding_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;
    net->num_layers = num_layers;
    
    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;
    
    // Allocate embedding table.
    int emb_table_size = input_dim * num_bins * embedding_dim;
    net->embedding_table = (float*)malloc(emb_table_size * sizeof(float));
    float emb_scale = 1.0f / sqrtf((float)num_bins);
    for (int i = 0; i < emb_table_size; i++) {
        net->embedding_table[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * emb_scale;
    }
    
    // Allocate layers.
    net->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    if (!net->layers) {
        fprintf(stderr, "Failed to allocate layers.\n");
        exit(EXIT_FAILURE);
    }
    for (int l = 0; l < num_layers; l++) {
        Layer* layer = &net->layers[l];
        int attn_dim = embedding_dim * embedding_dim;
        // Self–attention weights.
        layer->W_Q = (float*)malloc(attn_dim * sizeof(float));
        layer->W_K = (float*)malloc(attn_dim * sizeof(float));
        layer->W_V = (float*)malloc(attn_dim * sizeof(float));
        layer->W_Q_grad = (float*)malloc(attn_dim * sizeof(float));
        layer->W_K_grad = (float*)malloc(attn_dim * sizeof(float));
        layer->W_V_grad = (float*)malloc(attn_dim * sizeof(float));
        layer->W_Q_m = (float*)calloc(attn_dim, sizeof(float));
        layer->W_K_m = (float*)calloc(attn_dim, sizeof(float));
        layer->W_V_m = (float*)calloc(attn_dim, sizeof(float));
        layer->W_Q_v = (float*)calloc(attn_dim, sizeof(float));
        layer->W_K_v = (float*)calloc(attn_dim, sizeof(float));
        layer->W_V_v = (float*)calloc(attn_dim, sizeof(float));
        float scale = 1.0f / sqrtf((float)embedding_dim);
        for (int i = 0; i < attn_dim; i++) {
            layer->W_Q[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)* scale;
            layer->W_K[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)* scale;
            layer->W_V[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)* scale;
        }
        memset(layer->W_Q_grad, 0, attn_dim * sizeof(float));
        memset(layer->W_K_grad, 0, attn_dim * sizeof(float));
        memset(layer->W_V_grad, 0, attn_dim * sizeof(float));
        
        // Feed–forward network.
        layer->ff_hidden_dim = 4 * embedding_dim;
        int ff1_size = embedding_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * embedding_dim;
        layer->W_ff1 = (float*)malloc(ff1_size * sizeof(float));
        layer->W_ff1_grad = (float*)malloc(ff1_size * sizeof(float));
        layer->W_ff1_m = (float*)calloc(ff1_size, sizeof(float));
        layer->W_ff1_v = (float*)calloc(ff1_size, sizeof(float));
        layer->W_ff2 = (float*)malloc(ff2_size * sizeof(float));
        layer->W_ff2_grad = (float*)malloc(ff2_size * sizeof(float));
        layer->W_ff2_m = (float*)calloc(ff2_size, sizeof(float));
        layer->W_ff2_v = (float*)calloc(ff2_size, sizeof(float));
        float ff1_scale = 1.0f / sqrtf((float)embedding_dim);
        for (int i = 0; i < ff1_size; i++) {
            layer->W_ff1[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)* ff1_scale;
        }
        memset(layer->W_ff1_grad, 0, ff1_size * sizeof(float));
        float ff2_scale = 1.0f / sqrtf((float)layer->ff_hidden_dim);
        for (int i = 0; i < ff2_size; i++) {
            layer->W_ff2[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)* ff2_scale;
        }
        memset(layer->W_ff2_grad, 0, ff2_size * sizeof(float));
    }
    
    // Allocate intermediate buffers.
    int act_size = batch_size * input_dim * embedding_dim;
    net->layer_input = (float**)malloc((num_layers + 1) * sizeof(float*));
    for (int i = 0; i < num_layers + 1; i++) {
        net->layer_input[i] = (float*)malloc(act_size * sizeof(float));
    }
    // Self–attention intermediates.
    net->attn_Q_layers = (float**)malloc(num_layers * sizeof(float*));
    net->attn_K_layers = (float**)malloc(num_layers * sizeof(float*));
    net->attn_V_layers = (float**)malloc(num_layers * sizeof(float*));
    net->attn_scores_layers = (float**)malloc(num_layers * sizeof(float*));
    net->ff_residual_layers = (float**)malloc(num_layers * sizeof(float*));
    for (int i = 0; i < num_layers; i++) {
        net->attn_Q_layers[i] = (float*)malloc(act_size * sizeof(float));
        net->attn_K_layers[i] = (float*)malloc(act_size * sizeof(float));
        net->attn_V_layers[i] = (float*)malloc(act_size * sizeof(float));
        net->attn_scores_layers[i] = (float*)malloc(batch_size * input_dim * input_dim * sizeof(float));
        net->ff_residual_layers[i] = (float*)malloc(act_size * sizeof(float));
    }
    // Feed–forward intermediates.
    int ff_act_size = batch_size * input_dim * (4 * embedding_dim);
    net->ff_preact_layers = (float**)malloc(num_layers * sizeof(float*));
    net->ff_hidden_layers = (float**)malloc(num_layers * sizeof(float*));
    for (int i = 0; i < num_layers; i++) {
        net->ff_preact_layers[i] = (float*)malloc(ff_act_size * sizeof(float));
        net->ff_hidden_layers[i] = (float*)malloc(ff_act_size * sizeof(float));
    }
    
    // Allocate final predictions and error buffers.
    net->predictions = (float*)malloc(batch_size * input_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * input_dim * sizeof(float));
    
    // Allocate final projection weights (project from embedding_dim to scalar).
    net->W_out = (float*)malloc(embedding_dim * sizeof(float));
    net->W_out_grad = (float*)malloc(embedding_dim * sizeof(float));
    net->W_out_m = (float*)calloc(embedding_dim, sizeof(float));
    net->W_out_v = (float*)calloc(embedding_dim, sizeof(float));
    float wout_scale = 1.0f / sqrtf((float)embedding_dim);
    for (int i = 0; i < embedding_dim; i++) {
        net->W_out[i] = ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f) * wout_scale;
    }
    memset(net->W_out_grad, 0, embedding_dim * sizeof(float));
    
    return net;
}

void free_net(Net* net) {
    if(net) {
        free(net->embedding_table);
        for (int l = 0; l < net->num_layers; l++) {
            Layer* layer = &net->layers[l];
            free(layer->W_Q);
            free(layer->W_K);
            free(layer->W_V);
            free(layer->W_Q_grad);
            free(layer->W_K_grad);
            free(layer->W_V_grad);
            free(layer->W_Q_m);
            free(layer->W_Q_v);
            free(layer->W_K_m);
            free(layer->W_K_v);
            free(layer->W_V_m);
            free(layer->W_V_v);
            free(layer->W_ff1);
            free(layer->W_ff1_grad);
            free(layer->W_ff1_m);
            free(layer->W_ff1_v);
            free(layer->W_ff2);
            free(layer->W_ff2_grad);
            free(layer->W_ff2_m);
            free(layer->W_ff2_v);
        }
        free(net->layers);
        for (int i = 0; i < net->num_layers + 1; i++) {
            free(net->layer_input[i]);
        }
        free(net->layer_input);
        for (int i = 0; i < net->num_layers; i++) {
            free(net->attn_Q_layers[i]);
            free(net->attn_K_layers[i]);
            free(net->attn_V_layers[i]);
            free(net->attn_scores_layers[i]);
            free(net->ff_residual_layers[i]);
        }
        free(net->attn_Q_layers);
        free(net->attn_K_layers);
        free(net->attn_V_layers);
        free(net->attn_scores_layers);
        free(net->ff_residual_layers);
        for (int i = 0; i < net->num_layers; i++) {
            free(net->ff_preact_layers[i]);
            free(net->ff_hidden_layers[i]);
        }
        free(net->ff_preact_layers);
        free(net->ff_hidden_layers);
        free(net->predictions);
        free(net->error);
        // Free final projection weights.
        free(net->W_out);
        free(net->W_out_grad);
        free(net->W_out_m);
        free(net->W_out_v);
        free(net);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Embedding and Forward Pass
////////////////////////////////////////////////////////////////////////////////
/*
  embed_input: Maps each raw input (shape: [batch_size x input_dim]) into a token by quantizing 
  each feature and copying its embedding vector. The result is stored in net->layer_input[0] 
  (and also copied to the provided embedded_output).
*/
void embed_input(Net* net, float* raw_input, float* embedded_output, float* in_min, float* in_max) {
    for (int i = 0; i < net->batch_size; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            float val = raw_input[i * net->input_dim + j];
            int b = bin_value(val, in_min[j], in_max[j], net->num_bins);
            float* emb = net->embedding_table +
                         j * (net->num_bins * net->embedding_dim) +
                         b * net->embedding_dim;
            float* dest = net->layer_input[0] +
                          i * (net->input_dim * net->embedding_dim) +
                          j * net->embedding_dim;
            memcpy(dest, emb, net->embedding_dim * sizeof(float));
        }
    }
    memcpy(embedded_output, net->layer_input[0], net->batch_size * net->input_dim * net->embedding_dim * sizeof(float));
}

/*
  forward_pass: Processes the embedded input through each transformer layer.
  Each layer does:
    – Self–attention: computes Q, K, V; then scores = softmax((Q*K^T)/sqrt(d));
       attn_out = scores * V; residual: r = X + attn_out.
    – Feed–forward: computes hidden = ReLU(r * W_ff1); then f = hidden * W_ff2;
       final output = r + f.
  After the transformer layers, a final linear projection from embedding_dim down to scalar is applied for each token.
  The final predictions (shape: [batch_size x output_dim]) are stored in net->predictions.
*/
void forward_pass(Net* net) {
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int total_tokens = net->batch_size * tokens;
    int ff_hidden = 4 * d_dim;
    
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        float* X = net->layer_input[l];         // Input for layer l.
        float* out = net->layer_input[l+1];       // Output will be stored here.
        // Self–Attention:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, d_dim,
                    1.0f, X, d_dim,
                    layer->W_Q, d_dim,
                    0.0f, net->attn_Q_layers[l], d_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, d_dim,
                    1.0f, X, d_dim,
                    layer->W_K, d_dim,
                    0.0f, net->attn_K_layers[l], d_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, d_dim,
                    1.0f, X, d_dim,
                    layer->W_V, d_dim,
                    0.0f, net->attn_V_layers[l], d_dim);
        
        float scale = 1.0f / sqrtf((float)d_dim);
        for (int s = 0; s < net->batch_size; s++) {
            float* Q = net->attn_Q_layers[l] + s * tokens * d_dim;
            float* K = net->attn_K_layers[l] + s * tokens * d_dim;
            float* V = net->attn_V_layers[l] + s * tokens * d_dim;
            float* scores = net->attn_scores_layers[l] + s * tokens * tokens;
            // Compute scores = (Q*K^T)*scale.
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, tokens, d_dim,
                        scale, Q, d_dim,
                        K, d_dim,
                        0.0f, scores, tokens);
            // Apply softmax row–wise.
            for (int i = 0; i < tokens; i++) {
                float* score_row = scores + i * tokens;
                float* softmax_row = (float*)malloc(tokens * sizeof(float));
                softmax(score_row, softmax_row, tokens);
                memcpy(score_row, softmax_row, tokens * sizeof(float));
                free(softmax_row);
            }
            // Compute attention output: attn_out = scores * V.
            float* attn_out = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        1.0f, scores, tokens,
                        V, d_dim,
                        0.0f, attn_out, d_dim);
            // Residual: r = X_sample + attn_out.
            float* X_sample = X + s * tokens * d_dim;
            float* r = net->ff_residual_layers[l] + s * tokens * d_dim;
            for (int i = 0; i < tokens * d_dim; i++) {
                r[i] = X_sample[i] + attn_out[i];
            }
            free(attn_out);
        }
        
        // Feed–Forward Block:
        // Compute pre–activation: ff_preact = r * W_ff1.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, ff_hidden, d_dim,
                    1.0f, net->ff_residual_layers[l], d_dim,
                    layer->W_ff1, ff_hidden,
                    0.0f, net->ff_preact_layers[l], ff_hidden);
        // Apply ReLU.
        for (int i = 0; i < total_tokens * ff_hidden; i++) {
            net->ff_hidden_layers[l][i] = (net->ff_preact_layers[l][i] > 0.0f) ? net->ff_preact_layers[l][i] : 0.0f;
        }
        // Compute f = hidden * W_ff2.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_tokens, d_dim, ff_hidden,
                    1.0f, net->ff_hidden_layers[l], ff_hidden,
                    layer->W_ff2, d_dim,
                    0.0f, out, d_dim);
        // Final residual: add r.
        for (int i = 0; i < total_tokens * d_dim; i++) {
            out[i] += net->ff_residual_layers[l][i];
        }
    }
    // Final projection: for each token, project the transformer output (of dimension d_dim)
    // down to a scalar using W_out.
    for (int i = 0; i < total_tokens; i++) {
        float dot = 0.0f;
        float* token_vec = net->layer_input[net->num_layers] + i * d_dim;
        for (int k = 0; k < d_dim; k++) {
            dot += token_vec[k] * net->W_out[k];
        }
        net->predictions[i] = dot;
    }
}

/*
  calculate_loss: Computes the mean–squared error (MSE) loss.
  For each token the loss is 1/2 * (prediction – target)^2.
  It also sets net->error = (prediction – target).
*/
float calculate_loss(Net* net, float* targets) {
    float loss = 0.0f;
    int total = net->batch_size * net->output_dim;
    for (int i = 0; i < total; i++) {
        float diff = net->predictions[i] - targets[i];
        loss += 0.5f * diff * diff; // using 1/2 factor so that derivative is (p - t)
        net->error[i] = diff;
    }
    return loss / total;
}

/*
  zero_gradients: Resets the gradients for all layers and the final projection.
*/
void zero_gradients(Net* net) {
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    // Zero final projection gradients.
    memset(net->W_out_grad, 0, d_dim * sizeof(float));
    
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        memset(layer->W_Q_grad, 0, attn_dim * sizeof(float));
        memset(layer->W_K_grad, 0, attn_dim * sizeof(float));
        memset(layer->W_V_grad, 0, attn_dim * sizeof(float));
        int ff1_size = d_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * d_dim;
        memset(layer->W_ff1_grad, 0, ff1_size * sizeof(float));
        memset(layer->W_ff2_grad, 0, ff2_size * sizeof(float));
    }
}

////////////////////////////////////////////////////////////////////////////////
// Backward Pass (Multi–Layer)
//
/*
  backward_pass: Propagates the error (stored in net->error from calculate_loss)
  backward through each layer (from final layer down to the embedded input).
  Before back–propagating through the transformer layers, the final projection is handled.
  That is, for each token, if the transformer output (of dimension d_dim) is X and the learned projection W_out
  yields prediction = dot(X, W_out), then the gradient with respect to X is error * W_out and 
  the gradient with respect to W_out is accumulated as error * X.
*/
void backward_pass(Net* net) {
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int total_tokens = net->batch_size * tokens;
    int ff_hidden = 4 * d_dim;
    float inv_sqrt = 1.0f / sqrtf((float)d_dim);

    zero_gradients(net);

    // Backprop through final projection.
    float* dX = (float*)malloc(total_tokens * d_dim * sizeof(float));
    float* final_transformer_output = net->layer_input[net->num_layers];
    for (int s = 0; s < net->batch_size; s++) {
        for (int t = 0; t < tokens; t++) {
            int token_index = s * tokens + t;
            float err = net->error[token_index];  // scalar error
            for (int k = 0; k < d_dim; k++) {
                dX[token_index * d_dim + k] = err * net->W_out[k];
                net->W_out_grad[k] += err * final_transformer_output[token_index * d_dim + k];
            }
        }
    }
    
    // Loop backward over transformer layers.
    for (int l = net->num_layers - 1; l >= 0; l--) {
        Layer* layer = &net->layers[l];
        float* X = net->layer_input[l];            // Input to layer l.
        float* r = net->ff_residual_layers[l];       // r = X + attn_out.
        float* out = net->layer_input[l+1];          // Output of layer l.
        
        // --- Backprop through Feed–Forward Block ---
        // Out = r + f, with f = hidden * W_ff2.
        // (a) The direct residual path gives d_r_add = dX.
        float* d_r_add = (float*)malloc(total_tokens * d_dim * sizeof(float));
        memcpy(d_r_add, dX, total_tokens * d_dim * sizeof(float));
        // (b) Backprop through feed–forward branch.
        float* d_hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    total_tokens, ff_hidden, d_dim,
                    1.0f, dX, d_dim,
                    layer->W_ff2, d_dim,
                    0.0f, d_hidden, ff_hidden);
        float* d_preact = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
        float* ff_preact = net->ff_preact_layers[l];
        for (int i = 0; i < total_tokens * ff_hidden; i++) {
            d_preact[i] = (ff_preact[i] > 0.0f) ? d_hidden[i] : 0.0f;
        }
        // Gradient for W_ff1.
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    d_dim, ff_hidden, total_tokens,
                    1.0f, r, d_dim,
                    d_preact, ff_hidden,
                    1.0f, layer->W_ff1_grad, ff_hidden);
        float* d_r_ff = (float*)malloc(total_tokens * d_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    total_tokens, d_dim, ff_hidden,
                    1.0f, d_preact, ff_hidden,
                    layer->W_ff1, ff_hidden,
                    0.0f, d_r_ff, d_dim);
        // Gradient for W_ff2.
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ff_hidden, d_dim, total_tokens,
                    1.0f, net->ff_hidden_layers[l], ff_hidden,
                    dX, d_dim,
                    1.0f, layer->W_ff2_grad, d_dim);
        // Sum gradients from both residual and feed–forward branch.
        float* d_r = (float*)malloc(total_tokens * d_dim * sizeof(float));
        for (int i = 0; i < total_tokens * d_dim; i++) {
            d_r[i] = d_r_add[i] + d_r_ff[i];
        }
        free(d_r_add);
        free(d_r_ff);
        free(d_hidden);
        free(d_preact);
        
        // --- Backprop through Self–Attention Block ---
        // In the forward pass r = X + attn_out so the derivative through attn_out is d_r.
        float* d_attn_out = (float*)malloc(total_tokens * d_dim * sizeof(float));
        memcpy(d_attn_out, d_r, total_tokens * d_dim * sizeof(float));
        // Backprop attention per sample.
        for (int s = 0; s < net->batch_size; s++) {
            float* scores = net->attn_scores_layers[l] + s * tokens * tokens;
            float* d_attn_out_sample = d_attn_out + s * tokens * d_dim;
            float* V = net->attn_V_layers[l] + s * tokens * d_dim;
            float* Q = net->attn_Q_layers[l] + s * tokens * d_dim;
            float* K = net->attn_K_layers[l] + s * tokens * d_dim;
            
            // dV = scores^T * d_attn_out_sample.
            float* dV = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        1.0f, scores, tokens,
                        d_attn_out_sample, d_dim,
                        0.0f, dV, d_dim);
            
            // Intermediate: dA = d_attn_out_sample * V^T.
            float* dA = (float*)malloc(tokens * tokens * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, tokens, d_dim,
                        1.0f, d_attn_out_sample, d_dim,
                        V, d_dim,
                        0.0f, dA, tokens);
            // Softmax derivative per row.
            for (int i = 0; i < tokens; i++) {
                float sum_d = 0.0f;
                for (int j = 0; j < tokens; j++) {
                    sum_d += scores[i * tokens + j] * dA[i * tokens + j];
                }
                for (int j = 0; j < tokens; j++) {
                    dA[i * tokens + j] = scores[i * tokens + j] * (dA[i * tokens + j] - sum_d);
                }
            }
            // dQ = (dA * K)*inv_sqrt.
            float* dQ = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        inv_sqrt, dA, tokens,
                        K, d_dim,
                        0.0f, dQ, d_dim);
            // dK = (dA^T * Q)*inv_sqrt.
            float* dK = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        tokens, d_dim, tokens,
                        inv_sqrt, dA, tokens,
                        Q, d_dim,
                        0.0f, dK, d_dim);
            
            // Accumulate gradients for attention weights.
            float* X_sample = X + s * tokens * d_dim;
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d_dim, d_dim, tokens,
                        1.0f, X_sample, d_dim,
                        dQ, d_dim,
                        1.0f, layer->W_Q_grad, d_dim);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d_dim, d_dim, tokens,
                        1.0f, X_sample, d_dim,
                        dK, d_dim,
                        1.0f, layer->W_K_grad, d_dim);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        d_dim, d_dim, tokens,
                        1.0f, X_sample, d_dim,
                        dV, d_dim,
                        1.0f, layer->W_V_grad, d_dim);
            
            // Compute gradient wrt X from attention branch.
            float* temp = (float*)malloc(tokens * d_dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, d_dim, d_dim,
                        1.0f, dQ, d_dim,
                        layer->W_Q, d_dim,
                        0.0f, temp, d_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, d_dim, d_dim,
                        1.0f, dK, d_dim,
                        layer->W_K, d_dim,
                        1.0f, temp, d_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tokens, d_dim, d_dim,
                        1.0f, dV, d_dim,
                        layer->W_V, d_dim,
                        1.0f, temp, d_dim);
            for (int i = 0; i < tokens * d_dim; i++) {
                dX[s * tokens * d_dim + i] += temp[i];
            }
            free(temp);
            free(dA);
            free(dQ);
            free(dK);
            free(dV);
        }
        free(d_attn_out);
        free(d_r);
    }
    free(dX);
}

////////////////////////////////////////////////////////////////////////////////
// Weight Update using AdamW
////////////////////////////////////////////////////////////////////////////////
void update_weights(Net* net, float learning_rate) {
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        for (int i = 0; i < attn_dim; i++) {
            float grad = layer->W_Q_grad[i] / net->batch_size;
            layer->W_Q_m[i] = net->beta1 * layer->W_Q_m[i] + (1.0f - net->beta1) * grad;
            layer->W_Q_v[i] = net->beta2 * layer->W_Q_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_Q_m[i] / (sqrtf(layer->W_Q_v[i]) + net->epsilon);
            layer->W_Q[i] = layer->W_Q[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        for (int i = 0; i < attn_dim; i++) {
            float grad = layer->W_K_grad[i] / net->batch_size;
            layer->W_K_m[i] = net->beta1 * layer->W_K_m[i] + (1.0f - net->beta1) * grad;
            layer->W_K_v[i] = net->beta2 * layer->W_K_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_K_m[i] / (sqrtf(layer->W_K_v[i]) + net->epsilon);
            layer->W_K[i] = layer->W_K[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        for (int i = 0; i < attn_dim; i++) {
            float grad = layer->W_V_grad[i] / net->batch_size;
            layer->W_V_m[i] = net->beta1 * layer->W_V_m[i] + (1.0f - net->beta1) * grad;
            layer->W_V_v[i] = net->beta2 * layer->W_V_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_V_m[i] / (sqrtf(layer->W_V_v[i]) + net->epsilon);
            layer->W_V[i] = layer->W_V[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        int ff1_size = d_dim * layer->ff_hidden_dim;
        for (int i = 0; i < ff1_size; i++) {
            float grad = layer->W_ff1_grad[i] / net->batch_size;
            layer->W_ff1_m[i] = net->beta1 * layer->W_ff1_m[i] + (1.0f - net->beta1) * grad;
            layer->W_ff1_v[i] = net->beta2 * layer->W_ff1_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_ff1_m[i] / (sqrtf(layer->W_ff1_v[i]) + net->epsilon);
            layer->W_ff1[i] = layer->W_ff1[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
        int ff2_size = layer->ff_hidden_dim * d_dim;
        for (int i = 0; i < ff2_size; i++) {
            float grad = layer->W_ff2_grad[i] / net->batch_size;
            layer->W_ff2_m[i] = net->beta1 * layer->W_ff2_m[i] + (1.0f - net->beta1) * grad;
            layer->W_ff2_v[i] = net->beta2 * layer->W_ff2_v[i] + (1.0f - net->beta2) * grad * grad;
            float update_val = alpha_t * layer->W_ff2_m[i] / (sqrtf(layer->W_ff2_v[i]) + net->epsilon);
            layer->W_ff2[i] = layer->W_ff2[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
        }
    }
    // Update final projection weights.
    for (int i = 0; i < d_dim; i++) {
        float grad = net->W_out_grad[i] / net->batch_size;
        net->W_out_m[i] = net->beta1 * net->W_out_m[i] + (1.0f - net->beta1) * grad;
        net->W_out_v[i] = net->beta2 * net->W_out_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_out_m[i] / (sqrtf(net->W_out_v[i]) + net->epsilon);
        net->W_out[i] = net->W_out[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Save and Load Model
////////////////////////////////////////////////////////////////////////////////
void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->num_bins, sizeof(int), 1, file);
    fwrite(&net->embedding_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);
    fwrite(&net->num_layers, sizeof(int), 1, file);
    
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    // For each layer, save attention weights.
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        fwrite(layer->W_Q, sizeof(float), attn_dim, file);
        fwrite(layer->W_K, sizeof(float), attn_dim, file);
        fwrite(layer->W_V, sizeof(float), attn_dim, file);
    }
    fwrite(&net->t, sizeof(int), 1, file);
    // Save Adam state for attention in each layer.
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        fwrite(layer->W_Q_m, sizeof(float), attn_dim, file);
        fwrite(layer->W_Q_v, sizeof(float), attn_dim, file);
        fwrite(layer->W_K_m, sizeof(float), attn_dim, file);
        fwrite(layer->W_K_v, sizeof(float), attn_dim, file);
        fwrite(layer->W_V_m, sizeof(float), attn_dim, file);
        fwrite(layer->W_V_v, sizeof(float), attn_dim, file);
    }
    // Save final projection parameters.
    fwrite(net->W_out, sizeof(float), d_dim, file);
    fwrite(net->W_out_m, sizeof(float), d_dim, file);
    fwrite(net->W_out_v, sizeof(float), d_dim, file);
    
    // Save embedding table.
    int emb_table_size = net->input_dim * net->num_bins * net->embedding_dim;
    fwrite(net->embedding_table, sizeof(float), emb_table_size, file);
    // Save feed–forward network parameters for each layer.
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        fwrite(&layer->ff_hidden_dim, sizeof(int), 1, file);
        int ff1_size = d_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * d_dim;
        fwrite(layer->W_ff1, sizeof(float), ff1_size, file);
        fwrite(layer->W_ff2, sizeof(float), ff2_size, file);
        fwrite(layer->W_ff1_m, sizeof(float), ff1_size, file);
        fwrite(layer->W_ff1_v, sizeof(float), ff1_size, file);
        fwrite(layer->W_ff2_m, sizeof(float), ff2_size, file);
        fwrite(layer->W_ff2_v, sizeof(float), ff2_size, file);
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    int input_dim, num_bins, embedding_dim, output_dim, batch_size, num_layers;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&num_bins, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
    Net* net = init_net(input_dim, num_bins, embedding_dim, output_dim, batch_size, num_layers);
    int d_dim = net->embedding_dim;
    int attn_dim = d_dim * d_dim;
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        fread(layer->W_Q, sizeof(float), attn_dim, file);
        fread(layer->W_K, sizeof(float), attn_dim, file);
        fread(layer->W_V, sizeof(float), attn_dim, file);
    }
    fread(&net->t, sizeof(int), 1, file);
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        fread(layer->W_Q_m, sizeof(float), attn_dim, file);
        fread(layer->W_Q_v, sizeof(float), attn_dim, file);
        fread(layer->W_K_m, sizeof(float), attn_dim, file);
        fread(layer->W_K_v, sizeof(float), attn_dim, file);
        fread(layer->W_V_m, sizeof(float), attn_dim, file);
        fread(layer->W_V_v, sizeof(float), attn_dim, file);
    }
    // Load final projection parameters.
    fread(net->W_out, sizeof(float), d_dim, file);
    fread(net->W_out_m, sizeof(float), d_dim, file);
    fread(net->W_out_v, sizeof(float), d_dim, file);
    
    int emb_table_size = input_dim * num_bins * embedding_dim;
    fread(net->embedding_table, sizeof(float), emb_table_size, file);
    for (int l = 0; l < net->num_layers; l++) {
        Layer* layer = &net->layers[l];
        fread(&layer->ff_hidden_dim, sizeof(int), 1, file);
        int ff1_size = d_dim * layer->ff_hidden_dim;
        int ff2_size = layer->ff_hidden_dim * d_dim;
        fread(layer->W_ff1, sizeof(float), ff1_size, file);
        fread(layer->W_ff2, sizeof(float), ff2_size, file);
        fread(layer->W_ff1_m, sizeof(float), ff1_size, file);
        fread(layer->W_ff1_v, sizeof(float), ff1_size, file);
        fread(layer->W_ff2_m, sizeof(float), ff2_size, file);
        fread(layer->W_ff2_v, sizeof(float), ff2_size, file);
    }
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

////////////////////////////////////////////////////////////////////////////////
// Main Training and Evaluation
////////////////////////////////////////////////////////////////////////////////
int main() {
    srand((unsigned)time(NULL));
    openblas_set_num_threads(4);
    
    // PARAMETERS.
    const int input_dim = 4;            // Number of raw input features.
    const int num_samples = 1024;       // Number of samples.
    const int output_dim = input_dim;   // One output per input feature.
    const int num_bins = 64;            // Used for input embedding.
    const int embedding_dim = num_bins; // Embedding dimension.
    
    // Self–attention parameters.
    const int batch_size = num_samples; // Full–batch training.
    
    // DATA GENERATION.
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);
    
    // Compute per–feature min and max for inputs.
    float* input_min = (float*)malloc(input_dim * sizeof(float));
    float* input_max = (float*)malloc(input_dim * sizeof(float));
    compute_min_max(X, num_samples, input_dim, input_min, input_max);
    
    // Choose number of layers.
    int num_layers = 2;  // (Change to desired number.)
    
    // INITIALIZE NETWORK.
    Net* net = init_net(input_dim, num_bins, embedding_dim, output_dim, batch_size, num_layers);
    // Allocate embedded input buffer.
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
        embed_input(net, X, embedded_input, input_min, input_max);
        forward_pass(net);
        float loss = calculate_loss(net, y);
        zero_gradients(net);
        backward_pass(net);
        update_weights(net, learning_rate);
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], MSE Loss: %.8f\n", epoch + 1, num_epochs, loss);
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
    forward_pass(net);
    float verification_loss = calculate_loss(net, y);
    printf("MSE Loss with loaded model: %.8f\n", verification_loss);
    
    // EVALUATION.
    double sum_sq_error = 0.0, sum_abs_error = 0.0;
    int total_outputs = num_samples * output_dim;
    for (int i = 0; i < total_outputs; i++) {
        float diff = net->predictions[i] - y[i];
        sum_sq_error += diff * diff;
        sum_abs_error += fabs(diff);
    }
    float mse = (float)(sum_sq_error / total_outputs);
    float mae = (float)(sum_abs_error / total_outputs);
    printf("\nMean Squared Error (MSE): %.8f\n", mse);
    printf("Mean Absolute Error (MAE): %.8f\n", mae);
    
    // Print sample predictions (first 5 samples).
    printf("\nSample Predictions (first 5 samples):\n");
    for (int i = 0; i < 5; i++) {
        printf("Sample %d:\n", i);
        for (int d = 0; d < output_dim; d++) {
            int idx = i * output_dim + d;
            printf("  Output %d: Predicted = %.5f, True = %.5f\n",
                   d, net->predictions[idx], y[i * output_dim + d]);
        }
    }
    
    // CLEANUP.
    free(X);
    free(y);
    free(input_min);
    free(input_max);
    free(embedded_input);
    free_net(net);
    
    return 0;
} 