#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

/*
  In this network every raw input feature (a scalar) is first projected via a learned
  linear mapping (input_proj) into an embedding vector (of dimension embed_dim).
  Then, instead of using self–attention we mix the tokens using a token–mixing MLP (across the tokens axis)
  and feed them to a feed–forward network (which mixes channels). Both blocks include residual connections.
  Finally, each token’s final embedding is linearly projected (via output_proj) down to a scalar output.
  The loss is computed as mean–squared error (MSE) between the network outputs and the target continuous value.
  All activations use swish (swish(x)=x·sigmoid(x)) instead of ReLU.
*/

typedef struct {
    // Raw input and output dimensions.
    int input_dim;     // Number of raw input features = number of tokens.
    int output_dim;    // Number of continuous outputs = input_dim.

    int embedding_dim; // The projected dimension (learned representation for each token).

    // New learned input projection parameters.
    // For each token j: project raw scalar to an embedding vector of dimension embedding_dim.
    // Shape: [input_dim x embedding_dim]
    float* input_proj;
    float* input_proj_grad;
    float* input_proj_m;
    float* input_proj_v;

    // Token–mixing MLP parameters.
    // This MLP is applied “across” tokens (each token vector is of length = embedding_dim), 
    // independently for each channel.
    // First linear layer: [input_dim x token_hidden_dim] and second: [token_hidden_dim x input_dim].
    int token_hidden_dim;
    float* W_token1;       // [input_dim x token_hidden_dim]
    float* W_token1_grad;
    float* W_token1_m;
    float* W_token1_v;
    float* W_token2;       // [token_hidden_dim x input_dim]
    float* W_token2_grad;
    float* W_token2_m;
    float* W_token2_v;

    // Feed–forward network parameters (channel–mixing).
    // Two–layer network applied to each token independently.
    // First layer: [embedding_dim x ff_hidden_dim], second layer: [ff_hidden_dim x embedding_dim].
    int ff_hidden_dim;
    float* W_ff1;       // [embedding_dim x ff_hidden_dim]
    float* W_ff1_grad;
    float* W_ff1_m;
    float* W_ff1_v;
    float* W_ff2;       // [ff_hidden_dim x embedding_dim]
    float* W_ff2_grad;
    float* W_ff2_m;
    float* W_ff2_v;

    // Learned output projection.
    // For each token, the final embedding (of dim embedding_dim) is projected down to a scalar.
    // Shared across tokens.
    // Shape: [embedding_dim x 1] (can be thought of as a vector of length embedding_dim).
    float* output_proj;
    float* output_proj_grad;
    float* output_proj_m;
    float* output_proj_v;

    // Adam hyper–parameters.
    float beta1;
    float beta2;
    float epsilon;
    int   t;
    float weight_decay;

    // Helper arrays for forward/backward passes.
    // intermed: stores each token’s embedding AFTER the feed–forward (channel mixing) block.
    // Shape: [batch_size x (input_dim * embedding_dim)]
    float* predictions;
    // error: gradients for the intermediate token embeddings, under the feed–forward and mixing blocks.
    // Same shape as predictions.
    float* error;

    // ff_residual: buffer to store the output of the token–mixing block (with residual).
    // Shape: [batch_size x (input_dim * embedding_dim)]
    float* ff_residual;

    // final_output: final scalar outputs for each token.
    // Shape: [batch_size x output_dim]  (remember: output_dim == input_dim)
    float* final_output;
    // final_error: gradient for the final outputs.
    // Shape: [batch_size x output_dim]
    float* final_error;

    // Training batch size.
    int batch_size;  // Number of samples per batch.
} Net;

/* ---------------- Helper Activation Functions ---------------- */
static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float swishf(float x) {
    float s = sigmoidf(x);
    return x * s;
}

/* Swish derivative computed for pre–activation value z.
   Using d/dz swish(z) = sigmoid(z) + swish(z)·(1 - sigmoid(z)) */
static inline float swish_deriv(float z) {
    float s = sigmoidf(z);
    return s + swishf(z) * (1.0f - s);
}

/* ---------------- Network Initialization and Freeing ---------------- */

/*
  init_net: Initializes the network with a learned input projection, a token–mixing MLP,
  a feed–forward network, and a learned output projection.
  Parameters:
    input_dim: number of raw input features (tokens).
    embedding_dim: dimension for each token’s embedding.
    output_dim: number of continuous outputs (should equal input_dim).
    batch_size: number of samples per batch.
*/
static inline Net* init_net(int input_dim, int embedding_dim,
                              int output_dim, int batch_size) {
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) {
        fprintf(stderr, "Failed to allocate network.\n");
        exit(EXIT_FAILURE);
    }
    net->input_dim = input_dim;
    net->embedding_dim = embedding_dim;
    net->output_dim = output_dim;
    net->batch_size = batch_size;

    net->beta1 = 0.9f;
    net->beta2 = 0.999f;
    net->epsilon = 1e-8f;
    net->t = 0;
    net->weight_decay = 0.01f;

    // Allocate and initialize learned input projection.
    // Shape: [input_dim x embedding_dim]
    int inp_proj_size = input_dim * embedding_dim;
    net->input_proj = (float*)malloc(inp_proj_size * sizeof(float));
    net->input_proj_grad = (float*)malloc(inp_proj_size * sizeof(float));
    net->input_proj_m = (float*)calloc(inp_proj_size, sizeof(float));
    net->input_proj_v = (float*)calloc(inp_proj_size, sizeof(float));
    float inp_scale = 1.0f;  // fan_in=1 for scalar->vector.
    for (int i = 0; i < inp_proj_size; i++) {
        net->input_proj[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * inp_scale;
    }
    memset(net->input_proj_grad, 0, inp_proj_size * sizeof(float));

    // Allocate token–mixing MLP parameters.
    // Token mixing is done across the "tokens" dimension.
    // Set token_hidden_dim = 4 * input_dim.
    net->token_hidden_dim = 4 * input_dim;
    int token1_size = input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * input_dim;
    net->W_token1 = (float*)malloc(token1_size * sizeof(float));
    net->W_token1_grad = (float*)malloc(token1_size * sizeof(float));
    net->W_token1_m = (float*)calloc(token1_size, sizeof(float));
    net->W_token1_v = (float*)calloc(token1_size, sizeof(float));
    net->W_token2 = (float*)malloc(token2_size * sizeof(float));
    net->W_token2_grad = (float*)malloc(token2_size * sizeof(float));
    net->W_token2_m = (float*)calloc(token2_size, sizeof(float));
    net->W_token2_v = (float*)calloc(token2_size, sizeof(float));

    float token_scale = 1.0f / sqrtf((float)input_dim);
    for (int i = 0; i < token1_size; i++) {
        net->W_token1[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * token_scale;
    }
    memset(net->W_token1_grad, 0, token1_size * sizeof(float));
    float token2_scale = 1.0f / sqrtf((float)net->token_hidden_dim);
    for (int i = 0; i < token2_size; i++) {
        net->W_token2[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * token2_scale;
    }
    memset(net->W_token2_grad, 0, token2_size * sizeof(float));

    // Allocate helper arrays.
    // predictions and error: shape = [batch_size x (input_dim * embedding_dim)]
    net->predictions = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    net->error = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));
    // ff_residual: stored output of token–mixing (with residual).
    net->ff_residual = (float*)malloc(batch_size * input_dim * embedding_dim * sizeof(float));

    // Initialize feed–forward network parameters.
    net->ff_hidden_dim = 4 * embedding_dim;
    int ff1_size = embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * embedding_dim;
    net->W_ff1 = (float*)malloc(ff1_size * sizeof(float));
    net->W_ff1_grad = (float*)malloc(ff1_size * sizeof(float));
    net->W_ff1_m = (float*)calloc(ff1_size, sizeof(float));
    net->W_ff1_v = (float*)calloc(ff1_size, sizeof(float));
    net->W_ff2 = (float*)malloc(ff2_size * sizeof(float));
    net->W_ff2_grad = (float*)malloc(ff2_size * sizeof(float));
    net->W_ff2_m = (float*)calloc(ff2_size, sizeof(float));
    net->W_ff2_v = (float*)calloc(ff2_size, sizeof(float));

    float ff1_scale = 1.0f / sqrtf((float)embedding_dim);
    for (int i = 0; i < ff1_size; i++) {
        net->W_ff1[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * ff1_scale;
    }
    memset(net->W_ff1_grad, 0, ff1_size * sizeof(float));
    float ff2_scale = 1.0f / sqrtf((float)net->ff_hidden_dim);
    for (int i = 0; i < ff2_size; i++) {
        net->W_ff2[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * ff2_scale;
    }
    memset(net->W_ff2_grad, 0, ff2_size * sizeof(float));

    // Allocate learned output projection.
    // Shape: [embedding_dim x 1] (we treat this as a vector of length embedding_dim)
    net->output_proj = (float*)malloc(embedding_dim * sizeof(float));
    net->output_proj_grad = (float*)malloc(embedding_dim * sizeof(float));
    net->output_proj_m = (float*)calloc(embedding_dim, sizeof(float));
    net->output_proj_v = (float*)calloc(embedding_dim, sizeof(float));
    float out_scale = 1.0f / sqrtf((float)embedding_dim);
    for (int i = 0; i < embedding_dim; i++) {
        net->output_proj[i] = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * out_scale;
    }
    memset(net->output_proj_grad, 0, embedding_dim * sizeof(float));

    // Allocate final output and its gradient.
    // Shape: [batch_size x output_dim] with output_dim == input_dim.
    net->final_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    net->final_error = (float*)malloc(batch_size * output_dim * sizeof(float));

    return net;
}

/*
  free_net: Releases all memory allocated for the network.
*/
static inline void free_net(Net* net) {
    if(net) {
        free(net->input_proj);
        free(net->input_proj_grad);
        free(net->input_proj_m);
        free(net->input_proj_v);
        free(net->W_token1);
        free(net->W_token1_grad);
        free(net->W_token1_m);
        free(net->W_token1_v);
        free(net->W_token2);
        free(net->W_token2_grad);
        free(net->W_token2_m);
        free(net->W_token2_v);
        free(net->W_ff1);
        free(net->W_ff1_grad);
        free(net->W_ff1_m);
        free(net->W_ff1_v);
        free(net->W_ff2);
        free(net->W_ff2_grad);
        free(net->W_ff2_m);
        free(net->W_ff2_v);
        free(net->output_proj);
        free(net->output_proj_grad);
        free(net->output_proj_m);
        free(net->output_proj_v);
        free(net->predictions);
        free(net->error);
        free(net->ff_residual);
        free(net->final_output);
        free(net->final_error);
        free(net);
    }
}

/* ---------------- Input Projection and Forward Pass Functions ---------------- */

/*
  project_input: For each raw input (shape [batch_size x input_dim]),
  computes a learned linear projection from the scalar to an embedding vector.
  The resulting embedded output has shape [batch_size x (input_dim * embedding_dim)].
  For each sample and token j:
     embedded_output[i, j, :] = raw_input[i, j] * input_proj[j, :]
*/
static inline void project_input(Net* net, float* raw_input, float* embedded_output) {
    for (int i = 0; i < net->batch_size; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            float val = raw_input[i * net->input_dim + j];
            float* proj = net->input_proj + j * net->embedding_dim;
            float* dest = embedded_output + i * (net->input_dim * net->embedding_dim) + j * net->embedding_dim;
            for (int k = 0; k < net->embedding_dim; k++) {
                dest[k] = val * proj[k];
            }
        }
    }
}

/*
  forward_pass: Computes the forward pass.
  Steps:
    1. Token–mixing MLP: For each sample, for each channel (from 0 to embedding_dim-1),
       treat the tokens (of length input_dim) as a vector v and compute
         hidden = swish(v * W_token1)
         token_out = hidden * W_token2
         new token vector = v + token_out   (residual connection)
       The resulting matrix (shape [input_dim x embedding_dim]) is stored in ff_residual.
    2. Feed–forward network (channel mixing): For each token,
         pre_act = (token vector) * W_ff1, apply swish, then
         f = hidden * W_ff2, and add a residual connection:
         final = ff_residual + f.
       The intermediate token embeddings are stored in predictions.
    3. Output Projection: For each token, the final scalar output is computed by
         dot(token_embedding, output_proj)
       and stored in final_output.
*/
static inline void forward_pass(Net* net, float* X) {
    int tokens = net->input_dim;   // token count per sample
    int d_dim = net->embedding_dim; // embedding dimension
    int total_tokens = net->batch_size * tokens;
    int token_hidden = net->token_hidden_dim;
    int ff_hidden = net->ff_hidden_dim;

    // --- Token Mixing MLP ---
    // Process each sample independently
    for (int s = 0; s < net->batch_size; s++) {
        float* X_sample = X + s * tokens * d_dim;
        float* out_sample = net->ff_residual + s * tokens * d_dim;
        
        // Transpose the sample matrix to shape [d_dim x tokens]
        float* X_transposed = (float*)malloc(tokens * d_dim * sizeof(float));
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < d_dim; j++) {
                X_transposed[j * tokens + i] = X_sample[i * d_dim + j];
            }
        }

        // Compute hidden = X_transposed * W_token1; shape [d_dim x token_hidden]
        float* hidden = (float*)malloc(d_dim * token_hidden * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d_dim, token_hidden, tokens,
                    1.0f, X_transposed, tokens,
                    net->W_token1, token_hidden,
                    0.0f, hidden, token_hidden);

        // Apply swish activation
        for (int i = 0; i < d_dim * token_hidden; i++) {
            hidden[i] = swishf(hidden[i]);
        }

        // Compute token_out = hidden * W_token2; shape [d_dim x tokens]
        float* token_out = (float*)malloc(d_dim * tokens * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d_dim, tokens, token_hidden,
                    1.0f, hidden, token_hidden,
                    net->W_token2, tokens,
                    0.0f, token_out, tokens);

        // Transpose back and add residual connection: ff_residual = X_sample + token_out^T.
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < d_dim; j++) {
                out_sample[i * d_dim + j] = X_sample[i * d_dim + j] + token_out[j * tokens + i];
            }
        }

        free(X_transposed);
        free(hidden);
        free(token_out);
    }

    // --- Feed-Forward Network (Channel Mixing) ---
    // Compute hidden = swish(ff_residual * W_ff1)
    float* hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    float* pre_act = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->ff_residual, d_dim,
                net->W_ff1, ff_hidden,
                0.0f, pre_act, ff_hidden);
    
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        hidden[i] = swishf(pre_act[i]);
    }
    
    // Compute f = hidden * W_ff2; shape: [total_tokens x d_dim]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, d_dim, ff_hidden,
                1.0f, hidden, ff_hidden,
                net->W_ff2, d_dim,
                0.0f, net->predictions, d_dim);
    
    // Add residual connection from ff_residual:
    for (int i = 0; i < total_tokens * d_dim; i++) {
        net->predictions[i] += net->ff_residual[i];
    }
    
    free(hidden);
    free(pre_act);

    // --- Output Projection ---
    // For each sample and token, compute the scalar output = dot(token_embedding, output_proj).
    for (int s = 0; s < net->batch_size; s++) {
        for (int t = 0; t < tokens; t++) {
            float* token_emb = net->predictions + s * tokens * d_dim + t * d_dim;
            float dot = 0.0f;
            for (int k = 0; k < d_dim; k++) {
                dot += token_emb[k] * net->output_proj[k];
            }
            net->final_output[s * tokens + t] = dot;
        }
    }
}

/*
  calculate_loss: Computes mean–squared error (MSE) loss.
  For each sample and token, loss = 0.5*(prediction - target)^2.
  The average loss is returned and net->final_error is set with the gradient (prediction - target).
  y_target: array of shape [batch_size x output_dim] (flattened) with continuous target values.
*/
static inline float calculate_loss(Net* net, float* y_target) {
    float loss = 0.0f;
    int batch = net->batch_size;
    int total = net->output_dim;  // output_dim equals input_dim.
    int n = batch * total;
    for (int i = 0; i < n; i++) {
        float diff = net->final_output[i] - y_target[i];
        loss += diff * diff;
        net->final_error[i] = diff;  // derivative of 0.5*(diff^2) is diff.
    }
    return loss / n;
}

/*
  zero_gradients: Resets the gradients stored for the learned input projection, token–mixing,
  feed–forward, and output projection weights.
*/
static inline void zero_gradients(Net* net) {
    int token1_size = net->input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * net->input_dim;
    memset(net->W_token1_grad, 0, token1_size * sizeof(float));
    memset(net->W_token2_grad, 0, token2_size * sizeof(float));
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    memset(net->W_ff1_grad, 0, ff1_size * sizeof(float));
    memset(net->W_ff2_grad, 0, ff2_size * sizeof(float));

    int inp_proj_size = net->input_dim * net->embedding_dim;
    memset(net->input_proj_grad, 0, inp_proj_size * sizeof(float));
    memset(net->output_proj_grad, 0, net->embedding_dim * sizeof(float));
}

/*
  backward_pass: Backpropagates gradients through the output projection, feed–forward block,
  token–mixing block, and then through the input projection.
  X is the embedded input (i.e. the projected input from raw scalars).
*/
static inline void backward_pass(Net* net, float* X, float* y_target) {
    int tokens = net->input_dim;
    int d_dim = net->embedding_dim;
    int batch = net->batch_size;
    int total_tokens = batch * tokens;
    int ff_hidden = net->ff_hidden_dim;
    int token_hidden = net->token_hidden_dim;

    zero_gradients(net);

    // --- Backward Pass for Output Projection ---
    // For each sample and token, let token embedding = predictions (of shape d_dim)
    // and final output = dot(token_emb, output_proj). The loss gradient dL/d(final_output) is in final_error.
    // Compute gradients for output_proj and compute d_output = final_error * output_proj to propagate.
    // We accumulate these into an array that will become the gradient for the feed–forward block.
    for (int s = 0; s < batch; s++) {
        for (int t = 0; t < tokens; t++) {
            int idx_out = s * tokens + t; 
            float err = net->final_error[idx_out];
            float* token_emb = net->predictions + s * tokens * d_dim + t * d_dim;
            // Accumulate gradient for output projection.
            for (int k = 0; k < d_dim; k++) {
                net->output_proj_grad[k] += err * token_emb[k];
            }
            // Overwrite net->error with the backpropagated gradient from the output projection.
            for (int k = 0; k < d_dim; k++) {
                net->error[s * tokens * d_dim + t * d_dim + k] = err * net->output_proj[k];
            }
        }
    }

    // --- Backward Pass for Feed-Forward (Channel Mixing) Block ---
    float* hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    float* pre_act = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    
    // Recompute pre-activation: pre_act = ff_residual * W_ff1.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->ff_residual, d_dim,
                net->W_ff1, ff_hidden,
                0.0f, pre_act, ff_hidden);
    
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        hidden[i] = swishf(pre_act[i]);
    }
    
    // Compute d_hidden = error * W_ff2^T
    float* d_hidden = (float*)malloc(total_tokens * ff_hidden * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, ff_hidden, d_dim,
                1.0f, net->error, d_dim,
                net->W_ff2, d_dim,
                0.0f, d_hidden, ff_hidden);
    
    for (int i = 0; i < total_tokens * ff_hidden; i++) {
        d_hidden[i] *= swish_deriv(pre_act[i]);
    }
    
    // W_ff2 gradients
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ff_hidden, d_dim, total_tokens,
                1.0f, hidden, ff_hidden,
                net->error, d_dim,
                1.0f, net->W_ff2_grad, d_dim);
    
    // W_ff1 gradients
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_dim, ff_hidden, total_tokens,
                1.0f, net->ff_residual, d_dim,
                d_hidden, ff_hidden,
                1.0f, net->W_ff1_grad, ff_hidden);
    
    // Compute d_ff_input and update net->error accordingly.
    float* d_ff_input = (float*)malloc(total_tokens * d_dim * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, d_dim, ff_hidden,
                1.0f, d_hidden, ff_hidden,
                net->W_ff1, ff_hidden,
                0.0f, d_ff_input, d_dim);
    
    for (int i = 0; i < total_tokens * d_dim; i++) {
        net->error[i] += d_ff_input[i];
    }
    
    free(hidden);
    free(pre_act);
    free(d_hidden);
    free(d_ff_input);

    // --- Backward Pass for Token-Mixing Block ---
    // Process each sample independently.
    for (int s = 0; s < batch; s++) {
        float* X_sample = X + s * tokens * d_dim;
        float* d_token = net->error + s * tokens * d_dim;
        
        // Transpose input and gradient for this sample.
        float* X_trans = (float*)malloc(d_dim * tokens * sizeof(float));
        float* d_token_trans = (float*)malloc(d_dim * tokens * sizeof(float));
        for (int i = 0; i < tokens; i++) {
            for (int j = 0; j < d_dim; j++) {
                X_trans[j * tokens + i] = X_sample[i * d_dim + j];
                d_token_trans[j * tokens + i] = d_token[i * d_dim + j];
            }
        }

        // Forward pass (again) to get intermediates for token mixing.
        float* pre_token = (float*)malloc(d_dim * token_hidden * sizeof(float));
        float* hidden_token = (float*)malloc(d_dim * token_hidden * sizeof(float));
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d_dim, token_hidden, tokens,
                    1.0f, X_trans, tokens,
                    net->W_token1, token_hidden,
                    0.0f, pre_token, token_hidden);

        for (int i = 0; i < d_dim * token_hidden; i++) {
            hidden_token[i] = swishf(pre_token[i]);
        }

        // Backprop into token mixing.
        float* d_hidden_token = (float*)malloc(d_dim * token_hidden * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    d_dim, token_hidden, tokens,
                    1.0f, d_token_trans, tokens,
                    net->W_token2, tokens,
                    0.0f, d_hidden_token, token_hidden);

        for (int i = 0; i < d_dim * token_hidden; i++) {
            d_hidden_token[i] *= swish_deriv(pre_token[i]);
        }

        // W_token2 gradients.
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    token_hidden, tokens, d_dim,
                    1.0f, hidden_token, token_hidden,
                    d_token_trans, tokens,
                    1.0f, net->W_token2_grad, tokens);

        // W_token1 gradients.
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    tokens, token_hidden, d_dim,
                    1.0f, X_trans, tokens,
                    d_hidden_token, token_hidden,
                    1.0f, net->W_token1_grad, token_hidden);

        free(X_trans);
        free(d_token_trans);
        free(pre_token);
        free(hidden_token);
        free(d_hidden_token);
    }

    // --- Backward Pass for Input Projection ---
    // Propagate gradients from the initial projection.
    // Recall that: X_projected[i, j, :] = raw_input[i, j] * input_proj[j, :].
    // And dX/d(input_proj) = raw_input. So for each sample and token:
    //   input_proj_grad[j, k] += raw_input[i, j] * (gradient at X_projected[i, j, k]).
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < tokens; j++) {
            float raw_val = 0.0f; // raw_input value for sample i, token j.
            // The caller should have provided raw_input used in project_input.
            // Here we retrieve it from X. Note: X[i, j, k] = raw_input[i,j] * input_proj[j,k].
            // We can recover raw_input by dividing X[i,j,*] by input_proj[j,*] but since input_proj is not constant over channels,
            // we assume the same raw scalar multiplied all channels. So we average:
            float sum = 0.0f;
            for (int k = 0; k < d_dim; k++) {
                sum += X[i * tokens * d_dim + j * d_dim + k] / (net->input_proj[j * d_dim + k] + 1e-6f);
            }
            raw_val = sum / d_dim;
            // Now accumulate gradients.
            for (int k = 0; k < d_dim; k++) {
                net->input_proj_grad[j * d_dim + k] += raw_val * net->error[i * tokens * d_dim + j * d_dim + k];
            }
        }
    }
}

/*
  update_weights: Updates all network weights (input projection, token–mixing parameters,
  feed–forward parameters, and output projection) using AdamW.
  learning_rate is the base learning rate.
*/
static inline void update_weights(Net* net, float learning_rate) {
    net->t++;
    float beta1_t = powf(net->beta1, net->t);
    float beta2_t = powf(net->beta2, net->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update token–mixing parameters.
    int token1_size = net->input_dim * net->token_hidden_dim;
    for (int i = 0; i < token1_size; i++) {
        float grad = net->W_token1_grad[i] / net->batch_size;
        net->W_token1_m[i] = net->beta1 * net->W_token1_m[i] + (1.0f - net->beta1) * grad;
        net->W_token1_v[i] = net->beta2 * net->W_token1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_token1_m[i] / (sqrtf(net->W_token1_v[i]) + net->epsilon);
        net->W_token1[i] = net->W_token1[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    int token2_size = net->token_hidden_dim * net->input_dim;
    for (int i = 0; i < token2_size; i++) {
        float grad = net->W_token2_grad[i] / net->batch_size;
        net->W_token2_m[i] = net->beta1 * net->W_token2_m[i] + (1.0f - net->beta1) * grad;
        net->W_token2_v[i] = net->beta2 * net->W_token2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_token2_m[i] / (sqrtf(net->W_token2_v[i]) + net->epsilon);
        net->W_token2[i] = net->W_token2[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    // Update Feed–Forward weights.
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    for (int i = 0; i < ff1_size; i++) {
        float grad = net->W_ff1_grad[i] / net->batch_size;
        net->W_ff1_m[i] = net->beta1 * net->W_ff1_m[i] + (1.0f - net->beta1) * grad;
        net->W_ff1_v[i] = net->beta2 * net->W_ff1_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_ff1_m[i] / (sqrtf(net->W_ff1_v[i]) + net->epsilon);
        net->W_ff1[i] = net->W_ff1[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    for (int i = 0; i < ff2_size; i++) {
        float grad = net->W_ff2_grad[i] / net->batch_size;
        net->W_ff2_m[i] = net->beta1 * net->W_ff2_m[i] + (1.0f - net->beta1) * grad;
        net->W_ff2_v[i] = net->beta2 * net->W_ff2_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->W_ff2_m[i] / (sqrtf(net->W_ff2_v[i]) + net->epsilon);
        net->W_ff2[i] = net->W_ff2[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    // Update learned input projection.
    int inp_proj_size = net->input_dim * net->embedding_dim;
    for (int i = 0; i < inp_proj_size; i++) {
        float grad = net->input_proj_grad[i] / net->batch_size;
        net->input_proj_m[i] = net->beta1 * net->input_proj_m[i] + (1.0f - net->beta1) * grad;
        net->input_proj_v[i] = net->beta2 * net->input_proj_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->input_proj_m[i] / (sqrtf(net->input_proj_v[i]) + net->epsilon);
        net->input_proj[i] = net->input_proj[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
    // Update learned output projection.
    for (int i = 0; i < net->embedding_dim; i++) {
        float grad = net->output_proj_grad[i] / net->batch_size;
        net->output_proj_m[i] = net->beta1 * net->output_proj_m[i] + (1.0f - net->beta1) * grad;
        net->output_proj_v[i] = net->beta2 * net->output_proj_v[i] + (1.0f - net->beta2) * grad * grad;
        float update_val = alpha_t * net->output_proj_m[i] / (sqrtf(net->output_proj_v[i]) + net->epsilon);
        net->output_proj[i] = net->output_proj[i] * (1.0f - learning_rate * net->weight_decay) - update_val;
    }
}

/*
  save_model: Saves network dimensions, weights, Adam state, and all learned parameters into a binary file.
*/
static inline void save_model(Net* net, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->embedding_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->batch_size, sizeof(int), 1, file);

    // Save input projection.
    int inp_proj_size = net->input_dim * net->embedding_dim;
    fwrite(net->input_proj, sizeof(float), inp_proj_size, file);
    fwrite(net->input_proj_m, sizeof(float), inp_proj_size, file);
    fwrite(net->input_proj_v, sizeof(float), inp_proj_size, file);

    // Save token–mixing parameters.
    fwrite(&net->token_hidden_dim, sizeof(int), 1, file);
    int token1_size = net->input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * net->input_dim;
    fwrite(net->W_token1, sizeof(float), token1_size, file);
    fwrite(net->W_token2, sizeof(float), token2_size, file);
    fwrite(net->W_token1_m, sizeof(float), token1_size, file);
    fwrite(net->W_token1_v, sizeof(float), token1_size, file);
    fwrite(net->W_token2_m, sizeof(float), token2_size, file);
    fwrite(net->W_token2_v, sizeof(float), token2_size, file);

    // Save Adam state counter.
    fwrite(&net->t, sizeof(int), 1, file);

    // Save feed–forward network parameters.
    fwrite(&net->ff_hidden_dim, sizeof(int), 1, file);
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    fwrite(net->W_ff1, sizeof(float), ff1_size, file);
    fwrite(net->W_ff2, sizeof(float), ff2_size, file);
    fwrite(net->W_ff1_m, sizeof(float), ff1_size, file);
    fwrite(net->W_ff1_v, sizeof(float), ff1_size, file);
    fwrite(net->W_ff2_m, sizeof(float), ff2_size, file);
    fwrite(net->W_ff2_v, sizeof(float), ff2_size, file);

    // Save learned output projection.
    fwrite(net->output_proj, sizeof(float), net->embedding_dim, file);
    fwrite(net->output_proj_m, sizeof(float), net->embedding_dim, file);
    fwrite(net->output_proj_v, sizeof(float), net->embedding_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

/*
  load_model: Loads network dimensions, weights, Adam state, and all learned parameters from a binary file.
*/
static inline Net* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    int input_dim, embedding_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);

    Net* net = init_net(input_dim, embedding_dim, output_dim, batch_size);

    int inp_proj_size = net->input_dim * net->embedding_dim;
    fread(net->input_proj, sizeof(float), inp_proj_size, file);
    fread(net->input_proj_m, sizeof(float), inp_proj_size, file);
    fread(net->input_proj_v, sizeof(float), inp_proj_size, file);

    fread(&net->token_hidden_dim, sizeof(int), 1, file);
    int token1_size = net->input_dim * net->token_hidden_dim;
    int token2_size = net->token_hidden_dim * net->input_dim;
    fread(net->W_token1, sizeof(float), token1_size, file);
    fread(net->W_token2, sizeof(float), token2_size, file);
    fread(net->W_token1_m, sizeof(float), token1_size, file);
    fread(net->W_token1_v, sizeof(float), token1_size, file);
    fread(net->W_token2_m, sizeof(float), token2_size, file);
    fread(net->W_token2_v, sizeof(float), token2_size, file);

    fread(&net->t, sizeof(int), 1, file);

    fread(&net->ff_hidden_dim, sizeof(int), 1, file);
    int ff1_size = net->embedding_dim * net->ff_hidden_dim;
    int ff2_size = net->ff_hidden_dim * net->embedding_dim;
    fread(net->W_ff1, sizeof(float), ff1_size, file);
    fread(net->W_ff2, sizeof(float), ff2_size, file);
    fread(net->W_ff1_m, sizeof(float), ff1_size, file);
    fread(net->W_ff1_v, sizeof(float), ff1_size, file);
    fread(net->W_ff2_m, sizeof(float), ff2_size, file);
    fread(net->W_ff2_v, sizeof(float), ff2_size, file);

    fread(net->output_proj, sizeof(float), net->embedding_dim, file);
    fread(net->output_proj_m, sizeof(float), net->embedding_dim, file);
    fread(net->output_proj_v, sizeof(float), net->embedding_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    return net;
}

#endif /* MLP_H */