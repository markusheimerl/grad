#ifndef DATA_CUH
#define DATA_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_MIN -2.0
#define INPUT_MAX 2.0
#define OUTPUT_MIN -2.0
#define OUTPUT_MAX 2.0

typedef struct {
    float **inputs;        // [global_timesteps][features]
    float **targets;       // [global_timesteps][features]
    int window_size;       // size of sliding window
    int total_timesteps;   // total number of timesteps (window_size + n_sequences - 1)
    int n_sequences;       // number of complete sequences (windows)
    int n_inputs;         // number of input features
    int n_outputs;        // number of output features
    char **feature_names;  // Names of all features
} Dataset;

// Pattern types for generating synthetic data
typedef enum {
    PATTERN_A,    // Weighted combination of inputs
    PATTERN_B,    // Peak detection based
    PATTERN_C,    // Change frequency based
    PATTERN_D,    // Threshold crossing based
    N_PATTERNS
} PatternType;

static __host__ __device__ float compute_pattern(
    PatternType pattern,
    float** sequence,
    int sequence_length,
    int n_inputs,
    int output_idx
) {
    float result;
    
    switch(pattern) {
        case PATTERN_A: {
            float sum = 0;
            for(int t = 0; t < sequence_length; t++) {
                for(int f = 0; f < n_inputs; f++) {
                    sum += sequence[t][f] * sin(t * M_PI / sequence_length);
                }
            }
            result = tanh(sum / (sequence_length * n_inputs));
            break;
        }
        
        case PATTERN_B: {
            int peak_count = 0;
            for(int t = 1; t < sequence_length-1; t++) {
                for(int f = 0; f < n_inputs; f++) {
                    if(sequence[t][f] > sequence[t-1][f] && 
                       sequence[t][f] > sequence[t+1][f]) {
                        peak_count++;
                    }
                }
            }
            result = sin(peak_count * M_PI / (sequence_length * n_inputs));
            break;
        }
        
        case PATTERN_C: {
            float change_sum = 0;
            for(int t = 1; t < sequence_length; t++) {
                for(int f = 0; f < n_inputs; f++) {
                    change_sum += fabs(sequence[t][f] - sequence[t-1][f]);
                }
            }
            result = tanh(change_sum / (sequence_length * n_inputs));
            break;
        }
        
        case PATTERN_D: {
            int cross_count = 0;
            float threshold = 0.5;
            for(int t = 1; t < sequence_length; t++) {
                for(int f = 0; f < n_inputs; f++) {
                    if((sequence[t][f] > threshold && sequence[t-1][f] <= threshold) ||
                       (sequence[t][f] < threshold && sequence[t-1][f] >= threshold)) {
                        cross_count++;
                    }
                }
            }
            result = cos(cross_count * M_PI / (sequence_length * n_inputs));
            break;
        }
        
        default:
            result = 0.0;
    }
    
    return ((result + 1.0) / 2.0) * (OUTPUT_MAX - OUTPUT_MIN) + OUTPUT_MIN;
}

static Dataset* generate_data(
    int n_sequences,
    int window_size,
    int n_inputs,
    int n_outputs,
    float noise_level
) {
    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    data->window_size = window_size;
    data->n_sequences = n_sequences;
    data->n_inputs = n_inputs;
    data->n_outputs = n_outputs;
    data->total_timesteps = window_size + n_sequences - 1;
    
    // Allocate memory for continuous input series
    data->inputs = (float**)malloc(data->total_timesteps * sizeof(float*));
    for(int t = 0; t < data->total_timesteps; t++) {
        data->inputs[t] = (float*)malloc(n_inputs * sizeof(float));
    }
    
    // Allocate memory for outputs (one per timestep, including initial zeros)
    data->targets = (float**)malloc(data->total_timesteps * sizeof(float*));
    for(int t = 0; t < data->total_timesteps; t++) {
        data->targets[t] = (float*)malloc(n_outputs * sizeof(float));
        // Initialize all to zero
        for(int f = 0; f < n_outputs; f++) {
            data->targets[t][f] = 0.0f;
        }
    }
    
    // Generate feature names
    data->feature_names = (char**)malloc((n_inputs + n_outputs) * sizeof(char*));
    for(int i = 0; i < n_inputs + n_outputs; i++) {
        data->feature_names[i] = (char*)malloc(8);
        sprintf(data->feature_names[i], "%c%d", 
                i < n_inputs ? 'x' : 'y', 
                i < n_inputs ? i : i-n_inputs);
    }
    
    // Assign random patterns to outputs
    PatternType* patterns = (PatternType*)malloc(n_outputs * sizeof(PatternType));
    for(int i = 0; i < n_outputs; i++) {
        patterns[i] = (PatternType)(rand() % N_PATTERNS);
    }
    
    // Generate continuous input series
    for(int t = 0; t < data->total_timesteps; t++) {
        for(int f = 0; f < n_inputs; f++) {
            if(t == 0) {
                data->inputs[t][f] = (float)rand()/RAND_MAX * 
                                   (INPUT_MAX - INPUT_MIN) + INPUT_MIN;
            } else {
                data->inputs[t][f] = 0.8 * data->inputs[t-1][f] + 
                                   0.2 * ((float)rand()/RAND_MAX * 
                                   (INPUT_MAX - INPUT_MIN) + INPUT_MIN);
            }
        }
    }
    
    // Generate targets for each complete window
    for(int t = window_size - 1; t < data->total_timesteps; t++) {
        // Create temporary window array for pattern computation
        float** window = (float**)malloc(window_size * sizeof(float*));
        for(int w = 0; w < window_size; w++) {
            window[w] = data->inputs[t - window_size + 1 + w];
        }
        
        // Compute outputs for this window
        for(int f = 0; f < n_outputs; f++) {
            data->targets[t][f] = compute_pattern(patterns[f], window, 
                                                window_size, n_inputs, f);
            // Add noise
            data->targets[t][f] += ((float)rand()/RAND_MAX - 0.5) * noise_level;
        }
        
        free(window);
    }
    
    free(patterns);
    return data;
}

static void save_csv(const char* filename, Dataset* data) {
    FILE* fp = fopen(filename, "w");
    if(!fp) return;
    
    // Write header
    fprintf(fp, "sequence,timestep");
    for(int i = 0; i < data->n_inputs + data->n_outputs; i++) {
        fprintf(fp, ",%s", data->feature_names[i]);
    }
    fprintf(fp, "\n");
    
    // Write data
    for(int t = 0; t < data->total_timesteps; t++) {
        // Determine sequence number (0 until window_size-1, then increment)
        int seq_num = t < data->window_size - 1 ? 0 : t - data->window_size + 1;
        
        fprintf(fp, "%d,%d", seq_num, t);
        
        // Write inputs
        for(int f = 0; f < data->n_inputs; f++) {
            fprintf(fp, ",%.6f", data->inputs[t][f]);
        }
        
        // Write targets (zeros for initial timesteps, computed values after)
        for(int f = 0; f < data->n_outputs; f++) {
            fprintf(fp, ",%.6f", data->targets[t][f]);
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}

static void free_dataset(Dataset* data) {
    for(int t = 0; t < data->total_timesteps; t++) {
        free(data->inputs[t]);
        free(data->targets[t]);
    }
    free(data->inputs);
    free(data->targets);
    
    for(int i = 0; i < data->n_inputs + data->n_outputs; i++) {
        free(data->feature_names[i]);
    }
    free(data->feature_names);
    
    free(data);
}

#endif