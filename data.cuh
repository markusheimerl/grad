#ifndef DATA_CUH
#define DATA_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_MIN -5.0
#define INPUT_MAX 5.0
#define OUTPUT_MIN 30.0
#define OUTPUT_MAX 70.0

typedef struct {
    float ***inputs;       // [sequence][timesteps][features]
    float **targets;       // [sequence][features]
    int n_sequences;        // number of sequences
    int sequence_length;    // timesteps per sequence
    int n_inputs;          // number of input features
    int n_outputs;         // number of output features
    char **feature_names;   // Names of all features
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
            // Weighted combination with sine modulation
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
            // Count local maxima
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
            // Measure rate of change
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
            // Count threshold crossings
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
    
    // Scale to output range
    return ((result + 1.0) / 2.0) * (OUTPUT_MAX - OUTPUT_MIN) + OUTPUT_MIN;
}

static Dataset* generate_data(
    int n_sequences,
    int sequence_length,
    int n_inputs,
    int n_outputs,
    float noise_level
) {
    Dataset* data = (Dataset*)malloc(sizeof(Dataset));
    data->n_sequences = n_sequences;
    data->sequence_length = sequence_length;
    data->n_inputs = n_inputs;
    data->n_outputs = n_outputs;
    
    // Allocate memory
    data->inputs = (float***)malloc(n_sequences * sizeof(float**));
    data->targets = (float**)malloc(n_sequences * sizeof(float*));
    
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
    
    // Generate data
    for(int i = 0; i < n_sequences; i++) {
        // Generate input sequence
        data->inputs[i] = (float**)malloc(sequence_length * sizeof(float*));
        for(int t = 0; t < sequence_length; t++) {
            data->inputs[i][t] = (float*)malloc(n_inputs * sizeof(float));
            
            for(int f = 0; f < n_inputs; f++) {
                if(t == 0) {
                    data->inputs[i][t][f] = (float)rand()/RAND_MAX * 
                                          (INPUT_MAX - INPUT_MIN) + INPUT_MIN;
                } else {
                    // Add some temporal correlation
                    data->inputs[i][t][f] = 0.1 * data->inputs[i][t-1][f] + 
                                          0.9 * ((float)rand()/RAND_MAX * 
                                          (INPUT_MAX - INPUT_MIN) + INPUT_MIN);
                }
            }
        }
        
        // Generate targets
        data->targets[i] = (float*)malloc(n_outputs * sizeof(float));
        for(int f = 0; f < n_outputs; f++) {
            data->targets[i][f] = compute_pattern(patterns[f], data->inputs[i], 
                                                sequence_length, n_inputs, f);
            // Add noise
            data->targets[i][f] += ((float)rand()/RAND_MAX - 0.5) * noise_level;
        }
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
    for(int i = 0; i < data->n_sequences; i++) {
        for(int t = 0; t < data->sequence_length; t++) {
            fprintf(fp, "%d,%d", i, t);
            
            // Write inputs
            for(int f = 0; f < data->n_inputs; f++) {
                fprintf(fp, ",%.6f", data->inputs[i][t][f]);
            }
            
            // Write targets (only for last timestep)
            if(t == data->sequence_length - 1) {
                for(int f = 0; f < data->n_outputs; f++) {
                    fprintf(fp, ",%.6f", data->targets[i][f]);
                }
            } else {
                for(int f = 0; f < data->n_outputs; f++) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
        }
    }
    
    fclose(fp);
}

static void free_dataset(Dataset* data) {
    for(int i = 0; i < data->n_sequences; i++) {
        for(int t = 0; t < data->sequence_length; t++) {
            free(data->inputs[i][t]);
        }
        free(data->inputs[i]);
        free(data->targets[i]);
    }
    
    for(int i = 0; i < data->n_inputs + data->n_outputs; i++) {
        free(data->feature_names[i]);
    }
    
    free(data->feature_names);
    free(data->inputs);
    free(data->targets);
    free(data);
}

#endif