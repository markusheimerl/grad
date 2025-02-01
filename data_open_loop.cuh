#ifndef DATA_OPEN_LOOP_CUH
#define DATA_OPEN_LOOP_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_RANGE_MIN -3.0
#define INPUT_RANGE_MAX 3.0
#define OUTPUT_RANGE_MIN 30.0
#define OUTPUT_RANGE_MAX 70.0

typedef struct {
    double ***windows;      // [batch][window_size][input_features]
    double **outputs;       // [batch][output_features]
    int n;                  // number of samples
    int window_size;        // size of sliding window
    int input_features;     // number of input features
    int output_features;    // number of output features
    char **headers;         // Feature names
} OpenLoopData;

// Different pattern types for synthetic data
typedef enum {
    PATTERN_WEIGHTED_SUM,      // Output is weighted sum of window values
    PATTERN_PEAK_DETECT,       // Output depends on peaks in window
    PATTERN_FREQUENCY,         // Output depends on frequency patterns
    PATTERN_THRESHOLD,         // Output depends on threshold crossings
    NUM_PATTERNS
} OpenLoopPattern;

static __host__ __device__ double generate_pattern(
    OpenLoopPattern pattern,
    double** window,
    int window_size,
    int input_features,
    int output_idx
) {
    double raw_output;
    
    switch(pattern) {
        case PATTERN_WEIGHTED_SUM: {
            double sum = 0;
            for(int t = 0; t < window_size; t++) {
                for(int f = 0; f < input_features; f++) {
                    sum += window[t][f] * sin(t * M_PI / window_size);
                }
            }
            raw_output = tanh(sum / (window_size * input_features));
            break;
        }
        
        case PATTERN_PEAK_DETECT: {
            int peaks = 0;
            for(int t = 1; t < window_size-1; t++) {
                for(int f = 0; f < input_features; f++) {
                    if(window[t][f] > window[t-1][f] && 
                       window[t][f] > window[t+1][f]) {
                        peaks++;
                    }
                }
            }
            raw_output = sin(peaks * M_PI / (window_size * input_features));
            break;
        }
        
        case PATTERN_FREQUENCY: {
            double freq = 0;
            for(int t = 1; t < window_size; t++) {
                for(int f = 0; f < input_features; f++) {
                    freq += fabs(window[t][f] - window[t-1][f]);
                }
            }
            raw_output = tanh(freq / (window_size * input_features));
            break;
        }
        
        case PATTERN_THRESHOLD: {
            int crossings = 0;
            double threshold = 0.5;
            for(int t = 1; t < window_size; t++) {
                for(int f = 0; f < input_features; f++) {
                    if((window[t][f] > threshold && window[t-1][f] <= threshold) ||
                       (window[t][f] < threshold && window[t-1][f] >= threshold)) {
                        crossings++;
                    }
                }
            }
            raw_output = cos(crossings * M_PI / (window_size * input_features));
            break;
        }
        
        default:
            raw_output = 0.0;
    }
    
    // Scale the output from [-1,1] to [OUTPUT_RANGE_MIN, OUTPUT_RANGE_MAX]
    return ((raw_output + 1.0) / 2.0) * (OUTPUT_RANGE_MAX - OUTPUT_RANGE_MIN) + OUTPUT_RANGE_MIN;
}

static OpenLoopData* generate_open_loop_data(
    int n_samples,
    int window_size,
    int input_features,
    int output_features,
    double noise
) {
    OpenLoopData* d = (OpenLoopData*)malloc(sizeof(OpenLoopData));
    d->n = n_samples;
    d->window_size = window_size;
    d->input_features = input_features;
    d->output_features = output_features;
    
    // Allocate memory
    d->windows = (double***)malloc(n_samples * sizeof(double**));
    d->outputs = (double**)malloc(n_samples * sizeof(double*));
    
    // Generate headers
    d->headers = (char**)malloc((input_features + output_features) * sizeof(char*));
    for(int i = 0; i < input_features + output_features; i++) {
        d->headers[i] = (char*)malloc(8);
        sprintf(d->headers[i], "%c%d", i < input_features ? 'x' : 'y', 
                i < input_features ? i : i-input_features);
    }
    
    // Generate sequences
    for(int i = 0; i < n_samples; i++) {
        // Allocate window
        d->windows[i] = (double**)malloc(window_size * sizeof(double*));
        for(int t = 0; t < window_size; t++) {
            d->windows[i][t] = (double*)malloc(input_features * sizeof(double));
            
            // Generate random input features with some temporal correlation
            for(int f = 0; f < input_features; f++) {
                if(t == 0) {
                    d->windows[i][t][f] = (double)rand()/RAND_MAX * 
                                        (INPUT_RANGE_MAX - INPUT_RANGE_MIN) + 
                                        INPUT_RANGE_MIN;
                } else {
                    d->windows[i][t][f] = 0.1 * d->windows[i][t-1][f] + 
                                        0.9 * ((double)rand()/RAND_MAX * 
                                        (INPUT_RANGE_MAX - INPUT_RANGE_MIN) + 
                                        INPUT_RANGE_MIN);
                }
            }
        }
        
        // Generate outputs
        d->outputs[i] = (double*)malloc(output_features * sizeof(double));
        for(int f = 0; f < output_features; f++) {
            OpenLoopPattern pattern = (OpenLoopPattern)(rand() % NUM_PATTERNS);
            d->outputs[i][f] = generate_pattern(pattern, d->windows[i], 
                                              window_size, input_features, f);
            d->outputs[i][f] += ((double)rand()/RAND_MAX - 0.5) * noise;
        }
    }
    
    return d;
}

static void save_open_loop_csv(const char* f, OpenLoopData* d) {
    FILE* fp = fopen(f, "w");
    if(!fp) return;
    
    // Write headers
    fprintf(fp, "sequence,timestep");
    for(int i = 0; i < d->input_features + d->output_features; i++) {
        fprintf(fp, ",%s", d->headers[i]);
    }
    fprintf(fp, "\n");
    
    // Write data
    for(int i = 0; i < d->n; i++) {
        for(int t = 0; t < d->window_size; t++) {
            fprintf(fp, "%d,%d", i, t);
            
            // Write input features
            for(int f = 0; f < d->input_features; f++) {
                fprintf(fp, ",%.6f", d->windows[i][t][f]);
            }
            
            // Write outputs (only for last timestep)
            if(t == d->window_size - 1) {
                for(int f = 0; f < d->output_features; f++) {
                    fprintf(fp, ",%.6f", d->outputs[i][f]);
                }
            } else {
                for(int f = 0; f < d->output_features; f++) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
        }
    }
    
    fclose(fp);
}

static void free_open_loop_data(OpenLoopData* d) {
    for(int i = 0; i < d->n; i++) {
        for(int t = 0; t < d->window_size; t++) {
            free(d->windows[i][t]);
        }
        free(d->windows[i]);
        free(d->outputs[i]);
    }
    
    for(int i = 0; i < d->input_features + d->output_features; i++) {
        free(d->headers[i]);
    }
    
    free(d->headers);
    free(d->windows);
    free(d->outputs);
    free(d);
}

#endif