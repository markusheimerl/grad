#ifndef DATA_CUH
#define DATA_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Constants for data generation and handling
#define INPUT_RANGE_MIN -3.0
#define INPUT_RANGE_MAX 3.0
#define MAX_SYNTHETIC_OUTPUTS 4
#define MAX_HEADER_LENGTH 1024

// Main data structure for holding dataset
typedef struct { 
    double **X;        // Input features
    double **y;        // Output values
    int n;             // Number of samples
    int fx;            // Number of features
    int fy;            // Number of outputs
    char **headers;    // Column headers
} Data;

// Synthetic function for generating test data
static double synth_fn(const double* x, int fx, int dim) {
    switch (dim % MAX_SYNTHETIC_OUTPUTS) {
        case 0:
            return sin(x[0 % fx] * 2) * cos(x[1 % fx] * 1.5) + 
                   pow(x[2 % fx], 2) * x[3 % fx];
        case 1:
            return tanh(x[0 % fx] + x[1 % fx]) * sin(x[2 % fx] * 2) + 
                   log(fabs(x[3 % fx]) + 1) * cos(x[4 % fx]);
        case 2:
            return exp(-pow(x[0 % fx] - 0.5, 2)) * sin(x[1 % fx] * 3) + 
                   pow(cos(x[2 % fx]), 2) * x[3 % fx];
        default:
            return pow(sin(x[0 % fx] * x[1 % fx]), 2) + 
                   0.4 * tanh(x[2 % fx] + x[3 % fx] * x[4 % fx]);
    }
}

// Helper function to free allocated memory
static void free_data_internal(Data* d, int rows) {
    if (!d) return;
    
    if (d->X) {
        for (int i = 0; i < rows; i++) {
            if (d->X[i]) free(d->X[i]);
        }
        free(d->X);
    }
    
    if (d->y) {
        for (int i = 0; i < rows; i++) {
            if (d->y[i]) free(d->y[i]);
        }
        free(d->y);
    }
    
    if (d->headers) {
        for (int i = 0; i < d->fx + d->fy; i++) {
            if (d->headers[i]) free(d->headers[i]);
        }
        free(d->headers);
    }
    
    free(d);
}

// Generate synthetic dataset
Data* synth(int n, int fx, int fy, double noise) {
    if (n <= 0 || fx <= 0 || fy <= 0) return NULL;
    
    Data* d = (Data*)calloc(1, sizeof(Data));
    if (!d) return NULL;
    
    d->n = n;
    d->fx = fx;
    d->fy = fy;
    
    // Allocate headers
    d->headers = (char**)malloc((fx + fy) * sizeof(char*));
    if (!d->headers) {
        free_data_internal(d, 0);
        return NULL;
    }
    
    // Create header names
    for (int i = 0; i < fx + fy; i++) {
        d->headers[i] = (char*)malloc(MAX_HEADER_LENGTH);
        if (!d->headers[i]) {
            free_data_internal(d, 0);
            return NULL;
        }
        if (snprintf(d->headers[i], MAX_HEADER_LENGTH, "%c%d", 
                    i < fx ? 'x' : 'y', i < fx ? i : i-fx) < 0) {
            free_data_internal(d, 0);
            return NULL;
        }
    }
    
    // Allocate data arrays
    d->X = (double**)malloc(n * sizeof(double*));
    d->y = (double**)malloc(n * sizeof(double*));
    if (!d->X || !d->y) {
        free_data_internal(d, 0);
        return NULL;
    }
    
    // Generate data
    for (int i = 0; i < n; i++) {
        d->X[i] = (double*)malloc(fx * sizeof(double));
        d->y[i] = (double*)malloc(fy * sizeof(double));
        if (!d->X[i] || !d->y[i]) {
            free_data_internal(d, i);
            return NULL;
        }
        
        // Generate random inputs
        for (int j = 0; j < fx; j++) {
            d->X[i][j] = (double)rand()/RAND_MAX * 
                        (INPUT_RANGE_MAX - INPUT_RANGE_MIN) + INPUT_RANGE_MIN;
        }
        
        // Generate outputs
        for (int j = 0; j < fy; j++) {
            d->y[i][j] = synth_fn(d->X[i], fx, j) + 
                         ((double)rand()/RAND_MAX - 0.5) * noise;
        }
    }
    
    return d;
}

// Save dataset to CSV file
void save_csv(const char* filename, Data* d) {
    if (!filename || !d) return;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    
    // Write headers
    for (int i = 0; i < d->fx + d->fy; i++) {
        fprintf(fp, "%s%c", d->headers[i], i < d->fx + d->fy - 1 ? ',' : '\n');
    }
    
    // Write data
    for (int i = 0; i < d->n; i++) {
        for (int j = 0; j < d->fx; j++) {
            fprintf(fp, "%.17f,", d->X[i][j]);
        }
        for (int j = 0; j < d->fy; j++) {
            fprintf(fp, "%.17f%c", d->y[i][j], j == d->fy-1 ? '\n' : ',');
        }
    }
    
    fclose(fp);
}

// Load dataset from CSV file
Data* load_csv(const char* filename, int fx, int fy) {
    if (!filename || fx <= 0 || fy <= 0) return NULL;
    
    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;
    
    char line[4096];
    Data* d = (Data*)calloc(1, sizeof(Data));
    if (!d || !fgets(line, sizeof(line), fp)) {
        if (d) free(d);
        fclose(fp);
        return NULL;
    }
    
    d->fx = fx;
    d->fy = fy;
    
    // Allocate and read headers
    d->headers = (char**)malloc((fx + fy) * sizeof(char*));
    if (!d->headers) {
        free_data_internal(d, 0);
        fclose(fp);
        return NULL;
    }
    
    char* token = strtok(line, ",\n");
    for (int i = 0; i < fx + fy; i++) {
        if (!token) {
            free_data_internal(d, 0);
            fclose(fp);
            return NULL;
        }
        d->headers[i] = strdup(token);
        if (!d->headers[i]) {
            free_data_internal(d, 0);
            fclose(fp);
            return NULL;
        }
        token = strtok(NULL, ",\n");
    }
    
    // Count number of data rows
    d->n = 0;
    while (fgets(line, sizeof(line), fp)) {
        d->n++;
    }
    
    // Allocate data arrays
    d->X = (double**)malloc(d->n * sizeof(double*));
    d->y = (double**)malloc(d->n * sizeof(double*));
    if (!d->X || !d->y) {
        free_data_internal(d, 0);
        fclose(fp);
        return NULL;
    }
    
    // Read data
    rewind(fp);
    if (!fgets(line, sizeof(line), fp)) {  // Skip header line
        free_data_internal(d, 0);
        fclose(fp);
        return NULL;
    }
    
    for (int i = 0; i < d->n; i++) {
        if (!fgets(line, sizeof(line), fp)) {
            free_data_internal(d, i);
            fclose(fp);
            return NULL;
        }
        
        d->X[i] = (double*)malloc(fx * sizeof(double));
        d->y[i] = (double*)malloc(fy * sizeof(double));
        if (!d->X[i] || !d->y[i]) {
            free_data_internal(d, i);
            fclose(fp);
            return NULL;
        }
        
        token = strtok(line, ",");
        for (int j = 0; j < fx + fy; j++) {
            if (!token) {
                free_data_internal(d, i);
                fclose(fp);
                return NULL;
            }
            if (j < fx) {
                d->X[i][j] = atof(token);
            } else {
                d->y[i][j-fx] = atof(token);
            }
            token = strtok(NULL, ",");
        }
    }
    
    fclose(fp);
    return d;
}

// Free all allocated memory
void free_data(Data* d) {
    if (d) free_data_internal(d, d->n);
}

#endif