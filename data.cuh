#ifndef DATA_CUH
#define DATA_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_RANGE_MIN -3.0
#define INPUT_RANGE_MAX 3.0
#define MAX_SYNTHETIC_OUTPUTS 4
#define MAX_HEADER_LENGTH 1024

typedef struct { 
    double **X, **y;
    int n, fx, fy;
    char **headers;
} Data;

static double synth_fn(const double* x, int fx, int dim) {
    switch(dim % MAX_SYNTHETIC_OUTPUTS) {
        case 0: 
            return sin(x[0 % fx]*2)*cos(x[1 % fx]*1.5) + 
                   pow(x[2 % fx],2)*x[3 % fx] + 
                   exp(-pow(x[4 % fx]-x[5 % fx],2)) + 
                   0.5*sin(x[6 % fx]*x[7 % fx]*M_PI) +
                   tanh(x[8 % fx] + x[9 % fx]) +
                   0.3*cos(x[10 % fx]*x[11 % fx]) +
                   0.2*pow(x[12 % fx], 2) +
                   x[13 % fx]*sin(x[14 % fx]);
            
        case 1: 
            return tanh(x[0 % fx]+x[1 % fx])*sin(x[2 % fx]*2) + 
                   log(fabs(x[3 % fx])+1)*cos(x[4 % fx]) + 
                   0.3*pow(x[5 % fx]-x[6 % fx],3) +
                   exp(-pow(x[7 % fx],2)) +
                   sin(x[8 % fx]*x[9 % fx]*0.5) +
                   0.4*cos(x[10 % fx] + x[11 % fx]) +
                   pow(x[12 % fx]*x[13 % fx], 2) +
                   0.1*x[14 % fx];
            
        case 2: 
            return exp(-pow(x[0 % fx]-0.5,2))*sin(x[1 % fx]*3) + 
                   pow(cos(x[2 % fx]),2)*x[3 % fx] + 
                   0.2*sinh(x[4 % fx]*x[5 % fx]) +
                   0.5*tanh(x[6 % fx] + x[7 % fx]) +
                   pow(x[8 % fx], 3)*0.1 +
                   cos(x[9 % fx]*x[10 % fx]*M_PI) +
                   0.3*exp(-pow(x[11 % fx]-x[12 % fx],2)) +
                   0.2*(x[13 % fx] + x[14 % fx]);
            
        case 3:
            return pow(sin(x[0 % fx]*x[1 % fx]), 2) +
                   0.4*tanh(x[2 % fx] + x[3 % fx]*x[4 % fx]) +
                   exp(-fabs(x[5 % fx]-x[6 % fx])) +
                   0.3*cos(x[7 % fx]*x[8 % fx]*2) +
                   pow(x[9 % fx], 2)*sin(x[10 % fx]) +
                   0.2*log(fabs(x[11 % fx]*x[12 % fx])+1) +
                   0.1*(x[13 % fx] - x[14 % fx]);
            
        default: 
            return 0.0;
    }
}

static void free_partial_data(Data* d, int allocated_rows) {
    if (d->X) {
        for(int i = 0; i < allocated_rows; i++) {
            free(d->X[i]);
        }
        free(d->X);
    }
    if (d->y) {
        for(int i = 0; i < allocated_rows; i++) {
            free(d->y[i]);
        }
        free(d->y);
    }
    if (d->headers) {
        for(int i = 0; i < d->fx + d->fy; i++) {
            free(d->headers[i]);
        }
        free(d->headers);
    }
    free(d);
}

Data* synth(int n, int fx, int fy, double noise) {
    if (n <= 0 || fx <= 0 || fy <= 0) return NULL;
    
    Data* d = (Data*)malloc(sizeof(Data));
    if (!d) return NULL;
    
    d->n = n; d->fx = fx; d->fy = fy;
    d->headers = NULL;
    d->X = NULL;
    d->y = NULL;
    
    // Allocate headers
    d->headers = (char**)malloc((fx + fy) * sizeof(char*));
    if (!d->headers) {
        free_partial_data(d, 0);
        return NULL;
    }
    
    for(int i = 0; i < fx + fy; i++) {
        d->headers[i] = (char*)malloc(MAX_HEADER_LENGTH);
        if (!d->headers[i]) {
            free_partial_data(d, i);
            return NULL;
        }
        int written = snprintf(d->headers[i], MAX_HEADER_LENGTH, 
                             "%c%d", i < fx ? 'x' : 'y', i < fx ? i : i-fx);
        if (written < 0 || written >= MAX_HEADER_LENGTH) {
            free_partial_data(d, i + 1);
            return NULL;
        }
    }
    
    // Allocate data arrays
    d->X = (double**)malloc(n * sizeof(double*));
    if (!d->X) {
        free_partial_data(d, fx + fy);
        return NULL;
    }
    
    d->y = (double**)malloc(n * sizeof(double*));
    if (!d->y) {
        free_partial_data(d, 0);
        return NULL;
    }
    
    for(int i = 0; i < n; i++) {
        d->X[i] = (double*)malloc(fx * sizeof(double));
        if (!d->X[i]) {
            free_partial_data(d, i);
            return NULL;
        }
        
        d->y[i] = (double*)malloc(fy * sizeof(double));
        if (!d->y[i]) {
            free(d->X[i]);
            free_partial_data(d, i);
            return NULL;
        }
        
        for(int j = 0; j < fx; j++) {
            d->X[i][j] = (double)rand()/RAND_MAX * 
                        (INPUT_RANGE_MAX - INPUT_RANGE_MIN) + INPUT_RANGE_MIN;
        }
        
        for(int j = 0; j < fy; j++) {
            d->y[i][j] = synth_fn(d->X[i], fx, j);
            d->y[i][j] += ((double)rand()/RAND_MAX - 0.5) * noise;
        }
    }
    return d;
}

void save_csv(const char* f, Data* d) {
    if (!f || !d) return;
    
    FILE* fp = fopen(f, "w");
    if(!fp) return;
    
    for(int i = 0; i < d->fx + d->fy; i++) {
        if (fprintf(fp, "%s%c", d->headers[i], 
                   i < d->fx + d->fy - 1 ? ',' : '\n') < 0) {
            fclose(fp);
            return;
        }
    }
    
    for(int i = 0; i < d->n; i++) {
        for(int j = 0; j < d->fx; j++) {
            if (fprintf(fp, "%.17f,", d->X[i][j]) < 0) {
                fclose(fp);
                return;
            }
        }
        for(int j = 0; j < d->fy; j++) {
            if (fprintf(fp, "%.17f%c", d->y[i][j], 
                       j == d->fy-1 ? '\n' : ',') < 0) {
                fclose(fp);
                return;
            }
        }
    }
    fclose(fp);
}

Data* load_csv(const char* f, int fx, int fy) {
    if (!f || fx <= 0 || fy <= 0) return NULL;
    
    FILE* fp = fopen(f, "r");
    if(!fp) return NULL;
    
    Data* d = (Data*)malloc(sizeof(Data));
    if (!d) {
        fclose(fp);
        return NULL;
    }
    
    d->fx = fx; d->fy = fy;
    d->headers = NULL;
    d->X = NULL;
    d->y = NULL;
    
    char line[4096];
    char* result = fgets(line, sizeof(line), fp);
    if (!result) {
        fclose(fp);
        free(d);
        return NULL;
    }
    
    d->headers = (char**)malloc((fx + fy) * sizeof(char*));
    if (!d->headers) {
        fclose(fp);
        free(d);
        return NULL;
    }
    
    char* token = strtok(line, ",\n");
    for(int i = 0; i < fx + fy; i++) {
        if (!token) {
            fclose(fp);
            free_partial_data(d, i);
            return NULL;
        }
        d->headers[i] = strdup(token);
        if (!d->headers[i]) {
            fclose(fp);
            free_partial_data(d, i);
            return NULL;
        }
        token = strtok(NULL, ",\n");
    }
    
    // Count number of data rows
    d->n = 0;
    while(fgets(line, sizeof(line), fp)) d->n++;
    if (d->n == 0) {
        fclose(fp);
        free_partial_data(d, fx + fy);
        return NULL;
    }
    
    // Allocate data arrays
    d->X = (double**)malloc(d->n * sizeof(double*));
    d->y = (double**)malloc(d->n * sizeof(double*));
    if (!d->X || !d->y) {
        fclose(fp);
        free_partial_data(d, fx + fy);
        return NULL;
    }
    
    rewind(fp);
    result = fgets(line, sizeof(line), fp); // Skip header line
    if (!result) {
        fclose(fp);
        free_partial_data(d, fx + fy);
        return NULL;
    }
    
    for(int i = 0; i < d->n; i++) {
        d->X[i] = (double*)malloc(fx * sizeof(double));
        d->y[i] = (double*)malloc(fy * sizeof(double));
        if (!d->X[i] || !d->y[i]) {
            fclose(fp);
            free_partial_data(d, i);
            return NULL;
        }
        
        result = fgets(line, sizeof(line), fp);
        if (!result) {
            fclose(fp);
            free_partial_data(d, i);
            return NULL;
        }
        
        token = strtok(line, ",");
        if (!token) {
            fclose(fp);
            free_partial_data(d, i);
            return NULL;
        }
        
        for(int j = 0; j < fx; j++) {
            d->X[i][j] = atof(token);
            token = strtok(NULL, ",");
            if (!token && j < fx - 1) {
                fclose(fp);
                free_partial_data(d, i);
                return NULL;
            }
        }
        
        for(int j = 0; j < fy; j++) {
            if (!token) {
                fclose(fp);
                free_partial_data(d, i);
                return NULL;
            }
            d->y[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    
    fclose(fp);
    return d;
}

void free_data(Data* d) {
    if (!d) return;
    free_partial_data(d, d->n);
}

#endif