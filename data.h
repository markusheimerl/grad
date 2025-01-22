#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_RANGE_MIN -3.0
#define INPUT_RANGE_MAX 3.0
#define MAX_SYNTHETIC_OUTPUTS 4

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

Data* synth(int n, int fx, int fy, double noise) {
    Data* d = malloc(sizeof(Data));
    d->n = n; d->fx = fx; d->fy = fy;
    
    d->headers = malloc((fx + fy) * sizeof(char*));
    for(int i = 0; i < fx + fy; i++) {
        d->headers[i] = malloc(8);
        sprintf(d->headers[i], "%c%d", i < fx ? 'x' : 'y', i < fx ? i : i-fx);
    }
    
    d->X = malloc(n * sizeof(double*));
    d->y = malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++) {
        d->X[i] = malloc(fx * sizeof(double));
        d->y[i] = malloc(fy * sizeof(double));
        
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
    FILE* fp = fopen(f, "w");
    if(!fp) return;
    
    for(int i = 0; i < d->fx + d->fy; i++)
        fprintf(fp, "%s%c", d->headers[i], i < d->fx + d->fy - 1 ? ',' : '\n');
    
    for(int i = 0; i < d->n; i++) {
        for(int j = 0; j < d->fx; j++) fprintf(fp, "%.17f,", d->X[i][j]);
        for(int j = 0; j < d->fy; j++) fprintf(fp, "%.17f%c", d->y[i][j], j == d->fy-1 ? '\n' : ',');
    }
    fclose(fp);
}

Data* load_csv(const char* f, int fx, int fy) {
    FILE* fp = fopen(f, "r");
    if(!fp) return NULL;
    
    Data* d = malloc(sizeof(Data));
    d->fx = fx; d->fy = fy;
    
    char line[4096];
    fgets(line, sizeof(line), fp);
    
    d->headers = malloc((fx + fy) * sizeof(char*));
    char* token = strtok(line, ",\n");
    for(int i = 0; i < fx + fy; i++) {
        d->headers[i] = strdup(token);
        token = strtok(NULL, ",\n");
    }
    
    d->n = 0;
    while(fgets(line, sizeof(line), fp)) d->n++;
    rewind(fp);
    fgets(line, sizeof(line), fp);
    
    d->X = malloc(d->n * sizeof(double*));
    d->y = malloc(d->n * sizeof(double*));
    for(int i = 0; i < d->n; i++) {
        d->X[i] = malloc(fx * sizeof(double));
        d->y[i] = malloc(fy * sizeof(double));
        fgets(line, sizeof(line), fp);
        token = strtok(line, ",");
        for(int j = 0; j < fx; j++) { d->X[i][j] = atof(token); token = strtok(NULL, ","); }
        for(int j = 0; j < fy; j++) { d->y[i][j] = atof(token); token = strtok(NULL, ","); }
    }
    fclose(fp);
    return d;
}

void free_data(Data* d) {
    for(int i = 0; i < d->n; i++) { free(d->X[i]); free(d->y[i]); }
    for(int i = 0; i < d->fx + d->fy; i++) free(d->headers[i]);
    free(d->headers);
    free(d->X); free(d->y);
    free(d);
}

#endif