#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { double **X, **y; int n, fx, fy; } Data;

char* get_timestamp_filename(const char* prefix) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char *filename = malloc(100);
    strftime(filename, 100, "%Y%m%d_%H%M%S_", t);
    strcat(filename, prefix);
    return filename;
}

double complex_function(double x1, double x2, double x3, double x4, int output) {
    switch(output) {
        case 0:
            return sin(x1 * 2.0) * cos(x2 * 1.5) + 
                   pow(x3, 2) * x4 + 
                   exp(-pow(x1 - x2, 2)) +
                   0.5 * sin(x1 * x2 * 3.14159);
        case 1:
            return tanh(x1 + x2) * sin(x3 * 2.0) + 
                   log(fabs(x4) + 1) * cos(x1) +
                   0.3 * pow(x2 - x3, 3);
        case 2:
            return exp(-pow(x1 - 0.5, 2)) * sin(x2 * 3.0) +
                   pow(cos(x3), 2) * x4 +
                   0.2 * sinh(x1 * x2);
        default:
            return 0.0;
    }
}

Data* synth(int n, int fx, int fy, double noise) {
    Data* d = malloc(sizeof(Data));
    d->n = n; d->fx = fx; d->fy = fy;
    d->X = malloc(n * sizeof(double*));
    d->y = malloc(n * sizeof(double*));
    
    for(int i = 0; i < n; i++) {
        d->X[i] = malloc(fx * sizeof(double));
        d->y[i] = malloc(fy * sizeof(double));
        
        // Generate more complex input features
        for(int j = 0; j < fx; j++) {
            d->X[i][j] = (double)rand()/RAND_MAX * 4 - 2; // Range [-2, 2]
        }
        
        // Generate complex outputs
        for(int j = 0; j < fy; j++) {
            d->y[i][j] = complex_function(d->X[i][0], d->X[i][1], 
                                        d->X[i][2], d->X[i][3], j);
            
            // Add high-frequency components
            d->y[i][j] += 0.2 * sin(d->X[i][0] * 10.0);
            
            // Add noise
            d->y[i][j] += ((double)rand()/RAND_MAX - 0.5) * noise;
        }
    }
    return d;
}

char* save_csv(const char* f, Data* d, const char* header) {
    char* filename = get_timestamp_filename(f);
    FILE* fp = fopen(filename, "w");
    
    if(header) {
        fprintf(fp, "%s\n", header);
    } else {
        for(int i = 0; i < d->fx; i++) fprintf(fp, "x%d,", i+1);
        for(int i = 0; i < d->fy; i++) fprintf(fp, "y%d%c", i+1, i==d->fy-1?'\n':',');
    }
    
    for(int i = 0; i < d->n; i++) {
        for(int j = 0; j < d->fx; j++) fprintf(fp, "%.6f,", d->X[i][j]);
        for(int j = 0; j < d->fy; j++) fprintf(fp, "%.6f%c", d->y[i][j], j==d->fy-1?'\n':',');
    }
    fclose(fp);
    return filename;
}

Data* load_csv(const char* f) {
    FILE* fp = fopen(f, "r");
    if(!fp) return NULL;
    
    Data* d = malloc(sizeof(Data));
    char line[4096], *token;
    
    fgets(line, sizeof(line), fp);
    d->fx = 0; d->fy = 0;
    token = strtok(line, ",");
    while(token) {
        token[strcspn(token, "\n")] = 0;
        if(token[0] == 'x') d->fx++;
        if(token[0] == 'y') d->fy++;
        token = strtok(NULL, ",");
    }
    
    d->n = 0;
    while(fgets(line, sizeof(line), fp)) d->n++;
    rewind(fp);
    fgets(line, sizeof(line), fp);
    
    d->X = malloc(d->n * sizeof(double*));
    d->y = malloc(d->n * sizeof(double*));
    for(int i = 0; i < d->n; i++) {
        d->X[i] = malloc(d->fx * sizeof(double));
        d->y[i] = malloc(d->fy * sizeof(double));
    }
    
    for(int i = 0; i < d->n; i++) {
        fgets(line, sizeof(line), fp);
        token = strtok(line, ",");
        for(int j = 0; j < d->fx; j++) {
            d->X[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int j = 0; j < d->fy; j++) {
            d->y[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(fp);
    return d;
}

void free_data(Data* d) {
    for(int i = 0; i < d->n; i++) { free(d->X[i]); free(d->y[i]); }
    free(d->X); free(d->y); free(d);
}

#endif