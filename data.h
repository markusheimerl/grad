#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { double **X, **y; int n, fx, fy; } Data;

Data* synth(int n, int fx, int fy, double noise) {
    Data* d = malloc(sizeof(Data));
    d->n = n; d->fx = fx; d->fy = fy;
    d->X = malloc(n * sizeof(double*));
    d->y = malloc(n * sizeof(double*));
    double* w = malloc(fx * fy * sizeof(double));
    
    for(int i = 0; i < fx*fy; i++) w[i] = (double)rand()/RAND_MAX - 0.5;
    for(int i = 0; i < n; i++) {
        d->X[i] = malloc(fx * sizeof(double));
        d->y[i] = malloc(fy * sizeof(double));
        for(int j = 0; j < fx; j++) d->X[i][j] = (double)rand()/RAND_MAX - 0.5;
        for(int j = 0; j < fy; j++) {
            d->y[i][j] = 0;
            for(int k = 0; k < fx; k++) d->y[i][j] += d->X[i][k] * w[k*fy + j];
            d->y[i][j] += ((double)rand()/RAND_MAX - 0.5) * noise;
        }
    }
    free(w);
    return d;
}

void save_csv(const char* f, Data* d) {
    FILE* fp = fopen(f, "w");
    for(int i = 0; i < d->fx; i++) fprintf(fp, "x%d,", i+1);
    for(int i = 0; i < d->fy; i++) fprintf(fp, "y%d%c", i+1, i==d->fy-1?'\n':',');
    for(int i = 0; i < d->n; i++) {
        for(int j = 0; j < d->fx; j++) fprintf(fp, "%.6f,", d->X[i][j]);
        for(int j = 0; j < d->fy; j++) fprintf(fp, "%.6f%c", d->y[i][j], j==d->fy-1?'\n':',');
    }
    fclose(fp);
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