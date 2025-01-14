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
    for(int i = 0; i < d->n; i++) {
        for(int j = 0; j < d->fx; j++) fprintf(fp, "%.6f,", d->X[i][j]);
        for(int j = 0; j < d->fy; j++) fprintf(fp, "%.6f%c", d->y[i][j], j==d->fy-1?'\n':',');
    }
    fclose(fp);
}

void free_data(Data* d) {
    for(int i = 0; i < d->n; i++) { free(d->X[i]); free(d->y[i]); }
    free(d->X); free(d->y); free(d);
}

#endif