#ifndef GRAD_H
#define GRAD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "data.h"

#define B1 0.9
#define B2 0.999
#define EPS 1e-8
#define DECAY 0.01

typedef struct {
    int n, *sz;
    double **w, **b, **mw, **mb, **vw, **vb;
    double lr;
    int step;
} Net;

void adam(double *p, double *g, double *m, double *v, int n, int t, double lr) {
    double lrt = lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
    for(int i = 0; i < n; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lrt * (m[i] / (sqrt(v[i]) + EPS) + DECAY * p[i]);
    }
}

Net* init_net(int n, int *sz) {
    Net* net = malloc(sizeof(Net));
    net->n = n-1; net->sz = malloc(n * sizeof(int));
    memcpy(net->sz, sz, n * sizeof(int));
    
    net->w = malloc((n-1) * sizeof(double*));
    net->b = malloc((n-1) * sizeof(double*));
    net->mw = malloc((n-1) * sizeof(double*));
    net->mb = malloc((n-1) * sizeof(double*));
    net->vw = malloc((n-1) * sizeof(double*));
    net->vb = malloc((n-1) * sizeof(double*));
    
    for(int i = 0; i < n-1; i++) {
        int in = sz[i], out = sz[i+1];
        net->w[i] = malloc(in * out * sizeof(double));
        net->b[i] = calloc(out, sizeof(double));
        double s = sqrt(2.0/in);
        for(int j = 0; j < in*out; j++) {
            double u1 = (double)rand()/RAND_MAX, u2 = (double)rand()/RAND_MAX;
            net->w[i][j] = sqrt(-2*log(u1)) * cos(2*M_PI*u2) * s;
        }
        net->mw[i] = calloc(in*out, sizeof(double));
        net->mb[i] = calloc(out, sizeof(double));
        net->vw[i] = calloc(in*out, sizeof(double));
        net->vb[i] = calloc(out, sizeof(double));
    }
    net->lr = 0.001;
    net->step = 1;
    return net;
}

double lrelu(double x) { return x > 0 ? x : 0.1 * x; }
double dlrelu(double x) { return x > 0 ? 1.0 : 0.1; }

void fwd(Net* net, double* in, double** act) {
    memcpy(act[0], in, net->sz[0] * sizeof(double));
    for(int i = 0; i < net->n; i++) {
        int ni = net->sz[i], no = net->sz[i+1];
        for(int j = 0; j < no; j++) {
            act[i+1][j] = net->b[i][j];
            for(int k = 0; k < ni; k++) act[i+1][j] += net->w[i][j*ni + k] * act[i][k];
            if(i < net->n-1) act[i+1][j] = lrelu(act[i+1][j]);
        }
    }
}

void bwd(Net* net, double** act, double** grad) {
    for(int i = net->n-1; i >= 0; i--) {
        int ni = net->sz[i], no = net->sz[i+1];
        double* wg = malloc(ni * no * sizeof(double));
        
        if(i < net->n-1)
            for(int j = 0; j < no; j++)
                grad[i+1][j] *= dlrelu(act[i+1][j]);
        
        for(int j = 0; j < no; j++)
            for(int k = 0; k < ni; k++)
                wg[j*ni + k] = grad[i+1][j] * act[i][k];
        
        adam(net->w[i], wg, net->mw[i], net->vw[i], ni*no, net->step, net->lr);
        adam(net->b[i], grad[i+1], net->mb[i], net->vb[i], no, net->step, net->lr);
        
        if(i > 0) {
            for(int j = 0; j < ni; j++) {
                grad[i][j] = 0;
                for(int k = 0; k < no; k++)
                    grad[i][j] += grad[i+1][k] * net->w[i][k*ni + j];
            }
        }
        free(wg);
    }
    net->step++;
}

void save_weights(const char* f, Net* net) {
    FILE* fp = fopen(f, "wb");
    if(!fp) return;
    fwrite(&net->n, sizeof(int), 1, fp);
    fwrite(net->sz, sizeof(int), net->n + 1, fp);
    for(int i = 0; i < net->n; i++) {
        fwrite(net->w[i], sizeof(double), net->sz[i] * net->sz[i+1], fp);
        fwrite(net->b[i], sizeof(double), net->sz[i+1], fp);
    }
    fclose(fp);
}

Net* load_weights(const char* f) {
    FILE* fp = fopen(f, "rb");
    if(!fp) return NULL;
    
    int n;
    fread(&n, sizeof(int), 1, fp);
    int* sz = malloc((n + 1) * sizeof(int));
    fread(sz, sizeof(int), n + 1, fp);
    
    Net* net = init_net(n + 1, sz);
    for(int i = 0; i < net->n; i++) {
        fread(net->w[i], sizeof(double), net->sz[i] * net->sz[i+1], fp);
        fread(net->b[i], sizeof(double), net->sz[i+1], fp);
    }
    fclose(fp);
    free(sz);
    return net;
}

void free_net(Net* net) {
    for(int i = 0; i < net->n; i++) {
        free(net->w[i]); free(net->b[i]);
        free(net->mw[i]); free(net->mb[i]);
        free(net->vw[i]); free(net->vb[i]);
    }
    free(net->w); free(net->b);
    free(net->mw); free(net->mb);
    free(net->vw); free(net->vb);
    free(net->sz); free(net);
}

#endif