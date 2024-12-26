#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_FEATURES 8
#define TRAIN_SPLIT 0.8

void preprocess(const char* filename, 
    double** X_train, double* y_train, int n_train,
    double** X_test, double* y_test, int n_test) {
    
    // Load data
    FILE* f = fopen(filename, "r");
    char line[32768];
    int i = 0;
    while (fgets(line, 32768, f)) {
        if (line[0] == '/' || line[0] == '*' || line[0] == ' ') continue;
        char* tok = strtok(line, ",");
        double* row = i < n_train ? X_train[i] : X_test[i-n_train];
        for (int j = 0; j <= N_FEATURES; j++, tok = strtok(NULL, ","))
            j < N_FEATURES ? (row[j] = atof(tok)) : 
                (i < n_train ? (y_train[i] = atof(tok)) : (y_test[i-n_train] = atof(tok)));
        i++;
    }
    fclose(f);

    // Normalize
    double* means = calloc(N_FEATURES, sizeof(double));
    double* stds = calloc(N_FEATURES, sizeof(double));
    
    for (int j = 0; j < N_FEATURES; j++) {
        for (int i = 0; i < n_train + n_test; i++) {
            double val = i < n_train ? X_train[i][j] : X_test[i-n_train][j];
            means[j] += val;
            stds[j] += val * val;
        }
        means[j] /= (n_train + n_test);
        stds[j] = sqrt(stds[j]/(n_train + n_test) - means[j]*means[j]);
        if (stds[j] == 0) stds[j] = 1;

        for (int i = 0; i < n_train; i++)
            X_train[i][j] = (X_train[i][j] - means[j]) / stds[j];
        for (int i = 0; i < n_test; i++)
            X_test[i][j] = (X_test[i][j] - means[j]) / stds[j];
    }
    free(means); free(stds);
}

int main() {
    // Count samples
    FILE* f = fopen("data/cal_housing.data", "r");
    int n = 0;
    char line[32768];
    while (fgets(line, 32768, f)) 
        if (line[0] != '/' && line[0] != '*' && line[0] != ' ') n++;
    fclose(f);

    // Allocate
    int n_train = n * TRAIN_SPLIT, n_test = n - n_train;
    double** X_train = malloc(n_train * sizeof(double*));
    double** X_test = malloc(n_test * sizeof(double*));
    for (int i = 0; i < n_train; i++) X_train[i] = malloc(N_FEATURES * sizeof(double));
    for (int i = 0; i < n_test; i++) X_test[i] = malloc(N_FEATURES * sizeof(double));
    double* y_train = malloc(n_train * sizeof(double));
    double* y_test = malloc(n_test * sizeof(double));

    preprocess("data/cal_housing.data", X_train, y_train, n_train, X_test, y_test, n_test);

    // Free
    for (int i = 0; i < n_train; i++) free(X_train[i]);
    for (int i = 0; i < n_test; i++) free(X_test[i]);
    free(X_train); free(X_test);
    free(y_train); free(y_test);

    return 0;
}