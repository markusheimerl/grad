#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_FEATURES 8
#define MAX_LINE 32768

void load_cal_housing(const char* filename, double** X, double* y) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Error opening file"); exit(1); }

    char line[MAX_LINE];
    int i = 0;

    // Skip header and process data
    while (fgets(line, MAX_LINE, file)) {
        if (line[0] == '/' || line[0] == '*' || line[0] == ' ') continue;
        
        char* token = strtok(line, ",");
        for (int j = 0; j <= N_FEATURES; j++, token = strtok(NULL, ",")) {
            if (j < N_FEATURES) X[i][j] = atof(token);
            else y[i] = atof(token);
        }
        i++;
    }
    fclose(file);
}

int main() {
    FILE* file = fopen("data/cal_housing.data", "r");
    if (!file) { perror("Error opening file"); exit(1); }

    // Count data lines
    int n_samples = 0;
    char line[MAX_LINE];
    while (fgets(line, MAX_LINE, file)) {
        if (line[0] != '/' && line[0] != '*' && line[0] != ' ') n_samples++;
    }
    fclose(file);

    // Allocate memory
    double** X = malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) X[i] = malloc(N_FEATURES * sizeof(double));
    double* y = malloc(n_samples * sizeof(double));

    load_cal_housing("data/cal_housing.data", X, y);

    // Cleanup
    for (int i = 0; i < n_samples; i++) free(X[i]);
    free(X); free(y);

    return 0;
}