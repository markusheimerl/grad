#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_FEATURES 8
#define TRAIN_SPLIT 0.8
#define LEARNING_RATE 0.0001  // Reduced learning rate
#define N_EPOCHS 100

typedef struct {
    double* weights;
    double bias;
} LinearRegression;

void preprocess(const char* filename, double** X_train, double* y_train, int n_train,
                double** X_test, double* y_test, int n_test) {
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

    // Normalize features
    double* means = calloc(N_FEATURES + 1, sizeof(double));  // +1 for target
    double* stds = calloc(N_FEATURES + 1, sizeof(double));   // +1 for target
    
    // Calculate means and stds for features and target
    for (int j = 0; j <= N_FEATURES; j++) {
        for (int i = 0; i < n_train + n_test; i++) {
            double val;
            if (j < N_FEATURES) {
                val = i < n_train ? X_train[i][j] : X_test[i-n_train][j];
            } else {
                val = i < n_train ? y_train[i] : y_test[i-n_train];
            }
            means[j] += val;
            stds[j] += val * val;
        }
        means[j] /= (n_train + n_test);
        stds[j] = sqrt(stds[j]/(n_train + n_test) - means[j]*means[j]);
        if (stds[j] == 0) stds[j] = 1;
    }

    // Normalize features
    for (int j = 0; j < N_FEATURES; j++) {
        for (int i = 0; i < n_train; i++)
            X_train[i][j] = (X_train[i][j] - means[j]) / stds[j];
        for (int i = 0; i < n_test; i++)
            X_test[i][j] = (X_test[i][j] - means[j]) / stds[j];
    }

    // Normalize target
    for (int i = 0; i < n_train; i++)
        y_train[i] = (y_train[i] - means[N_FEATURES]) / stds[N_FEATURES];
    for (int i = 0; i < n_test; i++)
        y_test[i] = (y_test[i] - means[N_FEATURES]) / stds[N_FEATURES];

    free(means);
    free(stds);
}

double predict(LinearRegression* model, double* x) {
    double pred = model->bias;
    for (int j = 0; j < N_FEATURES; j++)
        pred += model->weights[j] * x[j];
    return pred;
}

void train(LinearRegression* model, double** X_train, double* y_train, int n_train,
           double** X_test, double* y_test, int n_test) {
    double best_test_mse = INFINITY;
    double* best_weights = malloc(N_FEATURES * sizeof(double));
    double best_bias = 0.0;

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        double train_mse = 0.0;
        
        // Shuffle training data
        for (int i = n_train-1; i > 0; i--) {
            int j = rand() % (i + 1);
            for (int k = 0; k < N_FEATURES; k++) {
                double temp = X_train[i][k];
                X_train[i][k] = X_train[j][k];
                X_train[j][k] = temp;
            }
            double temp = y_train[i];
            y_train[i] = y_train[j];
            y_train[j] = temp;
        }

        for (int i = 0; i < n_train; i++) {
            double pred = predict(model, X_train[i]);
            double error = pred - y_train[i];
            train_mse += error * error;
            
            model->bias -= LEARNING_RATE * error;
            for (int j = 0; j < N_FEATURES; j++)
                model->weights[j] -= LEARNING_RATE * error * X_train[i][j];
        }
        
        if ((epoch + 1) % 10 == 0) {
            double test_mse = 0.0;
            for (int i = 0; i < n_test; i++) {
                double error = predict(model, X_test[i]) - y_test[i];
                test_mse += error * error;
            }
            test_mse /= n_test;
            train_mse /= n_train;

            if (test_mse < best_test_mse) {
                best_test_mse = test_mse;
                memcpy(best_weights, model->weights, N_FEATURES * sizeof(double));
                best_bias = model->bias;
            }

            printf("Epoch %d: train_mse=%.4f, test_mse=%.4f\n", 
                   epoch + 1, train_mse, test_mse);
        }
    }

    // Restore best model
    memcpy(model->weights, best_weights, N_FEATURES * sizeof(double));
    model->bias = best_bias;
    free(best_weights);
}

int main() {
    srand(time(NULL));  // Initialize random seed

    FILE* f = fopen("data/cal_housing.data", "r");
    int n = 0;
    char line[32768];
    while (fgets(line, 32768, f)) 
        if (line[0] != '/' && line[0] != '*' && line[0] != ' ') n++;
    fclose(f);

    int n_train = n * TRAIN_SPLIT, n_test = n - n_train;
    double** X_train = malloc(n_train * sizeof(double*));
    double** X_test = malloc(n_test * sizeof(double*));
    for (int i = 0; i < n_train; i++) X_train[i] = malloc(N_FEATURES * sizeof(double));
    for (int i = 0; i < n_test; i++) X_test[i] = malloc(N_FEATURES * sizeof(double));
    double* y_train = malloc(n_train * sizeof(double));
    double* y_test = malloc(n_test * sizeof(double));

    preprocess("data/cal_housing.data", X_train, y_train, n_train, X_test, y_test, n_test);

    LinearRegression model = {.weights = calloc(N_FEATURES, sizeof(double)), .bias = 0.0};
    train(&model, X_train, y_train, n_train, X_test, y_test, n_test);

    free(model.weights);
    for (int i = 0; i < n_train; i++) free(X_train[i]);
    for (int i = 0; i < n_test; i++) free(X_test[i]);
    free(X_train); free(X_test);
    free(y_train); free(y_test);

    return 0;
}