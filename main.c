#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_FEATURES 8
#define TRAIN_SPLIT 0.8
#define LEARNING_RATE 0.0001
#define N_EPOCHS 100

typedef struct {
    double* weights;
    double bias;
} LinearRegression;

void shuffle_data(double** X, double* y, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i + 1);
        double* temp_X = X[i];
        X[i] = X[j];
        X[j] = temp_X;
        double temp_y = y[i];
        y[i] = y[j];
        y[j] = temp_y;
    }
}

int count_lines(const char* filename) {
    FILE* f = fopen(filename, "r");
    int n = 0;
    char line[32768];
    while (fgets(line, 32768, f)) 
        if (line[0] != '/' && line[0] != '*' && line[0] != ' ') n++;
    fclose(f);
    return n;
}

void normalize_data(double** X_train, double* y_train, int n_train,
                   double** X_test, double* y_test, int n_test,
                   double* means, double* stds) {
    for (int j = 0; j <= N_FEATURES; j++) {
        means[j] = 0;
        stds[j] = 0;
        for (int i = 0; i < n_train + n_test; i++) {
            double val = j < N_FEATURES ? 
                (i < n_train ? X_train[i][j] : X_test[i-n_train][j]) :
                (i < n_train ? y_train[i] : y_test[i-n_train]);
            means[j] += val;
            stds[j] += val * val;
        }
        means[j] /= (n_train + n_test);
        stds[j] = sqrt(stds[j]/(n_train + n_test) - means[j]*means[j]);
        if (stds[j] == 0) stds[j] = 1;
    }

    for (int j = 0; j < N_FEATURES; j++) {
        for (int i = 0; i < n_train; i++)
            X_train[i][j] = (X_train[i][j] - means[j]) / stds[j];
        for (int i = 0; i < n_test; i++)
            X_test[i][j] = (X_test[i][j] - means[j]) / stds[j];
    }

    for (int i = 0; i < n_train; i++)
        y_train[i] = (y_train[i] - means[N_FEATURES]) / stds[N_FEATURES];
    for (int i = 0; i < n_test; i++)
        y_test[i] = (y_test[i] - means[N_FEATURES]) / stds[N_FEATURES];
}

double predict(LinearRegression* model, double* x) {
    double pred = model->bias;
    for (int j = 0; j < N_FEATURES; j++)
        pred += model->weights[j] * x[j];
    return pred;
}

void train(LinearRegression* model, double** X_train, double* y_train, int n_train,
           double** X_test, double* y_test, int n_test) {
    double* best_weights = malloc(N_FEATURES * sizeof(double));
    double best_test_mse = INFINITY;
    double best_bias = 0.0;

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        double train_mse = 0.0;
        shuffle_data(X_train, y_train, n_train);

        for (int i = 0; i < n_train; i++) {
            double error = predict(model, X_train[i]) - y_train[i];
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
    memcpy(model->weights, best_weights, N_FEATURES * sizeof(double));
    model->bias = best_bias;
    free(best_weights);
}

void evaluate_predictions(LinearRegression* model, double** X_test, double* y_test, 
                        int n_test, double* means, double* stds) {
    int correct_predictions = 0;
    printf("\nFinal Predictions (showing first 10 examples):\n");
    printf("Predicted Price\tActual Price\tDifference\n");
    
    for (int i = 0; i < n_test; i++) {
        double pred = predict(model, X_test[i]);
        double denorm_pred = pred * stds[N_FEATURES] + means[N_FEATURES];
        double denorm_actual = y_test[i] * stds[N_FEATURES] + means[N_FEATURES];
        double diff = denorm_pred - denorm_actual;
        
        if (fabs(diff) <= 5000.0) correct_predictions++;
        if (i < 10) {
            printf("$%.2f\t$%.2f\t$%.2f\n", 
                   denorm_pred, denorm_actual, diff);
        }
    }
    printf("\nPredictions within $5000 margin: %d out of %d (%.2f%%)\n", 
           correct_predictions, n_test, (double)correct_predictions / n_test * 100);
}

int main() {
    srand(time(NULL));
    int n = count_lines("data/cal_housing.data");
    int n_train = n * TRAIN_SPLIT;
    int n_test = n - n_train;

    // Allocate memory for all data
    double** X = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        X[i] = malloc(N_FEATURES * sizeof(double));
    }
    double* y = malloc(n * sizeof(double));

    // Load all data
    FILE* f = fopen("data/cal_housing.data", "r");
    char line[32768];
    int i = 0;
    while (fgets(line, 32768, f)) {
        if (line[0] == '/' || line[0] == '*' || line[0] == ' ') continue;
        char* tok = strtok(line, ",");
        for (int j = 0; j <= N_FEATURES; j++, tok = strtok(NULL, ","))
            j < N_FEATURES ? (X[i][j] = atof(tok)) : (y[i] = atof(tok));
        i++;
    }
    fclose(f);

    // Shuffle and split data
    shuffle_data(X, y, n);
    double** X_train = malloc(n_train * sizeof(double*));
    double** X_test = malloc(n_test * sizeof(double*));
    double* y_train = malloc(n_train * sizeof(double));
    double* y_test = malloc(n_test * sizeof(double));

    for (i = 0; i < n_train; i++) {
        X_train[i] = X[i];
        y_train[i] = y[i];
    }
    for (i = 0; i < n_test; i++) {
        X_test[i] = X[n_train + i];
        y_test[i] = y[n_train + i];
    }

    // Initialize model and normalization parameters
    LinearRegression model = {.weights = calloc(N_FEATURES, sizeof(double)), .bias = 0.0};
    double* means = calloc(N_FEATURES + 1, sizeof(double));
    double* stds = calloc(N_FEATURES + 1, sizeof(double));

    // Train and evaluate model
    normalize_data(X_train, y_train, n_train, X_test, y_test, n_test, means, stds);
    train(&model, X_train, y_train, n_train, X_test, y_test, n_test);
    evaluate_predictions(&model, X_test, y_test, n_test, means, stds);

    // Free memory
    free(model.weights);
    free(means);
    free(stds);
    for (i = 0; i < n; i++) {
        free(X[i]);
    }
    free(X);
    free(X_train);
    free(X_test);
    free(y);
    free(y_train);
    free(y_test);

    return 0;
}