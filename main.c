#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct { double val, der; } Dual;

Dual sigmoid(Dual x) {
    double s = 1.0 / (1.0 + exp(-x.val));
    return (Dual){s, x.der * s * (1.0 - s)};
}

double random_range(double min, double max) {
    return (double)rand()/RAND_MAX * (max - min) + min;
}

int main() {
    srand(time(NULL));
    
    // Parameters
    double w1 = 0.1, w2 = 0.1, b = 0;
    const int EPOCHS = 1000, BATCH_SIZE = 1000;
    const double LR = 0.1;
    
    // Training
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0, dw1 = 0, dw2 = 0, db = 0;
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            double x = random_range(-1.5, 1.5);
            double y = random_range(-1.5, 1.5);
            int label = (x*x + y*y) <= 1.0;
            
            Dual pred = sigmoid((Dual){w1*x + w2*y + b, 1.0});
            double error = pred.val - label;
            
            double grad = 2 * error * pred.der;
            dw1 += grad * x;
            dw2 += grad * y;
            db += grad;
            loss += error * error;
        }
        
        w1 -= LR * dw1/BATCH_SIZE;
        w2 -= LR * dw2/BATCH_SIZE;
        b -= LR * db/BATCH_SIZE;
        
        if (epoch % 100 == 0)
            printf("Epoch %d, Loss: %f\n", epoch, loss/BATCH_SIZE);
    }
    
    // Testing
    int correct = 0;
    const int TEST_SIZE = 20;
    
    printf("\nTesting %d points:\n", TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        double x = random_range(-1.5, 1.5);
        double y = random_range(-1.5, 1.5);
        int true_label = (x*x + y*y) <= 1.0;
        int pred_label = 1.0/(1.0 + exp(-(w1*x + w2*y + b))) > 0.5;
        
        correct += (true_label == pred_label);
        printf("(%.2f, %.2f): %s, true=%s %s\n", 
               x, y, pred_label ? "Inside" : "Outside",
               true_label ? "Inside" : "Outside",
               true_label == pred_label ? "✓" : "✗");
    }
    
    printf("\nAccuracy: %.1f%% (%d/%d)\n", 
           100.0 * correct / TEST_SIZE, correct, TEST_SIZE);
    
    return 0;
}