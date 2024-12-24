#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct { double val, der; } Dual;

Dual sigmoid(Dual x) {
    double s = 1.0 / (1.0 + exp(-x.val));
    return (Dual){s, x.der * s * (1.0 - s)};
}

int main() {
    srand(time(NULL));
    
    // Parameters
    double w1 = 0.1, w2 = 0.1, w3 = -0.1, b = 0;
    const int N = 1000;
    const double lr = 0.1;
    
    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        double loss = 0;
        double dw1 = 0, dw2 = 0, dw3 = 0, db = 0;
        
        for (int i = 0; i < N; i++) {
            double x = (double)rand()/RAND_MAX * 3.0 - 1.5;
            double y = (double)rand()/RAND_MAX * 3.0 - 1.5;
            int label = (x*x + y*y) <= 1.0;
            
            Dual z = {w1*x + w2*y + w3*(x*x + y*y) + b, 1.0};
            Dual pred = sigmoid(z);
            
            double error = pred.val - label;
            loss += error * error;
            double grad = 2 * error * pred.der;
            
            dw1 += grad * x;
            dw2 += grad * y;
            dw3 += grad * (x*x + y*y);
            db += grad;
        }
        
        w1 -= lr * dw1/N;
        w2 -= lr * dw2/N;
        w3 -= lr * dw3/N;
        b -= lr * db/N;
        
        if (epoch % 100 == 0)
            printf("Epoch %d, Loss: %f\n", epoch, loss/N);
    }
    
    // Test random points
    printf("\nTesting 20 random points:\n");
    int correct = 0;
    const int TEST_POINTS = 20;
    
    for (int i = 0; i < TEST_POINTS; i++) {
        double x = (double)rand()/RAND_MAX * 3.0 - 1.5;
        double y = (double)rand()/RAND_MAX * 3.0 - 1.5;
        int true_label = (x*x + y*y) <= 1.0;
        
        double pred = 1.0/(1.0 + exp(-(w1*x + w2*y + w3*(x*x + y*y) + b)));
        int pred_label = pred > 0.5;
        
        correct += (true_label == pred_label);
        printf("(%.2f, %.2f): pred=%.3f %s, true=%s %s\n", 
               x, y, pred, 
               pred_label ? "Inside" : "Outside",
               true_label ? "Inside" : "Outside",
               true_label == pred_label ? "✓" : "✗");
    }
    
    printf("\nAccuracy: %.1f%% (%d/%d correct)\n", 
           100.0 * correct / TEST_POINTS, correct, TEST_POINTS);
    
    return 0;
}