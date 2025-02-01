#include "data_open_loop.cuh"
#include <time.h>

int main() {
    srand(time(NULL));
    
    // Generate synthetic open-loop data
    // 100 samples, window size 50, 3 input features, 2 output features, noise 0.1
    OpenLoopData* data = generate_open_loop_data(100, 50, 3, 2, 0.1);
    
    // Save data with timestamp
    time_t current_time = time(NULL);
    struct tm* timeinfo = localtime(&current_time);
    char data_fname[64];
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_open_loop_data.csv", timeinfo);
    
    save_open_loop_csv(data_fname, data);
    printf("Data saved to: %s\n", data_fname);
    
    // Print example sequence
    printf("\nExample sequence (first 5 timesteps of first sequence):\n");
    printf("Timestep | Input Features\n");
    printf("---------+---------------\n");
    
    for(int t = 0; t < 5; t++) {
        printf("%8d | ", t);
        for(int f = 0; f < data->input_features; f++) {
            printf("%6.3f ", data->windows[0][t][f]);
        }
        printf("\n");
    }
    
    printf("\nTarget outputs for this sequence:\n");
    for(int f = 0; f < data->output_features; f++) {
        printf("Output %d: %.3f\n", f, data->outputs[0][f]);
    }
    
    free_open_loop_data(data);
    return 0;
}