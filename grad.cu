#include "data.cuh"
#include <time.h>

int main() {
    srand(time(NULL));
    Dataset* data = generate_data(1000, 32, 6, 4, 0.1);
    
    time_t now = time(NULL);
    char fname[64];
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    save_csv(fname, data);
    printf("Data saved to: %s\n", fname);

    free_dataset(data);

    return 0;
}