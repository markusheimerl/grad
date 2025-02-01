CC = clang
NVCC = nvcc
CFLAGS = -O3 -march=native -ffast-math
NVCCFLAGS = -O3 -arch=sm_60
LDFLAGS = -flto -lm
CUDA_LDFLAGS = -lcudart

TARGET = grad.out
CUDA_TARGET = grad_cuda.out
SRC = grad.c
CUDA_SRC = grad.cu

.PHONY: clean run run_cuda

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(CUDA_TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $^ $(CUDA_LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

run_cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET)

clean:
	rm -f $(TARGET) $(CUDA_TARGET) *.csv *.bin