CC = clang
NVCC = nvcc
CFLAGS = -O3 -march=native -ffast-math
NVCCFLAGS = -O3 -arch=sm_60
LDFLAGS = -flto -lm
CUDA_LDFLAGS = -lcudart

.PHONY: clean run run_cuda run_open_loop

grad.out: grad.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

grad_cuda.out: grad.cu
	$(NVCC) $(NVCCFLAGS) $^ $(CUDA_LDFLAGS) -o $@

grad_open_loop.out: grad_open_loop.cu
	$(NVCC) $(NVCCFLAGS) $^ $(CUDA_LDFLAGS) -o $@

run: grad.out
	./grad.out

run_cuda: grad_cuda.out
	./grad_cuda.out

run_open_loop: grad_open_loop.out
	./grad_open_loop.out

clean:
	rm -f *.out *.csv *.bin