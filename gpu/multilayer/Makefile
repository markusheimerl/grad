CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_86 \
    -x cuda \
    -fcuda-flush-denormals-to-zero \
    -fcuda-approx-transcendentals \
    -Wno-unknown-cuda-version

CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

mlp.out: mlp.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $^ $(CUDALIBS) $(LDFLAGS) -o $@

run: mlp.out
	@time ./mlp.out
	@time python ../../mlp.py

clean:
	rm -f *.out *.csv *.bin