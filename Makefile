NVCC = nvcc
CFLAGS = -O3 -arch=sm_60
LDFLAGS = -lcudart

grad.out: grad.cu
	$(NVCC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: grad.out
	./grad.out

clean:
	rm -f *.out *.csv *.bin