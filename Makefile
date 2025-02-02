CC = nvcc
CFLAGS = -O3 -arch=sm_60
LDFLAGS = -lcudart -lcurand -lm

grad.out: grad.cu
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: grad.out
	./grad.out

clean:
	rm -f *.out *.csv *.bin