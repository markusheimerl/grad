CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -static -lopenblas -lm -flto

grad.out: grad.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: grad.out
	@time ./grad.out
	@time python grad.py
	
clean:
	rm -f *.out *.csv *.bin