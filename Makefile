CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -static -lopenblas -lm -flto

mlp.out: mlp.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: mlp.out
	@time ./mlp.out
	@time python mlp.py
	
clean:
	rm -f *.out *.csv *.bin