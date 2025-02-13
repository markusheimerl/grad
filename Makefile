CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lopenblas -lm -flto

mlp.out: mlp.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: mlp.out
	@time ./mlp.out
	
clean:
	rm -f *.out *.csv *.bin