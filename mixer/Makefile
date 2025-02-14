CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lopenblas -lm -flto

mlp_mixer.out: mlp_mixer.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: mlp_mixer.out
	@time ./mlp_mixer.out
	
clean:
	rm -f *.out *.csv *.bin