CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -static -lm -flto 

grad.out: grad.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: grad.out
	./grad.out

clean:
	rm -f *.out *.csv *.bin