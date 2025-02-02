CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -static -lm -flto 

TARGET = grad.out
SRC = grad.c

.PHONY: clean run

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.csv *.bin