CC = gcc
CFLAGS = -O3 -march=native -funroll-loops -Wall -Wextra
LDFLAGS = -flto -lm
TARGET = grad.out

.PHONY: all clean run

all: $(TARGET)

$(TARGET): grad.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)