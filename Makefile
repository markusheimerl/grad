CC = gcc
CFLAGS = -O3 -flto -ffast-math -funroll-loops -Wall -Wextra
LDFLAGS = -flto
TARGET = grad.out

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): grad.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)