CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -flto
LDFLAGS = -lm -flto

TARGET = autodiff.out
SRC = main.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ)

debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)