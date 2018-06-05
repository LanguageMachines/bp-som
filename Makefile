# the lines below assume the gnu c compiler.
CFLAGS = -O2
CC = gcc

DEPS = bp.h som.h

# You shouldn't need to make any more changes below this line.

all:	bpsom

%o: %c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

bpsom:	bpsom.o
	$(CC) -o $@ $^ -lm

clean:
	-rm bpsom bpsom.o
