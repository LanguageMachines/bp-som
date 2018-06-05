# the lines below assume the gnu c compiler.
C_ARGS = -O2
CC = gcc

# You shouldn't need to make any more changes below this line.

all:	bpsom

bpsom:	bpsom.c
	$(CC) $(C_ARGS) -o $@ $^ -lm

clean:
	-rm bpsom
