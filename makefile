CC=gcc
CFLAGS = -fopenmp

clean: 
	rm -f advection2D final.dat final.png 

advection2D: advection2D.c
	$(CC) $(CFLAGS) -o advection2D -std=c99 advection2D.c -lm


