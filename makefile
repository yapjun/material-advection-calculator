CC=gcc
CFLAGS = -fopenmp

clean: 
	rm -f advection2D final.* initial.* average.*

omp_advection2D: advection2D.c
	$(CC) $(CFLAGS) -o advection2D -std=c99 advection2D.c -lm

advection2D: advection2D.c
	$(CC) -o advection2D -std=c99 advection2D.c -lm

