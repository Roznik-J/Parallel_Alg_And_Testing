all: program

program: tcSquare.o
	g++ -o program -L/usr/local/cuda/lib64 -I./inc -I/usr/local/cuda/include main.cpp tcSquare.o -lcuda -lcudart

tcSquare.o: src/tcSquare.cu
	nvcc -c -I./inc ./src/tcSquare.cu

clean:
	rm -f *.o program