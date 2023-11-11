LIBRARY_NAME = libkernels.a

all: $(LIBRARY_NAME)

$(LIBRARY_NAME): tcSquare.o
	ar rcs $(LIBRARY_NAME) tcSquare.o

tcSquare.o: src/tcSquare.cu
	nvcc -c -I./inc ./src/tcSquare.cu

clean:
	rm -f *.o $(LIBRARY_NAME)
	