LIBRARY_NAME = libkernels.a

all: $(LIBRARY_NAME)

$(LIBRARY_NAME): tcSquare.o tcMatrixMultiply.o tcMatrixDiagonalSum.o
	ar rcs $(LIBRARY_NAME) tcSquare.o tcMatrixMultiply.o tcMatrixDiagonalSum.o

tcSquare.o: src/tcSquare.cu
	nvcc -c -I./inc ./src/tcSquare.cu

tcMatrixMultiply.o: src/tcMatrixMultiply.cu
	nvcc -c -I./inc ./src/tcMatrixMultiply.cu

tcMatrixDiagonalSum.o: src/tcMatrixDiagonalSum.cu
	nvcc -c -I./inc ./src/tcMatrixDiagonalSum.cu

clean:
	rm -f *.o $(LIBRARY_NAME)
	