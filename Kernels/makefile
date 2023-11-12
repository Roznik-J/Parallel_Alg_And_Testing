#*****************************************************
# Developed by J. Roznik
# 2023-11-11
# Make File for the Kernels Library
#*****************************************************

LIBRARY_NAME = libkernels.a

all: $(LIBRARY_NAME)

$(LIBRARY_NAME): tcSquare.o tcMatrixMultiply.o tcMatrixDiagonalSum.o tcBuildAdjacencyMatrix.o
	ar rcs $(LIBRARY_NAME) tcSquare.o tcMatrixMultiply.o tcMatrixDiagonalSum.o tcBuildAdjacencyMatrix.o

tcSquare.o: src/tcSquare.cu
	nvcc -c -I./inc ./src/tcSquare.cu

tcMatrixMultiply.o: src/tcMatrixMultiply.cu
	nvcc -c -I./inc ./src/tcMatrixMultiply.cu

tcMatrixDiagonalSum.o: src/tcMatrixDiagonalSum.cu
	nvcc -c -I./inc ./src/tcMatrixDiagonalSum.cu

tcBuildAdjacencyMatrix.o: src/tcBuildAdjacencyMatrix.cu
	nvcc -c -I./inc ./src/tcBuildAdjacencyMatrix.cu

clean:
	rm -f *.o $(LIBRARY_NAME)
	