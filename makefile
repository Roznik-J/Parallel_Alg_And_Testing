#makefile
CFLAGS = -g -Wall
LDFLAGS = -L./Kernels -lkernels

TARGET = output

FILES = $(wildcard nNotSparseFalse/*.txt) $(wildcard nNotSparseTrue/*.txt) $(wildcard nSparseFalse/*.txt) $(wildcard nSparseTrue/*.txt) $(wildcard pNotSparseFalse/*.txt) $(wildcard pNotSparseTrue/*.txt) $(wildcard pSparseFalse/*.txt) $(wildcard pSparseTrue/*.txt)

DIRECT = nNotSparseFalse nNotSparseTrue nSparseFalse nSparseTrue pNotSparseFalse pNotSparseTrue pSparseFalse pSparseTrue

DIRECTV2 = v2GraphsSparse v2GraphsDense

FILESV2 = $(wildcard v2GraphsSparse/*.txt) $(wildcard v2GraphsDense/*.txt)

NotValid:
	@echo "Please specificy graphsV2, main, mainV2, run, or clean."

graphs:
	@echo "Depreciated. Please use make graphsV2 instead"
	@# mkdir -p $(DIRECT)
	@# g++ -std=c++11 GraphGenerator.cpp -o $(TARGET)
	@# ./$(TARGET)

graphsV2:
	mkdir -p $(DIRECTV2)
	g++ -std=c++11 GraphGeneratorV2.cpp -o $(TARGET)
	./$(TARGET)

main: TestCase.cpp NonGpuAlgorithms/MatrixMultiplication.cpp
	g++ -std=c++17 main.cpp $(CFLAGS) TestCase.cpp -o $(TARGET) NonGpuAlgorithms/MatrixMultiplication.cpp

mainV2: Kernels/libkernels.a TestCaseV2.cpp
	g++ -std=c++17 mainV2.cpp $(CFLAGS) TestCaseV2.cpp -I./Kernels/inc -I/usr/local/cuda/include -o $(TARGET) -L./Kernels -L/usr/local/cuda/lib64 -lkernels -lcudart -lcuda -lcublas -Wl,-rpath=./Kernels

run: 
	@./$(TARGET) $(FILESV2)


clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
	rm -rf $(DIRECT)
	rm -rf $(DIRECTV2)
