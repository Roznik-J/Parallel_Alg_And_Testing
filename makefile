#makefile
CFLAGS = -g -Wall
LDFLAGS = -L./Kernels -lkernels

TARGET = output

TARGETCUDA = outputCUDA

FILES = $(wildcard nNotSparseFalse/*.txt) $(wildcard nNotSparseTrue/*.txt) $(wildcard nSparseFalse/*.txt) $(wildcard nSparseTrue/*.txt) $(wildcard pNotSparseFalse/*.txt) $(wildcard pNotSparseTrue/*.txt) $(wildcard pSparseFalse/*.txt) $(wildcard pSparseTrue/*.txt)

DIRECT = nNotSparseFalse nNotSparseTrue nSparseFalse nSparseTrue pNotSparseFalse pNotSparseTrue pSparseFalse pSparseTrue

DIRECTV2 = v2GraphsSparse v2GraphsDense v2GraphsSetTriangles

FILESV2 = $(wildcard v2GraphsSparse/*.txt) $(wildcard v2GraphsDense/*.txt)

FILESCALIBRATED = $(wildcard v2GraphsSetTriangles/*.txt)

VALGRIND_FLAGS = --tool=valgrind --leak-check=full

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

mainV2: CXXFLAGS += -g -rdynamic -DCOREDUMP_ENABLED
mainV2: Kernels/libkernels.a TestCaseV2.cpp
	g++ -std=c++17 $(CXXFLAGS) mainV2.cpp $(CFLAGS) TestCaseV2.cpp -I./Kernels/inc -I/usr/local/cuda/include -o $(TARGETCUDA) -L./Kernels -L/usr/local/cuda/lib64 -lkernels -lcudart -lcuda -lcublas -Wl,-rpath=./Kernels

run: 
	@./$(TARGET) $(FILESV2)

runCal:
	@./$(TARGET) $(FILESCALIBRATED)

runcuda: 
	@./$(TARGETCUDA) $(FILESV2)

runcudaCal: 
	@./$(TARGETCUDA) $(FILESCALIBRATED)

#valgrind: mainV2
#    valgrind $(VALGRIND_FLAGS) ./$(TARGETCUDA)

valgrind:
		valgrind ./$(TARGETCUDA) $(FILESCALIBRATED)

clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
	rm -rf $(DIRECT)
	rm -f $(TARGETCUDA)
	rm -rf $(DIRECTV2)


#.PHONY: valgrind