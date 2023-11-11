#makefile
CFLAGS = -g -Wall
LDFLAGS = -L./Kernels -lkernels

TARGET = output

FILES = $(wildcard nNotSparseFalse/*.txt) $(wildcard nNotSparseTrue/*.txt) $(wildcard nSparseFalse/*.txt) $(wildcard nSparseTrue/*.txt) $(wildcard pNotSparseFalse/*.txt) $(wildcard pNotSparseTrue/*.txt) $(wildcard pSparseFalse/*.txt) $(wildcard pSparseTrue/*.txt)

DIRECT = nNotSparseFalse nNotSparseTrue nSparseFalse nSparseTrue pNotSparseFalse pNotSparseTrue pSparseFalse pSparseTrue

NotValid:
	@echo "Please specificy graphs, main, run, or clean."

graphs:
	mkdir -p $(DIRECT)
	g++ -std=c++11 GraphGenerator.cpp -o $(TARGET)
	./$(TARGET)

main: Kernels/libkernels.a TestCase.cpp
	g++ -std=c++17 main.cpp $(CFLAGS) TestCase.cpp -I./Kernels/inc -I/usr/local/cuda/include -o $(TARGET) -L./Kernels -L/usr/local/cuda/lib64 -lkernels -lcudart -lcuda -Wl,-rpath=./Kernels

run: 
	./$(TARGET) $(FILES)


clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
	rm -rf $(DIRECT)