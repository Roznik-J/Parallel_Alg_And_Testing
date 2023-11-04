#makefile
CFLAGS = -g -Wall

DEPENDS = TestCase.cpp

TARGET = output

FILES = $(wildcard nNotSparseFalse/*.txt) $(wildcard nNotSparseTrue/*.txt) $(wildcard nSparseFalse/*.txt) $(wildcard nSparseTrue/*.txt) $(wildcard pNotSparseFalse/*.txt) $(wildcard pNotSparseTrue/*.txt) $(wildcard pSparseFalse/*.txt) $(wildcard pSparseTrue/*.txt)

DIRECT = nNotSparseFalse nNotSparseTrue nSparseFalse nSparseTrue pNotSparseFalse pNotSparseTrue pSparseFalse pSparseTrue

NotValid:
	@echo "Please specificy graphs, main, run, or clean."

graphs:
	mkdir -p $(DIRECT)
	g++ -std=c++11 GraphGenerator.cpp -o $(TARGET)
	./$(TARGET)

main: 
	g++ -std=c++17 main.cpp $(CFLAGS) $(DEPENDS) -o $(TARGET)

run: 
	./$(TARGET) $(FILES)


clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
	rm -rf $(DIRECT)
