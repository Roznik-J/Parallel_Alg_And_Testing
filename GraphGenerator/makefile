#*****************************************************
# Developed by J. Roznik
# 2023-11-24
# Make File to generate the graphs
#*****************************************************

DIRECT = ../GraphsSparse ../GraphsDense ../GraphsSetTriangles

TARGET = GenerateGraphs

NotValid:
	@echo "Please specificy Graphs, or clean."

Graphs:
	mkdir -p $(DIRECT)
	g++ -std=c++11 GraphGenerator.cpp -o $(TARGET)
	./$(TARGET)
	rm -f $(TARGET)


clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
	rm -rf $(DIRECT)
