#********************************************************************
#
# Make File for Graph Generator
#
# Instructions:
# >make Graphs; Build all graphs.
# >make clean; Delete all graphs.
#
# History:
#    Date         Name
#    ----------   ---------
#    11/24/2023 - J. Roznik
#        Moved Graph Generator to a seperated workspace.
#        Makefile changes.
#    11/--/2023   J. Roznik
#        Initial Development.
#********************************************************************

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
