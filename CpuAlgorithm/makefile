#********************************************************************
#
# Make File for CPU implementation
#
# Instructions:
# >make main; to build the Cpu Implementation.
# *Ensure that you have built all the graphs before running*
# >make run; to run CPU implementation with graphs.
# >make runCal; to run CPU implementation with
#     calibrated graphs.
# >make saveResults; move results to the Results/Data folder
# >make clean; remove binary
#
# History:
#    Date         Name
#    ----------   ---------
#    11/24/2023 - J. Roznik
#        Moved Cpu Algorithms to a seperated workspace.
#        Makefile changes.
#    11/--/2023   N. Do + J. Roznik
#        Initial Development.
#********************************************************************
CFLAGS = -g -Wall -O2

TARGET = CpuAlgorithm

FILES = $(wildcard ../GraphsSparse/*.txt) $(wildcard ../GraphsDense/*.txt)

FILESCALIBRATED = $(wildcard ../GraphsSetTriangles/*.txt)

NotValid:
	@echo "Please specificy main, run, runCal, saveResults, or clean."

main: TestCase.cpp MatrixMultiplication.cpp
	g++ -std=c++17 main.cpp $(CFLAGS) TestCase.cpp -o $(TARGET) MatrixMultiplication.cpp

run: 
	@./$(TARGET) $(FILES)

runCal:
	@./$(TARGET) $(FILESCALIBRATED)

saveResults: 
	@mv resultThreads* ../Results/Data

clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
