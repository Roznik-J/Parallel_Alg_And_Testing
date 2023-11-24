#********************************************************************
#
# Make File for Cuda implementation
#
# Instructions:
# >make main; to build the Gpu Implementation.
# *Ensure that you have built all the graphs before running*
# >make run; to run GPU implementation with graphs.
# >make runCal; to run GPU implementation with
#     calibrated graphs.
# >make valgrind; to run GPU implementation with valgrind
# >make clean; remove binary
#
# History:
#    Date         Name
#    ----------   ---------
#    11/24/2023 - J. Roznik
#        Moved Gpu Algorithms to a seperated workspace.
#        Makefile changes.
#    11/--/2023   J. Roznik
#        Initial Development.
#********************************************************************

CFLAGS = -g -Wall -O2
LDFLAGS = -L./Kernels -lkernels

TARGET = CudaImplementation

FILES = $(wildcard ../GraphsSparse/*.txt) $(wildcard ../GraphsDense/*.txt)

FILESCALIBRATED = $(wildcard ../GraphsSetTriangles/*.txt)

VALGRIND_FLAGS = --tool=valgrind --leak-check=full

NotValid:
	@echo "Please specificy graphsV2, main, mainV2, run, or clean."

main: CXXFLAGS += -g -rdynamic -DCOREDUMP_ENABLED
main: ../Kernels/libkernels.a TestCase.cpp
	g++ -std=c++17 $(CXXFLAGS) main.cpp $(CFLAGS) TestCase.cpp -I../Kernels/inc -I/usr/local/cuda/include -o $(TARGET) -L../Kernels -L/usr/local/cuda/lib64 -lkernels -lcudart -lcuda -lcublas -Wl,-rpath=../Kernels

run: 
	@./$(TARGET) $(FILES)

runCal: 
	@./$(TARGET) $(FILESCALIBRATED)

valgrind:
		valgrind ./$(TARGETCUDA) $(FILESCALIBRATED)

clean:
	@echo "Removing $(TARGET)"
	rm -f $(TARGET)
