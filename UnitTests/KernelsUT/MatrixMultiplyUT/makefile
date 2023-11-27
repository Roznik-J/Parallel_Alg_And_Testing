#********************************************************************
#
# Make File for Kernel_MatrixMultiply Unit Tests
#
# Instructions:
# >make all; build UT's
# >make runUT; run UT's
# >make clean; remove binary
#
# History:
#    Date         Name
#    ----------   ---------
#    11/26/2023   J. Roznik
#        Initial Development.
#********************************************************************

# Compiler options
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra

# calibrated triangle graphs
FILESCALIBRATED = $(wildcard ../../../GraphsSetTriangles/*.txt)

# Google Test directory
GTEST_DIR := ~/googletest/googletest

# Tactical directory
TACTICAL_DIR := ../../Kernels/inc

# Test source files
TEST_SRCS := tcMatrixMultiplyUT.cpp ../../../Kernels/inc/tcMatrixMultiply.hpp

# Test object files
TEST_OBJS := $(TEST_SRCS:.cpp=.o)

# Test executable
TEST_EXEC := Unit_Test_Kernel_tcMatrixMultiply

# Include directories
INCLUDES := -I$(GTEST_DIR)/include -I../../../Kernels/inc -I/usr/local/cuda/include -I$(TACTICAL_DIR) 

# Library directories
LDFLAGS := -L$(GTEST_DIR)/lib -L../../../Kernels -L/usr/local/cuda/lib64

# Libraries to link
LDLIBS := -lgtest -lgtest_main -lpthread -lcudart -lcuda -lcublas -lkernels

.PHONY: all clean

all: $(TEST_EXEC)

$(TEST_EXEC): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

runUT:
	./$(TEST_EXEC)

clean:
	rm -f $(TEST_OBJS) $(TEST_EXEC)