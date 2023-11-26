#********************************************************************
#
# Make File for CPU implementation Unit Test
#
# Instructions:
# >make all; build UT's
# >make runUT; run UT's
# >make clean; remove binary
#
# History:
#    Date         Name
#    ----------   ---------
#    11/25/2023   J. Roznik
#        Initial Development.
#********************************************************************

# Compiler options
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra

# calibrated triangle graphs
FILESCALIBRATED = $(wildcard ../../GraphsSetTriangles/*.txt)

# Google Test directory
GTEST_DIR := ~/googletest/googletest

# Tactical directory
TACTICAL_DIR := ../../CpuAlgorithm

# Test source files
TEST_SRCS := TestCaseUT.cpp ../../CpuAlgorithm/TestCase.cpp ../../CpuAlgorithm/MatrixMultiplication.cpp

# Test object files
TEST_OBJS := $(TEST_SRCS:.cpp=.o)

# Test executable
TEST_EXEC := Unit_Test_CpuAlgorithm_TestCase

# Include directories
INCLUDES := -I$(GTEST_DIR)/include -I$(TACTICAL_DIR) 

# Library directories
LDFLAGS := -L$(GTEST_DIR)/lib 

# Libraries to link
LDLIBS := -lgtest -lgtest_main -lpthread

.PHONY: all clean

all: $(TEST_EXEC)

$(TEST_EXEC): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

runUT:
	./$(TEST_EXEC) $(FILESCALIBRATED)

clean:
	rm -f $(TEST_OBJS) $(TEST_EXEC)