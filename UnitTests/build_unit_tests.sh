#********************************************************************
#
# Bash Script to build and clean all the unit tests
#
# Instructions:
# >build; build all UT's
# >clean; remove binary
#
# History:
#    Date         Name
#    ----------   ---------
#    11/26/2023   J. Roznik
#        Initial Development.
#********************************************************************

#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No command provided."
    exit 1
fi

command=$1

if [ "$command" == "build" ]; then
    echo "Building Unit Tests..."
    echo "Building CpuAlgorithm Unit Tests..."
    cd "CpuAlgorithmUT"
    make all
    cd ".."
    echo "Building GpuAlgorithm Unit Tests..."
    cd "CudaAlgorithmUT"
    make all
    cd ".."
    cd "KernelsUT"
    echo "Building KernelsUT"
    cd "BuildAdjacencyMatrixUT"
    echo "Building BuildAdjacencyMatrixUT"
    make all
    cd ".."
    cd "MatrixDiagonalSumUT"
    echo "Building MatrixDiagonalSumUT"
    make all
    cd ".."
    cd "MatrixMultiplyUT"
    echo "Building MatrixMultiplyUT"
    make all
    cd "../.."

elif [ "$command" == "clean" ]; then
    echo "Cleaning Unit Tests..."
    echo "Cleaning CpuAlgorithm Unit Tests..."
    cd "CpuAlgorithmUT"
    make clean
    cd ".."
    echo "Cleaning GpuAlgorithm Unit Tests..."
    cd "CudaAlgorithmUT"
    make clean
    cd ".."
    cd "KernelsUT"
    echo "Cleaning KernelsUT"
    cd "BuildAdjacencyMatrixUT"
    echo "Cleaning BuildAdjacencyMatrixUT"
    make clean
    cd ".."
    cd "MatrixDiagonalSumUT"
    echo "Cleaning MatrixDiagonalSumUT"
    make clean
    cd ".."
    cd "MatrixMultiplyUT"
    echo "Cleaning MatrixMultiplyUT"
    make clean
    cd "../.."
else
    echo "Unknown command: $command"
fi