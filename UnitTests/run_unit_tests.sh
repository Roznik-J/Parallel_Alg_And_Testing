#********************************************************************
#
# Bash Script to run all the unit tests
#
# History:
#    Date         Name
#    ----------   ---------
#    11/26/2023   J. Roznik
#        Initial Development.
#********************************************************************

#!/bin/bash

cd "CpuAlgorithmUT"
make runUT
cd ".."

cd "CudaAlgorithmUT"
make runUT
cd ".."

cd "KernelsUT"

cd "BuildAdjacencyMatrixUT"
make runUT
cd ".."

cd "MatrixDiagonalSumUT"
make runUT
cd ".."

cd "MatrixMultiplyUT"
make runUT
cd ".."