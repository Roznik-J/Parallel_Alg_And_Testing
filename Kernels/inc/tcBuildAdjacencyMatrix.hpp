//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Generates the Adjacency Matrix from a list of
// sources and destinations
//*****************************************************

#ifndef TCBUILDADJANCENCYMATRIX_HPP_
#define TCBUILDADJANCENCYMATRIX_HPP_

#include "cublas_v2.h"
#include <cuda_runtime_api.h>

namespace Kernel
{
namespace Matrix
{

void BuildAdjacencyMatrix(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, int anNumNodes, int* apnSources, int* apnDest, float* apfAdj);

}
}

#endif