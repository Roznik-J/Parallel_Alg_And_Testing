//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Sums the Diagonals of a Matrix
//*****************************************************

#ifndef TCSMATRIXDIAGONALSUM_HPP_
#define TCSMATRIXDIAGONALSUM_HPP_

#include "cublas_v2.h"
#include <cuda_runtime_api.h>

namespace Kernel
{
namespace Matrix
{

// Broken for matrices with size greater than 32. Leaving for the Software Testing Project
void SumDiagonals(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut);

void SumDiagonalsOptimized(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut, const int anMaxReduceThreads);

}
}

#endif