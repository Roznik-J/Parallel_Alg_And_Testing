//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// This is supposed to be used as a quick test to see
// if cuda functionality is working properly
//*****************************************************

#ifndef TCSQUARE_HPP_
#define TCSQUARE_HPP_

#include <cuda_runtime_api.h>

namespace Kernel
{
namespace Square 
{

void LaunchSquareValues(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut);

}
}

#endif