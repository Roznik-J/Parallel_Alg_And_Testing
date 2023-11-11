
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