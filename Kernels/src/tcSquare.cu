#include <stdio.h>
#include <vector>
#include <iostream>

#include <tcSquare.hpp>

namespace
{
__global__ void Kernel_SquareValues(int anNumThreads, float* apfDataIn, float* apfDataOut)
{
    int lnThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(lnThreadIdx < anNumThreads)
    {
        apfDataOut[lnThreadIdx] = apfDataIn[lnThreadIdx]*apfDataIn[lnThreadIdx];
    }
}
}

void Kernel::Square::LaunchSquareValues(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut)
{
    Kernel_SquareValues<<<asGrid, asBlock, 0, ahStream>>>(anNumThreads, apfDataIn, apfDataOut);

}
