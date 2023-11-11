
#include <tcMatrixDiagonalSum.hpp>
#include <iostream>

#define MAX_REDUCE_TILES 32

namespace
{

// For NxN matrix, launch N threads

static __global__ void Kernel_SumDiagonalEntries(const float* apnA, float* apnR, const int anNumSamples)
{
    __shared__ float lanShared[MAX_REDUCE_TILES];

    int lnThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int lnDiagEntry = lnThreadIdx * (anNumSamples + 1);

    float lnR = 0;

    if(lnThreadIdx < anNumSamples)
    {
        lanShared[lnThreadIdx] = apnA[lnDiagEntry];
    }

    __syncthreads();

    if(lnThreadIdx == 0)
    {
#pragma unroll
        for(int lnIdx = 0; lnIdx < MAX_REDUCE_TILES; lnIdx++)
        {
            lnR += lanShared[lnIdx];
        }
        apnR[0] = lnR;
    }
}

}

void Kernel::Matrix::SumDiagonals(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut)
{
    Kernel_SumDiagonalEntries<<<asGrid, asBlock, 0, ahStream>>>(apfDataIn, apfDataOut, anNumThreads);
}
