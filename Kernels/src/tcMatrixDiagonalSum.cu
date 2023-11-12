//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Sums the Diagonals of a Matrix
//*****************************************************

#include <tcMatrixDiagonalSum.hpp>
#include <iostream>
#include <stdio.h>

#define MAX_REDUCE_TILES 32

namespace
{

// For NxN matrix, launch N threads - Broken, major problem for matrixes > 32 nodes
// Even though this is broken, I will keep it for the Software Testing Project
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

static __global__ void Kernel_SumDiagonalEntries_Optimized(const float* apnA, float* apnR, const int anNumSamples, const int anMaxReduceThreads)
{
    extern __shared__ float lanShared[];

    int lnThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int lnBatchIdx = lnThreadIdx;
    float lrR = 0;
    int lnDiagEntry;
    
    for(; lnThreadIdx < anNumSamples; lnThreadIdx+=anMaxReduceThreads)
    {
        lnDiagEntry = lnThreadIdx * (anNumSamples + 1);
        lrR += apnA[lnDiagEntry];
    }

    lanShared[lnBatchIdx] = lrR;

    __syncthreads();

    if(lnBatchIdx == 0)
    {
        lrR = 0;

#pragma unroll

        for(int lnIdx = 0; lnIdx < anMaxReduceThreads; lnIdx++)
        {
            lrR += lanShared[lnIdx];
        }
        apnR[0] = lrR;
    }
}

}

void Kernel::Matrix::SumDiagonals(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut)
{
    Kernel_SumDiagonalEntries<<<asGrid, asBlock, 0, ahStream>>>(apfDataIn, apfDataOut, anNumThreads);
}

void Kernel::Matrix::SumDiagonalsOptimized(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, float* apfDataIn, float* apfDataOut, const int anMaxReduceThreads)
{
    size_t lcSharedMemorySize = sizeof(float)*anMaxReduceThreads;
    Kernel_SumDiagonalEntries_Optimized<<<asGrid, asBlock, lcSharedMemorySize, ahStream>>>(apfDataIn, apfDataOut, anNumThreads, anMaxReduceThreads);
}
