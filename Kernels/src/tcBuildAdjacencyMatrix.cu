//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Generates the Adjacency Matrix from a list of
// sources and destinations
//*****************************************************

#include <tcBuildAdjacencyMatrix.hpp>
#include <iostream>

namespace
{

__global__ void Kernel_GenerateAdjMatrix(int anNumThreads, int anNumNodes, int* apnSources, int* apnDest, float* apfAdj)
{
    int lnThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(lnThreadIdx < anNumThreads)
    {
        int lnSrc = apnSources[lnThreadIdx];
        int lnDst = apnDest[lnThreadIdx];
        int lnUpperIdx = anNumNodes*lnSrc + lnDst;
        int lnLowerIdx = anNumNodes*lnDst + lnSrc;

        apfAdj[lnUpperIdx] = 1.0f;
        apfAdj[lnLowerIdx] = 1.0f;
    }
}

}

void Kernel::Matrix::BuildAdjacencyMatrix(dim3 asGrid, dim3 asBlock, cudaStream_t ahStream,
                    int anNumThreads, int anNumNodes, int* apnSources, int* apnDest, float* apfAdj)
{
    Kernel_GenerateAdjMatrix<<<asGrid, asBlock, 0, ahStream>>>(anNumThreads, anNumNodes, apnSources, apnDest, apfAdj);
}