#include "TestCaseV2.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <stdio.h>

#include <tcBuildAdjacencyMatrix.hpp>
#include <tcMatrixMultiply.hpp>
#include <tcMatrixDiagonalSum.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

static const int snWarpSize = 32;

TestCaseV2::TestCaseV2(std::string& arcFileName)
{
    std::ifstream fileIn(arcFileName);

    fileIn >> mnNumNodes;
    fileIn >> mnNumEdges;
    fileIn >> mrSparsity;

    int lnSrc;
    int lnDst;
    for(int lnIdx = 0; lnIdx < mnNumEdges; lnIdx++)
    {
        fileIn >> lnSrc;
        fileIn >> lnDst;

        mcSources.push_back(lnSrc);
        mcDestinations.push_back(lnDst);
    }

    fileIn.close();

    // Debug only, remove later
    printEdges();
    printGraphInfo();

    auto handlereturn = cublasCreate(&mcCublasHandle);
    Kernel::Err::GetError(handlereturn);
    Kernel::Err::PrintError(handlereturn);

    cudaStreamCreate(&mcCudaStream);

    ConstructAdjacencyMatrix();

    // Debug Purposes Only, Expensive call!
    printAdjacencyMatrix();
}

TestCaseV2::~TestCaseV2(void)
{
    cublasDestroy(mcCublasHandle);
    cudaStreamDestroy(mcCudaStream);

    cudaFree(mpvSourcesGpu);
    cudaFree(mpvDestGpu);
    cudaFree(mpvAdjGpu);
}

void TestCaseV2::printEdges(void)
{
    for(int lnIdx=0; lnIdx < mcSources.size(); lnIdx++)
    {
        std::cout << mcSources[lnIdx] << " " << mcDestinations[lnIdx] << std::endl;
    }
}

void TestCaseV2::printGraphInfo(void)
{
    std::cout << "Number of Nodes: " << mnNumNodes;
    std::cout << " Number of Edges: " << mnNumEdges;
    std::cout << " Graph Sparsity: " << mrSparsity << std::endl;
}

void TestCaseV2::printAdjacencyMatrix(void)
{
    int lnDataSize = mnNumNodes*mnNumNodes;
    float* lpfRCpu = (float*)malloc(sizeof(float)*lnDataSize);
    cudaMemcpyAsync(lpfRCpu, mpvAdjGpu, sizeof(float)*lnDataSize, cudaMemcpyDeviceToHost, mcCudaStream);

    cudaStreamSynchronize(mcCudaStream);

    int lnIndex = 0;

    for(int lnIdy = 0; lnIdy < mnNumNodes; lnIdy++)
    {
        for(int lnIdx = 0; lnIdx < mnNumNodes; lnIdx++)
        {
            lnIndex = lnIdy*mnNumNodes + lnIdx;
            std::cout << lpfRCpu[lnIndex] << " ";
        }
        std::cout << std::endl;
    }

    free(lpfRCpu);
}

void TestCaseV2::ComputeNumTriangles(void)
{
    int InputDataSize = mnNumNodes*mnNumNodes;

    cudaMalloc(&mpvAdj2Gpu, sizeof(float)*InputDataSize);

    cudaMalloc(&mpvAdj3Gpu, sizeof(float)*InputDataSize);

    cudaMalloc(&mpvDiagOutput, sizeof(float)*1);

    const float lcScaleFactor = 1;
    const float lcZeroFactor = 0;

    int error = Kernel::Matrix::Multiply(mcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             mnNumNodes,mnNumNodes,mnNumNodes,
                             &lcScaleFactor,
                             static_cast<float*>(mpvAdjGpu),
                             mnNumNodes,
                             static_cast<float*>(mpvAdjGpu),
                             mnNumNodes,
                             &lcZeroFactor,
                             static_cast<float*>(mpvAdj2Gpu),
                             mnNumNodes);

    Kernel::Err::PrintError(error);

    error = Kernel::Matrix::Multiply(mcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             mnNumNodes,mnNumNodes,mnNumNodes,
                             &lcScaleFactor,
                             static_cast<float*>(mpvAdjGpu),
                             mnNumNodes,
                             static_cast<float*>(mpvAdj2Gpu),
                             mnNumNodes,
                             &lcZeroFactor,
                             static_cast<float*>(mpvAdj3Gpu),
                             mnNumNodes);

    Kernel::Err::PrintError(error);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(mnNumNodes)/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(mnNumNodes)/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;

    Kernel::Matrix::SumDiagonals(lsGridSize, lsBlockSize, mcCudaStream,
                    mnNumNodes, static_cast<float*>(mpvAdj3Gpu), static_cast<float*>(mpvDiagOutput));

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpyAsync(lpfRCpu, mpvDiagOutput, sizeof(float)*1, cudaMemcpyDeviceToHost, mcCudaStream);

    cudaStreamSynchronize(mcCudaStream);

    mnNumTriangles = (int)((*lpfRCpu)/6);

    std::cout << "Number of Triangles : " << mnNumTriangles << std::endl;
    
    free(lpfRCpu);

}

int TestCaseV2::GetNumTriangles(void)
{
    return mnNumTriangles;
}

void TestCaseV2::ConstructAdjacencyMatrix(void)
{
    int lnDataSize = mnNumNodes*mnNumNodes;

    cudaMalloc(&mpvSourcesGpu, sizeof(int)*lnDataSize);
    cudaMemcpyAsync(mpvSourcesGpu, mcSources.data(), sizeof(int)*lnDataSize, cudaMemcpyHostToDevice, mcCudaStream);

    cudaMalloc(&mpvDestGpu, sizeof(int)*lnDataSize);
    cudaMemcpyAsync(mpvDestGpu, mcDestinations.data(), sizeof(int)*lnDataSize, cudaMemcpyHostToDevice, mcCudaStream);

    cudaMalloc(&mpvAdjGpu, sizeof(int)*lnDataSize);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(mnNumEdges)/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(mnNumEdges)/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;

    Kernel::Matrix::BuildAdjacencyMatrix(lsGridSize, lsBlockSize, mcCudaStream, mnNumEdges, mnNumNodes, 
                                         static_cast<int*>(mpvSourcesGpu), static_cast<int*>(mpvDestGpu), 
                                         static_cast<float*>(mpvAdjGpu));

}