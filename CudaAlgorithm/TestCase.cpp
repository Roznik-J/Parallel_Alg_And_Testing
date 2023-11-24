//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Test Case class that handles Graph Testing with Cuda
//*****************************************************


#include "TestCase.hpp"

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

    if(!fileIn)
    {
        std::cout << "Not A File" << std::endl;
        throw std::runtime_error("error");
    }

    fileIn >> mnNumNodes;
    fileIn >> mnNumEdges;
    fileIn >> mrSparsity;

    int lnSrc;
    int lnDst;
    for(int lnIdx = 0; lnIdx < mnNumEdges; lnIdx++)
    {
        fileIn >> lnSrc;
        fileIn >> lnDst;

        if(lnSrc == -1 || lnDst == -1)
        {
            std::cout << "BAD DATA: " << lnSrc << " " << lnDst;
        }

        if(lnSrc >= mnNumNodes || lnDst >= mnNumNodes)
        {
            std::cout << "BAD DATA: " << lnSrc << " " << lnDst;
        }

        if(lnSrc < 0 || lnDst < 0)
        {
            std::cout << "BAD DATA: " << lnSrc << " " << lnDst;
        }

        mcSources.push_back(lnSrc);
        mcDestinations.push_back(lnDst);

        lnSrc = -1;
        lnDst = -1;
    }

    fileIn.close();

    // Debug only, remove later
    //printEdges();
    //printGraphInfo();

    auto handlereturn = cublasCreate(&mcCublasHandle);
    Kernel::Err::GetError(handlereturn);
    Kernel::Err::PrintError(handlereturn);

    cudaStreamCreate(&mcCudaStream);

    CreateCudaEvents();

    cudaEventRecord(mcStartProgram);

    ConstructAdjacencyMatrix();

    ComputeNumTriangles();

    cudaEventRecord(mcStopProgram);

    cudaEventSynchronize(mcStopProgram);

    cudaEventElapsedTime(&mrProgramTimeMS, mcStartProgram, mcStopProgram);

    // Debug Purposes Only, Expensive call!
    //printAdjacencyMatrix();
}

TestCaseV2::~TestCaseV2(void)
{
    auto handlereturn = cublasDestroy(mcCublasHandle);
    Kernel::Err::GetError(handlereturn);
    Kernel::Err::PrintError(handlereturn);

    cudaStreamDestroy(mcCudaStream);

    DestroyCudaEvents();

    cudaFree(mpvSourcesGpu);
    cudaFree(mpvDestGpu);
    cudaFree(mpvAdjGpu);
    cudaFree(mpvAdj2Gpu);
    cudaFree(mpvAdj3Gpu);
    cudaFree(mpvDiagOutput);

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

    if(mpvAdjGpu == nullptr || mpvAdj2Gpu == nullptr || mpvAdj3Gpu==nullptr || mpvDiagOutput==nullptr)
    {
        throw std::runtime_error("error");
    }

    const float lcScaleFactor = 1;
    const float lcZeroFactor = 0;

    //cudaEventRecord(mcStartAdjPowerOne);

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

    //cudaEventRecord(mcStopAdjPowerOne);
    //cudaEventSynchronize(mcStopAdjPowerOne);
    //cudaEventElapsedTime(&mrAdjPowerOneMS, mcStartAdjPowerOne, mcStopAdjPowerOne);

    Kernel::Err::PrintError(error);

    //cudaEventRecord(mcStartAdjPowerTwo);

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

    //cudaEventRecord(mcStopAdjPowerTwo);
    //cudaEventSynchronize(mcStopAdjPowerTwo);
    //cudaEventElapsedTime(&mrAdjPowerTwoMS, mcStartAdjPowerTwo, mcStopAdjPowerTwo);

    Kernel::Err::PrintError(error);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    int lnNumReduceThreads = std::ceil(std::sqrt(mnNumNodes));
    lsGridSize.x =std::ceil(static_cast<float>(lnNumReduceThreads)/lnThreadsPerBlock);
    lsBlockSize.x = lnNumReduceThreads;

    //cudaEventRecord(mcStartDiagSum);

    Kernel::Matrix::SumDiagonalsOptimized(lsGridSize,lsBlockSize, mcCudaStream,
                    mnNumNodes, static_cast<float*>(mpvAdj3Gpu), static_cast<float*>(mpvDiagOutput), lnNumReduceThreads);

    //cudaEventRecord(mcStopDiagSum);
    //cudaEventSynchronize(mcStopDiagSum);
    //cudaEventElapsedTime(&mrDiagSumMS, mcStartDiagSum, mcStopDiagSum);

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpyAsync(lpfRCpu, mpvDiagOutput, sizeof(float)*1, cudaMemcpyDeviceToHost, mcCudaStream);
    cudaStreamSynchronize(mcCudaStream);

    mnNumTriangles = (int)((*lpfRCpu)/6);
    
    free(lpfRCpu);

}

int TestCaseV2::GetNumTriangles(void)
{
    return mnNumTriangles;
}

int TestCaseV2::GetNodeSize(void)
{
    return mnNumNodes;
}

int TestCaseV2::GetNumEdges(void)
{
    return mnNumEdges;
}

void TestCaseV2::GetTimingInformation(std::vector<float>& arcTimeParameters)
{
    arcTimeParameters.clear();
    arcTimeParameters.push_back(mrAdjBuildSetupMS);
    arcTimeParameters.push_back(mcAdjBuildMS);
    arcTimeParameters.push_back(mrAdjPowerOneMS);
    arcTimeParameters.push_back(mrAdjPowerTwoMS);
    arcTimeParameters.push_back(mrDiagSumMS);
    arcTimeParameters.push_back(mrProgramTimeMS);
}

void TestCaseV2::ConstructAdjacencyMatrix(void)
{
    int lnDataSize = mnNumNodes*mnNumNodes;

    //cudaEventRecord(mcStartAdjBuildSetup);

    cudaMalloc(&mpvSourcesGpu, sizeof(int)*lnDataSize);
    cudaMemsetAsync(mpvSourcesGpu, 0, sizeof(int)*lnDataSize, mcCudaStream);
    cudaMemcpyAsync(mpvSourcesGpu, mcSources.data(), sizeof(int)*mcSources.size(), cudaMemcpyHostToDevice, mcCudaStream);

    cudaMalloc(&mpvDestGpu, sizeof(int)*lnDataSize);
    cudaMemsetAsync(mpvDestGpu, 0, sizeof(int)*lnDataSize, mcCudaStream);

    cudaMemcpyAsync(mpvDestGpu, mcDestinations.data(), sizeof(int)*mcDestinations.size(), cudaMemcpyHostToDevice, mcCudaStream);

    cudaMalloc(&mpvAdjGpu, sizeof(int)*lnDataSize);

    cudaMemsetAsync(mpvAdjGpu, 0, sizeof(int)*lnDataSize, mcCudaStream);


    if(mpvSourcesGpu == nullptr || mpvDestGpu == nullptr || mpvAdjGpu==nullptr)
    {
        throw std::runtime_error("error");
    }

    //cudaEventRecord(mcStopAdjBuildSetup);
    //cudaEventSynchronize(mcStopAdjBuildSetup);
    //cudaEventElapsedTime(&mrAdjBuildSetupMS, mcStartAdjBuildSetup, mcStopAdjBuildSetup);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(mnNumEdges)/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(mnNumEdges)/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;

    //cudaEventRecord(mcStartAdjBuild);

    Kernel::Matrix::BuildAdjacencyMatrix(lsGridSize, lsBlockSize, mcCudaStream, mnNumEdges, mnNumNodes, 
                                         static_cast<int*>(mpvSourcesGpu), static_cast<int*>(mpvDestGpu), 
                                         static_cast<float*>(mpvAdjGpu));

    //cudaEventRecord(mcStopAdjBuild);
    //cudaEventSynchronize(mcStopAdjBuild);
    //cudaEventElapsedTime(&mcAdjBuildMS, mcStartAdjBuild, mcStopAdjBuild);

}

void TestCaseV2::CreateCudaEvents(void)
{
    //cudaEventCreate(&mcStartAdjBuildSetup);
    //cudaEventCreate(&mcStopAdjBuildSetup);
    //cudaEventCreate(&mcStartAdjBuild);
    //cudaEventCreate(&mcStopAdjBuild);

    //cudaEventCreate(&mcStartAdjPowerOne);
    //cudaEventCreate(&mcStopAdjPowerOne);
    //cudaEventCreate(&mcStartAdjPowerTwo);
    //cudaEventCreate(&mcStopAdjPowerTwo);
    //cudaEventCreate(&mcStartDiagSum);
    //cudaEventCreate(&mcStopDiagSum);

    cudaEventCreate(&mcStartProgram);
    cudaEventCreate(&mcStopProgram);
}

void TestCaseV2::DestroyCudaEvents(void)
{
    //cudaEventDestroy(mcStartAdjBuildSetup);
    //cudaEventDestroy(mcStopAdjBuildSetup);
    //cudaEventDestroy(mcStartAdjBuild);
    //cudaEventDestroy(mcStopAdjBuild);

    //cudaEventDestroy(mcStartAdjPowerOne);
    //cudaEventDestroy(mcStopAdjPowerOne);
    //cudaEventDestroy(mcStartAdjPowerTwo);
    //cudaEventDestroy(mcStopAdjPowerTwo);
    //cudaEventDestroy(mcStartDiagSum);
    //cudaEventDestroy(mcStopDiagSum);

    cudaEventDestroy(mcStartProgram);
    cudaEventDestroy(mcStopProgram);
    
}