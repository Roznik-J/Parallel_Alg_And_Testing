//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Parallel Programming Project Main
//*****************************************************

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include <tcMatrixMultiply.hpp>
#include <tcMatrixDiagonalSum.hpp>
#include "TestCaseV2.hpp"

static const int snWarpSize = 32;

//#include "cublas_v2.h"
//#include <cuda_runtime_api.h>

void RunSimpleTest(void)
{
    // Matrix A 3 x 3 Matrix
    std::vector<float> A = {1,2,3,4,5,6,7,8,9};
    //std::vector<float> A = {1,4,7,2,5,8,3,6,9};

    // Matrix B 3 x 3 Matrix
    //std::vector<float> B = {1, 0.0, 3, 0.0, 5, 0.0, 7, 0.0, 9};
    std::vector<float> B = {1,2,3,4,5,6,7,8,9};
    //std::vector<float> B = {9,8,7,6,5,4,3,2,1};
    //std::vector<float> B = {1,0,0,0,1,0,0,0,1};

    // Expected Matrix C
    std::vector<float> C = {22,10,30,46,25,66,70,40,102};

    int lnNumCols = 3;
    int lnNumRows = 3;

    int InputDataSize = lnNumCols * lnNumRows;

    void* AGpu;
    //float* AGpuData;
    cudaMalloc(&AGpu, sizeof(float)*InputDataSize);
    cudaMemcpy(AGpu, A.data(), sizeof(float)*InputDataSize, cudaMemcpyHostToDevice);
    //cublasSetMatrix(lnNumRows, lnNumCols, sizeof(float)*InputDataSize,
    //            A.data(), lnNumCols, AGpu, lnNumCols);
    //cudaMemcpy2D (AGpu, sizeof(float)*lnNumCols, A.data(), sizeof(float)*lnNumCols, sizeof(float)*lnNumCols, sizeof(float)*lnNumCols, cudaMemcpyHostToDevice);

    void* BGpu;
    cudaMalloc(&BGpu, sizeof(float)*InputDataSize);
    cudaMemcpy(BGpu, B.data(), sizeof(float)*InputDataSize, cudaMemcpyHostToDevice);
    //cublasSetMatrix(lnNumRows, lnNumCols, sizeof(float)*InputDataSize,
    //            B.data(), lnNumCols, BGpu, lnNumCols);
    //cudaMemcpy2D (BGpu, sizeof(float)*lnNumCols, B.data(), sizeof(float)*lnNumCols, sizeof(float)*lnNumCols, sizeof(float)*lnNumCols, cudaMemcpyHostToDevice);

    void* CGpu;
    cudaMalloc(&CGpu, sizeof(float)*InputDataSize);

    cublasHandle_t lcCublasHandle;

    auto handlereturn = cublasCreate(&lcCublasHandle);
    std::cout << "handler error status" << std::endl;
    Kernel::Err::GetError(handlereturn);
    Kernel::Err::PrintError(handlereturn);

    const float lcScaleFactor = 1;
    const float lcZeroFactor = 0;

    int error = Kernel::Matrix::Multiply(lcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             3,3,3,
                             &lcScaleFactor,
                             static_cast<float*>(AGpu),
                             3,
                             static_cast<float*>(BGpu),
                             3,
                             &lcZeroFactor,
                             static_cast<float*>(CGpu),
                             3);

    Kernel::Err::PrintError(error);

    float* lpfOutputDataCpu = (float*)malloc(sizeof(float)*InputDataSize);
    cudaMemcpy(lpfOutputDataCpu, CGpu, sizeof(float)*InputDataSize, cudaMemcpyDeviceToHost);
    //cublasGetMatrix (lnNumRows,lnNumCols, sizeof(float)*InputDataSize, lpfOutputDataCpu ,lnNumRows,CGpu,lnNumRows);

    // Print MAtrix
    for(int lnIdy = 0; lnIdy < lnNumRows; lnIdy++)
    {
        for(int lnIdx = 0; lnIdx < lnNumCols; lnIdx++)
        {
            int Index = lnIdy*lnNumRows + lnIdx;
            std::cout << " idx " << Index << " " << lpfOutputDataCpu[Index] << " ";
        }
        std::cout << std::endl;
    }

    cublasDestroy(lcCublasHandle);
    cudaFree(AGpu);
    cudaFree(BGpu);
    cudaFree(CGpu);
    free(lpfOutputDataCpu);

}

void RunSimpleTest2(void)
{
    std::vector<float> A = {0,1,1,1,0,1,1,1,0};

    int lnNumCols = 3;
    int lnNumRows = 3;
    int InputDataSize = lnNumCols * lnNumRows;

    void* AGpu;
    cudaMalloc(&AGpu, sizeof(float)*InputDataSize);
    cudaMemcpy(AGpu, A.data(), sizeof(float)*InputDataSize, cudaMemcpyHostToDevice);

    void* CGpu;
    cudaMalloc(&CGpu, sizeof(float)*InputDataSize);

    cublasHandle_t lcCublasHandle;

    auto handlereturn = cublasCreate(&lcCublasHandle);
    std::cout << "handler error status" << std::endl;
    Kernel::Err::GetError(handlereturn);
    Kernel::Err::PrintError(handlereturn);

    const float lcScaleFactor = 1;
    const float lcZeroFactor = 0;

    int error = Kernel::Matrix::Multiply(lcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             3,3,3,
                             &lcScaleFactor,
                             static_cast<float*>(AGpu),
                             3,
                             static_cast<float*>(AGpu),
                             3,
                             &lcZeroFactor,
                             static_cast<float*>(CGpu),
                             3);

    Kernel::Err::PrintError(error);

    float* lpfCDataCpu = (float*)malloc(sizeof(float)*InputDataSize);
    cudaMemcpy(lpfCDataCpu, CGpu, sizeof(float)*InputDataSize, cudaMemcpyDeviceToHost);

    for(int lnIdy = 0; lnIdy < lnNumRows; lnIdy++)
    {
        for(int lnIdx = 0; lnIdx < lnNumCols; lnIdx++)
        {
            int Index = lnIdy*lnNumRows + lnIdx;
            std::cout << " idx " << Index << " " << lpfCDataCpu[Index] << " ";
        }
        std::cout << std::endl;
    }

    void* DGpu;
    cudaMalloc(&DGpu, sizeof(float)*InputDataSize);

    error = Kernel::Matrix::Multiply(lcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             3,3,3,
                             &lcScaleFactor,
                             static_cast<float*>(AGpu),
                             3,
                             static_cast<float*>(CGpu),
                             3,
                             &lcZeroFactor,
                             static_cast<float*>(DGpu),
                             3);

    Kernel::Err::PrintError(error);

    float* lpfDDataCpu = (float*)malloc(sizeof(float)*InputDataSize);
    cudaMemcpy(lpfDDataCpu, DGpu, sizeof(float)*InputDataSize, cudaMemcpyDeviceToHost);

    for(int lnIdy = 0; lnIdy < lnNumRows; lnIdy++)
    {
        for(int lnIdx = 0; lnIdx < lnNumCols; lnIdx++)
        {
            int Index = lnIdy*lnNumRows + lnIdx;
            std::cout << " idx " << Index << " " << lpfDDataCpu[Index] << " ";
        }
        std::cout << std::endl;
    }

    cublasDestroy(lcCublasHandle);
    cudaFree(AGpu);
    cudaFree(CGpu);
    cudaFree(DGpu);
    free(lpfCDataCpu);
    free(lpfDDataCpu);

}

void RunDiagonalSumTest(void)
{
    //std::vector<float> A = {1,1,1,  2,5,1,  4,9,15};
    std::vector<float>   A = {0,3,1,1,2,0,
                              3,2,6,6,2,2,
                              1,6,4,5,6,1,
                              1,6,5,4,6,1,
                              2,2,6,6,2,3,
                              0,2,1,1,3,0};

    int lnNumCols = 6;
    int lnNumRows = 6;
    int InputDataSize = lnNumCols * lnNumRows;

    void* AGpu;
    cudaMalloc(&AGpu, sizeof(float)*InputDataSize);
    cudaMemcpy(AGpu, A.data(), sizeof(float)*InputDataSize, cudaMemcpyHostToDevice);

    void* RGpu;
    cudaMalloc(&RGpu, sizeof(float)*1);

    cudaStream_t lcCudaStream;
    cudaStreamCreate(&lcCudaStream);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(lnNumCols)/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(lnNumCols)/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;

    Kernel::Matrix::SumDiagonals(lsGridSize, lsBlockSize, lcCudaStream,
                    lnNumCols, static_cast<float*>(AGpu), static_cast<float*>(RGpu));

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpy(lpfRCpu, RGpu, sizeof(float)*1, cudaMemcpyDeviceToHost);

    std::cout << "Value : " << *lpfRCpu << std::endl;

    cudaFree(AGpu);
    cudaFree(RGpu);
    free(lpfRCpu);

}

int GetNum3Cycles(void* apnA, int anNumNodes)
{
    void* AGpu;
    int InputDataSize = anNumNodes*anNumNodes;
    cudaMalloc(&AGpu, sizeof(float)*InputDataSize);
    cudaMemcpy(AGpu, apnA, sizeof(float)*InputDataSize, cudaMemcpyHostToDevice);

    cudaStream_t lcCudaStream;
    cudaStreamCreate(&lcCudaStream);
    cublasHandle_t lcCublasHandle;
    auto handlereturn = cublasCreate(&lcCublasHandle);

    void* A2Gpu;
    cudaMalloc(&A2Gpu, sizeof(float)*InputDataSize);

    void* A3Gpu;
    cudaMalloc(&A3Gpu, sizeof(float)*InputDataSize);

    void* DiagGpu;
    cudaMalloc(&DiagGpu, sizeof(float)*1);

    const float lcScaleFactor = 1;
    const float lcZeroFactor = 0;

    int error = Kernel::Matrix::Multiply(lcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             anNumNodes,anNumNodes,anNumNodes,
                             &lcScaleFactor,
                             static_cast<float*>(AGpu),
                             anNumNodes,
                             static_cast<float*>(AGpu),
                             anNumNodes,
                             &lcZeroFactor,
                             static_cast<float*>(A2Gpu),
                             anNumNodes);

    error = Kernel::Matrix::Multiply(lcCublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             anNumNodes,anNumNodes,anNumNodes,
                             &lcScaleFactor,
                             static_cast<float*>(AGpu),
                             anNumNodes,
                             static_cast<float*>(A2Gpu),
                             anNumNodes,
                             &lcZeroFactor,
                             static_cast<float*>(A3Gpu),
                             anNumNodes);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(anNumNodes)/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(anNumNodes)/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;

    Kernel::Matrix::SumDiagonals(lsGridSize, lsBlockSize, lcCudaStream,
                    anNumNodes, static_cast<float*>(A3Gpu), static_cast<float*>(DiagGpu));

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpy(lpfRCpu, DiagGpu, sizeof(float)*1, cudaMemcpyDeviceToHost);

    int lnNumCycles = (int)((*lpfRCpu)/6);

    cudaFree(AGpu);
    cudaFree(A2Gpu);
    cudaFree(A3Gpu);
    cudaFree(DiagGpu);
    free(lpfRCpu);

    return lnNumCycles;
}

void EntireTestCase(void)
{
    std::vector<float> lcA = {0,1,0,0,0,0,
                              1,0,1,1,0,0,
                              0,1,0,1,1,0,
                              0,1,1,0,1,0,
                              0,0,1,1,0,1,
                              0,0,0,0,1,0};
    int lnNumNodes = 6;

    int lnNum3Cycles = GetNum3Cycles(lcA.data(), lnNumNodes);

    std::cout << lnNum3Cycles << std::endl;

    cudaStream_t lcCudaStream;
    cudaStreamCreate(&lcCudaStream);
                        
}

void TestCase4(void)
{
    std::string lcFileName;
    lcFileName = "v2GraphsSparse/5.txt";

    std::ifstream fileIn(lcFileName);

    int lnNumNodes;
    int lnNumEdges;
    float lnSparsity;

    fileIn >> lnNumNodes;
    fileIn >> lnNumEdges;
    fileIn >> lnSparsity;

    std::vector<int> lcSources;
    std::vector<int> lcDestinations;

    std::cout << lnNumNodes << " " << lnNumEdges << " " << lnSparsity << std::endl;

    int lnSrc;
    int lnDst;
    for(int lnIdx = 0; lnIdx < lnNumEdges; lnIdx++)
    {
        fileIn >> lnSrc;
        fileIn >> lnDst;

        lcSources.push_back(lnSrc);
        lcDestinations.push_back(lnDst);
    }

    for(int lnIdx=0; lnIdx < lcSources.size(); lnIdx++)
    {
        std::cout << lcSources[lnIdx] << " " << lcDestinations[lnIdx] << std::endl;
    }

    fileIn.close();

}

int main(int argc, char * argv[]) 
{
    //RunSimpleTest();
    //RunSimpleTest2();
    //RunDiagonalSumTest();
    //EntireTestCase();
    //TestCase4();

    std::string lcTest = "v2GraphsSparse/5.txt";

    TestCaseV2 Test(lcTest);
    Test.ComputeNumTriangles();
    std::cout << Test.GetNumTriangles() << std::endl;

    return 0;

}