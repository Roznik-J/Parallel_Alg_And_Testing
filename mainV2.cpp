
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include <tcMatrixMultiply.hpp>

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

int main(int argc, char * argv[]) 
{
    //RunSimpleTest();
    RunSimpleTest2();

}