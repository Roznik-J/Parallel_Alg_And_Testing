//*****************************************************
// Developed by J. Roznik
// 2023-11-26
// UT for Kernel/tcMatrixMultiply
//*****************************************************

#include "gtest/gtest.h"
#include <tcMatrixMultiply.hpp>
#include <iostream>
#include <random>

// Add an epsilon for differences in floating point arithmetic between the CPU and GPU
bool ExpectNear(float lrActual, float lrTest, float lrTolerance)
{
    float lrT = (lrActual - lrTest)/lrActual;

    if((lrT < lrTolerance) && (lrT > -lrTolerance))
    {
        return true;
    }
    return false;
}

// For a square Matrix
bool ValidateResultsSquareMatrix(float* aprCpuData, float* aprGpuData, bool abIsRowMjr, int anSize)
{
    bool lbPredicate = true;

    for(int lnY = 0; lnY < anSize; lnY++)
    {
        for(int lnX = 0; lnX < anSize; lnX++)
        {
            int lnCpuIdx = lnY*anSize + lnX;
            int lnGpuIdx;
            // Cublas operations are mostly row major.
            if(abIsRowMjr)
            {
                lnGpuIdx = lnX*anSize + lnY; 
            }
            else
            {
                lnGpuIdx = lnCpuIdx;
            }

            if(!ExpectNear(aprCpuData[lnCpuIdx], aprGpuData[lnGpuIdx], 0.0001))
            {
                lbPredicate = false;
                std::cout << "EXPECTED : " << aprCpuData[lnCpuIdx] << " RECEIVED : " << aprGpuData[lnGpuIdx] << " ";
            }
        }
    }

    return lbPredicate;
}

TEST(tcMatrixMultiply, Small_ShouldPAss)
{
    cudaStream_t lcStream;
    cudaStreamCreate(&lcStream);
    cublasHandle_t lcCublasHandle;
    cublasCreate(&lcCublasHandle);

    std::vector<float> lcA = {1,2,3,4,5,
                              6,7,8,9,10,
                              11,12,13,14,15,
                              16,17,18,19,20,
                              21,22,23,24,25};

    std::vector<float> lcB = {25,24,23,22,21,
                              20,19,18,17,16,
                              15,14,13,12,11,
                              10,9,8,7,6,
                              5,4,3,2,1};

    std::vector<float> lcExpected = {175,160,145,130,115,
                                     550,510,470,430,390,
                                     925,860,795,730,665,
                                     1300,1210,1120,1030,940,
                                     1675,1560,1445,1330,1215};

    int lnNumRows = 5;
    int lnNumCols = lnNumRows;

    void* lpcAGpu;
    cudaMalloc(&lpcAGpu, sizeof(float)*lcA.size());
    cudaMemcpyAsync(lpcAGpu, lcA.data(), sizeof(float)*lcA.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpcBGpu;
    cudaMalloc(&lpcBGpu, sizeof(float)*lcA.size());
    cudaMemcpyAsync(lpcBGpu, lcB.data(), sizeof(float)*lcB.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpcCGpu;
    cudaMalloc(&lpcCGpu, sizeof(float)*lnNumRows*lnNumCols);

    float lcZeroFactor = 0;
    float lcScaleFactor = 1;
    
    int error = Kernel::Matrix::Multiply(lcCublasHandle, 
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_T,
                                         lnNumRows,lnNumRows,lnNumRows,
                                         &lcScaleFactor,
                                         static_cast<float*>(lpcAGpu),
                                         lnNumRows,
                                         static_cast<float*>(lpcBGpu),
                                         lnNumRows,
                                         &lcZeroFactor,
                                         static_cast<float*>(lpcCGpu),
                                         lnNumRows);

    // Expect a zero return value
    EXPECT_EQ(error, 0);

    float* lpcGpuResult = (float*)malloc(sizeof(float)*lnNumRows*lnNumRows);
    cudaMemcpyAsync(lpcGpuResult, lpcCGpu, sizeof(float)*lnNumRows*lnNumRows, cudaMemcpyDeviceToHost, lcStream);

    std::vector<float> lcGpuResult(lpcGpuResult, lpcGpuResult + lnNumRows*lnNumRows);

    EXPECT_TRUE(ValidateResultsSquareMatrix(lcExpected.data(), lpcGpuResult, true, lnNumRows));

    cudaFree(lpcAGpu);
    cudaFree(lpcBGpu);
    cudaFree(lpcCGpu);
    free(lpcGpuResult);
    cublasDestroy(lcCublasHandle);
    cudaStreamDestroy(lcStream);

}


int main(int argc, char *argv[]) 
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}