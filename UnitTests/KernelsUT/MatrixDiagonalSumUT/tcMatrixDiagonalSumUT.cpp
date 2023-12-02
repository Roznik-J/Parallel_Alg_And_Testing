//*****************************************************
// Developed by J. Roznik
// 2023-11-26
// UT for Kernel/tcMatrixDiagonalSum
//*****************************************************

#include "gtest/gtest.h"
#include <tcMatrixDiagonalSum.hpp>
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

float CpuDiagonalSum(float* aprData, int anSize)
{
    float lrSum = 0;
    int lnDiagEntry = 0;
    for(int lnIdx = 0; lnIdx < anSize; lnIdx++)
    {
        lnDiagEntry = lnIdx*anSize + lnIdx;
        lrSum += aprData[lnDiagEntry];
    }
    return lrSum;
}

TEST(tcMatrixDiagonalSum, Small_ShouldPass)
{
    cudaStream_t lcStream;
    cudaStreamCreate(&lcStream);

    int lnMSize = 5;
    std::vector<float> lrTestData = {5,1,1,1,1,
                                     1,5,1,1,1,
                                     1,1,5,1,1,
                                     1,1,1,5,1,
                                     1,1,1,1,5};

    float lrExpectedResult = CpuDiagonalSum(lrTestData.data(), lnMSize);

    EXPECT_EQ(25, lrExpectedResult);

    void* lpvDataGpu;
    cudaMalloc(&lpvDataGpu, sizeof(float)*lrTestData.size());
    cudaMemcpyAsync(lpvDataGpu, lrTestData.data(), sizeof(float)*lrTestData.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpvRGpu;
    cudaMalloc(&lpvRGpu, sizeof(float)*1);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    int lnNumReduceThreads = std::ceil(std::sqrt(lnMSize));
    lsGridSize.x =std::ceil(static_cast<float>(lnNumReduceThreads)/lnThreadsPerBlock);
    lsBlockSize.x = lnNumReduceThreads;
    Kernel::Matrix::SumDiagonalsOptimized(lsGridSize, lsBlockSize, lcStream,
                    lnMSize, static_cast<float*>(lpvDataGpu), static_cast<float*>(lpvRGpu), lnNumReduceThreads);

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpyAsync(lpfRCpu, lpvRGpu, sizeof(float)*1, cudaMemcpyDeviceToHost, lcStream);

    EXPECT_TRUE(ExpectNear(lrExpectedResult, *lpfRCpu, 0.0001));

    cudaFree(lpvDataGpu);
    cudaFree(lpvRGpu);
    free(lpfRCpu);
    cudaStreamDestroy(lcStream);
} 

TEST(tcMatrixDiagonalSum, Medium_ShouldPass)
{
    cudaStream_t lcStream;
    cudaStreamCreate(&lcStream);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    float min = -50.0;
    float max = 50.0;
    
    std::uniform_real_distribution<float> GetRandomFloat(min, max);

    int lnMSize = 100;
    std::vector<float> lrTestData;

    for(int lnIdx = 0; lnIdx < lnMSize*lnMSize; lnIdx++)
    {
        lrTestData.push_back(GetRandomFloat(gen));
    }

    float lrExpectedResult = CpuDiagonalSum(lrTestData.data(), lnMSize);

    void* lpvDataGpu;
    cudaMalloc(&lpvDataGpu, sizeof(float)*lrTestData.size());
    cudaMemcpyAsync(lpvDataGpu, lrTestData.data(), sizeof(float)*lrTestData.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpvRGpu;
    cudaMalloc(&lpvRGpu, sizeof(float)*1);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    int lnNumReduceThreads = std::ceil(std::sqrt(lnMSize));
    lsGridSize.x =std::ceil(static_cast<float>(lnNumReduceThreads)/lnThreadsPerBlock);
    lsBlockSize.x = lnNumReduceThreads;
    Kernel::Matrix::SumDiagonalsOptimized(lsGridSize, lsBlockSize, lcStream,
                    lnMSize, static_cast<float*>(lpvDataGpu), static_cast<float*>(lpvRGpu), lnNumReduceThreads);

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpyAsync(lpfRCpu, lpvRGpu, sizeof(float)*1, cudaMemcpyDeviceToHost, lcStream);

    EXPECT_TRUE(ExpectNear(lrExpectedResult, *lpfRCpu, 0.0001));

    cudaFree(lpvDataGpu);
    cudaFree(lpvRGpu);
    free(lpfRCpu);
    cudaStreamDestroy(lcStream);
} 

TEST(tcMatrixDiagonalSum, Large_ShouldPass)
{
    cudaStream_t lcStream;
    cudaStreamCreate(&lcStream);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    float min = -50.0;
    float max = 50.0;
    
    std::uniform_real_distribution<float> GetRandomFloat(min, max);

    int lnMSize = 2000;
    std::vector<float> lrTestData;

    for(int lnIdx = 0; lnIdx < lnMSize*lnMSize; lnIdx++)
    {
        lrTestData.push_back(GetRandomFloat(gen));
    }

    float lrExpectedResult = CpuDiagonalSum(lrTestData.data(), lnMSize);

    void* lpvDataGpu;
    cudaMalloc(&lpvDataGpu, sizeof(float)*lrTestData.size());
    cudaMemcpyAsync(lpvDataGpu, lrTestData.data(), sizeof(float)*lrTestData.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpvRGpu;
    cudaMalloc(&lpvRGpu, sizeof(float)*1);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    int lnNumReduceThreads = std::ceil(std::sqrt(lnMSize));
    lsGridSize.x =std::ceil(static_cast<float>(lnNumReduceThreads)/lnThreadsPerBlock);
    lsBlockSize.x = lnNumReduceThreads;
    Kernel::Matrix::SumDiagonalsOptimized(lsGridSize, lsBlockSize, lcStream,
                    lnMSize, static_cast<float*>(lpvDataGpu), static_cast<float*>(lpvRGpu), lnNumReduceThreads);

    float* lpfRCpu = (float*)malloc(sizeof(float)*1);
    cudaMemcpyAsync(lpfRCpu, lpvRGpu, sizeof(float)*1, cudaMemcpyDeviceToHost, lcStream);

    EXPECT_TRUE(ExpectNear(lrExpectedResult, *lpfRCpu, 0.0001));

    cudaFree(lpvDataGpu);
    cudaFree(lpvRGpu);
    free(lpfRCpu);
    cudaStreamDestroy(lcStream);
} 


int main(int argc, char *argv[]) 
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}