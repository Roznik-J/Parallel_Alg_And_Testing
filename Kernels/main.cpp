#include <stdio.h>
#include <vector>
#include <iostream>

#include <tcSquare.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

static const int snWarpSize = 32;

int main(void)
{
    cudaStream_t lcCudaStream;
    cudaStreamCreate(&lcCudaStream);

    // Input Data
    std::vector<float> lnInputData = {1,2,3,4,5,6,7,8};
    //float lnInputData[] = {1,2,3,4,5,7,8,9};
    int InputDataSize = lnInputData.size();

    // GPU DATA Management
    //     Input Data
    void* lpvGpuInputData;
    cudaMalloc(&lpvGpuInputData, sizeof(float)*InputDataSize);
    cudaMemcpy(lpvGpuInputData, lnInputData.data(), sizeof(float)*InputDataSize, cudaMemcpyHostToDevice);
    //     Output Data
    void* lpvGpuOutputData;
    cudaMalloc(&lpvGpuOutputData, sizeof(float)*InputDataSize);

    // Create Thread Structures
    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(InputDataSize)/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(InputDataSize)/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;

    // Launch Kernel
    Kernel::Square::LaunchSquareValues(lsGridSize, lsBlockSize, lcCudaStream, snWarpSize, reinterpret_cast<float*>(lpvGpuInputData), reinterpret_cast<float*>(lpvGpuOutputData));

    // Copy data from device to host
    float* lpfOutputDataCpu = (float*)malloc(sizeof(float)*InputDataSize);
    cudaMemcpy(lpfOutputDataCpu, lpvGpuOutputData, sizeof(float)*InputDataSize, cudaMemcpyDeviceToHost);

    // Expected Values
    std::vector<float> lnExpectedOut;
    for(auto val : lnInputData)
    {
        lnExpectedOut.push_back(val*val);
    }

    for(unsigned int lnIdx=0; lnIdx < InputDataSize; lnIdx++)
    {
        std::cout << "In : " << lnInputData[lnIdx] << " Out: " << lpfOutputDataCpu[lnIdx] << " Exp: " << lnExpectedOut[lnIdx] << std::endl;
    }

    //Destruction
    free(lpfOutputDataCpu);
    cudaFree(lpvGpuInputData);
    cudaFree(lpvGpuOutputData);
    cudaStreamDestroy(lcCudaStream);

}
