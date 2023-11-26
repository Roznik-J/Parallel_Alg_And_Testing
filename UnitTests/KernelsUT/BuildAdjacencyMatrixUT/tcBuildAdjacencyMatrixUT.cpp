//*****************************************************
// Developed by J. Roznik
// 2023-11-25
// UT for CudaAlgorithm/TestCase
//*****************************************************

#include "gtest/gtest.h"
#include <tcBuildAdjacencyMatrix.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <set>

std::set<std::pair<int,int>> constructTestEdges(std::vector<int>& arcSources, std::vector<int>& arcDest)
{
    std::set<std::pair<int,int>> lcEdges;
    if(arcSources.size() != arcDest.size())
    {
        std::cout << "Warning, sources and destinations are not the same size" << std::endl;
        return lcEdges;
    }
    for(int lnIdx = 0 ; lnIdx < arcSources.size(); lnIdx++)
    {
        lcEdges.insert(std::pair{arcSources.at(lnIdx),arcDest.at(lnIdx)});
        lcEdges.insert(std::pair{arcDest.at(lnIdx),arcSources.at(lnIdx)});
    }
    return lcEdges;
}

int snWarpSize = 32;

TEST(tcBuildAdjacencyMatrixUT, BuildAdjacencyMatrixSmall) 
{ 
    cudaStream_t lcStream;
    cudaStreamCreate(&lcStream);

    int lnNumNodes = 4;

    void* lpnA;
    std::vector<int> lnSources = {1,2};
    cudaMalloc(&lpnA, sizeof(int)*lnSources.size());
    cudaMemsetAsync(lpnA, 0, sizeof(int)*lnSources.size(), lcStream);
    cudaMemcpyAsync(lpnA, lnSources.data(), sizeof(int)*lnSources.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpnB;
    std::vector<int> lnDest = {2,3};
    cudaMalloc(&lpnB, sizeof(int)*lnDest.size());
    cudaMemsetAsync(lpnB, 0, sizeof(int)*lnDest.size(), lcStream);
    cudaMemcpyAsync(lpnB, lnDest.data(), sizeof(int)*lnDest.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpnC;
    cudaMalloc(&lpnC, sizeof(float)*lnNumNodes*lnNumNodes);
    cudaMemsetAsync(lpnC, 0, sizeof(float)*lnNumNodes*lnNumNodes, lcStream);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(lnSources.size())/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(lnSources.size())/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;
    Kernel::Matrix::BuildAdjacencyMatrix(lsGridSize, lsBlockSize, lcStream,
                    lnSources.size(), lnNumNodes, static_cast<int*>(lpnA), static_cast<int*>(lpnB), static_cast<float*>(lpnC));

    float* lpfRCpu = (float*)malloc(sizeof(float)*lnNumNodes*lnNumNodes);
    cudaMemcpyAsync(lpfRCpu, lpnC, sizeof(float)*lnNumNodes*lnNumNodes, cudaMemcpyDeviceToHost, lcStream);

    std::set<std::pair<int,int>> lcEdges = constructTestEdges(lnSources, lnDest);
    int lnIndex = 0;
    bool lnPredicate = true;
    for(int lnSrc = 0; lnSrc < lnNumNodes; lnSrc++)
    {
        for(int lnDst = 0; lnDst < lnNumNodes; lnDst++)
        {
            if((lcEdges.contains({lnSrc, lnDst}) && (lpfRCpu[lnIndex] == 0)) || 
               (!lcEdges.contains({lnSrc, lnDst}) && (lpfRCpu[lnIndex] == 1)))
            {
                lnPredicate = false;
            }

            // Fail the test is the results is some number other than 0 or 1
            if(lpfRCpu[lnIndex] != 0 && lpfRCpu[lnIndex] != 1)
            {
                lnPredicate = false;
            }

            lnIndex++;
        }
    }

    EXPECT_TRUE(lnPredicate);

    cudaFree(lpnA);
    cudaFree(lpnB);
    cudaFree(lpnC);
    free(lpfRCpu);

    cudaStreamDestroy(lcStream);

}

TEST(tcBuildAdjacencyMatrixUT, BuildAdjacencyMatrixLarge) 
{ 
    cudaStream_t lcStream;
    cudaStreamCreate(&lcStream);

    int lnNumNodes = 500;

    void* lpnA;
    std::vector<int> lnSources = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
    cudaMalloc(&lpnA, sizeof(int)*lnSources.size());
    cudaMemsetAsync(lpnA, 0, sizeof(int)*lnSources.size(), lcStream);
    cudaMemcpyAsync(lpnA, lnSources.data(), sizeof(int)*lnSources.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpnB;
    std::vector<int> lnDest = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26};
    cudaMalloc(&lpnB, sizeof(int)*lnDest.size());
    cudaMemsetAsync(lpnB, 0, sizeof(int)*lnDest.size(), lcStream);
    cudaMemcpyAsync(lpnB, lnDest.data(), sizeof(int)*lnDest.size(), cudaMemcpyHostToDevice, lcStream);

    void* lpnC;
    cudaMalloc(&lpnC, sizeof(float)*lnNumNodes*lnNumNodes);
    cudaMemsetAsync(lpnC, 0, sizeof(float)*lnNumNodes*lnNumNodes, lcStream);

    dim3 lsGridSize{};
    dim3 lsBlockSize{};
    int lnThreadsPerBlock = 1024;
    lsGridSize.x =std::ceil(static_cast<float>(lnSources.size())/lnThreadsPerBlock);
    int lnNumWarps = std::ceil((static_cast<float>(lnSources.size())/lsGridSize.x)/snWarpSize);
    lsBlockSize.x = lnNumWarps*snWarpSize;
    Kernel::Matrix::BuildAdjacencyMatrix(lsGridSize, lsBlockSize, lcStream,
                    lnSources.size(), lnNumNodes, static_cast<int*>(lpnA), static_cast<int*>(lpnB), static_cast<float*>(lpnC));

    float* lpfRCpu = (float*)malloc(sizeof(float)*lnNumNodes*lnNumNodes);
    cudaMemcpyAsync(lpfRCpu, lpnC, sizeof(float)*lnNumNodes*lnNumNodes, cudaMemcpyDeviceToHost, lcStream);

    std::set<std::pair<int,int>> lcEdges = constructTestEdges(lnSources, lnDest);
    int lnIndex = 0;
    bool lnPredicate = true;
    for(int lnSrc = 0; lnSrc < lnNumNodes; lnSrc++)
    {
        for(int lnDst = 0; lnDst < lnNumNodes; lnDst++)
        {
            if((lcEdges.contains({lnSrc, lnDst}) && (lpfRCpu[lnIndex] == 0)) || 
               (!lcEdges.contains({lnSrc, lnDst}) && (lpfRCpu[lnIndex] == 1)))
            {
                lnPredicate = false;
            }

            // Fail the test is the results is some number other than 0 or 1
            if(lpfRCpu[lnIndex] != 0 && lpfRCpu[lnIndex] != 1)
            {
                lnPredicate = false;
            }

            lnIndex++;
        }
    }

    EXPECT_TRUE(lnPredicate);

    cudaFree(lpnA);
    cudaFree(lpnB);
    cudaFree(lpnC);
    free(lpfRCpu);

    cudaStreamDestroy(lcStream);

}


int main(int argc, char *argv[]) 
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}