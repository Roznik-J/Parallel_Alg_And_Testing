#include "TestCase.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <stdio.h>

#include <tcSquare.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

static const int snWarpSize = 32;

using namespace std;

TestCaseV2::TestCaseV2(string fileName) : fileName(fileName) {
    ifstream fileIn(fileName);;

    // first line is the number of verticies in the graph
    int numVerticies;
    fileIn >> numVerticies;
    fileIn >> edgeSum;

    // resize the vector to hold `N` elements
    adjList.resize(numVerticies);

    // second line is whether or not it is an undirected graph (directed: false, undirected: true)
    bool isUndirected;
    fileIn >> std::boolalpha >> isUndirected;
    this->isUndirected = isUndirected;

    // read rest of lines and add edges to the graph
    int src, dest, weight;
    while (fileIn >> src >> dest >> weight) {
        adjList[src].push_back({dest, weight});
        if (isUndirected) {
            adjList[dest].push_back({src, weight});
        }
    }
}

int TestCaseV2::printSum()
{
    return edgeSum;
}

void TestCaseV2::printList() {
    if (isUndirected) {
        cout << endl << "------ Undirected Graph ";
    } else {
        cout << endl << "------ Directed Graph ";
    }
    cout << "from file: " << fileName << " ------" << endl;
    for (size_t i = 0; i < this->adjList.size(); i++) {
        cout << i << ": ";
        for (size_t j = 0; j < this->adjList.at(i).size(); j++) {
            Edge edge = this->adjList.at(i).at(j);
            if (j == 0) {
                cout << "(" << edge.dest << ", " << edge.weight << ")";
            } else {
                cout << " -> (" << edge.dest << ", " << edge.weight << ")";
            }
        }
        cout << endl;
    }
}

void TestCaseV2::RunSquareTest(void)
{
    cout << "here" << endl;

    cudaStream_t lcCudaStream;
    cudaStreamCreate(&lcCudaStream);

    // Input Data
    vector<float> lnInputData = {1,2,3,4,5,6,7,8};
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
    vector<float> lnExpectedOut;
    for(auto val : lnInputData)
    {
        lnExpectedOut.push_back(val*val);
    }

    for(int lnIdx=0; lnIdx < InputDataSize; lnIdx++)
    {
        std::cout << "In : " << lnInputData[lnIdx] << " Out: " << lpfOutputDataCpu[lnIdx] << " Exp: " << lnExpectedOut[lnIdx] << std::endl;
    }

    //Destruction
    free(lpfOutputDataCpu);
    cudaFree(lpvGpuInputData);
    cudaFree(lpvGpuOutputData);
    cudaStreamDestroy(lcCudaStream);
}