//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Test Case class that handles Graph Testing with Cuda
//*****************************************************

#ifndef TESTCASEV2_HPP
#define TESTCASEV2_HPP

#include <vector>
#include <string>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

class TestCaseV2 {

public:

    TestCaseV2(std::string& arcFileName);

    virtual ~TestCaseV2(void);

    void printEdges(void);

    void printGraphInfo(void);

    void printAdjacencyMatrix(void);

    void ComputeNumTriangles(void);

    int GetNumTriangles(void);

    int GetNodeSize(void);

private:

    void ConstructAdjacencyMatrix(void);

    std::vector<int> mcSources{};

    std::vector<int> mcDestinations{};

    int mnNumNodes{0};

    int mnNumEdges{0};

    float mrSparsity{0};

    cudaStream_t mcCudaStream;

    cublasHandle_t mcCublasHandle;

    void* mpvSourcesGpu{nullptr};

    void* mpvDestGpu{nullptr};

    void* mpvAdjGpu{nullptr};

    void* mpvAdj2Gpu{nullptr};

    void* mpvAdj3Gpu{nullptr};

    void* mpvDiagOutput{nullptr};

    int mnNumTriangles{0};
};

#endif