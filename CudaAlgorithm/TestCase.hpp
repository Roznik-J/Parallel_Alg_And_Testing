//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Test Case class that handles Graph Testing with Cuda
//*****************************************************

#ifndef TESTCASE_HPP
#define TESTCASE_HPP

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

    int GetNumEdges(void);

    void GetTimingInformation(std::vector<float>& arcTimeParameters);

private:

    void ConstructAdjacencyMatrix(void);

    void CreateCudaEvents(void);

    void DestroyCudaEvents(void);

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

    // Cuda Events For Adjacency Builder
    cudaEvent_t mcStartAdjBuildSetup;
    cudaEvent_t mcStopAdjBuildSetup;
    cudaEvent_t mcStartAdjBuild;
    cudaEvent_t mcStopAdjBuild;

    float mrAdjBuildSetupMS{0};
    float mcAdjBuildMS{0};

    // Cuda Events for Matrix Operations
    cudaEvent_t mcStartAdjPowerOne;
    cudaEvent_t mcStopAdjPowerOne;
    cudaEvent_t mcStartAdjPowerTwo;
    cudaEvent_t mcStopAdjPowerTwo;
    cudaEvent_t mcStartDiagSum;
    cudaEvent_t mcStopDiagSum;

    float mrAdjPowerOneMS{0};
    float mrAdjPowerTwoMS{0};
    float mrDiagSumMS{0};

    // Cuda Events for Overall Program - Note, best practise is to
    // comment out the other cudaEvent_t since cudaEventSynchronize()
    // Is a blocking call on the CPU, so it can alter results

    cudaEvent_t mcStartProgram;
    cudaEvent_t mcStopProgram;

    float mrProgramTimeMS{0};

};

#endif