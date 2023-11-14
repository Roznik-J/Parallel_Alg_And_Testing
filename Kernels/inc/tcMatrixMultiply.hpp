//*****************************************************
// Developed by J. Roznik
// 2023-11-11
// Multiplies Matrixes using the cubLAS library
//*****************************************************

#ifndef TCSMATRIXMULTIPLY_HPP_
#define TCSMATRIXMULTIPLY_HPP_

#include "cublas_v2.h"
#include <cuda_runtime_api.h>
#include <iostream>

namespace Kernel
{
namespace Matrix
{

int Multiply(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldct);

}
namespace Err
{
    int GetError(cublasStatus_t error);
    void PrintError(int anErr);
    void PrintCudaError(std::string& arcMsg, int anErr);
}
}

#endif