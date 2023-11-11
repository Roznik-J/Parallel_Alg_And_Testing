#include <tcMatrixMultiply.hpp>
#include <iostream>

int Kernel::Err::GetError(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return 0;

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return 1;

        case CUBLAS_STATUS_ALLOC_FAILED:
            return 2;

        case CUBLAS_STATUS_INVALID_VALUE:
            return 3;

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return 4;

        case CUBLAS_STATUS_MAPPING_ERROR:
            return 5;

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return 6;

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return 7;
    }
    return -1;
}

void Kernel::Err::PrintError(int anErr)
{
    switch (anErr)
    {
        case 0:
            //std::cout << "CUBLAS_STATUS_SUCCESS" << std::endl;
            break;

        case 1:
            std::cout << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;
            break;

        case 2:
            std::cout << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl;
            break;

        case 3:
            std::cout << "CUBLAS_STATUS_INVALID_VALUE" << std::endl;
            break;

        case 4:
            std::cout << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;
            break;

        case 5:
            std::cout << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl;
            break;

        case 6:
            std::cout << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;
            break;

        case 7:
            std::cout << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl;
            break;
        default:
            std::cout << "<unknown>" << std::endl;
    }
}

int Kernel::Matrix::Multiply(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    auto leResult = cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    int errorenum = Kernel::Err::GetError(leResult);
    return errorenum;
}
