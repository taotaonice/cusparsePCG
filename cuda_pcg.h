#ifndef CUDA_PCG_H
#define CUDA_PCG_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cuda.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusparse.h"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

struct csrformatstruct
{
    int row;
    int col;
    float value;
};

class CUDA_pcg
{
public:
    CUDA_pcg();
    ~CUDA_pcg();
    void pcg(int* cooRowIndexHostPtr, int* cooColIndexHostPtr,
             float* cooValHostPtr, int M, int N, int nz,
             float* x, float *rhs, float *csrValDiagAllCon);

    void setSortedCsrFormat(int* rowInd, int* colInd, float* value, int nnz,
                            int* sortedRowInd, int* sortedColInd, float* sortedValue);

    void setSortedCsrFormatTwice(int* rowInd, int* colInd, float* value, int nnz,
                                      int* sortedRowInd, int* sortedColInd, float* sortedValue);

    void cu_pcg(int* cooRowIndexHostPtr, int* cooColIndexHostPtr,
             float* cooValHostPtr, int M, int N, int nz,
             float* x, float *rhs, float *csrValDiagAllCon);

    // C = alphs*A + beta*B
//    void sparseAddSparse(int* csrRowPtrC, int* csrColIndC, float* csrValC, int& nnzC, cusparseMatDescr_t descrC,
//                         int* csrRowPtrA, int* csrColIndA, float* csrValA, int nnzA, cusparseMatDescr_t descrA,
//                         int* csrRowPtrB, int* csrColIndB, float* csrValB, int nnzB, cusparseMatDescr_t descrB,
//                         cusparseHandle_t cusparseHandle, cusparseStatus_t cusparseStatus,
//                         const float alpha, const float beta, int M, int N);

private:
    cudaError_t cudaStat1, cudaStat2, cudaStat3;
    cusparseStatus_t cusparseStatus;
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descrA, descrB, descrC, descrL, descrH;
    cublasHandle_t cublasHandle;
    cublasStatus_t cublasStatus;
};

#endif // CUDA_PCG_H
