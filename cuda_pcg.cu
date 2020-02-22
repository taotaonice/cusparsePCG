#include "cuda_pcg.h"
#include <ctime>

CUDA_pcg::CUDA_pcg()
{
    cublasStatus = cublasCreate(&cublasHandle);

    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseCreate failture  !!!" << std::endl;
        return ;
    }
    /* create and setup matrix descriptor */
    cusparseStatus = cusparseCreateMatDescr(&descrA);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseCreateMatDescrA failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseCreateMatDescr(&descrB);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseCreateMatDescrB failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseCreateMatDescr(&descrC);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseCreateMatDescrC failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseCreateMatDescr(&descrL);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseCreateMatDescrL failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseCreateMatDescr(&descrH);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseCreateMatDescrH failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatTypeA failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatTypeB failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatTypeC failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatTypeL failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatType(descrH, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatTypeH failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatIndexBaseA failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatIndexBaseB failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatIndexBaseC failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatIndexBaseL failture  !!!" << std::endl;
        return ;
    }
    cusparseStatus = cusparseSetMatIndexBase(descrH, CUSPARSE_INDEX_BASE_ZERO);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cusparseSetMatIndexBaseH failture  !!!" << std::endl;
        return ;
    }
}

CUDA_pcg::~CUDA_pcg()
{
    /* Destroy contexts */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
}

__global__ void cuSetValue(int* value, int data_len)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < data_len)
    {
        value[tid] = tid;
    }
}

__global__ void cuCopyCfsToR(csrformatstruct* cfs, int* rowInd, int data_len)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < data_len)
    {
        rowInd[tid] = cfs[tid].row;
    }
}

__global__ void cuCopyRcvToCfs(csrformatstruct* cfs, int* rowInd, int* colInd, float* value, int data_len)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < data_len)
    {
        cfs[tid].row = rowInd[tid];
        cfs[tid].col = colInd[tid];
        cfs[tid].value = value[tid];
    }
}

__global__ void cuCopyCfsToRcv(csrformatstruct* cfs, int* rowInd, int* colInd, float* value, int data_len)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < data_len)
    {
        rowInd[tid] = cfs[tid].row;
        colInd[tid] = cfs[tid].col;
        value[tid] = cfs[tid].value;
    }
}

// sum(C, 2) --- C = A+A';
__global__ void cuSumA2(float* D, int* row, int* col, float* A, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < length)
    {
        atomicAdd(&D[row[idx]], A[idx]);
        atomicAdd(&D[col[idx]], A[idx]);

//        if(row[idx] == col[idx])
//        {
//            atomicAdd(&D[row[idx]], -2 * A[idx]);
//        }
    }
}

__global__ void cuSetCarValD(float* D, float* csrValDiagAllCon, float lambda, int nz)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < nz)
    {
        D[idx] = lambda * csrValDiagAllCon[idx];
    }
}

void CUDA_pcg::setSortedCsrFormat(int* rowInd, int* colInd, float* value, int nnz,
                                  int* sortedRowInd, int* sortedColInd, float* sortedValue)
{
//    TimingGPU tg;

    int threads = 256;
    int blocks = nnz /threads + ((nnz % threads) ? 1:0);

    csrformatstruct* cfs = NULL;
    cudaMalloc((void**)&cfs, sizeof(csrformatstruct) * nnz);

    cuCopyRcvToCfs<<<blocks, threads>>>(cfs, rowInd, colInd, value, nnz);

    int* tmp_rowInd = NULL;
    cudaMalloc((void**)&tmp_rowInd, sizeof(int) * nnz);
    cudaMemcpy(tmp_rowInd, rowInd, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    //    tg.StartCounter();
    thrust::device_ptr<int> dev_row_ptr(tmp_rowInd);
    thrust::device_ptr<csrformatstruct> dev_csf_ptr(cfs);

//    tg.StartCounter();
    thrust::sort_by_key(dev_row_ptr, dev_row_ptr + nnz, dev_csf_ptr);
//    std::cout << "col thrust_time = " << tg.GetCounter() << " ms" << std::endl;

    csrformatstruct* first_sorted_cfs = thrust::raw_pointer_cast(dev_csf_ptr);

    cuCopyCfsToRcv<<<blocks, threads>>>(first_sorted_cfs, sortedRowInd, sortedColInd, sortedValue, nnz);

    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %d %s.\n", err, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }

    cudaFree(cfs);
    cudaFree(tmp_rowInd);
}


void CUDA_pcg::setSortedCsrFormatTwice(int* rowInd, int* colInd, float* value, int nnz,
                                  int* sortedRowInd, int* sortedColInd, float* sortedValue)
{
    int threads = 256;
    int blocks = nnz /threads + ((nnz % threads) ? 1:0);

    csrformatstruct* cfs = NULL;
    cudaMalloc((void**)&cfs, sizeof(csrformatstruct) * nnz);

    cuCopyRcvToCfs<<<blocks, threads>>>(cfs, rowInd, colInd, value, nnz);

    int* tmp_colInd = NULL;
    cudaMalloc((void**)&tmp_colInd, sizeof(int) * nnz);
    cudaMemcpy(tmp_colInd, colInd, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    thrust::device_ptr<int> dev_col_ptr(tmp_colInd);
    thrust::device_ptr<csrformatstruct> dev_csf_ptr(cfs);

    thrust::sort_by_key(dev_col_ptr, dev_col_ptr + nnz, dev_csf_ptr);

    csrformatstruct* first_sorted_cfs = thrust::raw_pointer_cast(dev_csf_ptr);

    int* first_sorted_row = NULL;
    cudaMalloc((void**)&first_sorted_row, sizeof(int) * nnz);
    cuCopyCfsToR<<<blocks, threads>>>(first_sorted_cfs, first_sorted_row, nnz);

    thrust::device_ptr<int> thrust_first_sorted_row(first_sorted_row);

    thrust::sort_by_key(thrust_first_sorted_row, thrust_first_sorted_row + nnz, dev_csf_ptr);


    csrformatstruct* second_sorted_cfs = thrust::raw_pointer_cast(dev_csf_ptr);

    cuCopyCfsToRcv<<<blocks, threads>>>(second_sorted_cfs, sortedRowInd, sortedColInd, sortedValue, nnz);

    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %d %s.\n", err, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }

    cudaFree(cfs);
    cudaFree(first_sorted_row);
    cudaFree(tmp_colInd);
}



void CUDA_pcg::pcg(int* cooRowIndexHostPtr, int* cooColIndexHostPtr,
         float* cooValHostPtr, int M, int N, int nz,
         float* x, float *rhs, float *csrValDiagAllCon)
{
//    TimingGPU tg;
    int threads = 256;
    int blocks = M / threads + ((M % threads) ? 1:0);

    int* sortedRowIndA = NULL;
    int* sortedColIndA = NULL;
    float* sortedValueA = NULL;

    int n = M;
    assert(M == N);
    int nnz = nz;

    /* allocate GPU memory and copy the matrix and vectors into it */
    cudaStat1 = cudaMalloc((void**)&sortedRowIndA, nnz*sizeof(sortedRowIndA[0]));
    cudaStat2 = cudaMalloc((void**)&sortedColIndA, nnz*sizeof(sortedColIndA[0]));
    cudaStat3 = cudaMalloc((void**)&sortedValueA, nnz*sizeof(sortedValueA[0]));

    if ((cudaStat1 != cudaSuccess) ||
            (cudaStat2 != cudaSuccess) ||
            (cudaStat3 != cudaSuccess))
    {
        std::cout << "malloc 2 failture  !!!" << std::endl;
        return ;
    }

    cudaStat1 = cudaMemcpy(sortedRowIndA, cooRowIndexHostPtr,
                           (size_t)(nnz*sizeof(sortedRowIndA[0])),
            cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(sortedColIndA, cooColIndexHostPtr,
                           (size_t)(nnz*sizeof(sortedColIndA[0])),
            cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(sortedValueA, cooValHostPtr,
                           (size_t)(nnz*sizeof(sortedValueA[0])),
            cudaMemcpyHostToDevice);

    if ((cudaStat1 != cudaSuccess) ||
            (cudaStat2 != cudaSuccess) ||
            (cudaStat3 != cudaSuccess))
    {
        std::cout << "cudaMemcpy 1 failture  !!!" << std::endl;
        return ;
    }

    /* get sorted col index and sorted key for ATranspose*/
//    std::cout << "get sorted row index and sorted key for AT: " << std::endl;
//    tg.StartCounter();

    int* sortedRowIndAT = NULL;
    int* sortedColIndAT = NULL;
    float* sortedValueAT = NULL;

    cudaMalloc((void**)&sortedRowIndAT, sizeof(int) * nnz);
    cudaMalloc((void**)&sortedColIndAT, sizeof(int) * nnz);
    cudaMalloc((void**)&sortedValueAT, sizeof(float) * nnz);


    setSortedCsrFormat(sortedColIndA, sortedRowIndA, sortedValueA, nnz,
                       sortedRowIndAT, sortedColIndAT, sortedValueAT);
//    std::cout << "sort time: " << tg.GetCounter() << std::endl;

//    // validate
//    int* h_sortedRowIndAT = new int[nnz];
//    int* h_sortedColIndAT = new int[nnz];
//    float* h_sortedValueAT = new float[nnz];

//    cudaMemcpy(h_sortedRowIndAT, sortedRowIndAT, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_sortedColIndAT, sortedColIndAT, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_sortedValueAT, sortedValueAT, sizeof(float) * nnz, cudaMemcpyDeviceToHost);

    // validate
//    std::cout << "h_sortedRowIndAT: " << std::endl;
//    for(int i = 0; i < 200; ++i)
//    {
//        std::cout << h_sortedRowIndAT[i] << "  " << h_sortedColIndAT[i] << "  " << h_sortedValueAT[i] << std::endl;
//    }

//    for(int i = 0; i < nnz; ++i)
//    {
//        std::cout << h_sortedRowIndAT[i] << "  " << h_sortedColIndAT[i] << "  " << h_sortedValueAT[i] << std::endl;
//    }

//    for(int i = 0; i < nnz; ++i)
//    {
//        std::cout << h_sortedValueAT[i] << std::endl;
//    }

    /* exercise conversion routines (convert matrix from COO 2 CSR format) */

        /* set csrRowPtrA */
//        tg.StartCounter();

        int* csrRowPtrA = NULL;
        cudaStat1 = cudaMalloc((void**)&csrRowPtrA, (n + 1)*sizeof(csrRowPtrA[0]));
        if (cudaStat1 != cudaSuccess)
        {
            std::cout << "cudaMalloc csrRowPtrA failture  !!!" << std::endl;
            return ;
        }

        cusparseStatus = cusparseXcoo2csr(cusparseHandle,
                                          sortedRowIndA,
                                          nnz,
                                          n,
                                          csrRowPtrA,
                                          CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcoo2csr csrRowPtrA failture  !!!" << std::endl;
            return ;
        }
//        std::cout << "set csrRowPtrA time = " << tg.GetCounter() << " ms" << std::endl;

//        // test  csrRowPtrHostA
//        int* csrRowPtrHostA = new int[n + 1];
//        cudaError_t(cudaMemcpy(csrRowPtrHostA, csrRowPtrA, (n + 1)*sizeof(csrRowPtrA[0]), cudaMemcpyDeviceToHost));
//        std::cout << "csrRowPtrHostA: " << std::endl;
//        for(int i = 0; i < n + 1; ++i)
//        {
//            std::cout << csrRowPtrHostA[i] << std::endl;
//        }

        /* set csrRowPtrAT */
//        tg.StartCounter();

        int* csrRowPtrAT = NULL;
        cudaStat1 = cudaMalloc((void**)&csrRowPtrAT, (n + 1)*sizeof(csrRowPtrA[0]));
        if (cudaStat2 != cudaSuccess)
        {
            std::cout << "cudaMalloc csrRowPtrAT failture  !!!" << std::endl;
            return ;
        }
        cusparseStatus = cusparseXcoo2csr(cusparseHandle,
                                          sortedRowIndAT,
                                          nnz,
                                          n,
                                          csrRowPtrAT,
                                          CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcoo2csr csrColPtr failture  !!!" << std::endl;
            return ;
        }
//        std::cout << "set csrRowPtrAT time = " << tg.GetCounter() << " ms" << std::endl;

//        // test  csrRowPtrHostA
//        int* csrRowPtrHostAT = new int[n + 1];
//        cudaError_t(cudaMemcpy(csrRowPtrHostAT, csrRowPtrAT, (n + 1)*sizeof(csrRowPtrAT[0]), cudaMemcpyDeviceToHost));
//        std::cout << "csrRowPtrHostAT: " << std::endl;
//        for(int i = 0; i < n + 1; ++i)
//        {
//            std::cout << csrRowPtrHostAT[i] << std::endl;
//        }

        /* A = A + A' */
//        tg.StartCounter();

        int baseC, nnzC;
        int *nnzTotalDevHostPtr = &nnzC;
        cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
        int* csrRowPtrC = NULL;
        cudaError_t(cudaMalloc((void**)&csrRowPtrC, sizeof(int) * (M + 1)));

        cusparseStatus = cusparseXcsrgeamNnz(cusparseHandle, M, N,
                                             descrA, nz,
                                             csrRowPtrA, sortedColIndA,
                                             descrB, nz,
                                             csrRowPtrAT, sortedColIndAT,
                                             descrC,
                                             csrRowPtrC, nnzTotalDevHostPtr);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcsrgeamNnz csrRowPtrC failture 1 !!!" << std::endl;
            return ;
        }

        if(NULL != nnzTotalDevHostPtr)
        {
            nnzC = *nnzTotalDevHostPtr;
        }
        else
        {
            cudaMemcpy(&nnzC, csrRowPtrC + M, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
            nnzC -= baseC;
        }

        const float A_alpha = 1.0;
        const float A_beta = 1.0;

        float* csrValC = NULL;
        int* csrColIndC = NULL;
        cudaError_t(cudaMalloc((void**)&csrValC, sizeof(float) * nnzC));
        cudaError_t(cudaMalloc((void**)&csrColIndC, sizeof(int) * nnzC));

        cusparseStatus = cusparseScsrgeam(cusparseHandle, M, N,
                         &A_alpha,
                         descrA, nz,
                         sortedValueA, csrRowPtrA, sortedColIndA,
                         &A_beta,
                         descrB, nz,
                         sortedValueAT, csrRowPtrAT, sortedColIndAT,
                         descrC,
                         csrValC, csrRowPtrC, csrColIndC);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseScsrgeam csrValC failture  !!!" << std::endl;
            return ;
        }
//        std::cout << "A + A' time = " << tg.GetCounter() << std::endl;

//        // validate csrRowPtrC
//        int* h_csrRowPtrC = new int[M + 1];
//        int* h_csrColIndC = new int[nnzC];
//        float* h_csrValC = new float[nnzC];

//        cudaMemcpy(h_csrRowPtrC, csrRowPtrC, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_csrColIndC, csrColIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_csrValC, csrValC, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);

//        std::cout << "h_csrRowPtrC: " << std::endl;
//        for(int i = 0; i < M + 1; ++i)
//        {
//            std::cout << h_csrRowPtrC[i] << std::endl;
//        }

//        std::cout << "h_csrColIndC h_csrValC: " << std::endl;
//        for(int i = 0; i < 500; ++i)
//        {
//            std::cout << h_csrColIndC[i] << "  " << h_csrValC[i] << std::endl;
//        }

//        std::cout << "h_csrColIndC h_csrValC: " << std::endl;
//        for(int i = nnzC - 500; i < nnzC; ++i)
//        {
//            std::cout << h_csrColIndC[i] << "  " << h_csrValC[i] << std::endl;
//        }
        ///////////////////////////////////////////
        ///
        ///
        ///             A + A'
        ///
        ///
        /// ///////////////////////////////////////////


//        tg.StartCounter();
        int* csrRowIndC = NULL;
        cudaMalloc((void**)&csrRowIndC, sizeof(int) * nnzC);

        cusparseStatus = cusparseXcsr2coo(cusparseHandle,
                                          csrRowPtrC,
                                          nnzC,
                                          M,
                                          csrRowIndC,
                                          CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseScsrgeam csrValC failture  !!!" << std::endl;
            return ;
        }
//        std::cout << "set A + A' row time " << tg.GetCounter() << std::endl;

//        int* h_csrRowIndC = new int[nnzC];
//        cudaMemcpy(h_csrRowIndC, csrRowIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
//        std::cout << "h_csrRowIndC: " << std::endl;
//        for(int i = 0; i < nnzC; ++i)
//        {
//            std::cout << h_csrRowIndC[i] << std::endl;
//        }


        // set sum(A, 2) and add lambda * spdiags(all_constraints, 0, n_pick, n_pick)
//        tg.StartCounter();
        float* csrValD  = NULL;
        cudaMalloc((void**)&csrValD, sizeof(int) * M);

        //float lambda = 100.0f;
        // inital carValD value to lambda * spdiags(all_constraints, 0, n_pick, n_pick)

        cuSetCarValD<<<blocks, threads>>>(csrValD, csrValDiagAllCon, 1, M);

//        float* h_csrValD = new float[M];
//        cudaMemcpy(h_csrValD, csrValD, sizeof(float) * M, cudaMemcpyDeviceToHost);
//        std::cout << "h_csrValD: " << std::endl;
//        for(int i = 0; i < M; ++i)
//        {
//            std::cout << h_csrValD[i] << std::endl;
//        }


//        cuSumA2<<<blocks, threads>>>(csrValD, csrRowIndC, csrColIndC, csrValC, nz);
        int blocks_sumA = nz / threads + ((nz % threads) ? 1:0);
        cuSumA2<<<blocks_sumA, threads>>>(csrValD, sortedRowIndA, sortedColIndA, sortedValueA, nz);
//        std::cout << "set csrValD time = " << tg.GetCounter() << " ms" << std::endl;

//        float* h_csrValD = new float[M];
//        cudaMemcpy(h_csrValD, csrValD, sizeof(float) * M, cudaMemcpyDeviceToHost);
//        std::cout << "h_csrValD: " << std::endl;
//        for(int i = 0; i < M; ++i)
//        {
//            std::cout << h_csrValD[i] << std::endl;
//        }


        ///////////////////////////////////////////
        ///
        ///
        ///             set D
        ///
        ///
        /// ///////////////////////////////////////////

//        tg.StartCounter();
        int* csrRowPtrD = NULL;
        int* csrColIndD = NULL;
        cudaMalloc((void**)&csrRowPtrD, sizeof(int) * (M + 1));
        cudaMalloc((void**)&csrColIndD, sizeof(int) * M);

        int blocks_ptr = (M + 1) / threads + (((M + 1) % threads) ? 1:0);
        cuSetValue<<<blocks_ptr, threads>>>(csrRowPtrD, M + 1);
        cuSetValue<<<blocks, threads>>>(csrColIndD, M);

//        std::cout << "cuSetValue time = " << tg.GetCounter() << " ms" << std::endl;

//        cusparseStatus = cusparseXcoo2csr(cusparseHandle,
//                                          csrRowIndC,
//                                          M,  // nnz
//                                          M,
//                                          csrRowPtrD,
//                                          CUSPARSE_INDEX_BASE_ZERO);
//        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
//        {
//            std::cout << "cusparseXcoo2csr csrRowPtrD failture  !!!" << std::endl;
//            return ;
//        }
//        std::cout << "set csrRowPtrD time = " << tg.GetCounter() << " ms" << std::endl;

//        int* h_csrRowPtrD = new int[M];
//        cudaMemcpy(h_csrRowPtrD, csrRowPtrD, sizeof(int) * M, cudaMemcpyDeviceToHost);
//        std::cout << "h_csrRowPtrD: " << std::endl;
//        for(int i = 0; i < M; ++i)
//        {
//            std::cout << h_csrRowPtrD[i] << std::endl;
//        }


        // L = D - A
//        tg.StartCounter();

        int* csrRowPtrL = NULL;
        int* csrColIndL = NULL;
        float* csrValL = NULL;
        int nnzL = 0;

        const float L_alpha = 1.0;
        const float L_beta = -1.0;

        int baseL;
        int *nnzTotalDevHostPtrL = &nnzL;
        cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
        cudaError_t(cudaMalloc((void**)&csrRowPtrL, sizeof(int) * (M + 1)));

        cusparseStatus = cusparseXcsrgeamNnz(cusparseHandle, M, N,
                                             descrA, M,
                                             csrRowPtrD, csrColIndD,
                                             descrC, nnzC,
                                             csrRowPtrC, csrColIndC,
                                             descrL,
                                             csrRowPtrL, nnzTotalDevHostPtrL);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcsrgeamNnz csrRowPtrC failture 2 !!!" << std::endl;
            return ;
        }

        if(NULL != nnzTotalDevHostPtrL)
        {
            nnzL = *nnzTotalDevHostPtrL;
        }
        else
        {
            cudaMemcpy(&nnzL, csrRowPtrL + M, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&baseL, csrRowPtrL, sizeof(int), cudaMemcpyDeviceToHost);
            nnzL -= baseL;
        }

        cudaError_t(cudaMalloc((void**)&csrValL, sizeof(float) * nnzL));
        cudaError_t(cudaMalloc((void**)&csrColIndL, sizeof(int) * nnzL));

        cusparseStatus = cusparseScsrgeam(cusparseHandle, M, N,
                                          &L_alpha,
                                          descrA, M,
                                          csrValD, csrRowPtrD, csrColIndD,
                                          &L_beta,
                                          descrC, nnzC,
                                          csrValC, csrRowPtrC, csrColIndC,
                                          descrL,
                                          csrValL, csrRowPtrL, csrColIndL);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseScsrgeam csrValC failture  !!!" << std::endl;
            return ;
        }
//        std::cout << "set L time = " << tg.GetCounter() << " ms" << std::endl;

//        // validate csrRowPtrL
//        int* h_csrRowPtrL = new int[M + 1];
//        int* h_csrColIndL = new int[nnzL];
//        float* h_csrValL = new float[nnzL];

//        cudaMemcpy(h_csrRowPtrL, csrRowPtrL, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_csrColIndL, csrColIndL, sizeof(int) * nnzL, cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_csrValL, csrValL, sizeof(float) * nnzL, cudaMemcpyDeviceToHost);

//        // validate csrRowPtrH
//        int* h_csrRowPtrL = new int[M + 1];
//        int* h_csrColIndL = new int[nnzL];
//        float* h_csrValL = new float[nnzL];

//        cudaMemcpy(h_csrRowPtrL, csrRowPtrL, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_csrColIndL, csrColIndL, sizeof(int) * nnzL, cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_csrValL, csrValL, sizeof(float) * nnzL, cudaMemcpyDeviceToHost);

//        std::cout << "h_csrRowPtrL: " << std::endl;
//        for(int i = 0; i < M + 1; ++i)
//        {
//            std::cout << h_csrRowPtrL[i] << std::endl;
//        }

//        std::cout << "h_csrColIndL h_csrValL: " << std::endl;
//        for(int i = 0; i < 500; ++i)
//        {
//            std::cout << h_csrColIndL[i] << "  " << h_csrValL[i] << std::endl;
//        }

//        std::cout << "h_csrColIndL h_csrValL: \n\n" << std::endl;

//        for(int i = nnzL - 500; i < nnzL; ++i)
//        {
//            std::cout << h_csrColIndL[i] << "  " << h_csrValL[i] << std::endl;
//        }



        /*******************************/
        /*******************************/
        /***********  pcg  *************/
        /*******************************/
        /*******************************/

        const float tol = 1e-2f;
        const int max_iter = 100;
        const float floatone = 1.0;
        const float floatzero = 0.0;

        float* d_x = NULL;
        float* d_r = NULL;
        float* d_p = NULL;
        float* d_omega = NULL;

        float r0, r1, alpha, beta;;
        int k;
        float dot, nalpha;

        cudaError_t(cudaMalloc((void **)&d_x, N*sizeof(float)));
        cudaError_t(cudaMalloc((void **)&d_r, N*sizeof(float)));
        cudaError_t(cudaMalloc((void **)&d_p, N*sizeof(float)));
        cudaError_t(cudaMalloc((void **)&d_omega, N*sizeof(float)));

        // validate rhs
    //    std::cout << "rhs: " << std::endl;
//        float* rhs = new float[N];
//        for(int i = 0; i < N; ++i)
//        {
//            rhs[i] = lambda * foreground[i];
//    //        std::cout << rhs[i] << std::endl;
//        }

        cudaError_t(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
        cudaError_t(cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice));


//        tg.StartCounter();

        k = 0;
        r0 = 0;
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

        while (r1 > tol*tol && k <= max_iter)
        {
            k++;
            if (k == 1)
            {
                cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
            }
            else
            {
                beta = r1/r0;
                cublasSscal(cublasHandle, N, &beta, d_p, 1);
                cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1) ;
            }

            cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzL, &floatone,
                           descrL, csrValL, csrRowPtrL, csrColIndL, d_p, &floatzero, d_omega);

            cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
            alpha = r1/dot;
            cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
            nalpha = -alpha;
            cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
            r0 = r1;
            cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        }

        printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));
//        std::cout << "get x time = " << tg.GetCounter() << " ms" << std::endl;

        cudaError_t(cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost));
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        /* check result */
//        err = 0.0;
//        for (int i = 0; i < N; i++)
//        {
//            rsum = 0.0;

//            for (int j = h_csrRowPtrL[i]; j < h_csrRowPtrL[i+1]; j++)
//            {
//                rsum += h_csrValL[j] * x[h_csrColIndL[j]];
//            }

//            diff = fabs(rsum - rhs[i]);

//            if (diff > err)
//            {
//                err = diff;
//            }
//        }
//        printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
//        nErrors += (k > max_iter) ? 1 : 0;
//        qaerr1 = err;

//        std::cout << "x....\n" ;
//        for(int i = 0; i < N; ++i)
//        {
//            printf(" %f \n", x[i]);
//        }

        cudaFree(sortedRowIndA);
        cudaFree(sortedColIndA);
        cudaFree(sortedValueA);

        cudaFree(sortedRowIndAT);
        cudaFree(sortedColIndAT);
        cudaFree(sortedValueAT);

        cudaFree(csrRowPtrA);
        cudaFree(csrRowPtrAT);
        cudaFree(csrRowPtrC);
        cudaFree(csrValC);
        cudaFree(csrColIndC);
        cudaFree(csrRowIndC);
        cudaFree(csrValD);
        cudaFree(csrRowPtrD);
        cudaFree(csrColIndD);
        cudaFree(csrRowPtrL);
        cudaFree(csrValL);
        cudaFree(csrColIndL);

        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_p);
        cudaFree(d_omega);
}


void CUDA_pcg::cu_pcg(int* sortedRowIndA, int* sortedColIndA,
         float* sortedValueA, int M, int N, int nz,
         float* x, float *rhs, float *csrValDiagAllCon)
{
//*********************************************
//    A = A + A';
//    D = spdiags(sum(A, 2), 0, N, N);
//    L = D - A;
//*********************************************

//*********************************************
//*********************************************
//*********************************************
//*********************************************
    int threads = 256;
    int blocks = M / threads + ((M % threads) ? 1:0);

    int n = M;
    assert(M == N);
    int nnz = nz;

    setSortedCsrFormat(sortedRowIndA, sortedColIndA, sortedValueA, nnz,
                       sortedRowIndA, sortedColIndA, sortedValueA);


    int* sortedRowIndAT = NULL;
    int* sortedColIndAT = NULL;
    float* sortedValueAT = NULL;

    cudaMalloc((void**)&sortedRowIndAT, sizeof(int) * nnz);
    cudaMalloc((void**)&sortedColIndAT, sizeof(int) * nnz);
    cudaMalloc((void**)&sortedValueAT, sizeof(float) * nnz);


    setSortedCsrFormat(sortedColIndA, sortedRowIndA, sortedValueA, nnz,
                       sortedRowIndAT, sortedColIndAT, sortedValueAT);

        /* set csrRowPtrA */

        int* csrRowPtrA = NULL;
        cudaStat1 = cudaMalloc((void**)&csrRowPtrA, (n + 1)*sizeof(csrRowPtrA[0]));
        if (cudaStat1 != cudaSuccess)
        {
            std::cout << "cudaMalloc csrRowPtrA failture  !!!" << std::endl;
            return ;
        }

        cusparseStatus = cusparseXcoo2csr(cusparseHandle,
                                          sortedRowIndA,
                                          nnz,
                                          n,
                                          csrRowPtrA,
                                          CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcoo2csr csrRowPtrA failture  !!!" << std::endl;
            return ;
        }

        /* set csrRowPtrAT */

        int* csrRowPtrAT = NULL;
        cudaStat1 = cudaMalloc((void**)&csrRowPtrAT, (n + 1)*sizeof(csrRowPtrA[0]));
        if (cudaStat1 != cudaSuccess)
        {
            std::cout << "cudaMalloc csrRowPtrAT failture  !!!" << std::endl;
            return ;
        }
        cusparseStatus = cusparseXcoo2csr(cusparseHandle,
                                          sortedRowIndAT,
                                          nnz,
                                          n,
                                          csrRowPtrAT,
                                          CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcoo2csr csrColPtr failture  !!!" << std::endl;
            return ;
        }

        /* A = A + A' */

        int baseC, nnzC;
        int *nnzTotalDevHostPtr = &nnzC;
        cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
        int* csrRowPtrC = NULL;
        cudaError_t(cudaMalloc((void**)&csrRowPtrC, sizeof(int) * (M + 1)));

        cusparseStatus = cusparseXcsrgeamNnz(cusparseHandle, M, N,
                                             descrA, nz,
                                             csrRowPtrA, sortedColIndA,
                                             descrB, nz,
                                             csrRowPtrAT, sortedColIndAT,
                                             descrC,
                                             csrRowPtrC, nnzTotalDevHostPtr);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcsrgeamNnz csrRowPtrC failture 3 !!!" << std::endl;
            return ;
        }

        if(NULL != nnzTotalDevHostPtr)
        {
            nnzC = *nnzTotalDevHostPtr;
        }
        else
        {
            cudaMemcpy(&nnzC, csrRowPtrC + M, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
            nnzC -= baseC;
        }

        const float A_alpha = 1.0;
        const float A_beta = 1.0;

        float* csrValC = NULL;
        int* csrColIndC = NULL;
        cudaError_t(cudaMalloc((void**)&csrValC, sizeof(float) * nnzC));
        cudaError_t(cudaMalloc((void**)&csrColIndC, sizeof(int) * nnzC));

        cusparseStatus = cusparseScsrgeam(cusparseHandle, M, N,
                         &A_alpha,
                         descrA, nz,
                         sortedValueA, csrRowPtrA, sortedColIndA,
                         &A_beta,
                         descrB, nz,
                         sortedValueAT, csrRowPtrAT, sortedColIndAT,
                         descrC,
                         csrValC, csrRowPtrC, csrColIndC);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseScsrgeam csrValC failture  !!!" << std::endl;
            return ;
        }

        ///////////////////////////////////////////
        ///
        ///
        ///             C = A + A'
        ///
        ///
        /// ///////////////////////////////////////////

        int* csrRowIndC = NULL;
        cudaMalloc((void**)&csrRowIndC, sizeof(int) * nnzC);

        cusparseStatus = cusparseXcsr2coo(cusparseHandle,
                                          csrRowPtrC,
                                          nnzC,
                                          M,
                                          csrRowIndC,
                                          CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseScsrgeam csrValC failture  !!!" << std::endl;
            return ;
        }

        // set sum(A, 2) and add lambda * spdiags(all_constraints, 0, n_pick, n_pick)
        float* csrValD  = NULL;
        cudaMalloc((void**)&csrValD, sizeof(int) * M);

        cudaMemcpy(csrValD, csrValDiagAllCon, sizeof(float) * M, cudaMemcpyDeviceToDevice);

        int blocks_sumA = nz / threads + ((nz % threads) ? 1:0);
        cuSumA2<<<blocks_sumA, threads>>>(csrValD, sortedRowIndA, sortedColIndA, sortedValueA, nz);

        ///////////////////////////////////////////
        ///
        ///
        ///             set D
        ///
        ///
        /// ///////////////////////////////////////////

        int* csrRowPtrD = NULL;
        int* csrColIndD = NULL;
        cudaMalloc((void**)&csrRowPtrD, sizeof(int) * (M + 1));
        cudaMalloc((void**)&csrColIndD, sizeof(int) * M);

        int blocks_ptr = (M + 1) / threads + (((M + 1) % threads) ? 1:0);
        cuSetValue<<<blocks_ptr, threads>>>(csrRowPtrD, M + 1);
        cuSetValue<<<blocks, threads>>>(csrColIndD, M);

        // L = D - C

        int* csrRowPtrL = NULL;
        int* csrColIndL = NULL;
        float* csrValL = NULL;
        int nnzL = 0;

        const float L_alpha = 1.0;
        const float L_beta = -1.0;

        int baseL;
        int *nnzTotalDevHostPtrL = &nnzL;
        cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
        cudaError_t(cudaMalloc((void**)&csrRowPtrL, sizeof(int) * (M + 1)));

        cusparseStatus = cusparseXcsrgeamNnz(cusparseHandle, M, N,
                                             descrA, M,
                                             csrRowPtrD, csrColIndD,
                                             descrC, nnzC,
                                             csrRowPtrC, csrColIndC,
                                             descrL,
                                             csrRowPtrL, nnzTotalDevHostPtrL);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseXcsrgeamNnz csrRowPtrC failture 4 !!!" << std::endl;
            return ;
        }

        if(NULL != nnzTotalDevHostPtrL)
        {
            nnzL = *nnzTotalDevHostPtrL;
        }
        else
        {
            cudaMemcpy(&nnzL, csrRowPtrL + M, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&baseL, csrRowPtrL, sizeof(int), cudaMemcpyDeviceToHost);
            nnzL -= baseL;
        }

        cudaError_t(cudaMalloc((void**)&csrValL, sizeof(float) * nnzL));
        cudaError_t(cudaMalloc((void**)&csrColIndL, sizeof(int) * nnzL));

        cusparseStatus = cusparseScsrgeam(cusparseHandle, M, N,
                                          &L_alpha,
                                          descrA, M,
                                          csrValD, csrRowPtrD, csrColIndD,
                                          &L_beta,
                                          descrC, nnzC,
                                          csrValC, csrRowPtrC, csrColIndC,
                                          descrL,
                                          csrValL, csrRowPtrL, csrColIndL);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "cusparseScsrgeam csrValC failture  !!!" << std::endl;
            return ;
        }
//*********************************************
//*********************************************
//*********************************************
//*********************************************


        /*******************************/
        /*******************************/
        /***********  pcg  *************/
        /*******************************/
        /*******************************/

        static float sum = 0;
        static int cnts = 0;
        long t1 = clock();

        const float tol = 1e-5f;
        const int max_iter = 100;
        const float floatone = 1.0;
        const float floatzero = 0.0;
        const float floatnone = -1.0;

        float* d_x = NULL;
        float* d_r = NULL;
        float* d_p = NULL;
        float* d_omega = NULL;

        float r0, r1, alpha;
        int k;
        float dot, nalpha;

        cudaError_t(cudaMalloc((void **)&d_x, N*sizeof(float)));
        cudaError_t(cudaMalloc((void **)&d_r, N*sizeof(float)));
        cudaError_t(cudaMalloc((void **)&d_p, N*sizeof(float)));
        cudaError_t(cudaMalloc((void **)&d_omega, N*sizeof(float)));

        cudaError_t(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyDeviceToDevice));
//        cudaError_t(cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice));

        cudaError_t(cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyDeviceToDevice));


        // omega = A*X
        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzL, &floatone,
                       descrL, csrValL, csrRowPtrL, csrColIndL, d_x, &floatzero, d_omega);
        // r = b - omega
        cublasSaxpy(cublasHandle, N, &floatnone, d_omega, 1, d_r, 1);
        // p = r
        cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        // r0 = r' * r
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r0);

        k = 0;

        while (k <= max_iter)
        {
            k++;

            // Ap = A*p
            cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzL, &floatone,
                           descrL, csrValL, csrRowPtrL, csrColIndL, d_p, &floatzero, d_omega);

            // dot = p' * Ap
            cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
            alpha = r0 / dot;
            cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
            nalpha = -alpha;
            cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
            //r0 = r1;
            cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
            if (r1 < tol*tol) break;
            // printf("r1: %f\n", r1);
            // p = r + (r1 / r0)*p
            cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
            alpha = (r1 / r0);
            cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_p, 1);
            r0 = r1;
        }

//        printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));
//        std::cout << "get x time = " << tg.GetCounter() << " ms" << std::endl;

        cudaError_t(cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToDevice));

        long t2 = clock();
        cnts++;
        sum += (t2-t1)*1000.0/CLOCKS_PER_SEC;
        std::cout<<cnts<<" times  ave_time: "<< sum/cnts<<" ms"<<std::endl;

        ///////////////////////////////////////////////////////////////////////////////////
        cudaFree(sortedRowIndAT);
        cudaFree(sortedColIndAT);
        cudaFree(sortedValueAT);

        cudaFree(csrRowPtrA);
        cudaFree(csrRowPtrAT);
        cudaFree(csrRowPtrC);
        cudaFree(csrValC);
        cudaFree(csrColIndC);
        cudaFree(csrRowIndC);
        cudaFree(csrValD);
        cudaFree(csrRowPtrD);
        cudaFree(csrColIndD);
        cudaFree(csrRowPtrL);
        cudaFree(csrValL);
        cudaFree(csrColIndL);

        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_p);
        cudaFree(d_omega);
}
