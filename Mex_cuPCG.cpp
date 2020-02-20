/*
 * File Type:     C/C++
 * Author:        Hutao {hutaonice@gmail.com}
 * Creation:      星期四 20/02/2020 23:10.
 * Last Revision: 星期四 20/02/2020 23:10.
 */

#include <iostream>
#include "MxArray.h"
#include "cuda_pcg.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    CUDA_pcg* pcg = new CUDA_pcg();

    std::vector<MxArray> rhs(prhs, prhs+nrhs);
    cv::Mat rowind(rhs[0].toMat());
    cv::Mat colind(rhs[1].toMat());
    cv::Mat val(rhs[2].toMat());
    int M = rhs[3].toInt();
    int N = rhs[4].toInt();
    int nz = rhs[5].toInt();
    cv::Mat x(rhs[6].toMat());
    cv::Mat rhs(rhs[7].toMat());
    cv::Mat diag(rhs[8].toMat());

    pcg->cu_pcg(rowind.data, colind.data, val.data, M, N, nz, x.data, rhs.data, diag.data);

    plhs[0] = MxArray(x);
    delete pcg;
}





/* EOF */

