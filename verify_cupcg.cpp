/*
 * File Type:     C/C++
 * Author:        Hutao {hutaonice@gmail.com}
 * Creation:      星期五 21/02/2020 18:23.
 * Last Revision: 星期五 21/02/2020 18:23.
 */

#include <iostream>
#include <fstream>
#include "cuda_pcg.h"

using namespace std;

int main(){
    CUDA_pcg* pcg = new CUDA_pcg();

    char data_path[] = "../data.bin";
    int m, n, nz;
    ifstream ifile(data_path, ios::in|ios::binary);
    if(!ifile){
        cout<<"open file failed."<<endl;
        exit(-1);
    }
    ifile.read((char*)&m, 4);
    ifile.read((char*)&n, 4);
    ifile.read((char*)&nz, 4);
    cout<<m<<endl<<n<<endl<<nz<<endl;
    int* row_ind = new int[nz];
    int* col_ind = new int[nz];
    ifile.read((char*)row_ind, nz*4);
    ifile.read((char*)col_ind, nz*4);

    float* val = new float[nz];
    float* cons = new float[m*n];
    float* Y = new float[m*n];
    float* X = new float[m*n];
    ifile.read((char*)val, nz*4);
    ifile.read((char*)cons, m*n*4);
    ifile.read((char*)Y, m*n*4);
    ifile.read((char*)X, m*n*4);

    ifile.close();

    int *row_ind_dev, *col_ind_dev;
    float *val_dev, *cons_dev, *Y_dev, *X_dev;
    cudaMalloc((void**)&row_ind_dev, nz*4);
    cudaMalloc((void**)&col_ind_dev, nz*4);
    cudaMalloc((void**)&val_dev, nz*4);
    cudaMalloc((void**)&cons_dev, m*n*4);
    cudaMalloc((void**)&Y_dev, m*n*4);
    cudaMalloc((void**)&X_dev, m*n*4);

    cudaMemcpy(row_ind_dev, row_ind, nz*4, cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind_dev, col_ind, nz*4, cudaMemcpyHostToDevice);
    cudaMemcpy(val_dev, val, nz*4, cudaMemcpyHostToDevice);
    cudaMemcpy(cons_dev, cons, m*n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(Y_dev, Y, m*n*4, cudaMemcpyHostToDevice);
    //cudaMemcpy(X_dev, X, m*n*4, cudaMemcpyHostToDevice);

    pcg->cu_pcg(row_ind_dev, col_ind_dev, val_dev, m*n, m*n, nz, X_dev, Y_dev, cons_dev);
    cudaMemcpy(X, X_dev, m*n*4, cudaMemcpyDeviceToHost);
    cv::Mat alpha(n, m, CV_32FC1, X);
    cv::imshow("alpha", alpha);
    cv::waitKey(0);

    delete pcg;
    return 0;
}






/* EOF */

