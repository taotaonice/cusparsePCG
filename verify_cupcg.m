% File Type:     Matlab
% Author:        Hutao {hutaonice@gmail.com}
% Creation:      星期五 21/02/2020 13:23.
% Last Revision: 星期五 21/02/2020 14:15.

clear;clc

rng(0)

MatrixSize = 640*320;
N_nonezero = 10000;

img_path = '/home/taotao/Documents/knn-matting-master/data/inputs/GT01.png';
trimap_path = '/home/taotao/Documents/knn-matting-master/data/trimaps/Trimap1/GT01.png';

img = im2double(imread(img_path));
trimap = im2double(imread(trimap_path));
%% generate a random matrix
row_ind = (randi(MatrixSize, N_nonezero, 1));
col_ind = (randi(MatrixSize, N_nonezero, 1));
value = (rand(N_nonezero, 1)*0.9+0.09);

X_init = rand(MatrixSize, 1);
Y = rand(MatrixSize, 1);
constraints = double(rand(MatrixSize, 1)>0.5);

A = sparse(row_ind, col_ind, value, MatrixSize, MatrixSize, N_nonezero);

A = A + A';
L = spdiags(sum(A, 2), 0, MatrixSize, MatrixSize);

X = pcg(L, Y, 1e-4, 1000);