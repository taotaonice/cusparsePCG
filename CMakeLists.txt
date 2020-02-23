cmake_minimum_required(VERSION 2.8)
project(cuPCG)

file(GLOB SRC_LIST *.c*)
file(GLOB INC_LIST *.h*)

find_package(CUDA)
set(CUDA_NVCC_FLAGS -O3 -gencode arch=compute_61,code=sm_61;)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler")
#set(CUDA_NVCC_FLAGS  -gencode arch=compute_61,code=sm_61)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

SET(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(CUDA_VISIBLE_DEVICES "1")

set(CMAKE_BUILD_TYPE "Release")

#set(thirdparty_root "/home/pansh/ssd_240/git/DeeparPolo/polo/Thirdparty")

# OpenCV
INCLUDE_DIRECTORIES(
    /usr/local/include
)
LINK_DIRECTORIES(
    /usr/local/lib
)
set(
    OPENCV_LIBRARIES
    opencv_highgui
    opencv_imgproc
    opencv_core
    opencv_imgcodecs
)

INCLUDE_DIRECTORIES(/usr/local/cuda/samples/common/inc)

INCLUDE_DIRECTORIES(/home/taotao/yaml-cpp-master/include)
LINK_DIRECTORIES(/home/taotao/yaml-cpp-master/build)

#INCLUDE_DIRECTORIES(/home/pansh/tools/ann_1.1.2/include)
#LINK_DIRECTORIES(/home/pansh/tools/ann_1.1.2/lib)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
LINK_DIRECTORIES(/usr/local/cuda/lib64)

# CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST} ${INC_LIST})
CUDA_ADD_LIBRARY(cuPCG SHARED cuda_pcg.cu)
CUDA_ADD_EXECUTABLE(verify_cupcg verify_cupcg.cpp)

target_link_libraries(
    cuPCG
    ${OPENCV_LIBRARIES}
    yaml-cpp
#    ANN
    cuda
    curand
    cusolver
    cublas
    cusparse
    gomp
)

target_link_libraries(
  verify_cupcg
  cuPCG
  )
