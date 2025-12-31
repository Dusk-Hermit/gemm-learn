#pragma once

#include "common.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void naive_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);
void cublas_gemm(cublasHandle_t handle, DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);
void optimized1_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);
void optimized2_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K, int block_size=32);
void optimized3_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K, int block_size=32);
