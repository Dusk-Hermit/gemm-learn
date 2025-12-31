#pragma once
#include "common.h"
#include "kernel.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE_M 16  // 块处理的行数
#define BLOCK_SIZE_N 16  // 块处理的列数
#define BLOCK_SIZE_K 16  // 块处理的K维度

// 使用 shared memory padding 避免 bank conflict
__global__ void optimized1_gemm_kernel(const DataType* __restrict__ A,
                                       const DataType* __restrict__ B,
                                       DataType* C,
                                       int M, int N, int K) {
    __shared__ DataType As[BLOCK_SIZE_M][BLOCK_SIZE_K + 1];
    __shared__ DataType Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + 1];

    // 块索引：行方向（M）用x，列方向（N）用y
    int block_row = blockIdx.x;  // 对应M方向
    int block_col = blockIdx.y;  // 对应N方向
    
    // 线程索引：行方向用x，列方向用y
    int thread_row = threadIdx.x;  // 块内行索引
    int thread_col = threadIdx.y;  // 块内列索引

    // 全局索引计算：行（M）在前，列（N）在后
    int row = block_row * BLOCK_SIZE_M + thread_row;
    int col = block_col * BLOCK_SIZE_N + thread_col;

    DataType sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE_K) {
        // 加载A tile: A[row][k0: k0+BLOCK_SIZE_K]
        // A是M×K矩阵，行主序
        if (row < M && (k0 + thread_col) < K) {
            As[thread_row][thread_col] = A[row * K + (k0 + thread_col)];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // 加载B tile: B[k0: k0+BLOCK_SIZE_K][col]
        // B是K×N矩阵，行主序
        if ((k0 + thread_row) < K && col < N) {
            Bs[thread_row][thread_col] = B[(k0 + thread_row) * N + col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ===================== Optimized GEMM Wrapper =====================
void optimized1_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K) {
    NVTX3_FUNC_RANGE();

    // block_dim: x方向处理行(M)，y方向处理列(N)
    dim3 block_dim(BLOCK_SIZE_M, BLOCK_SIZE_N); // threads per block
    
    // grid_dim: x方向是行方向(M)，y方向是列方向(N)
    dim3 grid_dim((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
                  (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

    optimized1_gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}