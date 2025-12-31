#pragma once
#include "common.h"
#include "kernel.h"
#include <cuda_runtime.h>


// ===================== Naive GEMM Kernel =====================
/**
 * @brief Naive implementation of matrix multiplication kernel (row-major, no shared memory optimization)
 */
__global__ void naive_gemm_kernel(const DataType* A, const DataType* B, DataType* C, int M, int N, int K) {
    // Calculate coordinates of C matrix element corresponding to current thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        DataType sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += gmem_get(A, row, k, M, K) * gmem_get(B, k, col, K, N);
            // sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ===================== Naive GEMM Wrapper =====================
void naive_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K) {
    NVTX3_FUNC_RANGE();
    // Thread block size (32x32 adapted to CUDA warp size)
    dim3 block_dim(32, 32);
    // Grid size (rounded up)
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x,
                  (N + block_dim.y - 1) / block_dim.y);
    
    // Launch kernel
    naive_gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    // Check kernel launch error
    CHECK_CUDA(cudaGetLastError());
}


