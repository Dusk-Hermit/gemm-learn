#pragma once
#include "common.h"
#include "kernel.h"
#include <cuda_runtime.h>

// ===================== Shared Memory Optimized GEMM Kernel (Template with Block Size) =====================
/**
 * @brief Optimized GEMM kernel with shared memory tiling (row-major, template for block size)
 * @tparam BLOCK_SIZE Tile size (sub-matrix size) for shared memory optimization, adapted to CUDA warp
 * @param A Input matrix A (M x K), device pointer
 * @param B Input matrix B (K x N), device pointer
 * @param C Output matrix C (M x N), device pointer
 * @param M Rows of matrix A and matrix C
 * @param N Columns of matrix B and matrix C
 * @param K Columns of matrix A and rows of matrix B
 */template <int BLOCK_SIZE>
__global__ void optimized2_gemm_kernel(const DataType* A, const DataType* B, DataType* C, int M, int N, int K) {
    // 使用2D线程块处理矩阵块
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 当前线程块处理的C矩阵块的起始位置
    const int c_row = by * BLOCK_SIZE;
    const int c_col = bx * BLOCK_SIZE;
    
    // 当前线程处理的C矩阵元素位置
    const int row = c_row + ty;
    const int col = c_col + tx;
    
    // 共享内存
    __shared__ DataType As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DataType Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    DataType Cvalue = 0.0;
    
    // 循环遍历K维度
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // 加载A矩阵块
        if (row < M && (k + tx) < K) {
            As[ty][tx] = A[row * K + k + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        // 加载B矩阵块
        if ((k + ty) < K && col < N) {
            Bs[ty][tx] = B[(k + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // 计算部分结果
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// ===================== Optional: Wrapper with Configurable Block Size (16/32) =====================
/**
 * @brief Overload with configurable block size (16 or 32) for flexibility
 * @param d_A Input matrix A (M x K), device pointer
 * @param d_B Input matrix B (K x N), device pointer
 * @param d_C Output matrix C (M x N), device pointer
 * @param M Rows of matrix A and matrix C
 * @param N Columns of matrix B and matrix C
 * @param K Columns of matrix A and rows of matrix B
 * @param block_size Block size (only 16 or 32 are supported)
 */
void optimized2_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K, int block_size) {
    NVTX3_FUNC_RANGE();

    // Thread block dimensions
    dim3 block_dim(block_size, block_size);
    // Grid dimensions (rounded up)
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (M + block_dim.y - 1) / block_dim.y);

    // Launch kernel with specified block size (only 16 and 32 are supported)
    if (block_size == 16) {
        optimized2_gemm_kernel<16><<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    } else if (block_size == 32) {
        optimized2_gemm_kernel<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        // Invalid block size, fallback to 32 and print warning
        printf("Warning: Invalid block size %d, using default 32 instead\n", block_size);
        optimized2_gemm_kernel<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    }

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
}