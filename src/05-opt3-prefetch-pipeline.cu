#pragma once
#include "common.h"
#include "kernel.h"
#include <cuda_runtime.h>
template <int BLOCK_SIZE>
__global__ void optimized3_gemm_kernel(const DataType* __restrict__ A, 
                                         const DataType* __restrict__ B, 
                                         DataType* __restrict__ C, 
                                         int M, int N, int K) {
    // 每个线程块负责C矩阵中的一个BLOCK_SIZE x BLOCK_SIZE的块
    const int blockRow = blockIdx.y * BLOCK_SIZE;
    const int blockCol = blockIdx.x * BLOCK_SIZE;
    
    // 每个线程在块内的位置
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    
    // C矩阵中当前线程负责的元素在全局内存中的行和列
    const int globalRow = blockRow + threadRow;
    const int globalCol = blockCol + threadCol;
    
    // 寄存器累加
    DataType Cvalue = 0.0;
    
    // 共享内存双缓冲（可选优化）
    __shared__ DataType As[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DataType Bs[2][BLOCK_SIZE][BLOCK_SIZE];
    
    // 预取第一块
    int tile = 0;
    int A_col = threadCol;
    int B_row = threadRow;
    
    if (globalRow < M && A_col < K) {
        As[0][threadRow][threadCol] = A[globalRow * K + A_col];
    } else {
        As[0][threadRow][threadCol] = 0.0;
    }
    
    if (B_row < K && globalCol < N) {
        Bs[0][threadRow][threadCol] = B[B_row * N + globalCol];
    } else {
        Bs[0][threadRow][threadCol] = 0.0;
    }
    
    __syncthreads();
    
    // 循环遍历K维度
    for (int tileIdx = 0; tileIdx < K; tileIdx += BLOCK_SIZE) {
        // 计算下一个tile的索引
        int nextTileIdx = tileIdx + BLOCK_SIZE;
        int nextBuffer = (tile + 1) % 2;
        
        // 预取下一个tile（如果还有）
        if (nextTileIdx < K) {
            int nextA_col = nextTileIdx + threadCol;
            int nextB_row = nextTileIdx + threadRow;
            
            if (globalRow < M && nextA_col < K) {
                As[nextBuffer][threadRow][threadCol] = A[globalRow * K + nextA_col];
            } else {
                As[nextBuffer][threadRow][threadCol] = 0.0;
            }
            
            if (nextB_row < K && globalCol < N) {
                Bs[nextBuffer][threadRow][threadCol] = B[nextB_row * N + globalCol];
            } else {
                Bs[nextBuffer][threadRow][threadCol] = 0.0;
            }
        }
        
        // 计算当前tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[tile][threadRow][k] * Bs[tile][k][threadCol];
        }
        
        // 切换buffer
        tile = (tile + 1) % 2;
        __syncthreads();
    }
    
    // 写入结果
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = Cvalue;
    }
}
void optimized3_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K, int block_size) {
    NVTX3_FUNC_RANGE();

    // Thread block dimensions
    dim3 block_dim(block_size, block_size);
    // Grid dimensions (rounded up)
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (M + block_dim.y - 1) / block_dim.y);

    // Launch kernel with specified block size (only 16 and 32 are supported)
    if (block_size == 16) {
        optimized3_gemm_kernel<16><<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    } else if (block_size == 32) {
        optimized3_gemm_kernel<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        // Invalid block size, fallback to 32 and print warning
        printf("Warning: Invalid block size %d, using default 32 instead\n", block_size);
        optimized3_gemm_kernel<32><<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    }

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
}