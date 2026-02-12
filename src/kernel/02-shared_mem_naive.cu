#include "gemm_kernel.h"
#include "test_utils.h"

#define BLOCK_SIZE 32
#define BLOCK_SIZE_K 32

// Naive GEMM kernel: each thread computes one element of C
// C = alpha * A * B + beta * C
// All matrices are in row-major format
__global__ void shared_mem_naive_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE];
    float acc = 0.0f;
    for(int kStart = 0; kStart < K; kStart += BLOCK_SIZE_K){
        // Load
        // One thread loads one element of A and B into shared memory
        // No decoupling of threadIdx and As/Bs indices for simplicity, which requires BLOCK_SIZE == BLOCK_SIZE_K
        if(row < M && kStart + threadIdx.x < K){
            As[threadIdx.y][threadIdx.x] = A[row * K + kStart + threadIdx.x];
        }else{
            As[threadIdx.y][threadIdx.x] = 0.0f; // Handle out-of-bounds
        }
        if(kStart + threadIdx.y < K && col < N){
            Bs[threadIdx.y][threadIdx.x] = B[(kStart + threadIdx.y) * N + col];
        }else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f; // Handle out-of-bounds
        }
        __syncthreads();

        // Compute
        for(int k = 0; k < BLOCK_SIZE_K; ++k){
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    // Write back
    if(row < M && col < N){
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// Wrapper function for the naive GEMM kernel
void shared_mem_naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    shared_mem_naive_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
}

// Register the kernel
REGISTER_GEMM_KERNEL("shared_mem_naive", shared_mem_naive_gemm, "Naive GEMM implementation - one thread per element");
