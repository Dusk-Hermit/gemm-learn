#include "gemm_kernel.h"
#include "test_utils.h"

// Naive GEMM kernel: each thread computes one element of C
// C = alpha * A * B + beta * C
// All matrices are in row-major format
__global__ void naive_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper function for the naive GEMM kernel
void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    naive_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
}

// Register the kernel
REGISTER_GEMM_KERNEL("naive", naive_gemm, "Naive GEMM implementation - one thread per element");
