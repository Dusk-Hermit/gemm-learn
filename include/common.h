#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <stdexcept>
#include <nvtx3/nvtx3.hpp>

// ===================== Config =====================
using DataType = float;
constexpr DataType DEFAULT_EPS = 1e-4f;

// ===================== Error Macros =====================
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error %s:%d: %d\n", __FILE__, __LINE__, s); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ===================== Params =====================
struct GemmParams {
    int M, N, K;
    GemmParams(int m, int n, int k) : M(m), N(n), K(k) {}
    GemmParams() = default;
};

enum class GemmBackend {
    Naive,
    CuBLAS,
    Opt1,
    Opt2,
    Opt3,
};

std::string gemm_backend_name(GemmBackend backend);


// ===================== Kernel Helper =====================

// 二维矩阵索引取数
__device__ __forceinline__ DataType gmem_get(const DataType* mat, int m, int n, int M, int N) {
    return mat[m * N + n];
}