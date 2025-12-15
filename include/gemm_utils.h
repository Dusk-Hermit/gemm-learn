#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cublas_v2.h>

// ===================== Configurable Parameters =====================
// Matrix Dimensions (A: M×K, B: K×N, C: M×N)
const int M = 1024;
const int N = 1024;
const int K = 1024;
// Number of repeated test iterations (average to reduce error)
const int TEST_ITERATIONS = 10;
// Data type (float/double optional)
using DataType = float;
// Result verification error threshold
const DataType EPS = 1e-4f;

// ===================== CUDA Error Checking Macro =====================
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS_ERROR(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ===================== Helper Function Declarations =====================
/**
 * @brief Initialize matrix (random values)
 */
void init_matrix(DataType* mat, int rows, int cols);

/**
 * @brief Verify if two matrices are approximately equal
 */
bool verify_result(const DataType* ref, const DataType* test, int size);

/**
 * @brief Calculate GFLOPS (GEMM floating-point operations: 2*M*N*K)
 */
double calculate_gflops(int iterations, double total_time_ms);

/**
 * @brief Naive GEMM wrapper function (external interface)
 */
void naive_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);

/**
 * @brief cuBLAS GEMM wrapper function (external interface)
 */
void cublas_gemm(cublasHandle_t handle, DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);

#endif // GEMM_UTILS_H