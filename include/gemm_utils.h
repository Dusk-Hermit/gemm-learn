#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cublas_v2.h>

// ===================== 可配置参数（按需修改）=====================
// 矩阵尺寸 (A: M×K, B: K×N, C: M×N)
const int M = 1024;
const int N = 1024;
const int K = 1024;
// 重复测试次数（取平均减少误差）
const int TEST_ITERATIONS = 10;
// 数据类型（float/double 可选）
using DataType = float;
// 结果验证误差阈值
const DataType EPS = 1e-4f;

// ===================== CUDA 错误检查宏 =====================
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

// ===================== 辅助函数声明 =====================
/**
 * @brief 初始化矩阵（随机值）
 */
void init_matrix(DataType* mat, int rows, int cols);

/**
 * @brief 验证两个矩阵是否近似相等
 */
bool verify_result(const DataType* ref, const DataType* test, int size);

/**
 * @brief 计算 GFLOPS (GEMM 浮点运算量：2*M*N*K)
 */
double calculate_gflops(int iterations, double total_time_ms);

/**
 * @brief Naive GEMM 封装函数（对外接口）
 */
void naive_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);

/**
 * @brief cuBLAS GEMM 封装函数（对外接口）
 */
void cublas_gemm(cublasHandle_t handle, DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K);

#endif // GEMM_UTILS_H