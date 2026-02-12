#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <nvtx3/nvToolsExt.h>

// Structure to hold test configuration
struct TestConfig {
    int M;
    int N;
    int K;
    float alpha;
    float beta;

    TestConfig(int m, int n, int k, float a = 1.0f, float b = 0.0f)
        : M(m), N(n), K(k), alpha(a), beta(b) {}
};

// Structure to hold performance results
struct PerfResult {
    std::string kernel_name;
    int M;
    int N;
    int K;
    float avg_time_ms;
    float gflops;
    size_t bytes;
    float bandwidth_gb_s;

    PerfResult(const std::string& name, int m, int n, int k, float time_ms)
        : kernel_name(name), M(m), N(n), K(k), avg_time_ms(time_ms) {

        // Calculate GFLOPS: 2 * M * N * K operations
        double flops = 2.0 * M * N * K;
        gflops = flops / (avg_time_ms * 1e-3) / 1e9;

        // Calculate data transferred (bytes)
        // Read A: M * K * sizeof(float)
        // Read B: K * N * sizeof(float)
        // Read C: M * N * sizeof(float) (if beta != 0)
        // Write C: M * N * sizeof(float)
        bytes = (M * K + K * N + 2 * M * N) * sizeof(float);
        bandwidth_gb_s = bytes / (avg_time_ms * 1e-3) / 1e9;
    }
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << ": error code " << err << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// Test utility class
class GemmTestUtils {
public:
    // Initialize matrix with random values
    static void init_matrix_random(float* mat, int rows, int cols, float min_val = -1.0f, float max_val = 1.0f);

    // Initialize matrix with constant value
    static void init_matrix_constant(float* mat, int rows, int cols, float value);

    // Calculate error between two matrices
    static float calculate_error(const float* ref, const float* test, int rows, int cols);

    // Compare two matrices with tolerance
    static bool compare_matrices(const float* ref, const float* test, int rows, int cols,
                                  float rtol = 1e-3f, float atol = 1e-5f);

    // Run cuBLAS GEMM (reference implementation)
    static void run_cublas_gemm(cublasHandle_t handle, const float* A, const float* B, float* C,
                                 int M, int N, int K, float alpha, float beta,
                                 cudaStream_t stream = 0);

    // Measure kernel execution time
    static float measure_time(GemmKernelFunc kernel, const float* A, const float* B, float* C,
                              int M, int N, int K, float alpha, float beta,
                              int warmup = 5, int iterations = 20,
                              cudaStream_t stream = 0);

    // Measure cuBLAS execution time
    static float measure_cublas_time(cublasHandle_t handle, const float* A, const float* B, float* C,
                                     int M, int N, int K, float alpha, float beta,
                                     int warmup = 5, int iterations = 20,
                                     cudaStream_t stream = 0);

    // Write performance results to file
    static void write_perf_results(const std::string& filename, const std::vector<PerfResult>& results);

    // Print performance results table
    static void print_perf_results(const std::vector<PerfResult>& results);
};

// Implementation

inline void GemmTestUtils::init_matrix_random(float* mat, int rows, int cols, float min_val, float max_val) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = min_val + (max_val - min_val) * (static_cast<float>(rand()) / RAND_MAX);
    }
}

inline void GemmTestUtils::init_matrix_constant(float* mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = value;
    }
}

inline float GemmTestUtils::calculate_error(const float* ref, const float* test, int rows, int cols) {
    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float error = std::abs(ref[i] - test[i]);
        max_error = (std::max)(max_error, error);
    }
    return max_error;
}

inline bool GemmTestUtils::compare_matrices(const float* ref, const float* test, int rows, int cols,
                                             float rtol, float atol) {
    for (int i = 0; i < rows * cols; ++i) {
        float error = std::abs(ref[i] - test[i]);
        float tolerance = atol + rtol * std::abs(ref[i]);
        if (error > tolerance) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref[i]
                      << ", test=" << test[i] << ", error=" << error << std::endl;
            return false;
        }
    }
    return true;
}

inline void GemmTestUtils::run_cublas_gemm(cublasHandle_t handle, const float* A, const float* B, float* C,
                                            int M, int N, int K, float alpha, float beta,
                                            cudaStream_t stream) {
    // cuBLAS uses column-major, so we need to transpose the operation
    // For row-major C = A * B where A is MxK, B is KxN, C is MxN:
    // In column-major: C^T = B^T * A^T where B^T is NxK, A^T is KxM, C^T is NxM

    // For row-major: compute C^T = beta * C^T + alpha * B^T * A^T
    // which in cuBLAS (column-major) is: C' = beta * C' + alpha * B' * A'
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,     // B is KxN row-major, treat as NxK column-major, leading dim = N
                             A, K,     // A is MxK row-major, treat as KxM column-major, leading dim = K
                             &beta,
                             C, N));   // C is MxN row-major, treat as NxM column-major, leading dim = N
}

inline float GemmTestUtils::measure_time(GemmKernelFunc kernel, const float* A, const float* B, float* C,
                                         int M, int N, int K, float alpha, float beta,
                                         int warmup, int iterations, cudaStream_t stream) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel(A, B, C, M, N, K, alpha, beta, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Measure
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    int target_perf_iter = iterations>1?2:1;
    for (int i = 0; i < iterations; ++i) {
        if(i == target_perf_iter) {
            nvtxRangePushA("Kernel_to_profile");
        }
        kernel(A, B, C, M, N, K, alpha, beta, stream);
        if(i == target_perf_iter) {
            nvtxRangePop();
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

inline float GemmTestUtils::measure_cublas_time(cublasHandle_t handle, const float* A, const float* B, float* C,
                                                 int M, int N, int K, float alpha, float beta,
                                                 int warmup, int iterations, cudaStream_t stream) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        run_cublas_gemm(handle, A, B, C, M, N, K, alpha, beta, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Measure
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    int target_perf_iter = iterations>1?2:1;
    for (int i = 0; i < iterations; ++i) {
        if(i == target_perf_iter) {
            nvtxRangePushA("Kernel_to_profile");
        }
        run_cublas_gemm(handle, A, B, C, M, N, K, alpha, beta, stream);
        if(i == target_perf_iter) {
            nvtxRangePop();
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

inline void GemmTestUtils::write_perf_results(const std::string& filename, const std::vector<PerfResult>& results) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write header
    out << "Kernel,M,N,K,Time(ms),GFLOPS,Bandwidth(GB/s)" << std::endl;

    // Write results
    for (const auto& result : results) {
        out << result.kernel_name << ","
            << result.M << ","
            << result.N << ","
            << result.K << ","
            << std::fixed << std::setprecision(4) << result.avg_time_ms << ","
            << std::fixed << std::setprecision(2) << result.gflops << ","
            << std::fixed << std::setprecision(2) << result.bandwidth_gb_s << std::endl;
    }

    out.close();
    std::cout << "Performance results written to: " << filename << std::endl;
}

inline void GemmTestUtils::print_perf_results(const std::vector<PerfResult>& results) {
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Kernel"
              << std::right << std::setw(6) << "M"
              << std::setw(6) << "N"
              << std::setw(6) << "K"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Bandwidth(GB/s)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.kernel_name
                  << std::right << std::setw(6) << result.M
                  << std::setw(6) << result.N
                  << std::setw(6) << result.K
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.avg_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gb_s
                  << std::endl;
    }
}
