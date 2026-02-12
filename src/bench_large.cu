#include "gemm_kernel.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

class LargeMatrixBenchmark {
public:
    LargeMatrixBenchmark() {
        CUDA_CHECK(cudaFree(0));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }

    ~LargeMatrixBenchmark() {
        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }

    void run() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "     Large Matrix Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;

        const auto& kernels = GemmKernelRegistry::instance().get_kernels();
        std::cout << "\nRegistered Kernels (" << kernels.size() << "):" << std::endl;
        for (const auto& kernel : kernels) {
            std::cout << "  - " << kernel.name << std::endl;
        }
        std::cout << "  - cuBLAS (reference)" << std::endl;

        // Large matrix configurations
        std::vector<TestConfig> large_configs = {
            // {2048, 2048, 2048, 1.0f, 0.0f},
            // {3072, 3072, 3072, 1.0f, 0.0f},
            {4096, 4096, 4096, 1.0f, 0.0f},
            // {2048, 4096, 3072, 1.0f, 0.0f},
            // {4096, 2048, 3072, 1.0f, 0.0f},
        };

        std::vector<PerfResult> all_results;

        int warmup = 10;
        int iterations = 25;
        for (const auto& config : large_configs) {
            int M = config.M;
            int N = config.N;
            int K = config.K;
            float alpha = config.alpha;
            float beta = config.beta;

            std::cout << "\n========================================" << std::endl;
            std::cout << "Benchmarking: M=" << M << ", N=" << N << ", K=" << K << std::endl;
            std::cout << "========================================" << std::endl;

            // Allocate host memory
            std::vector<float> h_A(M * K);
            std::vector<float> h_B(K * N);
            std::vector<float> h_C(M * N);

            // Initialize matrices
            GemmTestUtils::init_matrix_random(h_A.data(), M, K);
            GemmTestUtils::init_matrix_random(h_B.data(), K, N);
            GemmTestUtils::init_matrix_constant(h_C.data(), M, N, 0.0f);

            // Allocate device memory
            float *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

            // Benchmark cuBLAS (1 warmup + 1 measurement)
            float cublas_time = GemmTestUtils::measure_cublas_time(
                cublas_handle_, d_A, d_B, d_C, M, N, K, alpha, beta, warmup, iterations, 0);
            all_results.emplace_back("cuBLAS", M, N, K, cublas_time);
            std::cout << "cuBLAS: " << cublas_time << " ms" << std::endl;

            // Benchmark each registered kernel (1 warmup + 1 measurement)
            for (const auto& kernel : kernels) {
                float kernel_time = GemmTestUtils::measure_time(
                    kernel.func, d_A, d_B, d_C, M, N, K, alpha, beta, warmup, iterations, 0);
                all_results.emplace_back(kernel.name, M, N, K, kernel_time);
                std::cout << kernel.name << ": " << kernel_time << " ms" << std::endl;
            }

            // Cleanup
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }

        // Print and save results
        std::cout << "\n========================================" << std::endl;
        std::cout << "     Performance Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        GemmTestUtils::print_perf_results(all_results);
        GemmTestUtils::write_perf_results("large_matrix_benchmark.csv", all_results);
    }

private:
    cublasHandle_t cublas_handle_;
};

int main(int argc, char** argv) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Large Matrix GEMM Benchmark" << std::endl;
    std::cout << "  Configuration: 1 warmup + 1 run" << std::endl;
    std::cout << "========================================" << std::endl;

    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(0)));

    // Get CUDA device properties
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Memory: " << (prop.totalGlobalMem / 1024 / 1024 / 1024) << " GB" << std::endl;

    // Create benchmark runner
    LargeMatrixBenchmark bench;
    bench.run();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Benchmark completed!" << std::endl;
    std::cout << "  Results saved to: large_matrix_benchmark.csv" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
