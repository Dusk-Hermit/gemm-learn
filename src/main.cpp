#include "gemm_kernel.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

class GemmTestRunner {
public:
    GemmTestRunner() {
        CUDA_CHECK(cudaFree(0));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }

    ~GemmTestRunner() {
        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }

    // Run correctness test for a single kernel with specific configuration
    bool test_correctness(const GemmKernelInfo& kernel, const TestConfig& config) {
        int M = config.M;
        int N = config.N;
        int K = config.K;
        float alpha = config.alpha;
        float beta = config.beta;

        // Allocate host memory
        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_C_ref(M * N);
        std::vector<float> h_C_test(M * N);

        // Initialize matrices
        GemmTestUtils::init_matrix_random(h_A.data(), M, K);
        GemmTestUtils::init_matrix_random(h_B.data(), K, N);
        GemmTestUtils::init_matrix_random(h_C_ref.data(), M, N);
        h_C_test = h_C_ref;

        // Allocate device memory
        float *d_A, *d_B, *d_C_ref, *d_C_test;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_test, M * N * sizeof(float)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_test, h_C_test.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

        // Run cuBLAS (reference)
        GemmTestUtils::run_cublas_gemm(cublas_handle_, d_A, d_B, d_C_ref,
                                        M, N, K, alpha, beta, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run test kernel
        kernel.func(d_A, d_B, d_C_test, M, N, K, alpha, beta, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C_test.data(), d_C_test, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // Compare results
        bool passed = GemmTestUtils::compare_matrices(h_C_ref.data(), h_C_test.data(), M, N, 1e-2f, 2e-5f);

        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_ref));
        CUDA_CHECK(cudaFree(d_C_test));

        if (passed) {
            std::cout << "  [PASS] " << kernel.name << ": M=" << M << ", N=" << N << ", K=" << K
                      << ", alpha=" << alpha << ", beta=" << beta << std::endl;
        } else {
            std::cout << "  [FAIL] " << kernel.name << ": M=" << M << ", N=" << N << ", K=" << K
                      << ", alpha=" << alpha << ", beta=" << beta << std::endl;
        }

        return passed;
    }

    // Run correctness tests for all kernels
    void run_all_correctness_tests() {
        std::cout << "\n=== Correctness Tests ===" << std::endl;

        const auto& kernels = GemmKernelRegistry::instance().get_kernels();

        // Test configurations
        std::vector<TestConfig> test_configs = {
            {128, 128, 128, 1.0f, 0.0f},
            {256, 256, 256, 1.0f, 0.0f},
            {512, 512, 512, 1.0f, 0.0f},
            {1024, 1024, 1024, 1.0f, 0.0f}, // Larger size
            {256, 128, 512, 1.5f, 0.5f},   // Non-standard alpha/beta
            {2048, 64, 64, 1.0f, 1.0f},      // With beta scaling
            {64, 2048, 64, 1.0f, 0.0f},     // Tall and skinny
            {127, 257, 511, 1.0f, 0.0f},  // Non-power-of-2 sizes
        };

        int total_tests = 0;
        int passed_tests = 0;

        for (const auto& kernel : kernels) {
            std::cout << "\nTesting kernel: " << kernel.name;
            if (!kernel.description.empty()) {
                std::cout << " (" << kernel.description << ")";
            }
            std::cout << std::endl;

            for (const auto& config : test_configs) {
                total_tests++;
                if (test_correctness(kernel, config)) {
                    passed_tests++;
                }
            }
        }

        std::cout << "\n=== Correctness Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << total_tests << std::endl;
    }

    // Run performance tests
    void run_performance_tests() {
        std::cout << "\n=== Performance Tests ===" << std::endl;

        const auto& kernels = GemmKernelRegistry::instance().get_kernels();

        // Performance test configurations
        std::vector<TestConfig> perf_configs = {
            {128, 128, 128, 1.0f, 0.0f},
            {256, 256, 256, 1.0f, 0.0f},
            {512, 512, 512, 1.0f, 0.0f},
            {1024, 1024, 1024, 1.0f, 0.0f},
            {2048, 2048, 2048, 1.0f, 0.0f},
            {128, 2048, 128, 1.0f, 0.0f},
            {2048, 128, 128, 1.0f, 0.0f},
        };

        std::vector<PerfResult> all_results;

        for (const auto& config : perf_configs) {
            int M = config.M;
            int N = config.N;
            int K = config.K;
            float alpha = config.alpha;
            float beta = config.beta;

            std::cout << "\nTesting size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

            // Allocate host memory
            std::vector<float> h_A(M * K);
            std::vector<float> h_B(K * N);
            std::vector<float> h_C(M * N);

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

            // Measure cuBLAS performance
            float cublas_time = GemmTestUtils::measure_cublas_time(
                cublas_handle_, d_A, d_B, d_C, M, N, K, alpha, beta, 5, 20, 0);
            all_results.emplace_back("cuBLAS", M, N, K, cublas_time);
            std::cout << "  cuBLAS: " << cublas_time << " ms" << std::endl;

            // Measure each kernel
            for (const auto& kernel : kernels) {
                float kernel_time = GemmTestUtils::measure_time(
                    kernel.func, d_A, d_B, d_C, M, N, K, alpha, beta, 5, 20, 0);
                all_results.emplace_back(kernel.name, M, N, K, kernel_time);
                std::cout << "  " << kernel.name << ": " << kernel_time << " ms" << std::endl;
            }

            // Cleanup
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }

        // Print and save results
        GemmTestUtils::print_perf_results(all_results);
        GemmTestUtils::write_perf_results("performance_results.csv", all_results);
    }

private:
    cublasHandle_t cublas_handle_;
};

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "     GEMM Kernel Testing Framework     " << std::endl;
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

    // List registered kernels
    const auto& kernels = GemmKernelRegistry::instance().get_kernels();
    std::cout << "\nRegistered Kernels (" << kernels.size() << "):" << std::endl;
    for (const auto& kernel : kernels) {
        std::cout << "  - " << kernel.name;
        if (!kernel.description.empty()) {
            std::cout << ": " << kernel.description;
        }
        std::cout << std::endl;
    }

    // Create test runner
    GemmTestRunner runner;

    // Run correctness tests
    runner.run_all_correctness_tests();

    // Run performance tests
    runner.run_performance_tests();

    std::cout << "\n========================================" << std::endl;
    std::cout << "     All tests completed!     " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
