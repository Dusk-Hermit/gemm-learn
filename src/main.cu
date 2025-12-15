#include "gemm_utils.h"

int main() {
    // -------------------- 1. Host memory allocation and initialization --------------------
    DataType* h_A = new DataType[M * K];
    DataType* h_B = new DataType[K * N];
    DataType* h_C_naive = new DataType[M * N];
    DataType* h_C_cublas = new DataType[M * N];

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // -------------------- 2. Device memory allocation and data copy --------------------
    DataType *d_A, *d_B, *d_C_naive, *d_C_cublas;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(DataType)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(DataType)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C_naive, M * N * sizeof(DataType)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C_cublas, M * N * sizeof(DataType)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(DataType), cudaMemcpyHostToDevice));

    // -------------------- 3. Initialize cuBLAS handle --------------------
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    // -------------------- 4. Naive GEMM performance test --------------------
    std::cout << "=== Naive GEMM Test ===" << std::endl;
    // Warm-up (eliminate first launch overhead)
    naive_gemm(d_A, d_B, d_C_naive, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timing test
    cudaEvent_t start_naive, stop_naive;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_naive));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_naive));

    CHECK_CUDA_ERROR(cudaEventRecord(start_naive));
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        naive_gemm(d_A, d_B, d_C_naive, M, N, K);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop_naive));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_naive));

    // Calculate time and performance
    float naive_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time_ms, start_naive, stop_naive));
    double naive_avg_time_ms = naive_time_ms / TEST_ITERATIONS;
    double naive_gflops = calculate_gflops(TEST_ITERATIONS, naive_time_ms);

    // Copy results to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_naive, d_C_naive, M * N * sizeof(DataType), cudaMemcpyDeviceToHost));

    // -------------------- 5. cuBLAS GEMM performance test --------------------
    std::cout << "\n=== cuBLAS GEMM Test ===" << std::endl;
    // Warm-up
    cublas_gemm(cublas_handle, d_A, d_B, d_C_cublas, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timing test
    cudaEvent_t start_cublas, stop_cublas;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_cublas));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_cublas));

    CHECK_CUDA_ERROR(cudaEventRecord(start_cublas));
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        cublas_gemm(cublas_handle, d_A, d_B, d_C_cublas, M, N, K);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop_cublas));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_cublas));

    // Calculate time and performance
    float cublas_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&cublas_time_ms, start_cublas, stop_cublas));
    double cublas_avg_time_ms = cublas_time_ms / TEST_ITERATIONS;
    double cublas_gflops = calculate_gflops(TEST_ITERATIONS, cublas_time_ms);

    // Copy results to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(DataType), cudaMemcpyDeviceToHost));

    // -------------------- 6. Result verification --------------------
    std::cout << "\n=== Result Verification ===" << std::endl;
    if (verify_result(h_C_cublas, h_C_naive, M * N)) {
        std::cout << "✅ Naive GEMM results match cuBLAS!" << std::endl;
    } else {
        std::cerr << "❌ Naive GEMM results do NOT match cuBLAS!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // -------------------- 7. Performance comparison output --------------------
    std::cout << "\n=== Performance Comparison (Matrix Size: " << M << "×" << K << " × " << K << "×" << N << ") ===" << std::endl;
    std::cout << "Test Iterations: " << TEST_ITERATIONS << std::endl;
    std::cout << "┌─────────────┬──────────────┬─────────────┐" << std::endl;
    std::cout << "│ Implementation │ Avg Time(ms) │ GFLOPS      │" << std::endl;
    std::cout << "├─────────────┼──────────────┼─────────────┤" << std::endl;
    std::cout << "│ Naive GEMM  │ " << naive_avg_time_ms << " │ " << naive_gflops << " │" << std::endl;
    std::cout << "│ cuBLAS GEMM │ " << cublas_avg_time_ms << " │ " << cublas_gflops << " │" << std::endl;
    std::cout << "└─────────────┴──────────────┴─────────────┘" << std::endl;
    std::cout << "Performance Gap: cuBLAS is " << naive_avg_time_ms / cublas_avg_time_ms << " times faster than Naive" << std::endl;

    // -------------------- 8. Resource release --------------------
    // Device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C_naive));
    CHECK_CUDA_ERROR(cudaFree(d_C_cublas));
    // Host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_cublas;
    // cuBLAS handle and events
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_naive));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_naive));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_cublas));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_cublas));

    // Reset CUDA device
    CHECK_CUDA_ERROR(cudaDeviceReset());

    std::cout << "\n🎉 Test Completed!" << std::endl;
    return 0;
}