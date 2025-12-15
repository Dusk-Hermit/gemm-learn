#include "gemm_utils.h"

// ===================== Utility Functions =====================
void init_matrix(DataType* mat, int rows, int cols) {
    srand(time(nullptr));
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<DataType>(rand()) / RAND_MAX * 2.0 - 1.0; // [-1, 1] random numbers
    }
}

bool verify_result(const DataType* ref, const DataType* test, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(ref[i] - test[i]) > EPS) {
            std::cerr << "Result mismatch at index " << i 
                      << ": ref=" << ref[i] << ", test=" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

double calculate_gflops(int iterations, double total_time_ms) {
    double flops = 2.0 * M * N * K * iterations;
    double gflops = flops / (total_time_ms / 1000.0) / 1e9;
    return gflops;
}

// ===================== Naive GEMM Kernel =====================
/**
 * @brief Naive implementation of matrix multiplication kernel (row-major, no shared memory optimization)
 */
__global__ void naive_gemm_kernel(const DataType* A, const DataType* B, DataType* C, int M, int N, int K) {
    // Calculate coordinates of C matrix element corresponding to current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        DataType sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ===================== Naive GEMM Wrapper =====================
void naive_gemm(DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K) {
    // Thread block size (32x32 adapted to CUDA warp size)
    dim3 block_dim(32, 32);
    // Grid size (rounded up)
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (M + block_dim.y - 1) / block_dim.y);
    
    // Launch kernel
    naive_gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    // Check kernel launch error
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ===================== cuBLAS GEMM Wrapper =====================
void cublas_gemm(cublasHandle_t handle, DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K) {
    const DataType alpha = 1.0f;
    const DataType beta = 0.0f;

    // cuBLAS GEMM call (FP32 version)
    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                   CUBLAS_OP_N,  // A no transpose
                                   CUBLAS_OP_N,  // B no transpose
                                   M,            // m (rows of C)
                                   N,            // n (columns of C)
                                   K,            // k (columns of A / rows of B)
                                   &alpha,
                                   d_A, M,       // A matrix, leading dimension M
                                   d_B, K,       // B matrix, leading dimension K
                                   &beta,
                                   d_C, M));     // C matrix, leading dimension M
}