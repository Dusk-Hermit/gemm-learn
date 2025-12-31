#pragma once
#include "common.h"
#include "kernel.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ===================== cuBLAS GEMM Wrapper =====================
void cublas_gemm(cublasHandle_t handle, DataType* d_A, DataType* d_B, DataType* d_C, int M, int N, int K) {
    NVTX3_FUNC_RANGE();
    const DataType alpha = 1.0f;
    const DataType beta = 0.0f;

    // cuBLAS GEMM call (FP32 version)
    // CHECK_CUBLAS(cublasSgemm(handle,
    //     CUBLAS_OP_N,  // A no transpose
    //     CUBLAS_OP_N,  // B no transpose
    //     M,            // m (rows of C)
    //     N,            // n (columns of C)
    //     K,            // k (columns of A / rows of B)
    //     &alpha,
    //     d_A, M,       // A matrix, leading dimension M
    //     d_B, K,       // B matrix, leading dimension K
    //     &beta,
    //     d_C, M));     // C matrix, leading dimension M

    // 注意：交换 A / B 的顺序 + 都转置
    CHECK_CUBLAS(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,              // m = rows of C_cm
            M,              // n = cols of C_cm
            K,
            &alpha,
            d_B,            // A_cm
            N,              // lda
            d_A,            // B_cm
            K,              // ldb
            &beta,
            d_C,
            N               // ldc
        )
    );


    // C = A * B, row-major
    // 对应 cublasSgemm，需要做转置：
    // CHECK_CUBLAS(
    //     cublasSgemm(
    //         handle,
    //         CUBLAS_OP_T,  // A^T
    //         CUBLAS_OP_T,  // B^T
    //         M,            // rows of C
    //         N,            // cols of C
    //         K,
    //         &alpha,
    //         d_B, N,       // B^T, lda = N
    //         d_A, K,       // A^T, lda = K
    //         &beta,
    //         d_C, M        // C row-major, ldc = M
    //     )
    // );

}
