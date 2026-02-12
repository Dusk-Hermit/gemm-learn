#include "gemm_kernel.h"
#include "test_utils.h"
#include "cutlass/gemm/device/gemm.h"

// Wrapper function for the CUTLASS SGEMM kernel
void cutlass_sgemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {alpha, beta}
    );

    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM invocation failed");
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// Register the kernel
REGISTER_GEMM_KERNEL("cutlass", cutlass_sgemm, "CUTLASS GEMM Invocation");
