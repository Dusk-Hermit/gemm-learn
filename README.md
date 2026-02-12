# GEMM Kernel Testing Framework

A CUDA C++ framework for testing and benchmarking GEMM (General Matrix Multiply) kernels against cuBLAS reference implementation.

## Features

- **Automatic Kernel Registration**: Easy registration system for new GEMM kernels
- **Correctness Testing**: Verifies kernel results against cuBLAS for various matrix sizes and configurations
- **Performance Benchmarking**: Measures execution time, GFLOPS, and memory bandwidth
- **CSV Export**: Performance results automatically saved to `performance_results.csv`
- **Row-Major Layout**: All matrices use row-major format as preferred

## Project Structure

```
gemm-learn/
├── include/
│   ├── gemm_kernel.h      # Kernel registry and interface
│   └── test_utils.h       # Testing utilities and performance measurement
├── src/
│   ├── naive_gemm.cu      # Naive GEMM kernel implementation
│   └── main.cpp           # Main test program
├── build_simple.bat       # Build script for Windows
├── CMakeLists.txt         # CMake build configuration
└── README.md
```

## Building

### Windows (Recommended)

Run `build_simple.bat` directly or from Command Prompt:

```batch
build_simple.bat
```

This will:
1. Set up Visual Studio environment
2. Compile CUDA kernels
3. Compile C++ code
4. Link the executable
5. Run tests

### Using CMake

```bash
cmake --fresh -B build -S .
cmake --build build --config Release
```

Note: You may need to fix Windows SDK registration issues. Use `build_simple.bat` for a more reliable build.

## How to Add a New Kernel

1. Create a new `.cu` file in `src/` (e.g., `my_gemm.cu`)

2. Implement your kernel with this signature:

```cpp
#include "gemm_kernel.h"

__global__ void my_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Your kernel implementation
    // Remember: matrices are row-major
}

void my_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    // Launch your kernel here
    dim3 blockDim(...);
    dim3 gridDim(...);
    my_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

// Register your kernel
REGISTER_GEMM_KERNEL("my_kernel", my_gemm, "Description of my kernel");
```

3. Rebuild the project - your kernel will be automatically tested

## Kernel Interface

All GEMM kernels must use this function signature:

```cpp
void kernel_func(
    const float* A,    // Input matrix A (M x K, row-major)
    const float* B,    // Input matrix B (K x N, row-major)
    float* C,          // Output matrix C (M x N, row-major)
    int M,             // Rows of A and C
    int N,             // Columns of B and C
    int K,             // Columns of A and rows of B
    float alpha,       // Scalar multiplier for A*B
    float beta,        // Scalar multiplier for C
    cudaStream_t stream // CUDA stream for execution
);
```

The operation computes: `C = alpha * A * B + beta * C`

## Test Configurations

The framework automatically tests with:

**Correctness Tests:**
- Various matrix sizes (power-of-2 and non-power-of-2)
- Different alpha and beta values
- Tall/skinny matrices

**Performance Tests:**
- Matrix sizes from 128x128x128 to 2048x2048x2048
- Compares against cuBLAS performance
- Reports GFLOPS and bandwidth

## Example Output

```
========================================
     GEMM Kernel Testing Framework
========================================
CUDA Device: NVIDIA GeForce RTX 4060
Compute Capability: 8.9

Registered Kernels (1):
  - naive: Naive GEMM implementation

=== Correctness Tests ===
  [PASS] naive: M=128, N=128, K=128, alpha=1, beta=0
  [PASS] naive: M=256, N=256, K=256, alpha=1, beta=0
  ...

=== Performance Results ===
Kernel                   M     N     K    Time(ms)      GFLOPS  Bandwidth(GB/s)
--------------------------------------------------------------------------------
cuBLAS                 128   128   128      0.0460       42.23           2.64
naive                  128   128   128      0.1102       38.07           2.38
...
```

## Performance Results

Results are saved to `performance_results.csv` with columns:
- Kernel name
- Matrix dimensions (M, N, K)
- Execution time (ms)
- GFLOPS (billions of floating-point operations per second)
- Bandwidth (GB/s)

## Notes

- All matrices are stored in **row-major** format
- Tests use relative tolerance of 1e-2 and absolute tolerance of 2e-5
- Performance measurements use 5 warmup iterations and 20 timed iterations
- CUDA arch targets are sm_86 and sm_89 (can be modified in CMakeLists.txt)
