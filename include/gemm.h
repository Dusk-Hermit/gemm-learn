#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"
// #include <string>


// ===================== Runner =====================
class GemmRunner {
public:
    GemmRunner(const GemmParams& params, GemmBackend backend);
    ~GemmRunner();

    void init();
    void fill_random(unsigned int seed = 42);
    void run_once();
    double run_benchmark(int iterations);
    void fetch_result();
    void release();

    const DataType* host_C() const { return h_C_; }

    static bool compare(const GemmRunner& a,
                        const GemmRunner& b,
                        DataType eps = DEFAULT_EPS);
    
    GemmBackend backend() const { return backend_; }
    void print(int rows=4, int cols=4) const;

private:
    void launch_gemm();

private:
    GemmParams params_;
    GemmBackend backend_;

    DataType *h_A_{nullptr}, *h_B_{nullptr}, *h_C_{nullptr};
    DataType *d_A_{nullptr}, *d_B_{nullptr}, *d_C_{nullptr};

    cublasHandle_t cublas_{nullptr};
    bool initialized_{false};
};
