#include <cstdlib>
#include <cmath>
#include <iostream>
#include "gemm.h"
#include "kernel.h"

// ===================== ctor / dtor =====================
GemmRunner::GemmRunner(const GemmParams& params, GemmBackend backend)
    : params_(params), backend_(backend) {}

GemmRunner::~GemmRunner() {
    release();
}

// ===================== init =====================
void GemmRunner::init() {
    if (initialized_) return;

    int M = params_.M, N = params_.N, K = params_.K;

    h_A_ = new DataType[M * K];
    h_B_ = new DataType[K * N];
    h_C_ = new DataType[M * N];

    CHECK_CUDA(cudaMalloc(&d_A_, M * K * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&d_B_, K * N * sizeof(DataType)));
    CHECK_CUDA(cudaMalloc(&d_C_, M * N * sizeof(DataType)));

    if (backend_ == GemmBackend::CuBLAS) {
        CHECK_CUBLAS(cublasCreate(&cublas_));
    }

    initialized_ = true;
    // CHECK_CUDA(cudaDeviceSynchronize());
}

// ===================== random fill =====================
void GemmRunner::fill_random(unsigned int seed) {
    int M = params_.M, N = params_.N, K = params_.K;

    srand(seed);
    for (int i = 0; i < M * K; ++i)
        h_A_[i] = (rand() / float(RAND_MAX)) * 2 - 1;

    for (int i = 0; i < K * N; ++i)
        h_B_[i] = (rand() / float(RAND_MAX)) * 2 - 1;

    CHECK_CUDA(cudaMemcpy(d_A_, h_A_, M * K * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_, h_B_, K * N * sizeof(DataType), cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaDeviceSynchronize());
}

// ===================== GEMM dispatch =====================
void GemmRunner::launch_gemm() {
    int M = params_.M, N = params_.N, K = params_.K;

    if (backend_ == GemmBackend::Naive) {
        naive_gemm(d_A_, d_B_, d_C_, M, N, K);
    }
    else if(backend_ == GemmBackend::CuBLAS) {
        cublas_gemm(cublas_, d_A_, d_B_, d_C_, M, N, K);
    }
    else if(backend_ == GemmBackend::Opt1) {
        optimized1_gemm(d_A_, d_B_, d_C_, M, N, K);
    }
    else if(backend_ == GemmBackend::Opt2) {
        optimized2_gemm(d_A_, d_B_, d_C_, M, N, K);
    }
    else if(backend_ == GemmBackend::Opt3) {
        optimized3_gemm(d_A_, d_B_, d_C_, M, N, K);
    }
    else {
        std::cerr << "Invalid backend\n";
        std::exit(1);
    }
}

// ===================== run once =====================
void GemmRunner::run_once() {
    launch_gemm();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ===================== benchmark =====================
double GemmRunner::run_benchmark(int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    launch_gemm(); // warmup
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_gemm();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

// ===================== fetch result =====================
void GemmRunner::fetch_result() {
    int size = params_.M * params_.N;
    CHECK_CUDA(cudaMemcpy(h_C_, d_C_, size * sizeof(DataType), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaDeviceSynchronize());
}

// ===================== compare =====================
bool GemmRunner::compare(const GemmRunner& a,
                         const GemmRunner& b,
                         DataType eps) {
    if (a.params_.M != b.params_.M ||
        a.params_.N != b.params_.N ||
        a.params_.K != b.params_.K) {
        std::cerr << "MNK mismatch\n";
        return false;
    }

    int size = a.params_.M * a.params_.N;
    double max_diff = 0.0;

    for (int i = 0; i < size; ++i) {
        double diff = fabs(a.h_C_[i] - b.h_C_[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > eps) {
            std::cerr << "Mismatch at " << i << " diff=" << diff << "\n";
            return false;
        }
    }

    std::cout << "Compare OK, max diff = " << max_diff << "\n";
    return true;
}

// ===================== release =====================
void GemmRunner::release() {
    if (!initialized_) return;

    delete[] h_A_;
    delete[] h_B_;
    delete[] h_C_;

    CHECK_CUDA(cudaFree(d_A_));
    CHECK_CUDA(cudaFree(d_B_));
    CHECK_CUDA(cudaFree(d_C_));

    if (cublas_) {
        CHECK_CUBLAS(cublasDestroy(cublas_));
    }

    initialized_ = false;
}

void GemmRunner::print(int rows, int cols) const {
    // 打印fetech结果的h_C_的左上角rows*cols个元素，如果行或者列大于矩阵的行或者列，则取矩阵的行或者列
    rows = std::min(rows, params_.M);
    cols = std::min(cols, params_.N);
    std::cout << params_.M << "x" << params_.N << " h_C_ " << rows << "x" << cols << ":"<< std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_C_[i * params_.N + j] << " ";
        }
        std::cout << "\n";
    }
}