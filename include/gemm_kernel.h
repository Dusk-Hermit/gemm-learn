#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

// GEMM kernel function pointer type
// C = alpha * A * B + beta * C
// All matrices are in row-major format
using GemmKernelFunc = void(*)(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    cudaStream_t stream
);

// Structure to hold kernel information
struct GemmKernelInfo {
    std::string name;           // Kernel name
    GemmKernelFunc func;        // Kernel function pointer
    std::string description;    // Kernel description

    GemmKernelInfo(const std::string& n, GemmKernelFunc f, const std::string& desc = "")
        : name(n), func(f), description(desc) {}
};

// Global kernel registry
class GemmKernelRegistry {
public:
    static GemmKernelRegistry& instance() {
        static GemmKernelRegistry registry;
        return registry;
    }

    void register_kernel(const GemmKernelInfo& kernel) {
        kernels_.push_back(kernel);
    }

    const std::vector<GemmKernelInfo>& get_kernels() const {
        return kernels_;
    }

    GemmKernelFunc get_kernel(const std::string& name) const {
        for (const auto& kernel : kernels_) {
            if (kernel.name == name) {
                return kernel.func;
            }
        }
        return nullptr;
    }

private:
    GemmKernelRegistry() = default;
    std::vector<GemmKernelInfo> kernels_;
};

// Helper macro to register a kernel
#define REGISTER_GEMM_KERNEL(name, func, description) \
    namespace { \
        struct __KernelRegistrar_##func { \
            __KernelRegistrar_##func() { \
                GemmKernelRegistry::instance().register_kernel( \
                    GemmKernelInfo(name, func, description) \
                ); \
            } \
        }; \
        static __KernelRegistrar_##func __kernel_registrar_##func; \
    }
