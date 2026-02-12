cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Set cmake policy for CUDA integration
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 NEW)
endif()
# set(CMAKE_SYSTEM_VERSION 10.0.22621.0 CACHE STRING "" FORCE)
# set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION 10.0.22621.0 CACHE STRING "" FORCE)
project(gemm_test LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Set build type to Release
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

# if(ENABLE_CUDA_DEBUG)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Source files for main test executable
file(GLOB_RECURSE CPP_SRC
    src/*.cpp
)

file(GLOB_RECURSE CUDA_SRC_KERNEL
    src/kernel/*.cu
)

# Source files for large matrix benchmark
set(BENCH_LARGE_SRC
    src/bench_large.cu
    ${CUDA_SRC_KERNEL}
)

# Main test executable
add_executable(gemm_test
    ${CUDA_SRC_KERNEL}
    ${CPP_SRC}
)

# Large matrix benchmark executable
add_executable(gemm_bench_large
    ${BENCH_LARGE_SRC}
)

target_include_directories(gemm_test
PRIVATE
${CMAKE_SOURCE_DIR}/include
${CMAKE_SOURCE_DIR}/cutlass/include
)


get_target_property(_cublas CUDA::cublas IMPORTED_LOCATION)
message(STATUS "Using cuBLAS: ${_cublas}")

target_link_libraries(gemm_test
    PRIVATE
        CUDA::cudart
        CUDA::cublas
)

set_target_properties(gemm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Large matrix benchmark target configuration
target_include_directories(gemm_bench_large
PRIVATE
${CMAKE_SOURCE_DIR}/include
${CMAKE_SOURCE_DIR}/cutlass/include
)

target_link_libraries(gemm_bench_large
    PRIVATE
        CUDA::cudart
        CUDA::cublas
)

set_target_properties(gemm_bench_large PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Copy CUDA DLLs to output directory on Windows
if(WIN32)
    find_file(CUDART_DLL
        NAMES cudart64_13.dll
        PATHS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64
        REQUIRED
    )

    find_file(CUBLAS_DLL
        NAMES cublas64_13.dll
        PATHS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64
        REQUIRED
    )

    find_file(CUBLASLT_DLL
        NAMES cublasLt64_13.dll
        PATHS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64
        REQUIRED
    )

    add_custom_command(TARGET gemm_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUDART_DLL}
        # $<TARGET_FILE_DIR:gemm_test>
        "$<TARGET_FILE_DIR:gemm_test>/cudart64_12.dll"

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUBLAS_DLL}
        # $<TARGET_FILE_DIR:gemm_test>
        "$<TARGET_FILE_DIR:gemm_test>/cublas64_12.dll"

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUBLASLT_DLL}
        # $<TARGET_FILE_DIR:gemm_test>
        "$<TARGET_FILE_DIR:gemm_test>/cublasLt64_12.dll"
    )

    add_custom_command(TARGET gemm_bench_large POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUDART_DLL}
        $<TARGET_FILE_DIR:gemm_bench_large>

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUBLAS_DLL}
        $<TARGET_FILE_DIR:gemm_bench_large>

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUBLASLT_DLL}
        $<TARGET_FILE_DIR:gemm_bench_large>
    )


    # Required when compiled targets strongly needs to link *_12.dll 
    # So rename them
endif()
