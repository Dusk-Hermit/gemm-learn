cmake_minimum_required(VERSION 3.20)

project(gemm_test LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# if(MSVC)
# cmake_policy(SET CMP0091 NEW)
# cmake_policy(SET CMP0104 NEW)
#     add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
#     add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
#     # add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc /W1 /O2 /MD /GR>)
#     add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/EHsc /W1 /O2 /MD /GR">)

# endif()

add_executable(gemm_test
    src/naive_gemm.cu
    src/main.cu
)
target_include_directories(gemm_test
PRIVATE
${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(gemm_test
    PRIVATE
        CUDA::cudart
        CUDA::cublas
)

# ⭐ CUDA 在 Windows + VS 下必须
set_target_properties(gemm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


install(TARGETS gemm_test
    RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin
)
