cmake_minimum_required(VERSION 3.20)

# Set cmake policy for CUDA integration
# This addresses the CMP0146 warning by explicitly using the new approach
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 NEW)
endif()

project(gemm_test LANGUAGES C CXX CUDA)

# include(CTest)
# enable_testing()

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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3") 
endif()

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")


if(MSVC)
# add_compile_options(/utf-8) # 导致报错的原因
# add_compile_options(/wd4819) # 禁用C4819警告
endif()


##### 可执行文件

# set(EXE_SRC 
#     src/naive_gemm.cu
#     src/main.cu
#     src/gemm_utils.cu
# )

# add_executable(gemm_test ${EXE_SRC})

# target_include_directories(gemm_test
# PRIVATE
# ${CMAKE_SOURCE_DIR}/include
# )

# target_link_libraries(gemm_test
#     PRIVATE
#         CUDA::cudart
#         CUDA::cublas
# )

# # ⭐ CUDA 在 Windows + VS 下必须
# set_target_properties(gemm_test PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )


# install(TARGETS gemm_test
#     RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin
# )


##### pybind11 动态库
# pip install pybind11
find_package(Python3
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE CUDA_SRC
    src/*.cu
)

# set(CUDA_SRC
#     src/gemm.cu
#     src/01-naive.cu
#     src/02-cublas.cu
#     src/03-opt1.cu
# )

# find_path(NVTX_INCLUDE_DIR NAMES nvtx3/nvtx3.hpp)

set(LIB_SRC
    pybind11/gemm.cpp
    ${CUDA_SRC}
)

pybind11_add_module(gemm_ext ${LIB_SRC})
target_include_directories(gemm_ext
PRIVATE
${CMAKE_SOURCE_DIR}/include
# ${NVTX_INCLUDE_DIR}
)
target_link_libraries(gemm_ext
    PRIVATE
        CUDA::cudart
        CUDA::cublas
)
set_target_properties(gemm_ext PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


# ========== 关键新增：自动复制CUDA DLL到输出目录 ==========
if(WIN32)
    # 获取CUDA DLL路径
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

    # 复制DLL到输出目录（Release和Debug）
    add_custom_command(TARGET gemm_ext POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUDART_DLL}
        $<TARGET_FILE_DIR:gemm_ext>
        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUBLAS_DLL}
        $<TARGET_FILE_DIR:gemm_ext>
        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUBLASLT_DLL}
        $<TARGET_FILE_DIR:gemm_ext>
    )
endif()


# cmake --fresh -B build -S .
# cmake --build build --config Release

