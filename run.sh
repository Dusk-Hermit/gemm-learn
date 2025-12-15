#!/bin/bash
# 创建build目录（不存在则创建）
mkdir -p build
cd build
# 执行cmake配置
cmake ..
# 编译（-j后接CPU核心数，加速编译）
make -j$(nproc)
# 运行测试程序
./gemm_test