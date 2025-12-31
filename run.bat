@echo off
@REM :: 创建build目录（不存在则创建）
@REM if not exist build mkdir build
@REM cd build
@REM :: 执行cmake配置（默认使用Visual Studio编译器，自动匹配CUDA）
@REM cmake .. -DENABLE_CUDA_DEBUG=True
@REM :: 编译（--build . 等价于make，--config Release 指定Release模式）
@REM cmake --build . --config Debug
@REM :: 运行测试程序（Debug目录下的可执行文件）
@REM cd ..
@REM build\Debug\gemm_test.exe
@REM pause


@REM pip install pybind11
cmake --fresh -B build -S .
cmake --build build --config Release
set PYTHONPATH=build/Release
@REM python python/test.py
python python/test2.py
@REM python python/test2.py > test2.log

