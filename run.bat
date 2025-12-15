@echo off
:: 创建build目录（不存在则创建）
if not exist build mkdir build
cd build
:: 执行cmake配置（默认使用Visual Studio编译器，自动匹配CUDA）
cmake ..
:: 编译（--build . 等价于make，--config Release 指定Release模式）
cmake --build . --config Release
:: 运行测试程序（Release目录下的可执行文件）
cd ..
build\Release\gemm_test.exe
pause