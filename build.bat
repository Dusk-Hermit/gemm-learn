rmdir /s /q build
@echo off
echo ========================================
echo Building GEMM Test Framework
echo ========================================

echo.
echo Setting up Visual Studio 18 environment...
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

if %errorlevel% neq 0 (
    echo Failed to set up Visual Studio environment!
    pause
    exit /b 1
)

echo.
echo Step 1: Configure CMake
set BUILD_TYPE=Release
@REM cmake --fresh -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_VERBOSE_MAKEFILE=ON
cmake --fresh -B build -S . -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Build project...
cd build
@REM ninja -v
cmake --build . --config %BUILD_TYPE%
if %errorlevel% neq 0 (
    echo Build failed!
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Build completed successfully!
echo ========================================

echo.
echo Step 3: Copying CUDA DLLs...
for %%f in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudart64_*.dll") do copy /Y "%%f" build\ > nul
for %%f in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cublas64_*.dll") do copy /Y "%%f" build\ > nul
for %%f in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cublasLt64_*.dll") do copy /Y "%%f" build\ > nul
echo DLLs copied.

echo.
echo Step 4: Running tests...
build\gemm_test.exe

echo.
echo Step 5: Running large matrix benchmark...
build\gemm_bench_large.exe

echo.
echo ========================================
echo All done!
echo ========================================
@REM pause
