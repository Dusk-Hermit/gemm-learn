
D:\repos\gemm-learn>run.bat
-- Selecting Windows SDK version 10.0.26100.0 to target Windows 10.0.19045.
-- Configuring done (4.0s)
-- Generating done (0.0s)
-- Build files have been written to: D:/repos/gemm-learn/build
适用于 .NET Framework MSBuild 版本 17.14.8+a7a4d5af0

  Compiling CUDA source file ..\src\naive_gemm.cu...

  D:\repos\gemm-learn\build>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"  --use-local-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" -x cu
  -rdc=true  -I"D:\repos\gemm-learn\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"     --keep-dir gemm_test\x64\Release  -maxrregc
  ount=0    --machine 64 --compile -cudart static -Wno-deprecated-gpu-targets -rdc=true -lineinfo --extended-lambda -std=c++17 --generate-code=arch=compute_86,code=[compute_86,sm_86] --generate-code=arch=compute_89,code=[compute_8
  9,sm_89] /utf-8 -Xcompiler="/EHsc -Ob2 \"/EHsc /W1 /O2 /MD /GR\""   -D_WINDOWS -DNDEBUG -D"CMAKE_INTDIR=\"Release\"" -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Xcompiler "/EHsc /W1 /nologo /O2 /FS   /MD /GR" -Xcompiler "/Fdgemm_test.
  dir\Release\vc143.pdb" -o gemm_test.dir\Release\naive_gemm.obj "D:\repos\gemm-learn\src\naive_gemm.cu"
  nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.9.targets(801,9): error MSB3721: 命令“"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"  --use-local
-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" -x cu -rdc=true  -I"D:\repos\gemm-learn\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"
-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"     --keep-dir gemm_test\x64\Release  -maxrregcount=0    --machine 64 --compile -cudart static -Wno-deprecated-gpu-targets -rdc=true -lineinfo --extended-lambda
-std=c++17 --generate-code=arch=compute_86,code=[compute_86,sm_86] --generate-code=arch=compute_89,code=[compute_89,sm_89] /utf-8 -Xcompiler="/EHsc -Ob2 \"/EHsc /W1 /O2 /MD /GR\""   -D_WINDOWS -DNDEBUG -D"CMAKE_INTDIR=\"Release\""
 -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Xcompiler "/EHsc /W1 /nologo /O2 /FS   /MD /GR" -Xcompiler "/Fdgemm_test.dir\Release\vc143.pdb" -o gemm_test.dir\Release\naive_gemm.obj "D:\repos\gemm-learn\src\naive_gemm.cu"”已退出，返回代 码为 1。 [D
:\repos\gemm-learn\build\gemm_test.vcxproj]
  Compiling CUDA source file ..\src\main.cu...

  D:\repos\gemm-learn\build>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"  --use-local-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" -x cu
  -rdc=true  -I"D:\repos\gemm-learn\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"     --keep-dir gemm_test\x64\Release  -maxrregc
  ount=0    --machine 64 --compile -cudart static -Wno-deprecated-gpu-targets -rdc=true -lineinfo --extended-lambda -std=c++17 --generate-code=arch=compute_86,code=[compute_86,sm_86] --generate-code=arch=compute_89,code=[compute_8
  9,sm_89] /utf-8 -Xcompiler="/EHsc -Ob2 \"/EHsc /W1 /O2 /MD /GR\""   -D_WINDOWS -DNDEBUG -D"CMAKE_INTDIR=\"Release\"" -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Xcompiler "/EHsc /W1 /nologo /O2 /FS   /MD /GR" -Xcompiler "/Fdgemm_test.
  dir\Release\vc143.pdb" -o gemm_test.dir\Release\main.obj "D:\repos\gemm-learn\src\main.cu"
  nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.9.targets(801,9): error MSB3721: 命令“"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"  --use-local
-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" -x cu -rdc=true  -I"D:\repos\gemm-learn\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"
-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"     --keep-dir gemm_test\x64\Release  -maxrregcount=0    --machine 64 --compile -cudart static -Wno-deprecated-gpu-targets -rdc=true -lineinfo --extended-lambda
-std=c++17 --generate-code=arch=compute_86,code=[compute_86,sm_86] --generate-code=arch=compute_89,code=[compute_89,sm_89] /utf-8 -Xcompiler="/EHsc -Ob2 \"/EHsc /W1 /O2 /MD /GR\""   -D_WINDOWS -DNDEBUG -D"CMAKE_INTDIR=\"Release\""
 -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Xcompiler "/EHsc /W1 /nologo /O2 /FS   /MD /GR" -Xcompiler "/Fdgemm_test.dir\Release\vc143.pdb" -o gemm_test.dir\Release\main.obj "D:\repos\gemm-learn\src\main.cu"”已退出，返回代码为 1。 [D:\repos\gemm
-learn\build\gemm_test.vcxproj]