# gemm-learn
learn gemm optimization from scratch

https://zhuanlan.zhihu.com/p/521376336
chcp 65001 > nul

参考
https://github.com/danila-permogorskii/cuda-ml-example/blob/master/CMakeLists.txt


CMakeLists.txt的cuda写法，可以参考https://github.com/NVIDIA/cuda-samples/blob/master/Samples/CMakeLists.txt

中文注释没有问题，中文print会输出乱码。加上CMakeLists.txt的utf-8设置会导致报错


添加一个算子：
- 添加cu文件
- 修改common.h的注册枚举值
- 修改gemm.cpp的控制流
- 添加pybind11绑定

GEMM（General Matrix Multiplication）的 FLOPs 计算核心是统计矩阵乘法过程中**浮点乘法**与**浮点加法**的总次数，标准场景为矩阵 $A(m \times k)$ 与矩阵 $B(k \times n)$ 相乘得到矩阵 $C(m \times n)$。

1.  计算矩阵 $C$ 中**单个元素** $C_{i,j}$：需要对 $A$ 的第 $i$ 行和 $B$ 的第 $j$ 列做内积，包含 $k$ 次乘法和 $k-1$ 次加法，共 $2k-1$ 次 FLOPs。
2.  计算整个矩阵 $C$：$C$ 共有 $m \times n$ 个元素，总 FLOPs 为 $m \times n \times (2k-1)$。
3.  工程近似：当 $k$ 较大时，$k-1 \approx k$，因此总 FLOPs 可简化为 **$2mk n$**，这也是业界常用的近似计算公式。

举个例子：$A(100 \times 50)$ 与 $B(50 \times 200)$ 相乘，理论 FLOPs = $100\times200\times(2\times50-1)=1980000$，近似 FLOPs = $2\times100\times50\times200=2000000$，二者误差很小。


你提到的这两个概念确实容易混淆，核心区别如下：
1.  **FLOPs**（全大写）：全称是 **Floating-Point Operations**，指**浮点运算次数**，是衡量计算量大小的指标（比如 GEMM 的 $2mkn$ FLOPs）。
2.  **FLOPS**（全大写，发音“flops”）：全称是 **Floating-Point Operations Per Second**，指**每秒浮点运算次数**，是衡量计算速度/算力的量纲（单位如 GFLOPS、TFLOPS）。

日常交流中大家常把二者都读作“flops”，很容易搞混，区分的关键看是描述**运算次数**还是**运算速度**。

138.
NVIDIA GeForce RTX 4060
8GB GDDR6 - 2023.05
15.11
TFLOPS

https://www.topcpu.net/gpu-r/fp32-float-desktop


