cuda 13.1 没有官方识别 vs2026
https://www.reddit.com/r/CUDA/comments/1oura3a/when_can_cuda_support_for_vs_2026_be_expected/


nvtx 过滤，ncu只去profile nvtx打过点的kernel：参考这个
https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering

to use
git clone https://github.com/NVIDIA/cutlass.git