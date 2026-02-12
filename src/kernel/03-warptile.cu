#include "gemm_kernel.h"
#include "test_utils.h"
#include "common_utils.h"

#include<cstdio>

const int WARP_SIZE = 32;

#define BM_ 128
#define BK_ 16
#define BN_ 128
#define WM_ 64
#define WN_ 64
#define WMITER_ 1
#define WNITER_ 4
#define TM_ 8
#define TN_ 4
#define NUM_THREADS_ 128

static_assert(BM_ % WM_ == 0, "BM must be divisible by WM");
static_assert(BN_ % WN_ == 0, "BN must be divisible by WN");
static_assert(WM_ % TM_ == 0, "WM must be divisible by TM");
static_assert(WN_ % TN_ == 0, "WN must be divisible by TN");
static_assert(WM_ % WMITER_ == 0, "WM must be divisible by WMITER");
static_assert(WN_ % WNITER_ == 0, "WN must be divisible by WNITER");
static_assert(WM_ * WN_ == WARP_SIZE * WMITER_ * WNITER_ * TM_ * TN_, "WM * WN must be equal to WARP_SIZE * WMITER * WNITER * TM * TN");
static_assert(BK_%(128/8/sizeof(float))==0, "one row of As must be divisible by 128 bits");
static_assert(BN_%(128/8/sizeof(float))==0, "one row of Bs must be divisible by 128 bits");
static_assert(NUM_THREADS_ / WARP_SIZE == (BM_/WM_)*(BN_/WN_), "NUM_THREADS must be equal to (BM/WM)*(BN/WN), etc num_warps satisfies the warp-tile split of the block");


// Naive GEMM kernel: each thread computes one element of C
// C = alpha * A * B + beta * C
// All matrices are in row-major format
template<int BM, int BK, int BN, int WM, int WN, int WMITER, int WNITER, int TM, int TN, int NUM_THREADS>
__global__ void warptile_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // 1-d block, 2-d grid
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_offset = tid % WARP_SIZE;
    // const int num_warps = NUM_THREADS / WARP_SIZE;  // not used
    const int warpRow = warp_id / (BN / WN);
    const int warpCol = warp_id % (BN / WN);

    // the upper-left(first thread tiles) of each warp-tile
    // ThreadMInWarpTile and ThreadNInWarpTile are the size of the thread grid(total = 32threads)
    const int ThreadTileIterMLen = WM/WMITER;  // not used
    const int ThreadTileIterNLen = WN/WNITER;
    // const int ThreadMInWarpTile = ThreadTileIterMLen/TM;  // not used
    const int ThreadNInWarpTile = ThreadTileIterNLen/TN;
    const int thread_m_in_thread_tiles = warp_offset / ThreadNInWarpTile;
    const int thread_n_in_thread_tiles = warp_offset % ThreadNInWarpTile;

    // Load to shared memory
    // Each thread loads 128bit(4 floats). Then all warps take turns to load.
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    float regM[WMITER * TM]= {0.0f};
    float regN[WNITER * TN]= {0.0f};
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    for(int kTile = 0;kTile<K;kTile+=BK){
        // Load A Transpose(load as As[BK][BM])
        for(int start = tid*4;start<BM*BK;start+=NUM_THREADS*4){
            int Asrow = start / BK;
            int Ascol = start % BK;
            // A[M][K]
            auto AStart = A + (blockIdx.y * BM * K + kTile) + (Asrow * K + Ascol);  // don't make it wrong!
            // assume divisible and no need for if conditions
            // not this way, because As is transposed
            // auto AsStart = As + (Asrow * BK + Ascol);
            // reinterpret_cast<float4*>(AsStart)[0] = reinterpret_cast<const float4*>(AStart)[0];
            if(AStart +3-A<M*K && K%4==0){
                const float4 a = reinterpret_cast<const float4*>(AStart)[0];
                // transpose in register
                #pragma unroll
                for(int i=0;i<4;i++){
                    // BlockAs[Asrow][Ascol+ 0~3] = tmp[0~3] = As[Ascol+ 0~3][Asrow]
                    As[(Ascol + i)*BM + Asrow] = reinterpret_cast<const float*>(&a)[i];
                }
            }else{
                int end = M*K - (AStart-A);
                for(int i=0;i<4;i++){
                    As[(Ascol + i)*BM + Asrow] = i<end ? AStart[i] : 0.0f;
                }
            }
        }
        // Load B as Bs[BK][BN]
        for(int start = tid*4;start<BK*BN;start+=NUM_THREADS*4){
            int Bsrow = start / BN;
            int Bscol = start % BN;
            auto BsStart = Bs + (Bsrow * BN + Bscol);
            auto BStart = B + (blockIdx.x * BN + kTile * N) + (Bsrow * N + Bscol);  // don't make it wrong!
            // assume divisible and no need for if conditions
            if(BStart +3-B<N*K && N%4==0){
                reinterpret_cast<float4*>(BsStart)[0] = reinterpret_cast<const float4*>(BStart)[0];
            }else{
                int end = N*K - (BStart-B);
                for(int i=0;i<4;i++){
                    BsStart[i] = i<end ? BStart[i] : 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute
        // Load Warp-tile of A and B to registers, compute and accumulate
        // Cs: to load/store one element of Cs
        // m axis(row): + warpRow*WM (BlockTile), + i*ThreadTileIterMLen (WarpTile, for i in WMITER), + thread_m_in_thread_tiles * TM (ThreadTiles), + ii(ThreadTile, for ii in TM)
        // n axis(col): + warpCol*WN (BlockTile), + j*ThreadTileIterNLen (WarpTile, for j in WNITER), + thread_n_in_thread_tiles * TN (ThreadTiles), + jj(ThreadTile, for jj in TN)

        // load As to regM
        for(int kk=0;kk<BK && kk + kTile < K;kk++){
            // Load A warp-tile to register
            for(int iterM = 0;iterM<WMITER;iterM++){
                for(int tm=0;tm<TM;tm++){
                    // if As: As[BM][BK], then load: As[row][kk]
                    // but As is transposed, so load: As[kk][row]
                    int row_in_As = warpRow*WM + iterM*ThreadTileIterMLen + thread_m_in_thread_tiles * TM + tm;
                    if(row_in_As + BM * blockIdx.y < M)
                        regM[iterM*TM+tm] = As[kk*BM + row_in_As];
                    else regM[iterM*TM+tm] = 0.0f;
                }
            }
            // load Bs to regN
            // Load B warp-tile to register
            for(int iterN = 0;iterN<WNITER;iterN++){
                for(int tn=0;tn<TN;tn++){
                    // Bs: Bs[BK][BN], then load: Bs[kk][col]
                    int col_in_Bs = warpCol*WN + iterN*ThreadTileIterNLen + thread_n_in_thread_tiles * TN + tn;
                    if(col_in_Bs + BN * blockIdx.x < N)
                        regN[iterN*TN+tn] = Bs[kk*BN + col_in_Bs];
                    else regN[iterN*TN+tn] = 0.0f;
                }
            }
            // compute
            for(int iterM = 0;iterM<WMITER;iterM++){
                for(int iterN = 0;iterN<WNITER;iterN++){
                    // one thread is responsible for computing WMITER*WNITER thread tiles, each of size TM*TN
                    // these small matrix mult computations are done by multiplying regM and regN
                    for(int tm=0;tm<TM;tm++){
                        for(int tn=0;tn<TN;tn++){
                            // compute one thread tile: regM[iterM*TM+tm] * regN[iterN*TN+tn]
                            // accumulate to threadResults[iterM*TM*WNITER*TN + iterN*TM*TN + tm*TN + tn]
                            threadResults[(iterM*TM + tm)*WNITER*TN + (iterN*TN + tn)] += regM[iterM*TM+tm] * regN[iterN*TN+tn];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    // Write back results from threadResults to C
    for(int iterM = 0;iterM<WMITER;iterM++){
        for(int iterN = 0;iterN<WNITER;iterN++){
            for(int tm=0;tm<TM;tm++){
                for(int tn=0;tn<TN;tn++){
                    int cRow = blockIdx.y * BM + warpRow*WM + iterM*ThreadTileIterMLen + thread_m_in_thread_tiles * TM + tm;
                    int cCol = blockIdx.x * BN + warpCol*WN + iterN*ThreadTileIterNLen + thread_n_in_thread_tiles * TN + tn;
                    if(cRow < M && cCol < N){
                        C[cRow * N + cCol] = alpha * threadResults[(iterM*TM + tm)*WNITER*TN + (iterN*TN + tn)] + beta * C[cRow * N + cCol];
                    }
                }
            }
        }
    }
}




// Wrapper function for the naive GEMM kernel
void warptile_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 blockDim(NUM_THREADS_);
    dim3 gridDim(CEIL_DIV(N, BN_), CEIL_DIV(M, BM_));

    warptile_gemm_kernel<BM_, BK_, BN_, WM_, WN_, WMITER_, WNITER_, TM_, TN_, NUM_THREADS_><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize(); // for more accurate timing, can be removed if timing is done outside

    CUDA_CHECK(cudaGetLastError());
}

// Register the kernel
REGISTER_GEMM_KERNEL("warptile", warptile_gemm, "Warptile GEMM implementation, with warp-level tiling and register accumulation");
