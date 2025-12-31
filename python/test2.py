import gemm_ext as gx
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


backend_list = [
    gx.GemmBackend.CuBLAS,
    gx.GemmBackend.Naive,
    gx.GemmBackend.Opt1,
    gx.GemmBackend.Opt2,
    gx.GemmBackend.Opt3,
]

seed = 100
EPS = 0.001


def gemm_FLOPS(params, elapsed_time_ms):
    return params.M * params.N * params.K * 2 / (elapsed_time_ms * 1e-3)


def prepare_runner(backend, params):
    runner = gx.GemmRunner(params, backend)
    runner.init()
    runner.fill_random(seed)
    runner.run_once()
    runner.fetch_result()
    return runner


def prepare_mnk(use_random=False):
    mnk_sample_range = [1, 4096]
    mnk_list = [
        [256] * 3,
        [512] * 3,
        [1024] * 3,
        [2048] * 3,
        [4096] * 3,
    ]
    if use_random:
        mnk_list.extend(
            [[random.randint(*mnk_sample_range) for _ in range(3)] for _ in range(5)]
        )
    return mnk_list


def verify_backend(backend):
    # 对一个非CuBlas的kernel后端，对很多矩阵形状，验证结果是否正确
    if backend == gx.GemmBackend.CuBLAS:
        return
    mnk_list = prepare_mnk(use_random=True)

    print(f"verify backend: {gx.gemm_backend_name(backend)}")
    all_same = True
    for M, N, K in mnk_list:
        params = gx.GemmParams(M, N, K)
        baseline_runner = prepare_runner(gx.GemmBackend.CuBLAS, params)
        runner = prepare_runner(backend, params)
        same = gx.GemmRunner.compare(baseline_runner, runner, EPS)
        print(f"verify: {same}, m={params.M}, n={params.N}, k={params.K}")
        all_same &= same
        runner.release()
        baseline_runner.release()
    print(f"all verify: {all_same}")
    print()


def runner_benchmark(runner, params, num_runs=10):
    time_ms = runner.run_benchmark(num_runs)
    flops = gemm_FLOPS(params, time_ms)
    GFLOPS = flops / 1e9
    return time_ms, GFLOPS  # 返回耗时和GFLOPS，方便数据收集


# 函数1：执行基准测试，收集并打印结果（无绘图逻辑）
def run_all_benchmark(mnk_list, backend_list):
    """
    对所有backend在每个mnk参数下执行基准测试，收集并打印GFLOPS结果

    参数:
        mnk_list: list of list，每个元素是[M, N, K]的参数组合
        backend_list: list of gx.GemmBackend，需要测试的后端列表

    返回:
        benchmark_results: dict，键为backend名称，值为对应GFLOPS列表
        mnk_labels: list，格式化后的MNK参数标签（用于绘图横坐标）
    """
    # 1. 初始化数据收集容器
    benchmark_results = {}
    for backend in backend_list:
        backend_name = gx.gemm_backend_name(backend)
        benchmark_results[backend_name] = []

    # 2. 格式化MNK标签（后续绘图用）
    mnk_labels = [f"[{M},{N},{K}]" for M, N, K in mnk_list]

    # 3. 遍历所有MNK参数组合执行测试
    for idx, (M, N, K) in enumerate(mnk_list):
        print(
            f"===== 正在测试MNK参数: [{M}, {N}, {K}] (第{idx+1}/{len(mnk_list)}组) ====="
        )
        # 构造GemmParams
        params = gx.GemmParams(M, N, K)

        # 遍历所有backend测试
        for backend in backend_list:
            backend_name = gx.gemm_backend_name(backend)
            # 准备runner并执行基准测试
            runner = prepare_runner(backend, params)
            time_ms, gflops = runner_benchmark(runner, params)
            # 收集数据
            benchmark_results[backend_name].append(gflops)
            # 打印单条结果
            print(f"  {backend_name}: 耗时={time_ms:.3f} ms, GFLOPS={gflops:.3f}")
            # 释放资源
            runner.release()
        print()  # 空行分隔，优化输出格式

    # 4. 打印汇总结果表格
    print("=" * 60)
    print("所有基准测试汇总结果")
    print("=" * 60)
    backend_names = list(benchmark_results.keys())
    # 打印表头
    print(f"{'MNK参数':<20}", end="")
    for bk_name in backend_names:
        print(f"{bk_name:<15}", end="")
    print()
    print("-" * 60)
    # 打印每行数据
    for mnk_idx, mnk_label in enumerate(mnk_labels):
        print(f"{mnk_label:<20}", end="")
        for bk_name in backend_names:
            gflops_val = benchmark_results[bk_name][mnk_idx]
            print(f"{gflops_val:<15.3f}", end="")
        print()

    return benchmark_results, mnk_labels


# 函数2：专门用于绘制折线图（接收测试结果数据）
def plot_benchmark_results(benchmark_results, mnk_labels, save_path=None):
    """
    根据基准测试结果绘制折线图，每条折线对应一个backend

    参数:
        benchmark_results: dict，run_all_benchmark返回的测试结果
        mnk_labels: list，run_all_benchmark返回的MNK格式化标签
        save_path: str (可选)，图表保存路径（如"gemm_perf.png"），不传入则仅显示
    """
    print("\n正在绘制折线图...")
    # 配置绘图样式
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 中文支持可替换为"SimHei"
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # 遍历每个backend绘制折线
    backend_names = list(benchmark_results.keys())
    for bk_name in backend_names:
        gflops_list = benchmark_results[bk_name]
        plt.plot(
            mnk_labels,
            gflops_list,
            marker="o",  # 数据点标记
            linewidth=2,
            label=bk_name,
        )

    # 设置图表属性
    plt.xlabel("MNK Parameters", fontsize=12, fontweight="bold")
    plt.ylabel("GFLOPS", fontsize=12, fontweight="bold")
    plt.title(
        "GEMM Backend FLOPS Performance Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(rotation=45, ha="right")  # 横坐标旋转，避免重叠
    plt.legend(loc="best", fontsize=10)  # 显示图例
    plt.tight_layout()  # 自动调整布局

    # 保存图表（可选）
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    for backend in backend_list[1:]:
        backend_name = gx.gemm_backend_name(backend)
        print(f"===== 正在验证后端: {backend_name} =====")
        verify_backend(backend)

    test_mnk_list = prepare_mnk(use_random=False)
    benchmark_data, mnk_tags = run_all_benchmark(test_mnk_list, backend_list)
    # plot_benchmark_results(
    #     benchmark_data, mnk_tags, save_path="gemm_benchmark_perf.png"
    # )  # 显示并保存
