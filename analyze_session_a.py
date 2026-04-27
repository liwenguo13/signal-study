#!/usr/bin/env python3
"""
敲击事件检测脚本 - 用于从WAV音频文件中检测并定位敲击声（如敲门声）的时间点

工作流程：
1. 加载WAV文件并转换为PCM信号
2. 计算信号强度（取各通道最大绝对值）
3. 创建粗粒度包络（分块取最大值）
4. 在包络中检测超过阈值的局部峰值
5. 在粗检测点附近进行精细搜索，精确定位峰值时间
"""

import argparse
from pathlib import Path

import numpy as np

from wav_signal import load_signal


def block_max(signal: np.ndarray, block_size: int) -> np.ndarray:
    """
    对信号进行分块并取每块的最大值，创建粗粒度包络
    
    Args:
        signal: 输入信号数组
        block_size: 分块大小（样本点数）
        
    Returns:
        分块后的最大值数组（粗粒度包络）
        
    Note:
        如果信号长度不能被block_size整除，会在末尾补零
    """
    if block_size <= 1:
        return signal.copy()

    # 计算需要填充的样本数，使总长度能被block_size整除
    pad = (-len(signal)) % block_size
    if pad:
        signal = np.pad(signal, (0, pad), mode="constant")
    return signal.reshape(-1, block_size).max(axis=1)


def detect_knock_times(
    signal: np.ndarray,
    sample_rate: int,
    *,
    block_ms: float,
    threshold_ratio: float,
    refractory_ms: float,
    refine_ms: float,
) -> list[float]:
    """
    检测敲击事件的时间点
    
    Args:
        signal: 输入信号强度数组
        sample_rate: 采样率 (Hz)
        block_ms: 粗粒度包络的块大小（毫秒）
        threshold_ratio: 检测阈值比例（相对于全局最大值）
        refractory_ms: 折返期（毫秒），两次检测间的最小间隔
        refine_ms: 精细搜索半径（毫秒）
        
    Returns:
        检测到的敲击事件时间列表（秒）
        
    Algorithm:
        1. 创建粗粒度包络
        2. 设置检测阈值
        3. 寻找局部最大值且超过阈值的点
        4. 应用折返期约束避免重复检测
        5. 在粗检测点附近进行精细搜索
    """
    # 将毫秒参数转换为样本点数
    block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
    coarse_env = block_max(signal, block_size)
    threshold = float(coarse_env.max()) * threshold_ratio

    # 计算折返期对应的块数和精细搜索半径对应的样本数
    refractory_blocks = max(1, int(round(refractory_ms / block_ms)))
    refine_radius = max(block_size, int(round(sample_rate * refine_ms / 1000.0)))

    # 存储粗检测到的峰值索引
    coarse_indices: list[int] = []
    last_accepted = -refractory_blocks

    # 遍历粗粒度包络，寻找符合条件的局部最大值
    for idx in range(1, len(coarse_env) - 1):
        value = coarse_env[idx]
        # 跳过低于阈值的点
        if value < threshold:
            continue
        # 必须是局部最大值（比左右邻居都大）
        if value < coarse_env[idx - 1] or value < coarse_env[idx + 1]:
            continue
        # 必须满足折返期间隔要求
        if idx - last_accepted < refractory_blocks:
            continue
        coarse_indices.append(idx)
        last_accepted = idx

    # 在粗检测点附近进行精细搜索，找到精确的峰值位置
    peak_times: list[float] = []
    for idx in coarse_indices:
        center = idx * block_size                    # 粗检测点对应的原始信号位置
        start = max(0, center - refine_radius)       # 精细搜索起始位置
        stop = min(len(signal), center + refine_radius + 1)  # 精细搜索结束位置
        local_offset = int(np.argmax(signal[start:stop]))    # 在局部范围内找最大值
        sample_index = start + local_offset          # 精确的峰值样本索引
        peak_times.append(sample_index / sample_rate)        # 转换为时间（秒）

    return peak_times


def main() -> None:
    """主函数：解析命令行参数并执行敲击检测"""
    parser = argparse.ArgumentParser(
        description="Find knock event times in a WAV recording."
    )
    parser.add_argument("wav_path", type=Path, help="Path to the WAV file")
    parser.add_argument(
        "--block-ms",
        type=float,
        default=1.0,
        help="Coarse envelope block size in milliseconds. "
             "Smaller values = higher time resolution but more noise sensitivity. "
             "Typical range: 0.5-5.0 ms"
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.2,
        help="Detection threshold as a fraction of the global block maximum. "
             "Lower values = more sensitive (detect weaker knocks), "
             "higher values = less sensitive (ignore weaker knocks). "
             "Typical range: 0.1-0.5"
    )
    parser.add_argument(
        "--refractory-ms",
        type=float,
        default=150.0,
        help="Minimum gap between two knock detections in milliseconds. "
             "Prevents multiple detections from the same knock event. "
             "Adjust based on expected knock frequency. "
             "Typical range: 50-1000 ms"
    )
    parser.add_argument(
        "--refine-ms",
        type=float,
        default=20.0,
        help="Fine search radius around each coarse detection in milliseconds. "
             "Larger values = more accurate peak localization but slower processing. "
             "Should be larger than block-ms for best results. "
             "Typical range: 10-100 ms"
    )
    args = parser.parse_args()

    sample_rate, signal = load_signal(args.wav_path)
    peak_times = detect_knock_times(
        signal,
        sample_rate,
        block_ms=args.block_ms,
        threshold_ratio=args.threshold_ratio,
        refractory_ms=args.refractory_ms,
        refine_ms=args.refine_ms,
    )

    print(f"detected_peaks={len(peak_times)}")
    for index, peak_time in enumerate(peak_times, start=1):
        print(f"{index:02d} {peak_time:.6f}")


if __name__ == "__main__":
    main()
