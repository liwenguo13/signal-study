#!/usr/bin/env python3
"""
A/B 双端声学测距
=================

【核心原理】
  B 端敲击 → 声波经介质传播 → A 端接收。
  程序计算的是：传播时延 = A 收到时刻 - B 发出时刻。
  距离 = 传播时延 × 介质声速。

【数据流 / 函数调用链】
  main()
    ├── load_metadata()              # 读取 JSON 元数据
    ├── load_signal()                # 读取 WAV 音频
    ├── analyze_confirmed_events()   # ★ 事件检测与配对（analyze_session_a.py）
    └── compute_distances()          # ★ 互相关精炼 + 测距
          ├── refine_delay_xcorr()   #   每对事件的原始音频互相关
          └── 第一个事件做同步锚点     #   吸收 PTP 时钟偏差

【互相关测距原理】
  analyze_session_a.py 检测到 13 个拍手事件，两端各有对应的 onset 时间。
  程序在每个 onset 附近截取一小段原始音频（±10ms），
  用互相关找到两个录音中同一事件波形的精确对齐偏移。

  第一个事件的互相关偏移 = PTP 时钟偏差 + 传播时延 ≈ 同步锚点。
  后续事件的互相关偏移 - 锚点 = 纯传播时延。
  距离 = 纯传播时延 × 声速。

  互相关精度：192kHz 下每样本 5.2μs ≈ 1.8mm@340m/s。
  不依赖 PTP 时钟精度——直接比较原始音频波形。

【为什么不依赖标定/校准？】
  不依赖任何先验物理距离输入。不是"先射箭再画靶子"。
  第一个事件自动吸收 PTP 偏差，后续事件只反映传播时延。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from analyze_session_a import (
    AnalysisResult,
    EventTemplate,
    ReportEvent,
    analyze_file,
    build_template,
    classify_candidates,
    infer_auto_parameters,
    recover_missing_events,
    result_rank_key,
)
from wav_signal import load_signal


SOUND_SPEEDS: dict[str, float] = {
    "air":      340.0,
    "water":   1500.0,
    "iron":    5000.0,
    "wood":    3500.0,
    "concrete": 3200.0,
}


@dataclass
class SessionMetadata:
    """JSON 元数据。node_id: "A"/"B", sample_rate_hz: 采样率, start_time_realtime_ns: PTP 启动时间。"""
    node_id: str
    sample_rate_hz: int
    start_time_realtime_ns: int


def load_metadata(path: Path) -> SessionMetadata:
    """从 JSON 加载元数据。"""
    payload: dict = json.loads(path.read_text())
    return SessionMetadata(
        node_id=str(payload["node_id"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        start_time_realtime_ns=int(payload["start_time_realtime_ns"]),
    )


def analyze_confirmed_events(
    wav_paths: list[Path],
    *,
    block_ms: float = 10.0,
    min_score_ratio: float = 0.002,
    min_event_gap_ms: float = 300.0,
    edge_margin_ms: float = 200.0,
    max_candidates: int = 128,
) -> tuple[list[AnalysisResult], EventTemplate]:
    """
    两轮检测 + 自动参数学习 + 模板匹配 + 双端补漏。
    返回每个 WAV 的分析结果和共用的事件模板。

    参数:
      block_ms:              时间块大小 (ms)。【调大】→ 灵敏度降，【调小】→ 噪声增
      min_score_ratio:       候选分数下限比例。【调大】→ 门槛高，【调小】→ 更灵敏
      min_event_gap_ms:      事件最小间隔 (ms)。【调大】→ 合并激进，【调小】→ 能分开近事件
      edge_margin_ms:        首尾忽略区 (ms)。避免录音启动/停止伪峰
      max_candidates:        每文件候选上限
    """
    scout_results: list[AnalysisResult] = [
        analyze_file(wav_path, block_ms=block_ms, min_score_ratio=min_score_ratio,
                     edge_margin_ms=edge_margin_ms, max_candidates=max_candidates,
                     percentile_threshold=95.0,
                     candidate_gap_ms=max(60.0, block_ms * 6.0), candidate_merge_gap_ms=0.06)
        for wav_path in wav_paths
    ]
    auto_params = infer_auto_parameters(scout_results, requested_block_ms=block_ms,
                                        requested_min_score_ratio=min_score_ratio,
                                        requested_min_event_gap_ms=min_event_gap_ms)
    results: list[AnalysisResult] = [
        analyze_file(wav_path, block_ms=auto_params.block_ms,
                     min_score_ratio=auto_params.min_score_ratio,
                     edge_margin_ms=edge_margin_ms, max_candidates=max_candidates,
                     percentile_threshold=96.0,
                     candidate_gap_ms=auto_params.candidate_gap_ms,
                     candidate_merge_gap_ms=auto_params.candidate_merge_gap_ms)
        for wav_path in wav_paths
    ]
    template: EventTemplate = build_template(results)
    for result in results:
        classify_candidates(result, template,
                            min_event_gap_ms=auto_params.min_event_gap_ms,
                            echo_gap_ms=auto_params.echo_gap_ms)
    if len(results) == 2:
        ordered: list[AnalysisResult] = sorted(results, key=result_rank_key, reverse=True)
        recover_missing_events(ordered[0], ordered[1], template,
                               min_event_gap_ms=auto_params.min_event_gap_ms,
                               edge_margin_ms=edge_margin_ms,
                               recover_search_radius_ms=auto_params.recover_search_radius_ms)
        if result_rank_key(ordered[1]) > result_rank_key(ordered[0]):
            ordered = [ordered[1], ordered[0]]
        recover_missing_events(ordered[0], ordered[1], template,
                               min_event_gap_ms=auto_params.min_event_gap_ms,
                               edge_margin_ms=edge_margin_ms,
                               recover_search_radius_ms=auto_params.recover_search_radius_ms)
    return results, template


# ──────────────────────────────────────────────────────────────────────
# refine_delay_xcorr — 用原始音频互相关精炼事件时间差
# ──────────────────────────────────────────────────────────────────────
def refine_delay_xcorr(
    sig_b: np.ndarray,
    sig_a: np.ndarray,
    sr: int,
    onset_b: float,
    onset_a: float,
    window_ms: float = 20.0,
) -> float:
    """
    对一对事件，用原始音频互相关找到亚样本精度的时间偏移。

    参数:
      sig_b:     B 端原始音频信号 (int64 数组)
      sig_a:     A 端原始音频信号 (int64 数组)
      sr:        采样率 (Hz)，本实验为 192000
      onset_b:   B 端检测到的事件起始秒（相对 B 录音起点）
      onset_a:   A 端检测到的事件起始秒（相对 A 录音起点）
      window_ms: 截取窗口宽度 (ms)。默认 20ms（±10ms 各侧）

    返回:
      delay_s: 传播时延 (秒)。正值 = A 晚于 B 收到。

    算法:
      1. 以 onset_b 为中心从 B 截取 20ms 窗口作为"模板"
      2. 以 onset_a 为中心从 A 截取 40ms 窗口作为"搜索区"
         （搜索区比模板宽，容纳 PTP 偏差 + 检测抖动）
      3. 互相关找到模板在搜索区中的最佳对齐位置
      4. 峰值位置对应的偏移量 = 传播时延 (样本数)

    为什么用原始音频？
      192kHz 每样本 5.2μs，精度比 10ms block 高 2000 倍。
      互相关直接比较波形形状，不需要 PTP 时钟。
    """
    # half_window: 每侧截取的样本数
    half_window: int = int(window_ms / 2.0 * sr / 1000.0)

    # ── 从 B 截取模板 ──
    idx_b: int = int(onset_b * sr)
    tpl_start: int = max(0, idx_b - half_window)
    tpl_end: int = min(len(sig_b), idx_b + half_window)
    template: np.ndarray = sig_b[tpl_start:tpl_end].astype(np.float64)

    # ── 从 A 截取搜索区（比模板宽，容纳 PTP 偏差 ±15ms）──
    idx_a: int = int(onset_a * sr)
    search_margin: int = int(15.0 * sr / 1000.0)
    sch_start: int = max(0, idx_a - half_window - search_margin)
    sch_end: int = min(len(sig_a), idx_a + half_window + search_margin)
    search: np.ndarray = sig_a[sch_start:sch_end].astype(np.float64)

    # ── 边界检查 ──
    if len(template) < 10 or len(search) < len(template):
        return float("nan")

    # ── 互相关 ──
    # mode='valid': 输出长度 = len(search) - len(template) + 1
    # 每个输出位置 k 对应模板在 search[k : k+len(template)] 处对齐
    corr: np.ndarray = scipy_signal.correlate(search, template, mode="valid")
    k_peak: int = int(np.argmax(np.abs(corr)))

    # ── 计算传播时延 ──
    # 模板在 B 中的中心位置: tpl_start + half_window = idx_b
    # 模板在 search 中的最佳位置: sch_start + k_peak + half_window
    # 偏移 (样本) = (sch_start + k_peak + half_window) - (tpl_start + half_window)
    #             = sch_start + k_peak - tpl_start
    # 但 tpl_start = idx_b - half_window, sch_start = idx_a - half_window - search_margin
    # 偏移 = (idx_a - half_window - search_margin) + k_peak - (idx_b - half_window)
    #      = idx_a - idx_b - search_margin + k_peak
    # 传播时延 (秒) = 偏移 / sr
    delay_s: float = (idx_a - idx_b - search_margin + k_peak) / sr

    return delay_s


# ──────────────────────────────────────────────────────────────────────
# compute_distances — 互相关精炼 + 第一个事件做同步锚点
# ──────────────────────────────────────────────────────────────────────
def compute_distances(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    sr: int,
    events_a: list[ReportEvent],
    events_b: list[ReportEvent],
    speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用互相关精炼每对事件的传播时延，第一个事件做同步锚点。

    参数:
      sig_a:     A 端原始音频信号
      sig_b:     B 端原始音频信号
      sr:        采样率 (Hz)
      events_a:  A 端已确认事件列表（按 onset_s 排序）
      events_b:  B 端已确认事件列表（按 onset_s 排序）
      speed_mps: 介质声速 (m/s)

    返回:
      delays_s:    传播时延数组 (秒)
      distances_m: 距离数组 (米)

    算法:
      1. 第一个事件: xcorr_lag_0 = refine_delay_xcorr(事件0)
         → 这是"同步锚点"，包含 PTP 偏差 + 传播时延
      2. 后续事件: xcorr_lag_i = refine_delay_xcorr(事件i)
      3. 纯传播时延_i = xcorr_lag_i - xcorr_lag_0
         → PTP 偏差被减掉，只剩传播时延差
      4. 距离_i = 纯传播时延_i × 声速

    为什么第一个事件能做同步锚点？
      第一个事件在两个录音中的互相关偏移 = PTP 时钟偏差 + 传播时延。
      后续事件的互相关偏移也包含同样的 PTP 偏差。
      相减后 PTP 偏差被消除，只剩传播时延差。
    """
    n: int = min(len(events_a), len(events_b))
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # ── 对每个事件做互相关精炼 ──
    # xcorr_lags[i] = 第 i 个事件在 A 和 B 之间的互相关偏移 (秒)
    # 包含: PTP 偏差 + 传播时延 + 检测噪声
    xcorr_lags: np.ndarray = np.empty(n, dtype=np.float64)
    for i in range(n):
        xcorr_lags[i] = refine_delay_xcorr(
            sig_b, sig_a, sr,
            onset_b=events_b[i].onset_s,
            onset_a=events_a[i].onset_s,
        )

    # ── 第一个事件做同步锚点 ──
    anchor: float = xcorr_lags[0]
    delays_s: np.ndarray = xcorr_lags - anchor

    # ── 离群值剔除 ──
    # 互相关可能在某些事件上误匹配（波形相似但位置错误）。
    # 用中位数绝对偏差 (MAD) 检测离群值：偏差超过 3 倍 MAD 的事件标记为 NaN。
    median_delay: float = float(np.nanmedian(delays_s))
    mad: float = float(np.nanmedian(np.abs(delays_s - median_delay)))
    if mad > 0.0:
        threshold: float = 3.0 * mad
        outlier_mask: np.ndarray = np.abs(delays_s - median_delay) > threshold
        delays_s[outlier_mask] = float("nan")

    distances_m: np.ndarray = delays_s * speed_mps
    return delays_s, distances_m


def main() -> None:
    """命令行入口。解析参数 → 检测事件 → 互相关精炼 → 输出距离。"""

    parser = argparse.ArgumentParser(
        description="A/B 双端声学测距 —— 互相关精炼 + 同步锚点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
介质选项 (--medium):
  air      空气 20°C, 340 m/s
  water    水, 1500 m/s
  iron     铁/钢管, 5000 m/s
  wood     木材, 3500 m/s
  concrete 混凝土, 3200 m/s

示例:
  python estimate_distance.py session_A.wav session_B.wav \\
      --json-a session_A.json --json-b session_B.json
        """,
    )
    parser.add_argument("wav_a", type=Path, help="A 端 WAV 文件（接收端）")
    parser.add_argument("wav_b", type=Path, help="B 端 WAV 文件（声源端）")
    parser.add_argument("--json-a", type=Path, required=True, help="A 端元数据 JSON")
    parser.add_argument("--json-b", type=Path, required=True, help="B 端元数据 JSON")
    parser.add_argument("--medium", choices=list(SOUND_SPEEDS), default="air",
                        help="传播介质（影响声速）")
    parser.add_argument("--speed-mps", type=float,
                        help="自定义声速 m/s（覆盖 --medium）")
    args = parser.parse_args()

    speed_mps: float = args.speed_mps if args.speed_mps else SOUND_SPEEDS[args.medium]

    meta_a: SessionMetadata = load_metadata(args.json_a)
    meta_b: SessionMetadata = load_metadata(args.json_b)
    if meta_a.node_id != "A" or meta_b.node_id != "B":
        raise SystemExit("JSON node_id 必须分别为 A 和 B")

    sr_a: int
    sig_a: np.ndarray
    sr_b: int
    sig_b: np.ndarray
    sr_a, sig_a = load_signal(args.wav_a)
    sr_b, sig_b = load_signal(args.wav_b)
    if sr_a != sr_b:
        raise ValueError("A/B 两端采样率不一致。")
    if sr_a != meta_a.sample_rate_hz or sr_b != meta_b.sample_rate_hz:
        raise ValueError("WAV 与 JSON 采样率不一致。")

    # ── 事件检测（来自 analyze_session_a.py）──
    results: list[AnalysisResult]
    template: EventTemplate
    results, template = analyze_confirmed_events([args.wav_a, args.wav_b])
    result_by_path: dict[Path, AnalysisResult] = {r.wav_path.resolve(): r for r in results}
    events_a: list[ReportEvent] = result_by_path[args.wav_a.resolve()].confirmed_events
    events_b: list[ReportEvent] = result_by_path[args.wav_b.resolve()].confirmed_events

    # ── 互相关精炼测距 ──
    delays_s: np.ndarray
    distances_m: np.ndarray
    delays_s, distances_m = compute_distances(
        sig_a, sig_b, sr_a, events_a, events_b, speed_mps,
    )

    abs_delays: np.ndarray = np.abs(delays_s)
    abs_dists: np.ndarray = np.abs(distances_m)

    # ── 输出 ──
    print(f"介质: {args.medium} ({speed_mps:.0f} m/s)")
    print(f"A 端事件数: {len(events_a)}  B 端事件数: {len(events_b)}")
    print(f"有效配对: {len(delays_s)} 对")
    print(f"测距方法: 原始音频互相关 + 第一个事件同步锚点")
    print()
    print(f"传播时延  中位数={np.nanmedian(delays_s) * 1000:.3f}ms  "
          f"均值={np.nanmean(delays_s) * 1000:.3f}ms  "
          f"标准差={np.nanstd(delays_s) * 1000:.3f}ms")
    print(f"传播时延  |中位数|={np.nanmedian(abs_delays) * 1000:.3f}ms  "
          f"|均值|={np.nanmean(abs_delays) * 1000:.3f}ms")
    print()
    print(f"距离(有符号)  中位数={np.nanmedian(distances_m):.3f}m  "
          f"均值={np.nanmean(distances_m):.3f}m  "
          f"标准差={np.nanstd(distances_m):.3f}m")
    print(f"距离(绝对值)  中位数={np.nanmedian(abs_dists):.3f}m  "
          f"均值={np.nanmean(abs_dists):.3f}m")
    print()
    for i in range(len(delays_s)):
        print(
            f"{i + 1:02d}  "
            f"A_onset={events_a[i].onset_s:.6f}s  "
            f"B_onset={events_b[i].onset_s:.6f}s  "
            f"delay={delays_s[i] * 1000:.3f}ms  "
            f"dist={distances_m[i]:.3f}m  "
            f"A_confirm={events_a[i].confirmation}  "
            f"B_confirm={events_b[i].confirmation}"
        )


if __name__ == "__main__":
    main()
