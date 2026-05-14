#!/usr/bin/env python3
# 这是一个Python脚本的shebang行，告诉系统用python3来执行这个文件

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
# 上面是模块级文档字符串，描述了整个程序的核心原理、数据流、互相关测距原理
# 以及为什么不依赖标定/校准

from __future__ import annotations
# 从__future__模块导入annotations，允许使用Python 3.10+的类型注解语法

import argparse
# 导入argparse模块，用于解析命令行参数

import json
# 导入json模块，用于读取和解析JSON文件

from dataclasses import dataclass
# 从dataclasses模块导入dataclass装饰器，用于创建数据类

from pathlib import Path
# 从pathlib模块导入Path类，用于处理文件路径

import numpy as np
# 导入numpy库并重命名为np，用于数值计算

from scipy import signal as scipy_signal
# 从scipy库导入signal模块并重命名为scipy_signal，用于信号处理（互相关计算）

from analyze_session_a import (
    AnalysisResult,      # 分析结果数据类
    EventTemplate,       # 事件模板数据类
    ReportEvent,         # 报告事件数据类
    analyze_file,        # 分析单个WAV文件的函数
    build_template,      # 构建事件模板的函数
    classify_candidates, # 分类候选事件的函数
    infer_auto_parameters, # 推断自动参数的函数
    recover_missing_events, # 恢复缺失事件的函数
    result_rank_key,     # 结果排序键函数
)
# 从analyze_session_a模块导入所有需要的类和函数

from wav_signal import load_signal
# 从wav_signal模块导入load_signal函数，用于加载WAV音频文件


SOUND_SPEEDS: dict[str, float] = {
    "air":      340.0,    # 空气中的声速：340米/秒
    "water":   1500.0,    # 水中的声速：1500米/秒
    "iron":    5000.0,    # 铁/钢管中的声速：5000米/秒
    "wood":    3500.0,    # 木材中的声速：3500米/秒
    "concrete": 3200.0,   # 混凝土中的声速：3200米/秒
}
# 定义不同介质的声速字典，键是介质名称，值是声速（米/秒）


@dataclass
class SessionMetadata:
    """JSON 元数据。node_id: "A"/"B", sample_rate_hz: 采样率, start_time_realtime_ns: PTP 启动时间。"""
    # 使用@dataclass装饰器创建数据类，自动生成__init__等方法
    
    node_id: str
    # 节点ID，表示是A端还是B端，值为"A"或"B"
    
    sample_rate_hz: int
    # 采样率，单位是赫兹(Hz)，本实验为192000Hz
    
    start_time_realtime_ns: int
    # PTP启动时间，单位是纳秒(ns)，用于时间同步


def load_metadata(path: Path) -> SessionMetadata:
    """从 JSON 加载元数据。"""
    # 函数功能：从JSON文件加载元数据
    
    # 参数:
    #   path: JSON文件的路径
    
    # 返回:
    #   SessionMetadata: 包含node_id、sample_rate_hz、start_time_realtime_ns的数据类实例
    
    payload: dict = json.loads(path.read_text())
    # 读取JSON文件内容并解析为Python字典
    # path.read_text() 读取文件的全部文本内容
    # json.loads() 将JSON字符串解析为Python字典
    
    return SessionMetadata(
        node_id=str(payload["node_id"]),
        # 从字典中获取"node_id"键的值，并转换为字符串
        
        sample_rate_hz=int(payload["sample_rate_hz"]),
        # 从字典中获取"sample_rate_hz"键的值，并转换为整数
        
        start_time_realtime_ns=int(payload["start_time_realtime_ns"]),
        # 从字典中获取"start_time_realtime_ns"键的值，并转换为整数
    )
    # 返回一个SessionMetadata实例，包含所有元数据


def analyze_confirmed_events(
    wav_paths: list[Path],
    # 参数：WAV文件路径列表，包含一个或两个WAV文件
    
    *,
    # *表示后面的参数必须使用关键字方式传递
    
    block_ms: float = 10.0,
    # 时间块大小，单位是毫秒(ms)，默认值10.0ms
    # 调大：灵敏度降低，噪声减少，事件可能被合并
    # 调小：灵敏度升高，噪声增多，事件可能被拆开
    
    min_score_ratio: float = 0.002,
    # 候选分数下限比例，默认值0.002
    # 调大：门槛高，漏检多，只保留强事件
    # 调小：门槛低，误检多，弱事件也能进入
    
    min_event_gap_ms: float = 300.0,
    # 事件最小间隔，单位是毫秒(ms)，默认值300.0ms
    # 调大：合并激进，快速敲击可能被合并
    # 调小：能分开近事件，但可能多检
    
    edge_margin_ms: float = 200.0,
    # 首尾忽略区，单位是毫秒(ms)，默认值200.0ms
    # 用于避免录音启动/停止时的伪峰
    
    max_candidates: int = 128,
    # 每个文件的最大候选事件数，默认值128
    # 只是候选上限，不是最终事件数上限
    
) -> tuple[list[AnalysisResult], EventTemplate]:
    # 返回值：一个元组，包含两个元素：
    #   1. AnalysisResult列表：每个WAV文件的分析结果
    #   2. EventTemplate：共用的事件模板
    
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
    # 函数文档字符串，描述函数的功能和参数
    
    scout_results: list[AnalysisResult] = [
        analyze_file(wav_path, block_ms=block_ms, min_score_ratio=min_score_ratio,
                     edge_margin_ms=edge_margin_ms, max_candidates=max_candidates,
                     percentile_threshold=95.0,
                     candidate_gap_ms=max(60.0, block_ms * 6.0), candidate_merge_gap_ms=0.06)
        for wav_path in wav_paths
    ]
    # 第一轮：侦察模式
    # 对每个WAV文件调用analyze_file函数进行初步分析
    # percentile_threshold=95.0：使用95分位数作为阈值，比较宽松
    # candidate_gap_ms=max(60.0, block_ms * 6.0)：候选分组间隔，至少60ms或block_ms的6倍
    # candidate_merge_gap_ms=0.06：候选合并间隔，0.06秒=60ms
    # 结果存储在scout_results列表中
    
    auto_params = infer_auto_parameters(scout_results, requested_block_ms=block_ms,
                                        requested_min_score_ratio=min_score_ratio,
                                        requested_min_event_gap_ms=min_event_gap_ms)
    # 根据侦察结果自动推断参数
    # infer_auto_parameters函数会分析侦察结果，学习出更适合当前环境的参数
    # 返回一个AutoLearnedParameters对象，包含自动学习的参数
    
    results: list[AnalysisResult] = [
        analyze_file(wav_path, block_ms=auto_params.block_ms,
                     min_score_ratio=auto_params.min_score_ratio,
                     edge_margin_ms=edge_margin_ms, max_candidates=max_candidates,
                     percentile_threshold=96.0,
                     candidate_gap_ms=auto_params.candidate_gap_ms,
                     candidate_merge_gap_ms=auto_params.candidate_merge_gap_ms)
        for wav_path in wav_paths
    ]
    # 第二轮：正式模式
    # 使用自动学习的参数重新分析每个WAV文件
    # percentile_threshold=96.0：使用96分位数作为阈值，比侦察阶段稍严格
    # 这次的结果更准确，用于后续处理
    
    template: EventTemplate = build_template(results)
    # 从所有候选事件中构建事件模板
    # 模板表示"本次录音中这类事件的共同形状"
    # 用于后续的模板匹配和确认
    
    for result in results:
        # 遍历每个分析结果
        
        classify_candidates(result, template,
                            min_event_gap_ms=auto_params.min_event_gap_ms,
                            echo_gap_ms=auto_params.echo_gap_ms)
        # 对每个文件的候选事件进行分类和确认
        # 使用模板相似度、能量、宽度等条件筛选
        # echo_gap_ms用于抑制回响
    
    if len(results) == 2:
        # 如果有两个文件（A端和B端）
        
        ordered: list[AnalysisResult] = sorted(results, key=result_rank_key, reverse=True)
        # 按结果排序键排序，reverse=True表示降序
        # result_rank_key返回(确认事件数, 相似度总和+恢复奖励, 能量总和)
        # 排序后，ordered[0]是更强的参考端
        
        recover_missing_events(ordered[0], ordered[1], template,
                               min_event_gap_ms=auto_params.min_event_gap_ms,
                               edge_margin_ms=edge_margin_ms,
                               recover_search_radius_ms=auto_params.recover_search_radius_ms)
        # 第一轮复核：在ordered[1]（较弱端）中查找ordered[0]（较强端）有但ordered[1]没有的事件
        # 使用模板匹配和搜索半径来寻找弱端的缺失事件
        
        if result_rank_key(ordered[1]) > result_rank_key(ordered[0]):
            # 如果复核后ordered[1]的排序键大于ordered[0]
            # 说明ordered[1]现在更强了
            
            ordered = [ordered[1], ordered[0]]
            # 交换顺序，让更强的端作为reference
        
        recover_missing_events(ordered[0], ordered[1], template,
                               min_event_gap_ms=auto_params.min_event_gap_ms,
                               edge_margin_ms=edge_margin_ms,
                               recover_search_radius_ms=auto_params.recover_search_radius_ms)
        # 第二轮复核：再次在较弱端查找缺失事件
        # 这是为了确保两边的事件尽可能对齐
    
    return results, template
    # 返回分析结果列表和事件模板


# ──────────────────────────────────────────────────────────────────────
# refine_delay_xcorr — 用原始音频互相关精炼事件时间差
# ──────────────────────────────────────────────────────────────────────
def refine_delay_xcorr(
    sig_b: np.ndarray,
    # 参数：B端原始音频信号，int64数组
    
    sig_a: np.ndarray,
    # 参数：A端原始音频信号，int64数组
    
    sr: int,
    # 参数：采样率，单位是赫兹(Hz)，本实验为192000
    
    onset_b: float,
    # 参数：B端检测到的事件起始秒，相对于B端录音起点
    
    onset_a: float,
    # 参数：A端检测到的事件起始秒，相对于A端录音起点
    
    window_ms: float = 20.0,
    # 参数：截取窗口宽度，单位是毫秒(ms)，默认值20.0ms
    # 实际截取的是±10ms，总共20ms
    
) -> float:
    # 返回值：传播时延，单位是秒(s)
    # 正值表示A端晚于B端收到信号
    
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
    # 函数文档字符串，描述函数的功能、参数、返回值和算法
    
    # half_window: 每侧截取的样本数
    half_window: int = int(window_ms / 2.0 * sr / 1000.0)
    # 计算每侧截取的样本数
    # window_ms / 2.0 = 10ms（每侧）
    # sr / 1000.0 = 192（每毫秒的样本数）
    # 10ms * 192 = 1920个样本
    # int()取整，确保是整数

    # ── 从 B 截取模板 ──
    idx_b: int = int(onset_b * sr)
    # 将B端事件起始时间从秒转换为样本索引
    # onset_b是秒，sr是样本/秒，相乘得到样本索引
    
    tpl_start: int = max(0, idx_b - half_window)
    # 模板起始位置：事件中心向左half_window个样本
    # max(0, ...)确保不超出数组左边界
    
    tpl_end: int = min(len(sig_b), idx_b + half_window)
    # 模板结束位置：事件中心向右half_window个样本
    # min(len(sig_b), ...)确保不超出数组右边界
    
    template: np.ndarray = sig_b[tpl_start:tpl_end].astype(np.float64)
    # 从B端信号中截取模板
    # 转换为float64类型，便于后续计算

    # ── 从 A 截取搜索区（比模板宽，容纳 PTP 偏差 ±15ms）──
    idx_a: int = int(onset_a * sr)
    # 将A端事件起始时间从秒转换为样本索引
    
    search_margin: int = int(15.0 * sr / 1000.0)
    # 搜索区的额外边距：15ms对应的样本数
    # 15ms * 192样本/ms = 2880个样本
    # 用于容纳PTP时钟偏差和检测抖动
    
    sch_start: int = max(0, idx_a - half_window - search_margin)
    # 搜索区起始位置：事件中心向左half_window+search_margin个样本
    # max(0, ...)确保不超出数组左边界
    
    sch_end: int = min(len(sig_a), idx_a + half_window + search_margin)
    # 搜索区结束位置：事件中心向右half_window+search_margin个样本
    # min(len(sig_a), ...)确保不超出数组右边界
    
    search: np.ndarray = sig_a[sch_start:sch_end].astype(np.float64)
    # 从A端信号中截取搜索区
    # 转换为float64类型，便于后续计算

    # ── 边界检查 ──
    if len(template) < 10 or len(search) < len(template):
        return float("nan")
    # 如果模板长度小于10个样本，或者搜索区比模板还短
    # 返回NaN（Not a Number），表示无法计算

    # ── 互相关 ──
    # mode='valid': 输出长度 = len(search) - len(template) + 1
    # 每个输出位置 k 对应模板在 search[k : k+len(template)] 处对齐
    corr: np.ndarray = scipy_signal.correlate(search, template, mode="valid")
    # 计算互相关
    # search是搜索区，template是模板
    # mode='valid'表示只返回完全重叠的部分
    # 输出数组的每个位置k表示模板在search[k:k+len(template)]处的相关程度
    
    k_peak: int = int(np.argmax(np.abs(corr)))
    # 找到互相关峰值的位置
    # np.abs(corr)取绝对值，因为可能是正相关或负相关
    # np.argmax找到最大值的索引
    # int()转换为整数

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
    # 计算传播时延
    # idx_a - idx_b：A端和B端事件位置的差（样本数）
    # - search_margin：减去搜索区的额外边距
    # + k_peak：加上互相关找到的最佳对齐位置
    # / sr：转换为秒
    # 结果delay_s是传播时延，正值表示A端晚于B端收到信号

    return delay_s
    # 返回传播时延（秒）


# ──────────────────────────────────────────────────────────────────────
# compute_distances — 互相关精炼 + 第一个事件做同步锚点
# ──────────────────────────────────────────────────────────────────────
def compute_distances(
    sig_a: np.ndarray,
    # 参数：A端原始音频信号
    
    sig_b: np.ndarray,
    # 参数：B端原始音频信号
    
    sr: int,
    # 参数：采样率，单位是赫兹(Hz)
    
    events_a: list[ReportEvent],
    # 参数：A端已确认事件列表，按onset_s排序
    
    events_b: list[ReportEvent],
    # 参数：B端已确认事件列表，按onset_s排序
    
    speed_mps: float,
    # 参数：介质声速，单位是米/秒(m/s)
    
) -> tuple[np.ndarray, np.ndarray]:
    # 返回值：一个元组，包含两个numpy数组：
    #   1. delays_s：传播时延数组，单位是秒(s)
    #   2. distances_m：距离数组，单位是米(m)
    
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
    # 函数文档字符串，描述函数的功能、参数、返回值和算法
    
    n: int = min(len(events_a), len(events_b))
    # 计算有效事件对数：取A端和B端事件数的最小值
    # 例如：A端有13个事件，B端有13个事件，n=13
    
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    # 如果没有有效事件对，返回两个空数组

    # ── 对每个事件做互相关精炼 ──
    # xcorr_lags[i] = 第 i 个事件在 A 和 B 之间的互相关偏移 (秒)
    # 包含: PTP 偏差 + 传播时延 + 检测噪声
    xcorr_lags: np.ndarray = np.empty(n, dtype=np.float64)
    # 创建一个长度为n的空数组，用于存储每个事件的互相关延迟
    # np.empty创建未初始化的数组，比np.zeros稍快
    
    for i in range(n):
        # 遍历每个事件对
        
        xcorr_lags[i] = refine_delay_xcorr(
            sig_b, sig_a, sr,
            onset_b=events_b[i].onset_s,
            onset_a=events_a[i].onset_s,
        )
        # 调用refine_delay_xcorr函数计算第i个事件的互相关延迟
        # 传入B端和A端的信号、采样率、以及两端第i个事件的起始时间
        # 结果存储在xcorr_lags[i]中

    # ── 第一个事件做同步锚点 ──
    anchor: float = xcorr_lags[0]
    # 第一个事件的互相关延迟作为同步锚点
    # 包含：PTP时钟偏差 + 传播时延
    
    delays_s: np.ndarray = xcorr_lags - anchor
    # 所有事件的延迟减去锚点
    # 结果是纯传播时延差（PTP偏差被消除）
    # 第一个事件的delay变成0（自己减自己）
    # 后续事件的delay是相对于第一个事件的传播时延差

    # ── 离群值剔除 ──
    # 互相关可能在某些事件上误匹配（波形相似但位置错误）。
    # 用中位数绝对偏差 (MAD) 检测离群值：偏差超过 3 倍 MAD 的事件标记为 NaN。
    median_delay: float = float(np.nanmedian(delays_s))
    # 计算延迟数组的中位数
    # np.nanmedian忽略NaN值，只计算有效值的中位数
    
    mad: float = float(np.nanmedian(np.abs(delays_s - median_delay)))
    # 计算中位数绝对偏差 (Median Absolute Deviation)
    # np.abs(delays_s - median_delay)：每个值与中位数的绝对差
    # np.nanmedian：取这些绝对差的中位数
    # MAD是衡量数据离散程度的稳健指标
    
    if mad > 0.0:
        # 如果MAD大于0（说明有数据变异）
        
        threshold: float = 3.0 * mad
        # 设置离群值阈值：3倍MAD
        # 超过这个阈值的值被认为是离群值
        
        outlier_mask: np.ndarray = np.abs(delays_s - median_delay) > threshold
        # 创建布尔掩码，标记哪些值是离群值
        # np.abs(delays_s - median_delay) > threshold：绝对差超过阈值的位置为True
        
        delays_s[outlier_mask] = float("nan")
        # 将离群值替换为NaN
        # 这些值在后续计算中会被忽略

    distances_m: np.ndarray = delays_s * speed_mps
    # 计算距离：延迟 × 声速
    # 例如：延迟0.001秒 × 340米/秒 = 0.34米
    
    return delays_s, distances_m
    # 返回延迟数组和距离数组


def main() -> None:
    """命令行入口。解析参数 → 检测事件 → 互相关精炼 → 输出距离。"""
    # 主函数：程序的入口点
    
    parser = argparse.ArgumentParser(
        description="A/B 双端声学测距 —— 互相关精炼 + 同步锚点",
        # 程序描述
        
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # 使用原始格式的帮助信息，保留换行和缩进
        
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
        # 帮助信息的结尾部分，显示介质选项和示例
    )
    # 创建命令行参数解析器
    
    parser.add_argument("wav_a", type=Path, help="A 端 WAV 文件（接收端）")
    # 添加位置参数：A端WAV文件路径
    # type=Path：自动转换为Path对象
    # help：帮助信息
    
    parser.add_argument("wav_b", type=Path, help="B 端 WAV 文件（声源端）")
    # 添加位置参数：B端WAV文件路径
    
    parser.add_argument("--json-a", type=Path, required=True, help="A 端元数据 JSON")
    # 添加可选参数：A端元数据JSON文件路径
    # required=True：必须提供
    
    parser.add_argument("--json-b", type=Path, required=True, help="B 端元数据 JSON")
    # 添加可选参数：B端元数据JSON文件路径
    # required=True：必须提供
    
    parser.add_argument("--medium", choices=list(SOUND_SPEEDS), default="air",
                        help="传播介质（影响声速）")
    # 添加可选参数：传播介质
    # choices：只能选择SOUND_SPEEDS字典中的键
    # default="air"：默认使用空气
    
    parser.add_argument("--speed-mps", type=float,
                        help="自定义声速 m/s（覆盖 --medium）")
    # 添加可选参数：自定义声速
    # type=float：自动转换为浮点数
    # 如果提供，会覆盖--medium参数
    
    args = parser.parse_args()
    # 解析命令行参数，返回一个命名空间对象

    speed_mps: float = args.speed_mps if args.speed_mps else SOUND_SPEEDS[args.medium]
    # 确定使用的声速
    # 如果提供了--speed-mps，使用自定义声速
    # 否则，根据--medium参数从SOUND_SPEEDS字典中获取声速

    meta_a: SessionMetadata = load_metadata(args.json_a)
    # 加载A端的元数据
    
    meta_b: SessionMetadata = load_metadata(args.json_b)
    # 加载B端的元数据
    
    if meta_a.node_id != "A" or meta_b.node_id != "B":
        raise SystemExit("JSON node_id 必须分别为 A 和 B")
    # 检查元数据中的node_id是否正确
    # A端的node_id必须是"A"，B端的node_id必须是"B"
    # 如果不正确，程序退出并显示错误信息

    sr_a: int
    # A端采样率
    
    sig_a: np.ndarray
    # A端音频信号
    
    sr_b: int
    # B端采样率
    
    sig_b: np.ndarray
    # B端音频信号
    
    sr_a, sig_a = load_signal(args.wav_a)
    # 加载A端的WAV文件，返回采样率和信号
    
    sr_b, sig_b = load_signal(args.wav_b)
    # 加载B端的WAV文件，返回采样率和信号
    
    if sr_a != sr_b:
        raise ValueError("A/B 两端采样率不一致。")
    # 检查两端的采样率是否一致
    # 如果不一致，程序报错
    
    if sr_a != meta_a.sample_rate_hz or sr_b != meta_b.sample_rate_hz:
        raise ValueError("WAV 与 JSON 采样率不一致。")
    # 检查WAV文件的采样率是否与JSON元数据中的采样率一致
    # 如果不一致，程序报错

    # ── 事件检测（来自 analyze_session_a.py）──
    results: list[AnalysisResult]
    # 分析结果列表
    
    template: EventTemplate
    # 事件模板
    
    results, template = analyze_confirmed_events([args.wav_a, args.wav_b])
    # 调用analyze_confirmed_events函数进行事件检测
    # 传入两个WAV文件路径
    # 返回分析结果列表和事件模板
    
    result_by_path: dict[Path, AnalysisResult] = {r.wav_path.resolve(): r for r in results}
    # 创建一个字典，键是文件路径（解析后的绝对路径），值是对应的分析结果
    # 便于后续根据文件路径查找对应的分析结果
    
    events_a: list[ReportEvent] = result_by_path[args.wav_a.resolve()].confirmed_events
    # 获取A端的已确认事件列表
    
    events_b: list[ReportEvent] = result_by_path[args.wav_b.resolve()].confirmed_events
    # 获取B端的已确认事件列表

    # ── 互相关精炼测距 ──
    delays_s: np.ndarray
    # 传播时延数组
    
    distances_m: np.ndarray
    # 距离数组
    
    delays_s, distances_m = compute_distances(
        sig_a, sig_b, sr_a, events_a, events_b, speed_mps,
    )
    # 调用compute_distances函数计算距离
    # 传入两端的信号、采样率、事件列表和声速
    # 返回延迟数组和距离数组

    abs_delays: np.ndarray = np.abs(delays_s)
    # 计算延迟的绝对值
    
    abs_dists: np.ndarray = np.abs(distances_m)
    # 计算距离的绝对值

    # ── 输出 ──
    # 使用中位数（绝对值）作为最终结果，对离群值最稳健
    final_distance = float(np.nanmedian(abs_dists))
    # 计算距离绝对值的中位数作为最终结果
    # 中位数对离群值更稳健，比均值更可靠
    
    confidence = "高" if np.nanstd(delays_s) < 0.005 else "中" if np.nanstd(delays_s) < 0.02 else "低"
    # 计算置信度
    # np.nanstd计算延迟的标准差（忽略NaN值）
    # 如果标准差 < 0.005秒，置信度为"高"
    # 如果标准差 < 0.02秒，置信度为"中"
    # 否则，置信度为"低"
    
    print("=" * 50)
    # 打印分隔线
    
    print(f"测距结果")
    # 打印标题
    
    print("=" * 50)
    # 打印分隔线
    
    print(f"距离: {final_distance:.2f} 米")
    # 打印最终距离，保留2位小数
    
    print(f"置信度: {confidence}")
    # 打印置信度
    
    print(f"介质: {args.medium} ({speed_mps:.0f} m/s)")
    # 打印介质类型和声速，声速保留0位小数
    
    print(f"有效事件: {len(delays_s)} 对")
    # 打印有效事件对数
    
    print("=" * 50)
    # 打印分隔线


if __name__ == "__main__":
    # 如果这个文件是直接运行的（而不是被导入的）
    
    main()
    # 调用主函数
