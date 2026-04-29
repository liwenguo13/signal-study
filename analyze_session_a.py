#!/usr/bin/env python3
"""
用途：
1. 对 1 个或 2 个 WAV 录音做“特殊瞬态事件”检测。
2. 这里的“特殊事件”指人耳能明确数出来的拍手、敲击、撞击这类短促事件。
3. 当传感器距离声源远、事件很弱时，会先在强侧学出“本次事件长什么样”，再去弱侧复核。

主流程：
1. `main()` 先做一轮宽松侦察，收集候选事件
2. `infer_auto_parameters()` 根据侦察结果自动估计本次环境下的参数
3. 再用自动参数重新分析每个 WAV
4. `build_template()` 从候选里学出“本次会话的事件模板”
5. `classify_candidates()` 做单文件直接确认
6. 如果有两个文件，再调 `recover_missing_events()` 做弱侧本地复核
7. 最后 `print_result()` 输出中文结果，可选写入 JSON

设计原则：
1. 不写死事件个数
2. 不因为另一端有事件，就强行给本端“补假数”
3. 本端必须出现自己的波形证据，才允许计入最终事件数
4. 默认走“冲击自动模式”，尽量让客户不需要改参数
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wav_signal import load_signal


@dataclass
class CandidateEvent:
    # 候选事件：只是“可能是一次拍手/敲击”，还没有进入最终结果。
    block_index: int
    onset_s: float
    peak_s: float
    score: float
    width_ms: float
    local_energy: float
    attack_ratio: float
    decay_ratio: float
    shape_vector: np.ndarray
    template_similarity: float = 0.0


@dataclass
class ReportEvent:
    # 输出事件：已经通过模板确认，或者通过双文件复核后确认。
    onset_s: float
    peak_s: float
    score: float
    width_ms: float
    local_energy: float
    template_similarity: float
    confirmation: str
    confidence: str


@dataclass
class EventTemplate:
    # 事件模板：不是固定声音文件，而是“本次录音里这一类事件的共同形状”。
    vector: np.ndarray
    confirm_similarity: float
    recover_similarity: float
    median_width_ms: float
    median_attack_ratio: float
    median_decay_ratio: float
    min_confirm_energy: float
    min_recover_energy: float


@dataclass
class AutoLearnedParameters:
    # 本次录音自动学习出来的内部工作参数。
    block_ms: float
    min_score_ratio: float
    candidate_gap_ms: float
    candidate_merge_gap_ms: float
    min_event_gap_ms: float
    echo_gap_ms: float
    recover_search_radius_ms: float
    impact_mode_ok: bool
    impact_mode_reason: str
    scout_candidate_count: int


@dataclass
class AnalysisResult:
    # 单个 WAV 文件的完整分析结果。
    wav_path: Path
    sample_rate: int
    signal: np.ndarray
    envelope: np.ndarray
    feature: np.ndarray
    block_ms: float
    duration_s: float
    candidates: list[CandidateEvent]
    confirmed_events: list[ReportEvent]
    recovered_events: int
    suspected_missing_s: list[float]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    # 简单滑动平均。用于把粗糙包络平滑一点，压低微小抖动。
    # window 越大：越平滑，但会损失时间分辨率。
    # window 越小：越灵敏，但更容易把噪声凸显出来。
    if window <= 1:
        # 当窗口长度 <= 1 时，平均就失去意义，所以直接返回原数组的浮点副本。
        return values.astype(np.float64, copy=True)
    # 这里构造一个“每个位置权重都一样”的平均核。
    # 例如 window=4 时，kernel 就是 [0.25, 0.25, 0.25, 0.25]。
    kernel = np.ones(window, dtype=np.float64) / window
    # 用 same 模式做卷积，表示输出长度与输入一致，便于后续逐点比较。
    return np.convolve(values, kernel, mode="same")


def block_max(signal: np.ndarray, block_size: int) -> np.ndarray:
    # 按 block_size 分块，每块取最大绝对幅值，得到“粗包络”。
    # block_size 越大：抗噪声更强，但多个靠得很近的小峰可能被揉在一起。
    # block_size 越小：时间定位更细，但更容易把回响/噪声也当成独立峰。
    # (-len(signal)) % block_size 的作用：
    # 1. 如果长度本来就能整除，结果就是 0
    # 2. 如果不能整除，结果就是“还差多少个样本才能补齐最后一块”
    pad = (-len(signal)) % block_size
    if pad:
        # 末尾补零是为了后面的 reshape 不报错。
        # 补零只发生在最后一块，不会影响前面真实波形的时间位置。
        signal = np.pad(signal, (0, pad), mode="constant")
    # reshape(-1, block_size) 表示按 block_size 切成很多行，每行是一小段时间块。
    # max(axis=1) 表示每一行取最大值，从而得到“每个时间块里的最大振幅”。
    return signal.reshape(-1, block_size).max(axis=1)


def build_feature(signal: np.ndarray, sample_rate: int, block_ms: float) -> tuple[np.ndarray, np.ndarray, int]:
    # 构建本脚本最核心的特征：
    # 1. envelope：分块后的粗包络
    # 2. feature：强调“突然变大”的新事件得分
    #
    # 这里的 feature 不是直接看振幅，而是看：
    # - 短窗口平均是否明显高于长窗口平均（novelty）
    # - 上升沿是否足够陡（rise）
    #
    # block_ms 是最关键参数之一：
    # - 调大：更适合“按人耳数拍手”，不容易把一次拍手后的反射拆成多次
    # - 调小：更容易把弱小事件拉出来，但也更容易多检
    #
    # 下面这个式子是“毫秒 -> 样本数”的标准换算：
    # sample_rate 的单位是 “样本/秒”
    # block_ms 的单位是 “毫秒”
    # 1 秒 = 1000 毫秒
    # 所以 sample_rate * block_ms / 1000.0 的单位就是 “样本”
    #
    # 例如：
    # sample_rate = 192000 Hz
    # block_ms = 10 ms
    # block_size = 192000 * 10 / 1000 = 1920 个样本
    block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
    # 用分块最大值构造粗包络，先把原始高速采样信号压缩到较慢时间尺度。
    envelope = block_max(signal.astype(np.float64), block_size)
    # 这里的窗口长度都是“块数”，不是毫秒。
    # 例如 block_ms=10 时，10.0 / block_ms = 1，表示短窗口大约覆盖 10ms。
    short_window = max(1, int(round(10.0 / block_ms)))
    # 长窗口故意取更长时间，用来表示背景/慢变化。
    # short_window + 1 是为了保证长窗口一定比短窗口长，否则 novelty 没意义。
    long_window = max(short_window + 1, int(round(120.0 / block_ms)))
    # smooth_short 看局部短时能量，smooth_long 看更慢的背景趋势。
    smooth_short = moving_average(envelope, short_window)
    smooth_long = moving_average(envelope, long_window)
    # novelty = max(短时平均 - 长时平均, 0)
    # 含义：只有当“当前局部明显高于背景”时，才认为这里有新事件抬头。
    # 用 max(..., 0) 是因为我们只关心“突然增强”，不关心“突然变弱”。
    novelty = np.maximum(smooth_short - smooth_long, 0.0)
    # diff 计算相邻块的增量；prepend=smooth_short[0] 用来保持长度不变。
    # 再用 max(..., 0) 保留正向上升，忽略下降。
    rise = np.maximum(np.diff(smooth_short, prepend=smooth_short[0]), 0.0)
    # 用 95 分位数而不是最大值，是为了避免个别超强峰把整体比例拉得太离谱。
    # +1.0 是保险项，避免 rise 全是 0 时除零。
    rise_scale = np.percentile(rise, 95) + 1.0
    # 最终特征 = novelty * (1 + rise/rise_scale)
    # 含义：
    # 1. 先要求“当前局部高于背景”
    # 2. 再奖励“上升沿更陡”的位置
    # 这样比单纯看振幅更适合抓短促冲击事件。
    feature = novelty * (1.0 + rise / rise_scale)
    return envelope, feature, block_size


def extract_window(values: np.ndarray, center_index: int, left: int, right: int) -> np.ndarray:
    # 以 center_index 为中心取一个局部窗口。
    # 这个函数主要被 `build_candidate()` 调用，用来截取候选事件附近的局部形状。
    # start/stop 是窗口理想边界，允许越界，后面再统一做补零。
    start = center_index - left
    stop = center_index + right + 1
    # 如果窗口左侧超出了数组开头，就要在左边补 pad_left 个 0。
    pad_left = max(0, -start)
    # 如果窗口右侧超出了数组末尾，就要在右边补 pad_right 个 0。
    pad_right = max(0, stop - len(values))
    # 先截出数组里实际存在的部分。
    clipped = values[max(0, start) : min(len(values), stop)].astype(np.float64, copy=False)
    if pad_left or pad_right:
        # 对靠近边界的窗口做零填充，保证所有 shape_vector 长度一致，后面才能比较相似度。
        clipped = np.pad(clipped, (pad_left, pad_right), mode="constant")
    return clipped


def find_onset(signal: np.ndarray, peak_index: int, search_radius: int) -> int:
    # 已知峰值点后，向前找“真正开始抬头”的起始点。
    # search_radius 越大：更有机会找到真实起点，但也可能被更早的干扰拖走。
    # search_radius 越小：定位更保守，可能把起点报晚。
    # 从峰值往前最多回看 search_radius 个样本。
    start = max(0, peak_index - search_radius)
    local = signal[start : peak_index + 1].astype(np.float64)
    if local.size <= 1:
        return peak_index

    # local 的一阶差分，表示“每一步涨了多少”。
    diff = np.maximum(np.diff(local, prepend=local[0]), 0.0)
    # 用 80 分位数而不是固定值，是为了自适应当前局部波形尺度。
    threshold = np.percentile(diff, 80)
    # 找到最早一个“涨幅已经达到局部较高水平”的点，把它当作起点。
    above = np.flatnonzero(diff >= threshold)
    if len(above) == 0:
        return peak_index
    return start + int(above[0])


def candidate_width_ms(feature: np.ndarray, index: int, block_ms: float) -> float:
    # 估算一个候选峰有多“宽”。
    # 很宽但不尖的波形，更像背景起伏/拖尾，不像一次明确拍手。
    peak = float(feature[index])
    # 这里把峰值的 25% 当成“峰的有效边界”。
    # 比例太高会让宽度偏窄；比例太低会把拖尾算得过长。
    threshold = peak * 0.25
    left = index
    while left > 0 and feature[left - 1] >= threshold:
        left -= 1
    right = index
    while right + 1 < len(feature) and feature[right + 1] >= threshold:
        right += 1
    # (right - left + 1) 是覆盖了多少个 block，再乘每个 block 的毫秒数得到最终宽度。
    return (right - left + 1) * block_ms


def candidate_local_energy(feature: np.ndarray, index: int) -> float:
    # 看候选峰附近一小段区域的总能量。
    # 它比单点峰值更稳，常用来区分“真实小事件”和“单点毛刺”。
    start = max(0, index - 2)
    stop = min(len(feature), index + 3)
    # 这里不是积分物理能量，只是把局部特征值加起来，作为“这一小片区域有多强”的近似指标。
    return float(np.sum(feature[start:stop]))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # 模板相似度。1.0 表示形状非常像，越低表示越不像。
    # 这里只比较“形状”，不是比较绝对音量，所以远端弱信号也有机会匹配到。
    # 余弦相似度公式：
    # dot(left, right) / (||left|| * ||right||)
    # 含义：只比较两个向量方向是否一致，不关心整体大小比例。
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        # 如果某个向量几乎全是 0，就没法定义方向，相似度直接按 0 处理。
        return 0.0
    return float(np.dot(left, right) / denom)


def candidate_is_plausible(candidate: CandidateEvent, *, duration_s: float, edge_margin_s: float) -> bool:
    # 候选级过滤：
    # 这里只做“明显不靠谱”的剔除，尽量不要在这里误杀真实事件。
    #
    # 过滤内容：
    # 1. 录音刚开始/刚结束的截断伪峰
    # 2. 很宽又很弱的慢变化
    # 3. 极弱的底噪尖刺
    if candidate.onset_s < edge_margin_s:
        return False
    if duration_s - candidate.peak_s < edge_margin_s:
        return False
    if candidate.width_ms >= 45.0 and candidate.local_energy < 700000.0:
        return False
    if candidate.score < 12000.0 and candidate.local_energy < 60000.0:
        return False
    return True


def build_candidate(
    signal: np.ndarray,
    sample_rate: int,
    envelope: np.ndarray,
    feature: np.ndarray,
    *,
    block_ms: float,
    block_index: int,
) -> CandidateEvent:
    # 把一个 block 索引展开成“候选事件对象”。
    # 调用链：
    # `collect_candidates()` -> `build_candidate()`
    #
    # 这里会补出候选的完整信息：
    # 1. onset_s / peak_s：事件起点和峰值时刻
    # 2. width_ms / local_energy：形状与强度
    # 3. shape_vector：后面给模板匹配用的局部形状向量
    #
    # 再次提醒：
    # sample_rate 单位是 “样本/秒”
    # block_ms 单位是 “毫秒”
    # 所以 sample_rate * block_ms / 1000.0 是 “每个块对应多少个样本”
    block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
    # block_index 只是第几个块；乘 block_size 才能换成原始信号里的样本位置。
    peak_center = block_index * block_size
    # 不是只看一个点，而是在当前块前后各看一个 block 范围，避免粗定位偏一点就错过真峰。
    peak_start = max(0, peak_center - block_size)
    peak_stop = min(len(signal), peak_center + block_size + 1)
    # np.argmax(...) 找到局部窗口内最大值的位置，再加回偏移量得到全局 peak_index。
    peak_index = peak_start + int(np.argmax(signal[peak_start:peak_stop]))
    # 15ms 是经验性的“峰前回看范围”。
    # 太短可能找不到真正起点，太长则可能被更早的杂波影响。
    onset_index = find_onset(signal, peak_index, max(1, int(round(sample_rate * 15.0 / 1000.0))))
    width_ms = candidate_width_ms(feature, block_index, block_ms)
    local_energy = candidate_local_energy(feature, block_index)

    # 以候选点为中心取左右固定长度窗口，构造局部形状。
    # left=4, right=8 表示更关注“峰前少、峰后稍长”的不对称冲击结构。
    env_window = extract_window(envelope, block_index, left=4, right=8)
    feat_window = extract_window(feature, block_index, left=4, right=8)
    # 归一化是为了“比较形状而不是比较绝对响度”。
    # +1.0 是为了避免 max 为 0 时除零。
    env_norm = env_window / (float(np.max(env_window)) + 1.0)
    feat_norm = feat_window / (float(np.max(feat_window)) + 1.0)
    # 把包络形状和特征形状拼接在一起，得到一个更完整的局部描述向量。
    shape_vector = np.concatenate([env_norm, feat_norm])

    # pre_energy：峰前局部总量
    # peak_energy：峰中心块的量
    # post_energy：峰后拖尾局部总量
    # 这些值后面用来描述“上升多陡、尾巴多长”。
    pre_energy = float(np.sum(env_window[:4])) + 1.0
    peak_energy = float(env_window[4]) + 1.0
    post_energy = float(np.sum(env_window[5:])) + 1.0

    return CandidateEvent(
        block_index=block_index,
        onset_s=onset_index / sample_rate,
        peak_s=peak_index / sample_rate,
        score=float(feature[block_index]),
        width_ms=width_ms,
        local_energy=local_energy,
        # attack_ratio 越大，表示“峰中心相对峰前更突然地冒出来”。
        attack_ratio=peak_energy / pre_energy,
        # decay_ratio 越大，表示峰后拖尾相对更长。
        decay_ratio=post_energy / peak_energy,
        shape_vector=shape_vector,
    )


def collect_candidate_indices(
    feature: np.ndarray,
    *,
    min_score_ratio: float,
    percentile_threshold: float,
    candidate_gap_ms: float,
    block_ms: float,
    max_candidates: int,
) -> list[int]:
    # 从 feature 里先找出“可能有事”的块索引。
    # 调用链：
    # `collect_candidates()` -> `collect_candidate_indices()`
    #
    # 参数影响：
    # 1. min_score_ratio
    #    - 调大：只保留更强的候选，漏检风险上升，误检下降
    #    - 调小：弱事件更容易进来，但噪声/回响也更容易进来
    # 2. percentile_threshold
    #    - 调大：阈值更严格，候选更少
    #    - 调小：候选更多，更依赖后续模板筛选
    # 3. candidate_gap_ms
    #    - 调大：近邻峰更早被合并，不容易把一次拍手拆成多次
    #    - 调小：容易多检，特别是在空气传播和有回响时
    # 4. max_candidates
    #    - 只是候选上限，不是最终事件数上限
    # 两个阈值取 max 的原因：
    # 1. percentile_threshold 防止“整体都很弱时”阈值太低
    # 2. min_score_ratio 防止“有极强峰时”阈值完全脱离主峰尺度
    threshold = max(
        float(np.percentile(feature, percentile_threshold)),
        float(np.max(feature)) * min_score_ratio,
    )
    local_maxima = [
        idx
        for idx in range(1, len(feature) - 1)
        if feature[idx] >= threshold
        and feature[idx] >= feature[idx - 1]
        and feature[idx] >= feature[idx + 1]
    ]
    if not local_maxima:
        return []

    # candidate_gap_ms 是毫秒，要先除以 block_ms，才能换成“隔多少个 block 才算新事件”。
    refractory_blocks = max(1, int(round(candidate_gap_ms / block_ms)))
    grouped: list[list[int]] = []
    for idx in local_maxima:
        if not grouped or idx - grouped[-1][-1] >= refractory_blocks:
            grouped.append([idx])
        else:
            grouped[-1].append(idx)

    chosen = [max(group, key=lambda index: feature[index]) for group in grouped]
    # 先按强度排序再截断，是为了优先保留更强的候选，防止 max_candidates 很小时随机丢掉主事件。
    chosen.sort(key=lambda index: feature[index], reverse=True)
    chosen = chosen[:max_candidates]
    # 最后再按时间排序，便于后面顺序处理。
    chosen.sort()
    return chosen


def merge_close_candidates(candidates: list[CandidateEvent], merge_gap_s: float) -> list[CandidateEvent]:
    # 把时间上过近的两个候选合并成一个。
    # 调用位置有两处：
    # 1. `collect_candidates()`：候选阶段的近邻合并
    # 2. `classify_candidates()`：最终确认前再做一次合并
    #
    # merge_gap_s 越大：越不容易多检，但过大时会把两次真实快速拍手并成一次。
    merged: list[CandidateEvent] = []
    for candidate in sorted(candidates, key=lambda item: item.onset_s):
        if not merged:
            merged.append(candidate)
            continue
        previous = merged[-1]
        if candidate.onset_s - previous.onset_s < merge_gap_s:
            # 如果两个候选太近，就只留一个。
            # 排序准则依次看：
            # 1. score：特征峰值强不强
            # 2. local_energy：周围整体是不是也强
            # 3. attack_ratio：是不是更像突然冲击
            prev_rank = (previous.score, previous.local_energy, previous.attack_ratio)
            curr_rank = (candidate.score, candidate.local_energy, candidate.attack_ratio)
            if curr_rank > prev_rank:
                merged[-1] = candidate
            continue
        merged.append(candidate)
    return merged


def collect_candidates(
    signal: np.ndarray,
    sample_rate: int,
    *,
    block_ms: float,
    min_score_ratio: float,
    edge_margin_ms: float,
    max_candidates: int,
    percentile_threshold: float = 96.0,
    candidate_gap_ms: float | None = None,
    candidate_merge_gap_ms: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, list[CandidateEvent]]:
    # 单文件候选提取总入口。
    # 调用链：
    # `analyze_file()` -> `collect_candidates()`
    # `collect_candidates()` -> `build_feature()` -> `collect_candidate_indices()` -> `build_candidate()`
    #
    # 这一步只负责“找可能的事件”，不负责做最终确认。
    envelope, feature, _ = build_feature(signal, sample_rate, block_ms)
    if candidate_gap_ms is None:
        # 如果没显式给候选分组间隔，就按 block_ms 自动给一个默认值。
        # block_ms 越大，允许的组内距离也应跟着大一点。
        candidate_gap_ms = max(80.0, block_ms * 8.0)

    indices = collect_candidate_indices(
        feature,
        min_score_ratio=min_score_ratio,
        percentile_threshold=percentile_threshold,
        candidate_gap_ms=candidate_gap_ms,
        block_ms=block_ms,
        max_candidates=max_candidates,
    )

    duration_s = len(signal) / sample_rate
    # edge_margin_ms 是毫秒，要除以 1000 才能与 onset_s / peak_s 这种“秒”单位比较。
    edge_margin_s = edge_margin_ms / 1000.0
    candidates: list[CandidateEvent] = []
    for block_index in indices:
        candidate = build_candidate(
            signal,
            sample_rate,
            envelope,
            feature,
            block_ms=block_ms,
            block_index=block_index,
        )
        if candidate_is_plausible(candidate, duration_s=duration_s, edge_margin_s=edge_margin_s):
            candidates.append(candidate)

    candidates = merge_close_candidates(candidates, merge_gap_s=candidate_merge_gap_ms)
    return envelope, feature, candidates


def safe_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    # 对空数组取分位数会报错，所以这里统一包一层“空则返回 fallback”。
    if values.size == 0:
        return fallback
    return float(np.percentile(values, percentile))


def infer_auto_parameters(
    scout_results: list[AnalysisResult],
    *,
    requested_block_ms: float,
    requested_min_score_ratio: float,
    requested_min_event_gap_ms: float,
) -> AutoLearnedParameters:
    # 自动学习阶段：
    # 1. 先看一眼所有文件的候选分布
    # 2. 再自动反推出“本次环境更合适的内部参数”
    #
    # 这一步的目标不是完美分类，而是把后续正式检测的尺度调到更接近当前介质。
    pool = [candidate for result in scout_results for candidate in result.candidates]
    if not pool:
        return AutoLearnedParameters(
            block_ms=requested_block_ms,
            min_score_ratio=requested_min_score_ratio,
            candidate_gap_ms=max(80.0, requested_block_ms * 8.0),
            candidate_merge_gap_ms=0.08,
            min_event_gap_ms=requested_min_event_gap_ms,
            echo_gap_ms=180.0,
            recover_search_radius_ms=180.0,
            impact_mode_ok=False,
            impact_mode_reason="候选事件太少，无法建立冲击模板",
            scout_candidate_count=0,
        )

    widths = np.array([candidate.width_ms for candidate in pool], dtype=np.float64)
    scores = np.array([candidate.score for candidate in pool], dtype=np.float64)
    attacks = np.array([candidate.attack_ratio for candidate in pool], dtype=np.float64)
    energies = np.array([candidate.local_energy for candidate in pool], dtype=np.float64)

    interval_pool: list[float] = []
    for result in scout_results:
        if len(result.candidates) > 1:
            times = np.array([candidate.onset_s for candidate in result.candidates], dtype=np.float64)
            # np.diff(times) 计算相邻候选之间的时间间隔，后面用来估计“真实事件通常隔多久”。
            interval_pool.extend(np.diff(times).tolist())
    intervals = np.array(interval_pool, dtype=np.float64)

    # 用中位数而不是平均数，是为了降低极端宽峰/窄峰的影响。
    width_median = safe_percentile(widths, 50, requested_block_ms)
    width_p90 = safe_percentile(widths, 90, width_median)
    # block_ms 自动值限制在 [6, 15]ms：
    # - 太小会过分敏感
    # - 太大又会抹平短促冲击
    learned_block_ms = float(np.clip(width_median, 6.0, 15.0))

    max_score = float(np.max(scores)) if scores.size else 1.0
    score_p15 = safe_percentile(scores, 15, max_score * requested_min_score_ratio)
    # score_p15 / max_score 得到“较弱但仍像真实候选的相对强度”。
    # 再乘 0.60，表示正式阶段阈值比这再稍微放宽一点，避免把弱端真实事件全砍掉。
    learned_min_score_ratio = float(
        np.clip(min(requested_min_score_ratio, score_p15 / max(max_score, 1.0) * 0.60), 0.0005, 0.01)
    )

    # intervals 是“秒”，这里乘 1000.0 转成“毫秒”，因为下面所有自动间隔参数都用毫秒表达。
    interval_p10_ms = safe_percentile(intervals * 1000.0, 10, requested_min_event_gap_ms)
    interval_median_ms = safe_percentile(intervals * 1000.0, 50, requested_min_event_gap_ms)
    # 这里一组系数本质上是在表达经验关系：
    # 1. 候选分组间隔要明显小于真实事件间隔
    # 2. 最终最小事件间隔要比候选分组更大，用来避免多检
    # 3. 回响抑制间隔通常比最终事件间隔略小，用来消掉一次敲击后的近邻反射
    candidate_gap_ms = float(np.clip(min(interval_p10_ms * 0.18, interval_median_ms * 0.15), 70.0, 160.0))
    # candidate_merge_gap_ms 这里换成“秒”，因为 merge_close_candidates 里是按秒比较的。
    candidate_merge_gap_ms = float(np.clip(candidate_gap_ms / 1000.0 * 1.10, 0.07, 0.14))
    min_event_gap_ms = float(np.clip(min(interval_p10_ms * 0.45, interval_median_ms * 0.50), 180.0, 420.0))
    echo_gap_ms = float(np.clip(min(interval_p10_ms * 0.34, interval_median_ms * 0.30), 100.0, 240.0))
    # 复核搜索半径不宜太小，否则另一端稍有时差就找不到；
    # 也不宜太大，否则会在太大范围里捞噪声，所以做上下限裁剪。
    recover_search_radius_ms = float(np.clip(max(140.0, min_event_gap_ms * 0.55), 140.0, 260.0))

    impact_mode_ok = True
    reason = "已自动学习冲击模式参数"
    if len(pool) < 4:
        impact_mode_ok = False
        reason = "候选事件数量不足，当前录音不适合冲击自动模式"
    elif width_median > 35.0 or width_p90 > 55.0:
        impact_mode_ok = False
        reason = "候选事件过宽，当前录音更像连续扰动而不是短促冲击"
    elif safe_percentile(attacks, 50, 0.0) < 0.08 and safe_percentile(energies, 90, 0.0) < 250000.0:
        impact_mode_ok = False
        reason = "候选上升沿不明显，当前录音不适合冲击自动模式"

    return AutoLearnedParameters(
        block_ms=learned_block_ms,
        min_score_ratio=learned_min_score_ratio,
        candidate_gap_ms=candidate_gap_ms,
        candidate_merge_gap_ms=candidate_merge_gap_ms,
        min_event_gap_ms=min_event_gap_ms,
        echo_gap_ms=echo_gap_ms,
        recover_search_radius_ms=recover_search_radius_ms,
        impact_mode_ok=impact_mode_ok,
        impact_mode_reason=reason,
        scout_candidate_count=len(pool),
    )


def build_template(results: list[AnalysisResult]) -> EventTemplate:
    # 从本次会话的所有候选里学出“这次特殊事件的典型形状”。
    # 调用链：
    # `main()` -> `build_template()`
    #
    # 重要说明：
    # 1. 不是加载固定模板文件
    # 2. 不是只识别一种永远不变的声音
    # 3. 而是对“这一次录音里反复出现的同类事件”做自适应建模
    #
    # 例如：
    # - 这一轮你拍手，它就学拍手
    # - 下一轮你敲桌子，它就学敲桌子
    #
    # 这里还会自动导出几个阈值：
    # - confirm_similarity：直接确认阈值
    # - recover_similarity：双文件复核时使用的较低阈值
    # - min_confirm_energy / min_recover_energy：对应的能量门槛
    pool = [candidate for result in results for candidate in result.candidates]
    if not pool:
        zero = np.zeros(26, dtype=np.float64)
        return EventTemplate(
            vector=zero,
            confirm_similarity=1.0,
            recover_similarity=1.0,
            median_width_ms=0.0,
            median_attack_ratio=0.0,
            median_decay_ratio=0.0,
            min_confirm_energy=float("inf"),
            min_recover_energy=float("inf"),
        )

    energies = np.array([candidate.local_energy for candidate in pool], dtype=np.float64)
    # 35 分位数表示“排除最弱那一截候选”，避免模板被明显弱噪声污染。
    energy_floor = safe_percentile(energies, 35, float(np.min(energies)))
    eligible = [
        candidate
        for candidate in pool
        if candidate.width_ms <= 25.0 and candidate.local_energy >= energy_floor
    ]
    if len(eligible) < 3:
        eligible = sorted(pool, key=lambda item: item.score, reverse=True)[: min(24, len(pool))]
    else:
        eligible = sorted(eligible, key=lambda item: item.score, reverse=True)[: min(24, len(eligible))]

    matrix = np.stack([candidate.shape_vector for candidate in eligible])
    # 每个 shape_vector 的长度可能一样，但模长大小不同，所以先各自单位化。
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    normalized = matrix / norms
    # similarity[i, j] 就是第 i 个候选和第 j 个候选之间的余弦相似度。
    similarity = normalized @ normalized.T

    # 先用较严格阈值聚类；如果太严格导致没有成团，再自动放宽一点。
    cluster_threshold = 0.82
    cluster_masks = similarity >= cluster_threshold
    if int(np.max(np.sum(cluster_masks, axis=1))) < 3:
        cluster_threshold = 0.74
        cluster_masks = similarity >= cluster_threshold

    best_index = 0
    best_size = -1
    best_weight = -1.0
    for index, mask in enumerate(cluster_masks):
        size = int(np.sum(mask))
        # weight 不是计数，而是把簇里成员分数加总。
        # 这样在“簇大小相同”时，优先选整体更强、更稳定的那一簇。
        weight = float(np.sum([eligible[item].score for item in np.flatnonzero(mask)]))
        if size > best_size or (size == best_size and weight > best_weight):
            best_index = index
            best_size = size
            best_weight = weight

    cluster = [eligible[index] for index in np.flatnonzero(cluster_masks[best_index])]
    # 用 sqrt(score) 做权重，而不是直接用 score：
    # 1. 强事件确实更可靠，应当权重大一些
    # 2. 但如果直接用 score，超强峰会一票否决其它成员
    # 3. 开平方是个折中，让强事件更重要，但不会压得太狠
    weights = np.sqrt(np.array([candidate.score for candidate in cluster], dtype=np.float64))
    template_vector = np.average(
        np.stack([candidate.shape_vector for candidate in cluster]),
        axis=0,
        weights=weights,
    )
    template_norm = np.linalg.norm(template_vector)
    if template_norm > 0.0:
        # 模板最后再归一化一次，保证后面算余弦相似度时尺度一致。
        template_vector = template_vector / template_norm

    cluster_similarities = np.array(
        [cosine_similarity(candidate.shape_vector, template_vector) for candidate in cluster],
        dtype=np.float64,
    )
    # 20 分位数代表“模板簇里偏弱那一截但仍属于同类”的相似度。
    # 再减 0.05，表示正式确认阈值稍微留一点余地。
    confirm_similarity = float(np.clip(np.percentile(cluster_similarities, 20) - 0.05, 0.70, 0.90))
    # 复核阈值比直接确认阈值更低，因为弱端证据天然更弱。
    recover_similarity = float(np.clip(confirm_similarity - 0.10, 0.60, 0.84))

    cluster_energies = np.array([candidate.local_energy for candidate in cluster], dtype=np.float64)
    # 10 分位数表示“模板簇里比较弱但仍可信的那一截能量”。
    # 再乘 0.35，是为了给远端/弱端留出生存空间。
    # 最后 clip 到 [120000, 420000]，防止阈值被极端样本拉得太离谱。
    min_confirm_energy = float(
        np.clip(np.percentile(cluster_energies, 10) * 0.35, 120000.0, 420000.0)
    )
    # 复核能量门槛再低一些，因为复核阶段本来就是给弱端事件机会。
    min_recover_energy = float(
        np.clip(min_confirm_energy * 0.55, 70000.0, 260000.0)
    )

    return EventTemplate(
        vector=template_vector,
        confirm_similarity=confirm_similarity,
        recover_similarity=recover_similarity,
        median_width_ms=float(np.median([candidate.width_ms for candidate in cluster])),
        median_attack_ratio=float(np.median([candidate.attack_ratio for candidate in cluster])),
        median_decay_ratio=float(np.median([candidate.decay_ratio for candidate in cluster])),
        min_confirm_energy=min_confirm_energy,
        min_recover_energy=min_recover_energy,
    )


def confidence_label(similarity: float, template: EventTemplate) -> str:
    # 这里只是给人看结果时的“相对置信等级”。
    # 注意：
    # - `低` 不代表一定是假
    # - 只是说它离直接确认阈值比较近，证据没那么强
    if similarity >= template.confirm_similarity + 0.12:
        return "高"
    if similarity >= template.confirm_similarity + 0.05:
        return "中"
    return "低"


def report_from_candidate(
    candidate: CandidateEvent,
    *,
    template: EventTemplate,
    confirmation: str,
) -> ReportEvent:
    # 把内部候选对象转换成最终输出对象。
    # confirmation 用来区分：
    # - 直接确认：本文件自己证据就足够
    # - 复核确认：本文件本地证据偏弱，但在双文件节奏对齐后复核通过
    return ReportEvent(
        onset_s=candidate.onset_s,
        peak_s=candidate.peak_s,
        score=candidate.score,
        width_ms=candidate.width_ms,
        local_energy=candidate.local_energy,
        template_similarity=candidate.template_similarity,
        confirmation=confirmation,
        confidence=confidence_label(candidate.template_similarity, template),
    )


def suppress_echo_clusters(candidates: list[CandidateEvent], echo_gap_s: float) -> list[CandidateEvent]:
    # 回响抑制：
    # 一次真实敲击之后，常会在很近的时间里出现 1 到多次较弱反射。
    # 这里先按“近邻团簇”分组，再保留每簇里最像主事件的那个。
    if not candidates:
        return candidates

    groups: list[list[CandidateEvent]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        # 如果两个候选间隔不超过 echo_gap_s，就假设它们更可能属于同一次冲击后的回响团簇。
        if candidate.onset_s - groups[-1][-1].onset_s <= echo_gap_s:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    reduced: list[CandidateEvent] = []
    for group in groups:
        if len(group) == 1:
            reduced.append(group[0])
            continue
        best = max(
            group,
            key=lambda item: (
                # 先看“像模板且局部也强不强”，这是最核心的排序依据。
                item.template_similarity * item.local_energy,
                # 再看局部能量本身。
                item.local_energy,
                # 再看峰值特征分数。
                item.score,
                # 最后更偏好宽度小一点的，因为更像短促主冲击而不是拖尾。
                -item.width_ms,
            ),
        )
        reduced.append(best)
    reduced.sort(key=lambda item: item.onset_s)
    return reduced


def classify_candidates(
    result: AnalysisResult,
    template: EventTemplate,
    *,
    min_event_gap_ms: float,
    echo_gap_ms: float,
) -> None:
    # 单文件的“直接确认”阶段。
    # 调用链：
    # `main()` -> `classify_candidates()`
    #
    # 通过条件：
    # 1. 模板相似度够高
    # 2. 局部能量达到直接确认门槛
    # 3. 宽度不能明显偏离模板
    #
    # 参数影响：
    # min_event_gap_ms
    # - 调大：默认更贴近“人耳数事件”，多检更少
    # - 调小：适合极快节奏敲击，但空气传播时更容易把回响拆开
    confirmed_candidates: list[CandidateEvent] = []
    # 宽度上限 = max(40ms, 模板中位宽度 * 2.5)
    # 含义：模板允许有波动，但不能宽得离谱。
    width_limit = max(40.0, template.median_width_ms * 2.5)
    # min_event_gap_ms 是毫秒，这里除以 1000 换成秒。
    # 再乘 0.70，表示最终合并间隔通常设成“事件最小间隔”的七成左右。
    merge_gap_s = max(0.16, min_event_gap_ms / 1000.0 * 0.70)
    # echo_gap_ms 同样先从毫秒换成秒，因为时间戳都是按秒存的。
    echo_gap_s = echo_gap_ms / 1000.0

    for candidate in result.candidates:
        # 每个候选都要先和模板比较一次，得到它到底“像不像这次主事件”。
        candidate.template_similarity = cosine_similarity(candidate.shape_vector, template.vector)
        if candidate.template_similarity < template.confirm_similarity:
            continue
        if candidate.local_energy < template.min_confirm_energy:
            continue
        if candidate.width_ms > width_limit:
            continue
        confirmed_candidates.append(candidate)

    confirmed_candidates.sort(key=lambda item: item.onset_s)
    confirmed_candidates = suppress_echo_clusters(confirmed_candidates, echo_gap_s=echo_gap_s)
    confirmed_candidates = merge_close_candidates(confirmed_candidates, merge_gap_s=merge_gap_s)
    result.confirmed_events = [
        report_from_candidate(candidate, template=template, confirmation="直接确认")
        for candidate in confirmed_candidates
    ]
    result.recovered_events = 0
    result.suspected_missing_s = []


def estimate_offset(reference_events: list[ReportEvent], target_events: list[ReportEvent]) -> float | None:
    # 估计两个文件之间的大致时间偏移。
    # 这里只是为了“双文件复核时知道去哪里找”，不是做距离估计。
    if not reference_events or not target_events:
        return None

    reference_times = np.array([event.onset_s for event in reference_events], dtype=np.float64)
    target_times = np.array([event.onset_s for event in target_events], dtype=np.float64)
    # pairwise[i, j] 表示 target 第 i 个事件比 reference 第 j 个事件晚了多少秒。
    pairwise = target_times[:, None] - reference_times[None, :]
    # /0.01 再 *0.01 相当于把偏移量量化到 10ms 网格。
    # 这样做的目的是：
    # 1. 允许小量抖动被视为同一个 offset
    # 2. 用投票法找出最常见的整体时间差
    rounded = np.round(pairwise / 0.01) * 0.01
    values, counts = np.unique(rounded, return_counts=True)
    if len(values) == 0:
        return None
    return float(values[np.argmax(counts)])


def match_events_by_offset(
    reference_events: list[ReportEvent],
    target_events: list[ReportEvent],
    *,
    offset_s: float,
    tolerance_s: float,
) -> tuple[set[int], set[int]]:
    # 基于估计出的 offset，把两边已确认事件先做一轮粗配对。
    # 哪些 reference 事件没有配上 target，后面就会进入“弱侧补找/复核”流程。
    matched_reference: set[int] = set()
    matched_target: set[int] = set()
    target_used = [False] * len(target_events)

    for reference_index, reference_event in enumerate(reference_events):
        # predicted = “如果两边这个事件是同一个，那么 target 这边应该出现在什么时刻”
        predicted = reference_event.onset_s + offset_s
        best_target_index = None
        best_delta = None
        for target_index, target_event in enumerate(target_events):
            if target_used[target_index]:
                continue
            delta = abs(target_event.onset_s - predicted)
            if delta > tolerance_s:
                continue
            if best_delta is None or delta < best_delta:
                best_target_index = target_index
                best_delta = delta
        if best_target_index is None:
            continue
        target_used[best_target_index] = True
        matched_reference.add(reference_index)
        matched_target.add(best_target_index)

    return matched_reference, matched_target


def search_candidate_near_time(
    result: AnalysisResult,
    template: EventTemplate,
    *,
    center_time_s: float,
    radius_ms: float,
    edge_margin_ms: float,
) -> CandidateEvent | None:
    # 已知另一边某个事件的大致时刻后，在本文件附近小范围搜索一个最像模板的弱候选。
    # 调用链：
    # `recover_missing_events()` -> `search_candidate_near_time()`
    #
    # radius_ms 越大：
    # - 更不怕起录时差和粗略对齐误差
    # - 但也更容易搜到无关噪声
    # center_time_s 是秒，要先换回“第几个 block”，才能在 feature 上做局部搜索。
    center_index = int(round(center_time_s / result.block_ms * 1000.0))
    # radius_ms / result.block_ms 把搜索半径从“毫秒”换成“多少个 block”。
    radius_blocks = max(1, int(round(radius_ms / result.block_ms)))
    start = max(1, center_index - radius_blocks)
    stop = min(len(result.feature) - 1, center_index + radius_blocks + 1)
    if stop <= start:
        return None

    local_maxima = [
        index
        for index in range(start, stop)
        if result.feature[index] >= result.feature[index - 1]
        and result.feature[index] >= result.feature[index + 1]
    ]
    if not local_maxima:
        return None

    ranked_indices = sorted(local_maxima, key=lambda index: result.feature[index], reverse=True)[:8]
    best_candidate = None
    best_rank = None
    edge_margin_s = edge_margin_ms / 1000.0
    for block_index in ranked_indices:
        candidate = build_candidate(
            result.signal,
            result.sample_rate,
            result.envelope,
            result.feature,
            block_ms=result.block_ms,
            block_index=block_index,
        )
        if not candidate_is_plausible(candidate, duration_s=result.duration_s, edge_margin_s=edge_margin_s):
            continue
        candidate.template_similarity = cosine_similarity(candidate.shape_vector, template.vector)
        # distance_penalty 的作用是：同样像模板时，优先选离预测时刻更近的那个候选。
        #
        # 分子 abs(candidate.onset_s - center_time_s)：
        # - 候选真实起点与“预测起点”之间差了多少秒
        #
        # 分母 radius_ms / 1000.0：
        # - 把搜索半径从毫秒换成秒，作为归一化尺度
        #
        # 这样 distance_penalty 就变成一个“相对偏离比例”：
        # - 值接近 0：非常靠近预测位置
        # - 值接近 1：已经偏到搜索半径边缘
        #
        # max(..., 1e-6) 是防止半径极小时分母变成 0。
        distance_penalty = abs(candidate.onset_s - center_time_s) / max(radius_ms / 1000.0, 1e-6)
        # rank = 模板相似度 - 距离惩罚 * 0.20
        # 含义：模板相似度仍是主导，距离只作为轻度扣分项，而不是一票否决。
        # 0.20 是经验权重：够用来区分“近的”和“远的”，但不会压过相似度本身。
        rank = candidate.template_similarity - 0.20 * distance_penalty
        if best_rank is None or rank > best_rank:
            best_candidate = candidate
            best_rank = rank
    return best_candidate


def append_missing_time(target: AnalysisResult, predicted_time: float) -> None:
    # 这里做去重，避免同一个疑似漏检位置被重复写入很多次。
    # 0.05 秒是一个经验容差：50ms 内视为同一个缺失点。
    if not any(abs(predicted_time - current) <= 0.05 for current in target.suspected_missing_s):
        target.suspected_missing_s.append(predicted_time)


def recover_missing_events(
    reference: AnalysisResult,
    target: AnalysisResult,
    template: EventTemplate,
    *,
    min_event_gap_ms: float,
    edge_margin_ms: float,
    recover_search_radius_ms: float,
) -> None:
    # 双文件复核阶段。
    # 调用链：
    # `main()` -> `recover_missing_events()`
    #
    # 核心原则：
    # 1. 只有 reference 端比 target 端确认事件更多时，才尝试在 target 端补找
    # 2. 另一端只提供“去哪里找”的线索
    # 3. target 端必须自己出现本地证据，才允许补成“复核确认”
    # 4. 如果找不到足够证据，就记到 `suspected_missing_s`，不会硬加到最终结果
    if len(reference.confirmed_events) <= len(target.confirmed_events):
        return

    offset_s = estimate_offset(reference.confirmed_events, target.confirmed_events)
    if offset_s is None:
        return

    tolerance_s = max(0.12, min_event_gap_ms / 1000.0 * 0.75)
    # tolerance_s 越大，两边更容易配上；但过大时会把不该配的事件也配进去。
    matched_reference, _ = match_events_by_offset(
        reference.confirmed_events,
        target.confirmed_events,
        offset_s=offset_s,
        tolerance_s=tolerance_s,
    )

    existing_times = [event.onset_s for event in target.confirmed_events]
    # gap_s 是复核补找时避免“补到已有事件身上”的保护半径。
    gap_s = max(0.08, min_event_gap_ms / 1000.0 * 0.50)

    for reference_index, reference_event in enumerate(reference.confirmed_events):
        if reference_index in matched_reference:
            continue

        predicted_time = reference_event.onset_s + offset_s
        if any(abs(predicted_time - current) <= gap_s for current in existing_times):
            continue

        candidate = search_candidate_near_time(
            target,
            template,
            center_time_s=predicted_time,
            radius_ms=recover_search_radius_ms,
            edge_margin_ms=edge_margin_ms,
        )
        if candidate is None:
            append_missing_time(target, predicted_time)
            continue

        if candidate.template_similarity < template.recover_similarity:
            append_missing_time(target, predicted_time)
            continue
        if candidate.local_energy < template.min_recover_energy:
            append_missing_time(target, predicted_time)
            continue
        if candidate.width_ms > max(45.0, template.median_width_ms * 3.0):
            append_missing_time(target, predicted_time)
            continue
        if any(abs(candidate.onset_s - current) <= gap_s for current in existing_times):
            continue

        target.confirmed_events.append(
            report_from_candidate(candidate, template=template, confirmation="复核确认")
        )
        target.recovered_events += 1
        existing_times.append(candidate.onset_s)

    target.confirmed_events.sort(key=lambda event: event.onset_s)
    target.suspected_missing_s.sort()


def result_rank_key(result: AnalysisResult) -> tuple[int, float, float]:
    # 给两个文件排“谁更像强参考端”的简单排序分数。
    # 一般会让确认数更多、整体相似度更高、能量更强的一边先作为 reference。
    # recovered_bonus 是给“已经通过复核找回的结果”一点点加分，但不会喧宾夺主。
    recovered_bonus = result.recovered_events * 0.2
    similarity_sum = float(np.sum([event.template_similarity for event in result.confirmed_events]))
    energy_sum = float(np.sum([event.local_energy for event in result.confirmed_events]))
    return (len(result.confirmed_events), similarity_sum + recovered_bonus, energy_sum)


def analyze_file(
    wav_path: Path,
    *,
    block_ms: float,
    min_score_ratio: float,
    edge_margin_ms: float,
    max_candidates: int,
    percentile_threshold: float = 96.0,
    candidate_gap_ms: float | None = None,
    candidate_merge_gap_ms: float = 0.08,
) -> AnalysisResult:
    # 单个 WAV 文件分析入口。
    # 调用链：
    # `main()` -> `analyze_file()` -> `collect_candidates()`
    sample_rate, signal = load_signal(wav_path)
    envelope, feature, candidates = collect_candidates(
        signal,
        sample_rate,
        block_ms=block_ms,
        min_score_ratio=min_score_ratio,
        edge_margin_ms=edge_margin_ms,
        max_candidates=max_candidates,
        percentile_threshold=percentile_threshold,
        candidate_gap_ms=candidate_gap_ms,
        candidate_merge_gap_ms=candidate_merge_gap_ms,
    )
    return AnalysisResult(
        wav_path=wav_path,
        sample_rate=sample_rate,
        signal=signal,
        envelope=envelope,
        feature=feature,
        block_ms=block_ms,
        duration_s=len(signal) / sample_rate,
        candidates=candidates,
        confirmed_events=[],
        recovered_events=0,
        suspected_missing_s=[],
    )


def event_to_payload(event: ReportEvent) -> dict[str, object]:
    # 把输出事件转成 JSON 可写入的普通字典。
    return {
        "onset_s": event.onset_s,
        "peak_s": event.peak_s,
        "score": event.score,
        "width_ms": event.width_ms,
        "local_energy": event.local_energy,
        "template_similarity": event.template_similarity,
        "confirmation": event.confirmation,
        "confidence": event.confidence,
    }


def compute_next_onset_metrics(events: list[ReportEvent], index: int) -> tuple[float | None, float | None]:
    # 计算当前事件相对“下一个事件”的两个展示指标。
    #
    # 第 1 个指标：interval_s
    # - 公式：下一个起始时刻 - 当前起始时刻
    # - 单位：秒
    # - 用途：这才是最直接的“两个事件间隔了多少秒”
    #
    # 第 2 个指标：onset_ratio_percent
    # - 公式：当前起始时刻 / 下一个起始时刻 * 100
    # - 单位：百分比
    # - 这就是你说的“拿上一个除以下一个得出的数据”
    #
    # 注意：
    # - 百分比反映的是“两个绝对时间戳之间的比例”
    # - 它不是标准节奏间隔公式，所以只适合作为辅助观察
    # - 真正判断相邻事件间隔，核心仍然看 interval_s
    if index + 1 >= len(events):
        # 最后一个事件后面没有“下一个事件”，所以两个值都返回空。
        return None, None

    current_onset_s = events[index].onset_s
    next_onset_s = events[index + 1].onset_s
    interval_s = next_onset_s - current_onset_s

    # 理论上 next_onset_s 不会是 0，但这里仍做保护，避免极端情况下除零。
    if abs(next_onset_s) <= 1e-12:
        onset_ratio_percent = None
    else:
        onset_ratio_percent = current_onset_s / next_onset_s * 100.0
    return interval_s, onset_ratio_percent


def result_to_payload(result: AnalysisResult) -> dict[str, object]:
    # 把单文件分析结果转成 JSON 可写入的普通字典。
    events_payload = []
    for index, event in enumerate(result.confirmed_events):
        interval_s, onset_ratio_percent = compute_next_onset_metrics(result.confirmed_events, index)
        payload = event_to_payload(event)
        payload["next_interval_s"] = interval_s
        payload["next_onset_ratio_percent"] = onset_ratio_percent
        events_payload.append(payload)
    return {
        "wav_path": str(result.wav_path),
        "duration_s": result.duration_s,
        "candidate_count": len(result.candidates),
        "confirmed_count": len(result.confirmed_events),
        "recovered_count": result.recovered_events,
        "suspected_missing_s": result.suspected_missing_s,
        "events": events_payload,
    }


def print_result(result: AnalysisResult) -> None:
    # 中文终端输出。
    # 这是人直接看的版本，重点显示：
    # 1. 候选事件数
    # 2. 已确认特殊事件数
    # 3. 其中有多少是复核补找成功
    # 4. 还有多少疑似漏检
    print(f"文件：{result.wav_path}")
    print(f"录音时长={result.duration_s:.3f}秒")
    print(f"候选事件数={len(result.candidates)}")
    print(f"已确认特殊事件数={len(result.confirmed_events)}")
    print(f"其中复核补找成功={result.recovered_events}")
    print(f"疑似漏检数={len(result.suspected_missing_s)}")
    for zero_based_index, event in enumerate(result.confirmed_events):
        interval_s, onset_ratio_percent = compute_next_onset_metrics(result.confirmed_events, zero_based_index)
        index = zero_based_index + 1
        if interval_s is None or onset_ratio_percent is None:
            next_info = "到下一个起始间隔=无 起始时刻除以下一个=无"
        else:
            next_info = (
                f"到下一个起始间隔={interval_s:.6f}秒 "
                f"起始时刻除以下一个={onset_ratio_percent:.2f}%"
            )
        print(
            f"{index:02d} 判定={event.confirmation} "
            f"起始时刻={event.onset_s:.6f}秒 "
            f"{next_info} "
            f"峰值时刻={event.peak_s:.6f}秒 "
            f"模板相似度={event.template_similarity:.3f} "
            f"分数={event.score:.3f} "
            f"宽度={event.width_ms:.1f}毫秒 "
            f"局部能量={event.local_energy:.1f} "
            f"置信度={event.confidence}"
        )
    if result.suspected_missing_s:
        formatted = "、".join(f"{time_s:.6f}秒" for time_s in result.suspected_missing_s)
        print(f"疑似漏检位置：{formatted}")


def print_auto_summary(auto_params: AutoLearnedParameters, template: EventTemplate) -> None:
    # 这个输出是给人看的“自动学习摘要”。
    # 目的是让你知道：程序这次到底自动学成了什么参数，而不是黑箱运行。
    print(
        f"当前模式=冲击自动模式 判断={'适合' if auto_params.impact_mode_ok else '不适合'} "
        f"原因={auto_params.impact_mode_reason}"
    )
    print(
        "自动学习参数："
        f"block_ms={auto_params.block_ms:.1f} "
        f"候选分组间隔_ms={auto_params.candidate_gap_ms:.1f} "
        f"最终最小事件间隔_ms={auto_params.min_event_gap_ms:.1f} "
        f"回响抑制间隔_ms={auto_params.echo_gap_ms:.1f} "
        f"复核搜索半径_ms={auto_params.recover_search_radius_ms:.1f}"
    )
    print(
        "模板确认参数："
        f"直接确认阈值={template.confirm_similarity:.3f} "
        f"复核阈值={template.recover_similarity:.3f} "
        f"直接确认最小局部能量={template.min_confirm_energy:.1f}"
    )


def main() -> None:
    # 总入口。
    # 推荐阅读顺序：
    # 1. 先看这里，理解主流程和参数
    # 2. 再看 `analyze_file()` / `collect_candidates()`
    # 3. 再看 `infer_auto_parameters()` / `build_template()` / `classify_candidates()`
    # 4. 最后看 `recover_missing_events()`
    parser = argparse.ArgumentParser(description="检测一个或两个 WAV 文件中的人耳可辨特殊事件。")
    parser.add_argument("wav_paths", nargs="+", type=Path, help="一个或两个 WAV 文件路径")
    parser.add_argument(
        "--block-ms",
        type=float,
        default=10.0,
        help=(
            "侦察阶段的初始时间块大小，单位毫秒。"
            "程序后续会自动学习正式值；通常不用改。"
        ),
    )
    parser.add_argument(
        "--min-score-ratio",
        type=float,
        default=0.002,
        help=(
            "侦察阶段的初始候选阈值比例。"
            "程序后续会自动学习正式值；通常不用改。"
        ),
    )
    parser.add_argument(
        "--min-event-gap-ms",
        type=float,
        default=300.0,
        help=(
            "最终事件最小间隔的侦察基准值，单位毫秒。"
            "程序后续会自动学习正式值；通常不用改。"
        ),
    )
    parser.add_argument(
        "--edge-margin-ms",
        type=float,
        default=200.0,
        help=(
            "忽略录音开头和结尾附近的边界伪峰，单位毫秒。"
            "调大更安全，但真实事件如果非常靠边，也可能被忽略。"
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=128,
        help=(
            "每个文件最多保留多少个候选事件。"
            "一般不需要改；只有事件特别多，或你故意把阈值调得很低时才可能碰到。"
        ),
    )
    parser.add_argument("--output-json", type=Path, help="可选：输出 JSON 文件路径")
    args = parser.parse_args()

    if not (1 <= len(args.wav_paths) <= 2):
        raise SystemExit("请传入 1 个或 2 个 WAV 文件。")

    # 第一遍：侦察模式
    # 目标不是给最终答案，而是尽量多看一些候选，用来估计本次环境下的参数。
    scout_results = [
        analyze_file(
            wav_path,
            block_ms=args.block_ms,
            min_score_ratio=args.min_score_ratio,
            edge_margin_ms=args.edge_margin_ms,
            max_candidates=args.max_candidates,
            percentile_threshold=95.0,
            candidate_gap_ms=max(60.0, args.block_ms * 6.0),
            candidate_merge_gap_ms=0.06,
        )
        for wav_path in args.wav_paths
    ]

    auto_params = infer_auto_parameters(
        scout_results,
        requested_block_ms=args.block_ms,
        requested_min_score_ratio=args.min_score_ratio,
        requested_min_event_gap_ms=args.min_event_gap_ms,
    )

    # 第二遍：正式模式
    # 把第一遍学出来的参数真正用于候选提取和后续模板确认。
    results = [
        analyze_file(
            wav_path,
            block_ms=auto_params.block_ms,
            min_score_ratio=auto_params.min_score_ratio,
            edge_margin_ms=args.edge_margin_ms,
            max_candidates=args.max_candidates,
            percentile_threshold=96.0,
            candidate_gap_ms=auto_params.candidate_gap_ms,
            candidate_merge_gap_ms=auto_params.candidate_merge_gap_ms,
        )
        for wav_path in args.wav_paths
    ]

    template = build_template(results)
    for result in results:
        classify_candidates(
            result,
            template,
            min_event_gap_ms=auto_params.min_event_gap_ms,
            echo_gap_ms=auto_params.echo_gap_ms,
        )

    if len(results) == 2:
        # 双文件场景默认让“更强、更完整”的那一边先当 reference。
        ordered = sorted(results, key=result_rank_key, reverse=True)
        recover_missing_events(
            ordered[0],
            ordered[1],
            template,
            min_event_gap_ms=auto_params.min_event_gap_ms,
            edge_margin_ms=args.edge_margin_ms,
            recover_search_radius_ms=auto_params.recover_search_radius_ms,
        )
        if result_rank_key(ordered[1]) > result_rank_key(ordered[0]):
            ordered = [ordered[1], ordered[0]]
        # 第一轮复核后，弱侧可能已经补回一些事件，所以再允许重排一次，再做第二轮复核。
        recover_missing_events(
            ordered[0],
            ordered[1],
            template,
            min_event_gap_ms=auto_params.min_event_gap_ms,
            edge_margin_ms=args.edge_margin_ms,
            recover_search_radius_ms=auto_params.recover_search_radius_ms,
        )

    print_auto_summary(auto_params, template)
    for index, result in enumerate(results, start=1):
        if index > 1:
            print()
        print_result(result)

    if args.output_json:
        payload = {
            "auto_parameters": {
                "block_ms": auto_params.block_ms,
                "min_score_ratio": auto_params.min_score_ratio,
                "candidate_gap_ms": auto_params.candidate_gap_ms,
                "candidate_merge_gap_ms": auto_params.candidate_merge_gap_ms,
                "min_event_gap_ms": auto_params.min_event_gap_ms,
                "echo_gap_ms": auto_params.echo_gap_ms,
                "recover_search_radius_ms": auto_params.recover_search_radius_ms,
                "impact_mode_ok": auto_params.impact_mode_ok,
                "impact_mode_reason": auto_params.impact_mode_reason,
                "scout_candidate_count": auto_params.scout_candidate_count,
            },
            "template": {
                "confirm_similarity": template.confirm_similarity,
                "recover_similarity": template.recover_similarity,
                "median_width_ms": template.median_width_ms,
                "median_attack_ratio": template.median_attack_ratio,
                "median_decay_ratio": template.median_decay_ratio,
                "min_confirm_energy": template.min_confirm_energy,
                "min_recover_energy": template.min_recover_energy,
            },
            "files": [result_to_payload(result) for result in results],
        }
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
