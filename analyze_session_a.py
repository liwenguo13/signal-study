#!/usr/bin/env python3
"""
用途：
1. 对 1 个或 2 个 WAV 录音做“特殊瞬态事件”检测。
2. 这里的“特殊事件”指人耳能明确数出来的拍手、敲击、撞击这类短促事件。
3. 当传感器距离声源远、事件很弱时，会先在强侧学出“本次事件长什么样”，再去弱侧复核。

主流程：
1. `main()` 解析参数，逐个 WAV 调用 `analyze_file()`
2. `analyze_file()` 内部先调 `collect_candidates()` 找候选事件
3. 所有文件的候选合并后，`build_template()` 生成“本次会话的事件模板”
4. 每个文件再调 `classify_candidates()`，把候选分成“直接确认”与“淘汰”
5. 如果有两个文件，再调 `recover_missing_events()`，只在本地找到弱证据时补成“复核确认”
6. 最后 `print_result()` 输出中文结果，可选写入 JSON

设计原则：
1. 不写死事件个数
2. 不因为另一端有事件，就强行给本端“补假数”
3. 本端必须出现自己的波形证据，才允许计入最终事件数
"""

"""
  - 主流程怎么走：main() -> analyze_file() -> collect_candidates() -> build_template() -> classify_candidates() -> recover_missing_events()
  - 函数之间怎么调用、每个函数负责什么
  - 关键参数调大/调小分别会带来什么后果，尤其是 --block-ms、--min-score-ratio、--min-event-gap-ms、--edge-margin-ms、--max-candidates
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
        return values.astype(np.float64, copy=True)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def block_max(signal: np.ndarray, block_size: int) -> np.ndarray:
    # 按 block_size 分块，每块取最大绝对幅值，得到“粗包络”。
    # block_size 越大：抗噪声更强，但多个靠得很近的小峰可能被揉在一起。
    # block_size 越小：时间定位更细，但更容易把回响/噪声也当成独立峰。
    pad = (-len(signal)) % block_size
    if pad:
        signal = np.pad(signal, (0, pad), mode="constant")
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
    block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
    envelope = block_max(signal.astype(np.float64), block_size)
    short_window = max(1, int(round(10.0 / block_ms)))
    long_window = max(short_window + 1, int(round(120.0 / block_ms)))
    smooth_short = moving_average(envelope, short_window)
    smooth_long = moving_average(envelope, long_window)
    novelty = np.maximum(smooth_short - smooth_long, 0.0)
    rise = np.maximum(np.diff(smooth_short, prepend=smooth_short[0]), 0.0)
    rise_scale = np.percentile(rise, 95) + 1.0
    feature = novelty * (1.0 + rise / rise_scale)
    return envelope, feature, block_size


def extract_window(values: np.ndarray, center_index: int, left: int, right: int) -> np.ndarray:
    # 以 center_index 为中心取一个局部窗口。
    # 这个函数主要被 `build_candidate()` 调用，用来截取候选事件附近的局部形状。
    start = center_index - left
    stop = center_index + right + 1
    pad_left = max(0, -start)
    pad_right = max(0, stop - len(values))
    clipped = values[max(0, start) : min(len(values), stop)].astype(np.float64, copy=False)
    if pad_left or pad_right:
        clipped = np.pad(clipped, (pad_left, pad_right), mode="constant")
    return clipped


def find_onset(signal: np.ndarray, peak_index: int, search_radius: int) -> int:
    # 已知峰值点后，向前找“真正开始抬头”的起始点。
    # search_radius 越大：更有机会找到真实起点，但也可能被更早的干扰拖走。
    # search_radius 越小：定位更保守，可能把起点报晚。
    start = max(0, peak_index - search_radius)
    local = signal[start : peak_index + 1].astype(np.float64)
    if local.size <= 1:
        return peak_index

    diff = np.maximum(np.diff(local, prepend=local[0]), 0.0)
    threshold = np.percentile(diff, 80)
    above = np.flatnonzero(diff >= threshold)
    if len(above) == 0:
        return peak_index
    return start + int(above[0])


def candidate_width_ms(feature: np.ndarray, index: int, block_ms: float) -> float:
    # 估算一个候选峰有多“宽”。
    # 很宽但不尖的波形，更像背景起伏/拖尾，不像一次明确拍手。
    peak = float(feature[index])
    threshold = peak * 0.25
    left = index
    while left > 0 and feature[left - 1] >= threshold:
        left -= 1
    right = index
    while right + 1 < len(feature) and feature[right + 1] >= threshold:
        right += 1
    return (right - left + 1) * block_ms


def candidate_local_energy(feature: np.ndarray, index: int) -> float:
    # 看候选峰附近一小段区域的总能量。
    # 它比单点峰值更稳，常用来区分“真实小事件”和“单点毛刺”。
    start = max(0, index - 2)
    stop = min(len(feature), index + 3)
    return float(np.sum(feature[start:stop]))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # 模板相似度。1.0 表示形状非常像，越低表示越不像。
    # 这里只比较“形状”，不是比较绝对音量，所以远端弱信号也有机会匹配到。
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
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
    block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
    peak_center = block_index * block_size
    peak_start = max(0, peak_center - block_size)
    peak_stop = min(len(signal), peak_center + block_size + 1)
    peak_index = peak_start + int(np.argmax(signal[peak_start:peak_stop]))
    onset_index = find_onset(signal, peak_index, max(1, int(round(sample_rate * 15.0 / 1000.0))))
    width_ms = candidate_width_ms(feature, block_index, block_ms)
    local_energy = candidate_local_energy(feature, block_index)

    env_window = extract_window(envelope, block_index, left=4, right=8)
    feat_window = extract_window(feature, block_index, left=4, right=8)
    env_norm = env_window / (float(np.max(env_window)) + 1.0)
    feat_norm = feat_window / (float(np.max(feat_window)) + 1.0)
    shape_vector = np.concatenate([env_norm, feat_norm])

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
        attack_ratio=peak_energy / pre_energy,
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

    refractory_blocks = max(1, int(round(candidate_gap_ms / block_ms)))
    grouped: list[list[int]] = []
    for idx in local_maxima:
        if not grouped or idx - grouped[-1][-1] >= refractory_blocks:
            grouped.append([idx])
        else:
            grouped[-1].append(idx)

    chosen = [max(group, key=lambda index: feature[index]) for group in grouped]
    chosen.sort(key=lambda index: feature[index], reverse=True)
    chosen = chosen[:max_candidates]
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
) -> tuple[np.ndarray, np.ndarray, list[CandidateEvent]]:
    # 单文件候选提取总入口。
    # 调用链：
    # `analyze_file()` -> `collect_candidates()`
    # `collect_candidates()` -> `build_feature()` -> `collect_candidate_indices()` -> `build_candidate()`
    #
    # 这一步只负责“找可能的事件”，不负责做最终确认。
    envelope, feature, _ = build_feature(signal, sample_rate, block_ms)
    indices = collect_candidate_indices(
        feature,
        min_score_ratio=min_score_ratio,
        percentile_threshold=96.0,
        candidate_gap_ms=max(80.0, block_ms * 8.0),
        block_ms=block_ms,
        max_candidates=max_candidates,
    )

    duration_s = len(signal) / sample_rate
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

    candidates = merge_close_candidates(candidates, merge_gap_s=0.08)
    return envelope, feature, candidates


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
    energy_floor = float(np.percentile(energies, 35)) if len(energies) > 2 else float(np.min(energies))
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
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    normalized = matrix / norms
    similarity = normalized @ normalized.T

    cluster_threshold = 0.82
    cluster_masks = similarity >= cluster_threshold
    cluster_sizes = np.sum(cluster_masks, axis=1)
    if int(np.max(cluster_sizes)) < 3:
        cluster_threshold = 0.74
        cluster_masks = similarity >= cluster_threshold

    best_index = 0
    best_size = -1
    best_weight = -1.0
    for index, mask in enumerate(cluster_masks):
        size = int(np.sum(mask))
        weight = float(np.sum([eligible[item].score for item in np.flatnonzero(mask)]))
        if size > best_size or (size == best_size and weight > best_weight):
            best_index = index
            best_size = size
            best_weight = weight

    cluster = [eligible[index] for index in np.flatnonzero(cluster_masks[best_index])]
    weights = np.sqrt(np.array([candidate.score for candidate in cluster], dtype=np.float64))
    template_vector = np.average(np.stack([candidate.shape_vector for candidate in cluster]), axis=0, weights=weights)
    template_norm = np.linalg.norm(template_vector)
    if template_norm > 0.0:
        template_vector = template_vector / template_norm

    cluster_similarities = np.array(
        [cosine_similarity(candidate.shape_vector, template_vector) for candidate in cluster],
        dtype=np.float64,
    )
    confirm_similarity = float(np.clip(np.percentile(cluster_similarities, 20) - 0.05, 0.68, 0.88))
    recover_similarity = float(np.clip(confirm_similarity - 0.10, 0.58, 0.82))

    cluster_energies = np.array([candidate.local_energy for candidate in cluster], dtype=np.float64)
    min_confirm_energy = float(
        np.clip(np.percentile(cluster_energies, 10) * 0.35, 120000.0, 420000.0)
    )
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


def classify_candidates(
    result: AnalysisResult,
    template: EventTemplate,
    *,
    min_event_gap_ms: float,
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
    width_limit = max(40.0, template.median_width_ms * 2.5)
    merge_gap_s = max(0.20, min_event_gap_ms / 1000.0 * 0.75)

    for candidate in result.candidates:
        candidate.template_similarity = cosine_similarity(candidate.shape_vector, template.vector)
        if candidate.template_similarity < template.confirm_similarity:
            continue
        if candidate.local_energy < template.min_confirm_energy:
            continue
        if candidate.width_ms > width_limit:
            continue
        confirmed_candidates.append(candidate)

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
    pairwise = target_times[:, None] - reference_times[None, :]
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
    center_index = int(round(center_time_s / result.block_ms * 1000.0))
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
        distance_penalty = abs(candidate.onset_s - center_time_s) / max(radius_ms / 1000.0, 1e-6)
        rank = candidate.template_similarity - 0.20 * distance_penalty
        if best_rank is None or rank > best_rank:
            best_candidate = candidate
            best_rank = rank
    return best_candidate


def recover_missing_events(
    reference: AnalysisResult,
    target: AnalysisResult,
    template: EventTemplate,
    *,
    min_event_gap_ms: float,
    edge_margin_ms: float,
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
    matched_reference, _ = match_events_by_offset(
        reference.confirmed_events,
        target.confirmed_events,
        offset_s=offset_s,
        tolerance_s=tolerance_s,
    )

    existing_times = [event.onset_s for event in target.confirmed_events]
    gap_s = max(0.08, min_event_gap_ms / 1000.0 * 0.5)

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
            radius_ms=180.0,
            edge_margin_ms=edge_margin_ms,
        )
        if candidate is None:
            if not any(abs(predicted_time - current) <= 0.05 for current in target.suspected_missing_s):
                target.suspected_missing_s.append(predicted_time)
            continue

        if candidate.template_similarity < template.recover_similarity:
            if not any(abs(predicted_time - current) <= 0.05 for current in target.suspected_missing_s):
                target.suspected_missing_s.append(predicted_time)
            continue
        if candidate.local_energy < template.min_recover_energy:
            if not any(abs(predicted_time - current) <= 0.05 for current in target.suspected_missing_s):
                target.suspected_missing_s.append(predicted_time)
            continue
        if candidate.width_ms > max(45.0, template.median_width_ms * 3.0):
            if not any(abs(predicted_time - current) <= 0.05 for current in target.suspected_missing_s):
                target.suspected_missing_s.append(predicted_time)
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


def result_to_payload(result: AnalysisResult) -> dict[str, object]:
    # 把单文件分析结果转成 JSON 可写入的普通字典。
    return {
        "wav_path": str(result.wav_path),
        "duration_s": result.duration_s,
        "candidate_count": len(result.candidates),
        "confirmed_count": len(result.confirmed_events),
        "recovered_count": result.recovered_events,
        "suspected_missing_s": result.suspected_missing_s,
        "events": [event_to_payload(event) for event in result.confirmed_events],
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
    for index, event in enumerate(result.confirmed_events, start=1):
        print(
            f"{index:02d} 判定={event.confirmation} "
            f"起始时刻={event.onset_s:.6f}秒 "
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


def main() -> None:
    # 总入口。
    # 推荐阅读顺序：
    # 1. 先看这里，理解主流程和参数
    # 2. 再看 `analyze_file()` / `collect_candidates()`
    # 3. 再看 `build_template()` / `classify_candidates()`
    # 4. 最后看 `recover_missing_events()`
    parser = argparse.ArgumentParser(description="检测一个或两个 WAV 文件中的人耳可辨特殊事件。")
    parser.add_argument("wav_paths", nargs="+", type=Path, help="一个或两个 WAV 文件路径")
    parser.add_argument(
        "--block-ms",
        type=float,
        default=10.0,
        help=(
            "特征计算的时间块大小，单位毫秒。"
            "调大更不容易把一次拍手后的回响拆成多次；"
            "调小时间定位更细，但更容易多检。"
        ),
    )
    parser.add_argument(
        "--min-score-ratio",
        type=float,
        default=0.002,
        help=(
            "候选检测时，最弱特征分数相对最强分数的下限比例。"
            "调大更严格、候选更少；调小更容易保住弱事件，但噪声也更容易进来。"
        ),
    )
    parser.add_argument(
        "--min-event-gap-ms",
        type=float,
        default=300.0,
        help=(
            "两个最终事件之间的最小间隔，单位毫秒。"
            "调大可减少多检；调得过大时，节奏很快的真实连续敲击可能被并成一次。"
        ),
    )
    parser.add_argument(
        "--edge-margin-ms",
        type=float,
        default=200.0,
        help=(
            "忽略录音开头和结尾附近的边界伪峰，单位毫秒。"
            "调大更安全，但如果真实事件恰好非常靠近录音边缘，也可能被忽略。"
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

    results = [
        analyze_file(
            wav_path,
            block_ms=args.block_ms,
            min_score_ratio=args.min_score_ratio,
            edge_margin_ms=args.edge_margin_ms,
            max_candidates=args.max_candidates,
        )
        for wav_path in args.wav_paths
    ]

    template = build_template(results)
    for result in results:
        classify_candidates(
            result,
            template,
            min_event_gap_ms=args.min_event_gap_ms,
        )

    if len(results) == 2:
        ordered = sorted(results, key=result_rank_key, reverse=True)
        recover_missing_events(
            ordered[0],
            ordered[1],
            template,
            min_event_gap_ms=args.min_event_gap_ms,
            edge_margin_ms=args.edge_margin_ms,
        )
        if result_rank_key(ordered[1]) > result_rank_key(ordered[0]):
            ordered = [ordered[1], ordered[0]]
        recover_missing_events(
            ordered[0],
            ordered[1],
            template,
            min_event_gap_ms=args.min_event_gap_ms,
            edge_margin_ms=args.edge_margin_ms,
        )

    print(
        "模板确认参数："
        f"直接确认阈值={template.confirm_similarity:.3f} "
        f"复核阈值={template.recover_similarity:.3f} "
        f"直接确认最小局部能量={template.min_confirm_energy:.1f}"
    )
    for index, result in enumerate(results, start=1):
        if index > 1:
            print()
        print_result(result)

    if args.output_json:
        payload = {
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

