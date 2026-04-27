#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wav_signal import load_signal


@dataclass
class CandidateEvent:
    block_index: int
    time_s: float
    onset_s: float
    peak_s: float
    score: float
    signature: np.ndarray


@dataclass
class MatchedEvent:
    reference: CandidateEvent
    target: CandidateEvent
    offset_s: float
    signature_distance: float


def block_max(signal: np.ndarray, block_size: int) -> np.ndarray:
    pad = (-len(signal)) % block_size
    if pad:
        signal = np.pad(signal, (0, pad), mode="constant")
    return signal.reshape(-1, block_size).max(axis=1)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def build_feature(signal: np.ndarray, sample_rate: int, block_ms: float) -> tuple[np.ndarray, np.ndarray, int]:
    block_size = max(1, int(round(sample_rate * block_ms / 1000.0)))
    env = block_max(signal.astype(np.float64), block_size)
    short_window = max(1, int(round(10.0 / block_ms)))
    long_window = max(short_window + 1, int(round(120.0 / block_ms)))
    smooth_short = moving_average(env, short_window)
    smooth_long = moving_average(env, long_window)
    novelty = np.maximum(smooth_short - smooth_long, 0.0)
    diff = np.maximum(np.diff(smooth_short, prepend=smooth_short[0]), 0.0)
    feature = novelty * (1.0 + diff / (np.percentile(diff, 95) + 1.0))
    return env, feature, block_size


def normalize_signature(window: np.ndarray) -> np.ndarray:
    centered = window.astype(np.float64) - np.mean(window)
    scale = np.linalg.norm(centered)
    if scale <= 1e-12:
        return np.zeros_like(centered)
    return centered / scale


def find_onset(signal: np.ndarray, peak_index: int, search_radius: int) -> int:
    start = max(0, peak_index - search_radius)
    local = signal[start : peak_index + 1].astype(np.float64)
    energy = np.abs(np.diff(local, prepend=local[0]))
    threshold = np.percentile(energy, 80)
    above = np.flatnonzero(energy >= threshold)
    if len(above) == 0:
        return peak_index
    return start + int(above[0])


def extract_candidates(
    signal: np.ndarray,
    sample_rate: int,
    *,
    block_ms: float,
    refractory_ms: float,
    max_candidates: int,
    signature_ms: float,
    min_score_ratio: float,
) -> list[CandidateEvent]:
    env, feature, block_size = build_feature(signal, sample_rate, block_ms)
    refractory_blocks = max(1, int(round(refractory_ms / block_ms)))
    score_threshold = max(
        np.percentile(feature, 95),
        float(np.max(feature)) * min_score_ratio,
    )

    local_maxima = [
        idx
        for idx in range(1, len(feature) - 1)
        if feature[idx] >= score_threshold
        and feature[idx] >= feature[idx - 1]
        and feature[idx] >= feature[idx + 1]
    ]
    local_maxima.sort(key=lambda idx: feature[idx], reverse=True)

    chosen: list[int] = []
    for idx in local_maxima:
        if any(abs(idx - prev) < refractory_blocks for prev in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= max_candidates:
            break
    chosen.sort()

    signature_radius = max(1, int(round(sample_rate * signature_ms / 1000.0 / 2.0)))
    onset_radius = max(1, int(round(sample_rate * 15.0 / 1000.0)))

    candidates: list[CandidateEvent] = []
    for block_index in chosen:
        peak_center = block_index * block_size
        peak_start = max(0, peak_center - block_size)
        peak_stop = min(len(signal), peak_center + block_size + 1)
        peak_index = peak_start + int(np.argmax(signal[peak_start:peak_stop]))
        onset_index = find_onset(signal, peak_index, onset_radius)

        sig_start = max(0, onset_index - signature_radius)
        sig_stop = min(len(signal), onset_index + signature_radius)
        signature = normalize_signature(signal[sig_start:sig_stop])

        candidates.append(
            CandidateEvent(
                block_index=block_index,
                time_s=block_index * block_size / sample_rate,
                onset_s=onset_index / sample_rate,
                peak_s=peak_index / sample_rate,
                score=float(feature[block_index]),
                signature=signature,
            )
        )
    return candidates


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return float("inf")
    aa = a[:length]
    bb = b[:length]
    return float(np.linalg.norm(aa - bb))


def estimate_global_offset(
    reference_events: list[CandidateEvent],
    target_events: list[CandidateEvent],
    *,
    max_offset_s: float,
    signature_gate: float,
    offset_bin_s: float,
) -> tuple[float, list[tuple[int, int, float, float]]]:
    pair_votes: list[tuple[int, int, float, float]] = []
    for ref_index, ref_event in enumerate(reference_events):
        for tgt_index, tgt_event in enumerate(target_events):
            distance = signature_distance(ref_event.signature, tgt_event.signature)
            if distance > signature_gate:
                continue
            offset = tgt_event.onset_s - ref_event.onset_s
            if abs(offset) > max_offset_s:
                continue
            pair_votes.append((ref_index, tgt_index, offset, distance))

    if not pair_votes:
        raise ValueError("no candidate event pairs survived signature gating")

    bins: dict[float, list[tuple[int, int, float, float]]] = {}
    for pair in pair_votes:
        _, _, offset, _ = pair
        bin_key = round(offset / offset_bin_s) * offset_bin_s
        bins.setdefault(bin_key, []).append(pair)

    def score_bin(items: list[tuple[int, int, float, float]]) -> tuple[int, float]:
        count = len(items)
        mean_distance = float(np.mean([item[3] for item in items]))
        return (count, -mean_distance)

    best_key = max(bins, key=lambda key: score_bin(bins[key]))
    best_items = bins[best_key]
    refined_offset = float(np.median([item[2] for item in best_items]))
    return refined_offset, best_items


def greedy_match(
    reference_events: list[CandidateEvent],
    target_events: list[CandidateEvent],
    *,
    offset_s: float,
    offset_tolerance_s: float,
    signature_gate: float,
) -> list[MatchedEvent]:
    pair_scores: list[tuple[float, float, int, int]] = []
    for ref_index, ref_event in enumerate(reference_events):
        for tgt_index, tgt_event in enumerate(target_events):
            dt = tgt_event.onset_s - ref_event.onset_s
            offset_error = abs(dt - offset_s)
            if offset_error > offset_tolerance_s:
                continue
            distance = signature_distance(ref_event.signature, tgt_event.signature)
            if distance > signature_gate:
                continue
            pair_scores.append((offset_error, distance, ref_index, tgt_index))

    pair_scores.sort()
    used_ref: set[int] = set()
    used_tgt: set[int] = set()
    matches: list[MatchedEvent] = []
    for offset_error, distance, ref_index, tgt_index in pair_scores:
        if ref_index in used_ref or tgt_index in used_tgt:
            continue
        used_ref.add(ref_index)
        used_tgt.add(tgt_index)
        ref_event = reference_events[ref_index]
        tgt_event = target_events[tgt_index]
        matches.append(
            MatchedEvent(
                reference=ref_event,
                target=tgt_event,
                offset_s=tgt_event.onset_s - ref_event.onset_s,
                signature_distance=distance,
            )
        )
    matches.sort(key=lambda item: item.reference.onset_s)
    return matches


def filter_matches_by_offset(
    matches: list[MatchedEvent],
    *,
    cluster_tolerance_s: float,
) -> list[MatchedEvent]:
    if not matches:
        return matches
    offsets = np.array([match.offset_s for match in matches], dtype=np.float64)
    median_offset = float(np.median(offsets))
    filtered = [
        match
        for match in matches
        if abs(match.offset_s - median_offset) <= cluster_tolerance_s
    ]
    return filtered


def load_start_time_ns(path: Path | None) -> int | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    value = data.get("start_time_realtime_ns")
    if value is None:
        raise ValueError(f"{path} does not contain start_time_realtime_ns")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match arbitrary transient events across two WAV recordings and estimate time/distance differences."
    )
    parser.add_argument("reference_wav", type=Path, help="Reference WAV path")
    parser.add_argument("target_wav", type=Path, help="Target WAV path")
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="Optional JSON metadata for the reference WAV containing start_time_realtime_ns",
    )
    parser.add_argument(
        "--target-json",
        type=Path,
        help="Optional JSON metadata for the target WAV containing start_time_realtime_ns",
    )
    parser.add_argument("--block-ms", type=float, default=5.0, help="Feature block size in milliseconds")
    parser.add_argument(
        "--refractory-ms",
        type=float,
        default=250.0,
        help="Minimum separation between candidate events in one file",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=32,
        help="Maximum number of candidate events to retain from each file",
    )
    parser.add_argument(
        "--signature-ms",
        type=float,
        default=40.0,
        help="Local waveform window used to build each event signature",
    )
    parser.add_argument(
        "--min-score-ratio",
        type=float,
        default=0.008,
        help="Minimum candidate score as a fraction of the file's strongest event score",
    )
    parser.add_argument(
        "--signature-gate",
        type=float,
        default=1.6,
        help="Maximum signature distance allowed when considering a pair of events",
    )
    parser.add_argument(
        "--max-offset-s",
        type=float,
        default=30.0,
        help="Maximum whole-file start offset to consider during matching",
    )
    parser.add_argument(
        "--offset-bin-ms",
        type=float,
        default=10.0,
        help="Histogram bin size used when voting for the global start offset",
    )
    parser.add_argument(
        "--offset-tolerance-ms",
        type=float,
        default=120.0,
        help="Allowed deviation from the voted global offset when keeping a match",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=343.0,
        help="Propagation speed in m/s for converting time difference into path difference",
    )
    args = parser.parse_args()

    reference_rate, reference_signal = load_signal(args.reference_wav)
    target_rate, target_signal = load_signal(args.target_wav)
    if reference_rate != target_rate:
        raise ValueError("sample rates must match")

    reference_events = extract_candidates(
        reference_signal,
        reference_rate,
        block_ms=args.block_ms,
        refractory_ms=args.refractory_ms,
        max_candidates=args.max_candidates,
        signature_ms=args.signature_ms,
        min_score_ratio=args.min_score_ratio,
    )
    target_events = extract_candidates(
        target_signal,
        target_rate,
        block_ms=args.block_ms,
        refractory_ms=args.refractory_ms,
        max_candidates=args.max_candidates,
        signature_ms=args.signature_ms,
        min_score_ratio=args.min_score_ratio,
    )

    offset_s, voted_pairs = estimate_global_offset(
        reference_events,
        target_events,
        max_offset_s=args.max_offset_s,
        signature_gate=args.signature_gate,
        offset_bin_s=args.offset_bin_ms / 1000.0,
    )
    matches = greedy_match(
        reference_events,
        target_events,
        offset_s=offset_s,
        offset_tolerance_s=args.offset_tolerance_ms / 1000.0,
        signature_gate=args.signature_gate,
    )
    matches = filter_matches_by_offset(
        matches,
        cluster_tolerance_s=max(args.offset_tolerance_ms / 1000.0 / 2.0, 0.04),
    )

    print(f"reference_candidates={len(reference_events)}")
    print(f"target_candidates={len(target_events)}")
    print(f"offset_vote_pairs={len(voted_pairs)}")
    print(f"estimated_start_offset_s={offset_s:.6f}")
    print(f"matched_events={len(matches)}")

    ref_start_ns = load_start_time_ns(args.reference_json)
    tgt_start_ns = load_start_time_ns(args.target_json)
    have_absolute_time = ref_start_ns is not None and tgt_start_ns is not None

    for index, match in enumerate(matches, start=1):
        relative_delta_s = match.offset_s
        line = (
            f"{index:02d} ref_onset={match.reference.onset_s:.6f} "
            f"target_onset={match.target.onset_s:.6f} "
            f"relative_offset_s={relative_delta_s:.6f} "
            f"signature_distance={match.signature_distance:.4f}"
        )
        if have_absolute_time:
            ref_abs_ns = ref_start_ns + int(round(match.reference.onset_s * 1e9))
            tgt_abs_ns = tgt_start_ns + int(round(match.target.onset_s * 1e9))
            absolute_delta_s = (tgt_abs_ns - ref_abs_ns) / 1e9
            distance_delta_m = absolute_delta_s * args.speed
            line += (
                f" absolute_delta_s={absolute_delta_s:.9f}"
                f" absolute_delta_m={distance_delta_m:.6f}"
            )
        print(line)

    if matches:
        relative_offsets = np.array([match.offset_s for match in matches], dtype=np.float64)
        print(f"median_relative_offset_s={np.median(relative_offsets):.6f}")
        print(f"mean_relative_offset_s={np.mean(relative_offsets):.6f}")
        if have_absolute_time:
            absolute_offsets = []
            for match in matches:
                ref_abs_ns = ref_start_ns + int(round(match.reference.onset_s * 1e9))
                tgt_abs_ns = tgt_start_ns + int(round(match.target.onset_s * 1e9))
                absolute_offsets.append((tgt_abs_ns - ref_abs_ns) / 1e9)
            absolute_offsets = np.array(absolute_offsets, dtype=np.float64)
            print(f"median_absolute_delta_s={np.median(absolute_offsets):.9f}")
            print(f"mean_absolute_delta_s={np.mean(absolute_offsets):.9f}")
            print(f"median_absolute_delta_m={np.median(absolute_offsets) * args.speed:.6f}")
            print(f"mean_absolute_delta_m={np.mean(absolute_offsets) * args.speed:.6f}")


if __name__ == "__main__":
    main()
