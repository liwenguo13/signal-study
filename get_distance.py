#!/usr/bin/env python3
"""Estimate underwater A/B hydrophone distance from WAV and start-time JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal


NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class Recording:
    path: Path
    json_path: Path
    samples: np.ndarray
    sample_rate: int
    start_time_ns: int


@dataclass(frozen=True)
class EventEstimate:
    index: int
    a_onset_s: float
    expected_b_onset_s: float
    b_delay_s: float | None
    distance_m: float | None
    bands_agree: int
    bands_total: int
    score: float
    status: str
    reason: str
    broadband_delay_s: float | None = None
    debug_detail: str = ""


def read_json_start_time(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return int(data["start_time_realtime_ns"])
    except KeyError as exc:
        raise ValueError(f"{path} missing start_time_realtime_ns") from exc


def read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width == 3:
        u8 = np.frombuffer(raw, dtype=np.uint8)
        if len(u8) % (channels * 3) != 0:
            raise ValueError(f"{path} has incomplete 24-bit frames")
        triplets = u8.reshape(-1, channels, 3).astype(np.uint32)
        values = triplets[:, :, 0] | (triplets[:, :, 1] << 8) | (triplets[:, :, 2] << 16)
        signed = ((values ^ 0x800000) - 0x800000).astype(np.int32)
        samples = signed.astype(np.float32) / float(1 << 23)
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32)
        samples /= float(1 << 15)
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").reshape(-1, channels).astype(np.float32)
        samples /= float(1 << 31)
    else:
        raise ValueError(f"{path} uses unsupported sample width: {sample_width} bytes")

    return samples, sample_rate


def load_recording(wav_path: Path, json_path: Path) -> Recording:
    samples, sample_rate = read_pcm_wav(wav_path)
    start_time_ns = read_json_start_time(json_path)
    return Recording(wav_path, json_path, samples, sample_rate, start_time_ns)


def transient_contrast(y: np.ndarray, sample_rate: int) -> float:
    y = np.asarray(y, dtype=np.float32)
    y = y - float(np.mean(y))
    frame = max(16, int(round(sample_rate * 0.005)))
    hop = max(1, int(round(sample_rate * 0.001)))
    if len(y) < frame:
        return 0.0
    power = np.square(y.astype(np.float64))
    cumsum = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, len(y) - frame + 1, hop)
    energy = (cumsum[starts + frame] - cumsum[starts]) / frame
    z = robust_z(10.0 * np.log10(energy + 1e-20))
    return float(np.percentile(z, 99.5))


def choose_analysis_signal(samples: np.ndarray, sample_rate: int, mode: str) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("expected samples shaped as frames x channels")
    if mode == "auto":
        candidates = [np.mean(samples, axis=1)]
        candidates.extend(samples[:, idx] for idx in range(samples.shape[1]))
        scores = [transient_contrast(candidate, sample_rate) for candidate in candidates]
        y = candidates[int(np.argmax(scores))]
    elif mode == "mean":
        y = np.mean(samples, axis=1)
    elif mode == "best-rms":
        rms = np.sqrt(np.mean(np.square(samples.astype(np.float64)), axis=0))
        y = samples[:, int(np.argmax(rms))]
    elif mode == "first":
        y = samples[:, 0]
    else:
        raise ValueError(f"unknown channel mode: {mode}")
    y = np.asarray(y, dtype=np.float32)
    return y - float(np.mean(y))


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-12)
    return (values - median) / scale


def detect_a_events(
    y: np.ndarray,
    sample_rate: int,
    min_spacing_s: float,
    frame_ms: float,
    hop_ms: float,
    threshold_z: float,
    max_events: int | None,
) -> list[float]:
    frame = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if len(y) < frame:
        return []

    starts = np.arange(0, len(y) - frame + 1, hop)
    power = np.square(y.astype(np.float64))
    cumsum = np.concatenate(([0.0], np.cumsum(power)))
    energy = (cumsum[starts + frame] - cumsum[starts]) / frame
    log_energy = 10.0 * np.log10(energy + 1e-20)
    z = robust_z(log_energy)

    min_distance = max(1, int(round(min_spacing_s / (hop / sample_rate))))
    peaks, props = signal.find_peaks(z, height=threshold_z, distance=min_distance)

    if len(peaks) == 0:
        fallback = max(threshold_z * 0.65, 5.0)
        peaks, props = signal.find_peaks(z, height=fallback, distance=min_distance)

    events: list[float] = []
    for peak in peaks:
        local_start = max(0, peak - int(round(0.05 / (hop / sample_rate))))
        local = z[local_start : peak + 1]
        onset_candidates = np.flatnonzero(local >= max(3.0, threshold_z * 0.35))
        onset_frame = local_start + int(onset_candidates[0]) if len(onset_candidates) else int(peak)
        events.append(float(starts[onset_frame] / sample_rate))

    events = sorted(events)
    if max_events is not None:
        events = events[:max_events]
    return events


def bandpass_filter(y: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray | None:
    nyquist = sample_rate / 2.0
    low = max(1.0, low_hz)
    high = min(high_hz, nyquist * 0.95)
    if low >= high:
        return None
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def normalize_segment(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = y - float(np.mean(y))
    std = float(np.std(y))
    if std < 1e-9:
        return y * 0.0
    return y / std


def gcc_phat_scores(
    reference: np.ndarray,
    search: np.ndarray,
    sample_rate: int,
    min_delay_s: float,
    max_delay_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = normalize_segment(reference)
    search = normalize_segment(search)
    if len(reference) == 0 or len(search) == 0 or np.all(reference == 0) or np.all(search == 0):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    n = int(2 ** math.ceil(math.log2(len(reference) + len(search) - 1)))
    ref_fft = np.fft.rfft(reference, n=n)
    search_fft = np.fft.rfft(search, n=n)
    cross_power = search_fft * np.conj(ref_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    corr = np.fft.irfft(cross_power, n=n)
    corr = np.concatenate((corr[-(len(reference) - 1) :], corr[: len(search)]))
    lags = np.arange(-(len(reference) - 1), len(search), dtype=np.int64)
    delays = lags / float(sample_rate)

    mask = (delays >= min_delay_s) & (delays <= max_delay_s)
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    scores = np.abs(corr[mask])
    score_max = float(np.max(scores)) if len(scores) else 0.0
    if score_max > 0:
        scores = scores / score_max
    return delays[mask], scores


def pick_band_candidate(
    delays: np.ndarray,
    scores: np.ndarray,
    peak_threshold: float,
    edge_guard_s: float,
) -> tuple[float, float] | None:
    if len(delays) == 0:
        return None
    delay_span_s = float(delays[-1] - delays[0]) if len(delays) > 1 else 0.0
    effective_edge_guard_s = edge_guard_s if delay_span_s > edge_guard_s * 5.0 else 0.0
    if effective_edge_guard_s > 0 and len(delays) > 2:
        lo = float(delays[0]) + effective_edge_guard_s
        hi = float(delays[-1]) - effective_edge_guard_s
        inner = (delays >= lo) & (delays <= hi)
        if np.any(inner):
            delays = delays[inner]
            scores = scores[inner]
    peaks, props = signal.find_peaks(scores, height=peak_threshold)
    if len(peaks) == 0:
        best = int(np.argmax(scores))
        if float(scores[best]) >= peak_threshold:
            return float(delays[best]), float(scores[best])
        return None
    order = np.argsort(delays[peaks])
    first = int(peaks[int(order[0])])
    return float(delays[first]), float(scores[first])


def window_samples(y: np.ndarray, sample_rate: int, center_s: float, pre_s: float, post_s: float) -> tuple[np.ndarray, float]:
    start_s = center_s - pre_s
    end_s = center_s + post_s
    start = max(0, int(round(start_s * sample_rate)))
    end = min(len(y), int(round(end_s * sample_rate)))
    actual_start_s = start / float(sample_rate)
    if end <= start:
        return np.array([], dtype=np.float32), actual_start_s
    return y[start:end], actual_start_s


def estimate_event(
    index: int,
    a_onset_s: float,
    a_y: np.ndarray,
    b_y: np.ndarray,
    sample_rate: int,
    start_delta_s: float,
    min_prop_s: float,
    max_prop_s: float,
    bands: list[tuple[float, float]],
    ref_pre_s: float,
    ref_post_s: float,
    search_pad_s: float,
    band_peak_threshold: float,
    edge_guard_s: float,
    consensus_s: float,
    min_bands_agree: int,
    sound_speed: float,
    debug: bool,
    allow_broadband_conflict: bool,
    broadband_conflict_s: float,
) -> EventEstimate:
    # B local time for the same absolute A onset, before propagation delay.
    b_same_absolute_s = a_onset_s - start_delta_s
    b_search_start_s = b_same_absolute_s + min_prop_s
    b_search_end_s = b_same_absolute_s + max_prop_s

    if b_search_end_s <= 0 or b_search_start_s >= len(b_y) / sample_rate:
        return EventEstimate(
            index,
            a_onset_s,
            b_same_absolute_s,
            None,
            None,
            0,
            len(bands),
            0.0,
            "skip",
            "B search window outside recording",
        )

    a_ref, _ = window_samples(a_y, sample_rate, a_onset_s, ref_pre_s, ref_post_s)
    b_search, b_start_actual_s = window_samples(
        b_y,
        sample_rate,
        (b_search_start_s + b_search_end_s) / 2.0,
        (b_search_end_s - b_search_start_s) / 2.0 + search_pad_s,
        (b_search_end_s - b_search_start_s) / 2.0 + search_pad_s,
    )
    if len(a_ref) < int(0.01 * sample_rate) or len(b_search) < int(0.01 * sample_rate):
        return EventEstimate(
            index,
            a_onset_s,
            b_same_absolute_s,
            None,
            None,
            0,
            len(bands),
            0.0,
            "skip",
            "insufficient samples around event",
        )

    min_delay_local_s = max(0.0, b_search_start_s - b_start_actual_s)
    max_delay_local_s = min(len(b_search) / sample_rate, b_search_end_s - b_start_actual_s)
    candidates: list[tuple[float, float, tuple[float, float]]] = []
    broadband_delay_s: float | None = None

    broad_delays, broad_scores = gcc_phat_scores(a_ref, b_search, sample_rate, min_delay_local_s, max_delay_local_s)
    broad = pick_band_candidate(broad_delays, broad_scores, band_peak_threshold, edge_guard_s)
    if broad is not None:
        broadband_delay_s = (b_start_actual_s + broad[0]) - b_same_absolute_s

    for low_hz, high_hz in bands:
        a_band = bandpass_filter(a_ref, sample_rate, low_hz, high_hz)
        b_band = bandpass_filter(b_search, sample_rate, low_hz, high_hz)
        if a_band is None or b_band is None:
            continue
        delays, scores = gcc_phat_scores(a_band, b_band, sample_rate, min_delay_local_s, max_delay_local_s)
        candidate = pick_band_candidate(delays, scores, band_peak_threshold, edge_guard_s)
        if candidate is None:
            continue
        delay_local_s, score = candidate
        propagation_delay_s = (b_start_actual_s + delay_local_s) - b_same_absolute_s
        candidates.append((propagation_delay_s, score, (low_hz, high_hz)))

    if len(candidates) < min_bands_agree:
        return EventEstimate(
            index,
            a_onset_s,
            b_same_absolute_s,
            None,
            None,
            len(candidates),
            len(bands),
            float(np.mean([c[1] for c in candidates])) if candidates else 0.0,
            "weak",
            "too few frequency bands found a peak",
            broadband_delay_s,
            ", ".join(f"{lo:.0f}-{hi:.0f}:{d*1000:.2f}ms/{s:.2f}" for d, s, (lo, hi) in candidates),
        )

    candidates.sort(key=lambda item: item[0])
    candidate_detail = ", ".join(f"{lo:.0f}-{hi:.0f}:{d*1000:.2f}ms/{s:.2f}" for d, s, (lo, hi) in candidates)
    for delay_s, score, _band in candidates:
        supporters = [c for c in candidates if abs(c[0] - delay_s) <= consensus_s]
        if len(supporters) >= min_bands_agree:
            best_delay = float(np.median([c[0] for c in supporters]))
            best_score = float(np.mean([c[1] for c in supporters]))
            distance_m = sound_speed * best_delay
            if (
                not allow_broadband_conflict
                and broadband_delay_s is not None
                and best_delay - broadband_delay_s > broadband_conflict_s
            ):
                detail = f"broadband earliest {broadband_delay_s*1000:.2f}ms vs multiband {best_delay*1000:.2f}ms"
                return EventEstimate(
                    index,
                    a_onset_s,
                    b_same_absolute_s,
                    None,
                    None,
                    len(supporters),
                    len(bands),
                    best_score,
                    "unstable",
                    f"earlier broadband peak conflicts with multiband candidate ({detail})",
                    broadband_delay_s,
                    candidate_detail,
                )
            return EventEstimate(
                index,
                a_onset_s,
                b_same_absolute_s,
                best_delay,
                distance_m,
                len(supporters),
                len(bands),
                best_score,
                "ok",
                "ok",
                broadband_delay_s,
                candidate_detail,
            )

    reason = f"bands disagree: {candidate_detail}" if debug else "frequency bands disagree"
    return EventEstimate(
        index,
        a_onset_s,
        b_same_absolute_s,
        None,
        None,
        len(candidates),
        len(bands),
        float(np.mean([c[1] for c in candidates])),
        "unstable",
        reason,
        broadband_delay_s,
        candidate_detail,
    )


def median_abs_deviation(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def parse_bands(text: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            low, high = item.split("-", 1)
            bands.append((float(low), float(high)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid band {item!r}; expected LOW-HIGH") from exc
    if not bands:
        raise argparse.ArgumentTypeError("at least one band is required")
    return bands


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate A/B hydrophone distance from WAV files and PTP-aligned start-time JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a-wav", type=Path, default=Path("record_A/session_A.wav"))
    parser.add_argument("--a-json", type=Path, default=Path("record_A/session_A.json"))
    parser.add_argument("--b-wav", type=Path, default=Path("record_B/session_B.wav"))
    parser.add_argument("--b-json", type=Path, default=Path("record_B/session_B.json"))
    parser.add_argument("--min-distance", type=float, default=0.0)
    parser.add_argument("--max-distance", type=float, default=500.0)
    parser.add_argument("--sound-speed", type=float, default=1500.0)
    parser.add_argument(
        "--b-start-offset-ms",
        type=float,
        default=0.0,
        help="correction added to B JSON start time; positive means B audio starts later than the JSON timestamp",
    )
    parser.add_argument(
        "--calibration-distance",
        type=float,
        default=None,
        help="known A/B distance in meters; prints the B start-time correction needed to match it",
    )
    parser.add_argument("--channel-mode", choices=["auto", "mean", "best-rms", "first"], default="auto")
    parser.add_argument("--event-min-spacing", type=float, default=0.55, help="minimum spacing between A events, seconds")
    parser.add_argument("--event-frame-ms", type=float, default=5.0)
    parser.add_argument("--event-hop-ms", type=float, default=1.0)
    parser.add_argument("--event-threshold-z", type=float, default=8.0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--bands", type=parse_bands, default=parse_bands("300-800,800-1500,1500-3000,3000-8000,8000-20000"))
    parser.add_argument("--ref-pre", type=float, default=0.010, help="A reference window before onset, seconds")
    parser.add_argument("--ref-post", type=float, default=0.080, help="A reference window after onset, seconds")
    parser.add_argument("--search-pad", type=float, default=0.030, help="extra B search padding around distance window, seconds")
    parser.add_argument("--band-peak-threshold", type=float, default=0.45)
    parser.add_argument("--edge-guard-ms", type=float, default=0.0, help="ignore correlation peaks this close to search-window edges")
    parser.add_argument("--consensus-ms", type=float, default=2.0)
    parser.add_argument("--min-bands-agree", type=int, default=3)
    parser.add_argument("--max-mad-m", type=float, default=10.0)
    parser.add_argument(
        "--allow-broadband-conflict",
        action="store_true",
        help="allow a later multiband peak even when broadband GCC finds a much earlier peak",
    )
    parser.add_argument("--broadband-conflict-ms", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.min_distance < 0 or args.max_distance <= args.min_distance:
        raise ValueError("--max-distance must be greater than --min-distance and distances must be non-negative")
    if args.sound_speed <= 0:
        raise ValueError("--sound-speed must be positive")
    if args.min_bands_agree < 1:
        raise ValueError("--min-bands-agree must be at least 1")
    if args.min_bands_agree > len(args.bands):
        raise ValueError("--min-bands-agree cannot exceed number of bands")
    if args.edge_guard_ms < 0:
        raise ValueError("--edge-guard-ms must be non-negative")
    if args.broadband_conflict_ms < 0:
        raise ValueError("--broadband-conflict-ms must be non-negative")


def classify_distance_inliers(distances: np.ndarray, max_deviation_m: float) -> tuple[float, float, np.ndarray]:
    if len(distances) == 0:
        return float("nan"), float("nan"), np.array([], dtype=bool)
    median = float(np.median(distances))
    mad = median_abs_deviation(distances)
    if not math.isfinite(mad):
        return median, mad, np.zeros(len(distances), dtype=bool)
    if mad < 1e-9:
        threshold = max(max_deviation_m, 1.0)
    else:
        threshold = max(max_deviation_m, 3.0 * 1.4826 * mad)
    inliers = np.abs(distances - median) <= threshold
    if np.any(inliers):
        median = float(np.median(distances[inliers]))
        mad = median_abs_deviation(distances[inliers])
    return median, mad, inliers


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    rec_a = load_recording(args.a_wav, args.a_json)
    rec_b = load_recording(args.b_wav, args.b_json)
    if rec_a.sample_rate != rec_b.sample_rate:
        raise ValueError(f"sample rate mismatch: A={rec_a.sample_rate}, B={rec_b.sample_rate}")

    sample_rate = rec_a.sample_rate
    a_y = choose_analysis_signal(rec_a.samples, sample_rate, args.channel_mode)
    b_y = choose_analysis_signal(rec_b.samples, sample_rate, args.channel_mode)

    events = detect_a_events(
        a_y,
        sample_rate,
        args.event_min_spacing,
        args.event_frame_ms,
        args.event_hop_ms,
        args.event_threshold_z,
        args.max_events,
    )
    if not events:
        print("status: error")
        print("reason: no A-side knock events found")
        return 2

    start_delta_s = (rec_b.start_time_ns - rec_a.start_time_ns) / NS_PER_S + args.b_start_offset_ms / 1000.0
    min_prop_s = args.min_distance / args.sound_speed
    max_prop_s = args.max_distance / args.sound_speed
    consensus_s = args.consensus_ms / 1000.0
    edge_guard_s = args.edge_guard_ms / 1000.0

    estimates = [
        estimate_event(
            idx,
            onset_s,
            a_y,
            b_y,
            sample_rate,
            start_delta_s,
            min_prop_s,
            max_prop_s,
            args.bands,
            args.ref_pre,
            args.ref_post,
            args.search_pad,
            args.band_peak_threshold,
            edge_guard_s,
            consensus_s,
            args.min_bands_agree,
            args.sound_speed,
            args.debug,
            args.allow_broadband_conflict,
            args.broadband_conflict_ms / 1000.0,
        )
        for idx, onset_s in enumerate(events, start=1)
    ]

    ok_estimates = [e for e in estimates if e.status == "ok" and e.distance_m is not None]
    ok_distances = np.array([e.distance_m for e in ok_estimates], dtype=np.float64)
    median_distance, mad_m, inlier_mask = classify_distance_inliers(ok_distances, args.max_mad_m)
    inlier_event_indexes = {
        estimate.index for estimate, is_inlier in zip(ok_estimates, inlier_mask, strict=False) if bool(is_inlier)
    }
    outlier_count = len(ok_distances) - len(inlier_event_indexes)
    if len(ok_distances) == 0:
        overall_status = "unstable"
        overall_reason = "no event produced a credible multi-band delay"
    elif len(inlier_event_indexes) < 3:
        overall_status = "low_confidence"
        overall_reason = "fewer than 3 consistent valid events"
    elif mad_m > args.max_mad_m:
        overall_status = "unstable"
        overall_reason = f"event distances are dispersed, MAD {mad_m:.2f}m > {args.max_mad_m:.2f}m"
    elif outlier_count:
        overall_status = "ok"
        overall_reason = f"ok, ignored {outlier_count} outlier event(s)"
    else:
        overall_status = "ok"
        overall_reason = "ok"

    print("event  a_onset_s  b_delay_s  distance_m  bands_agree  score  status  reason")
    for e in estimates:
        display_status = "outlier" if e.status == "ok" and e.index not in inlier_event_indexes else e.status
        display_reason = "distance outside event consensus" if display_status == "outlier" else e.reason
        print(
            f"{e.index:<5d}  "
            f"{e.a_onset_s:<9.4f}  "
            f"{format_float(e.b_delay_s, 5):<9}  "
            f"{format_float(e.distance_m, 2):<10}  "
            f"{e.bands_agree}/{e.bands_total:<10}  "
            f"{e.score:<5.2f}  "
            f"{display_status:<8}  "
            f"{display_reason}"
        )
        if args.debug and e.broadband_delay_s is not None:
            print(f"# event {e.index} broadband_earliest_peak_delay_s={e.broadband_delay_s:.6f}")
        if args.debug and e.debug_detail:
            print(f"# event {e.index} band_candidates={e.debug_detail}")

    print()
    print(f"median_distance_m: {format_float(median_distance, 2)}")
    print(f"mad_m: {format_float(mad_m, 2)}")
    print(f"valid_events: {len(inlier_event_indexes)}/{len(estimates)}")
    print(f"outlier_events: {outlier_count}")
    if args.calibration_distance is not None and math.isfinite(median_distance):
        measured_delay_s = median_distance / args.sound_speed
        expected_delay_s = args.calibration_distance / args.sound_speed
        needed_b_offset_ms = (expected_delay_s - measured_delay_s) * 1000.0
        print(f"calibration_distance_m: {args.calibration_distance:.2f}")
        print(f"b_start_offset_for_calibration_ms: {needed_b_offset_ms:.3f}")
    print(f"status: {overall_status}")
    print(f"reason: {overall_reason}")

    if args.debug:
        print()
        print(f"# sample_rate_hz: {sample_rate}")
        print(f"# start_delta_s_B_minus_A: {start_delta_s:.9f}")
        print(f"# b_start_offset_ms: {args.b_start_offset_ms:.3f}")
        print(f"# propagation_window_s: {min_prop_s:.6f}-{max_prop_s:.6f}")
        print(f"# edge_guard_s: {edge_guard_s:.6f}")
        print(f"# detected_a_events_s: {', '.join(f'{e:.4f}' for e in events)}")
        print(f"# bands_hz: {', '.join(f'{lo:.0f}-{hi:.0f}' for lo, hi in args.bands)}")

    return 0 if overall_status in {"ok", "low_confidence"} else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("status: error", file=sys.stderr)
        print(f"reason: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())