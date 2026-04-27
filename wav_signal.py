from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def decode_pcm(raw: bytes, sampwidth: int, channels: int) -> np.ndarray:
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2")
    elif sampwidth == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        data = (data ^ sign_bit) - sign_bit
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    return data.reshape(-1, channels)


def load_signal(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    pcm = decode_pcm(raw, sample_width, channels)
    signal = np.max(np.abs(pcm.astype(np.int64)), axis=1)
    return sample_rate, signal
