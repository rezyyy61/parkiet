from __future__ import annotations

import numpy as np


def waveform_to_pcm16(waveform: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def pcm16_frame_bytes(waveform: np.ndarray) -> bytes:
    return waveform_to_pcm16(waveform).tobytes()


def ensure_fixed_frame_size(samples: np.ndarray, samples_per_frame: int) -> np.ndarray:
    frame = np.asarray(samples, dtype=np.float32).reshape(-1)
    if frame.shape[0] == samples_per_frame:
        return frame
    if frame.shape[0] > samples_per_frame:
        return frame[:samples_per_frame]
    padded = np.zeros(samples_per_frame, dtype=np.float32)
    padded[: frame.shape[0]] = frame
    return padded
