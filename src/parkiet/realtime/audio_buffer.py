from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

from .config import RealtimeTTSConfig
from .frame_encoder import ensure_fixed_frame_size
from .types import AudioChunk


@dataclass(slots=True)
class BufferedAudioFrame:
    samples: np.ndarray
    is_silence: bool
    source_phrase_index: int | None


class AudioFrameBuffer:
    def __init__(self, config: RealtimeTTSConfig):
        self.config = config
        self._buffer = np.zeros(0, dtype=np.float32)
        self._phrase_index_by_frame: deque[int | None] = deque()

    def push_chunk(self, chunk: AudioChunk) -> int:
        waveform = np.asarray(chunk.waveform, dtype=np.float32).reshape(-1)
        if chunk.sample_rate != self.config.output_sample_rate:
            waveform = self._resample(waveform, chunk.sample_rate, self.config.output_sample_rate)
        if waveform.size == 0:
            return 0

        self._buffer = np.concatenate([self._buffer, waveform])
        frame_count = int(np.ceil(waveform.size / self.config.samples_per_frame))
        self._phrase_index_by_frame.extend([chunk.phrase_index] * frame_count)
        return frame_count

    def has_buffered_audio(self) -> bool:
        return self._buffer.size > 0

    def available_frames(self) -> int:
        if self._buffer.size == 0:
            return 0
        return int(np.ceil(self._buffer.size / self.config.samples_per_frame))

    def available_complete_frames(self) -> int:
        return self._buffer.size // self.config.samples_per_frame

    def buffer_level_ms(self) -> float:
        if self._buffer.size == 0:
            return 0.0
        return 1000.0 * float(self._buffer.size) / float(self.config.output_sample_rate)

    def has_prebuffer(self) -> bool:
        return self.buffer_level_ms() >= self.config.prebuffer_ms

    def read_frame(
        self,
        *,
        return_silence_when_empty: bool | None = None,
    ) -> BufferedAudioFrame | None:
        use_silence = (
            self.config.return_silence_when_empty
            if return_silence_when_empty is None
            else return_silence_when_empty
        )
        frame_size = self.config.samples_per_frame

        if self._buffer.size >= frame_size:
            frame_samples = self._buffer[:frame_size]
            self._buffer = self._buffer[frame_size:]
            source_phrase_index = self._phrase_index_by_frame.popleft() if self._phrase_index_by_frame else None
            return BufferedAudioFrame(
                samples=ensure_fixed_frame_size(frame_samples, frame_size),
                is_silence=False,
                source_phrase_index=source_phrase_index,
            )

        if self._buffer.size > 0:
            frame_samples = ensure_fixed_frame_size(self._buffer, frame_size)
            self._buffer = np.zeros(0, dtype=np.float32)
            source_phrase_index = self._phrase_index_by_frame.popleft() if self._phrase_index_by_frame else None
            self._phrase_index_by_frame.clear()
            return BufferedAudioFrame(
                samples=frame_samples,
                is_silence=False,
                source_phrase_index=source_phrase_index,
            )

        if not use_silence:
            return None

        return BufferedAudioFrame(
            samples=np.zeros(frame_size, dtype=np.float32),
            is_silence=True,
            source_phrase_index=None,
        )

    def flush(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._phrase_index_by_frame.clear()

    def _resample(self, waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        tensor = torch.from_numpy(waveform).reshape(1, -1)
        resampled = torchaudio.functional.resample(tensor, source_rate, target_rate)
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
