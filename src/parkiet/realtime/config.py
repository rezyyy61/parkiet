from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class RealtimeTTSConfig:
    internal_sample_rate: int = 44100
    output_sample_rate: int = 16000
    frame_duration_ms: int = 20
    min_phrase_chars: int = 24
    max_phrase_chars: int = 180
    prebuffer_ms: int = 120
    start_playback_when_buffer_ms: int = 120
    return_silence_when_empty: bool = True
    output_sample_format: Literal["pcm16", "float32"] = "pcm16"
    voice_tag: str = "[S1]"
    cfg_scale: float = 3.0
    temperature: float = 1.8
    top_p: float = 0.90
    cfg_filter_top_k: int = 50

    @property
    def samples_per_frame(self) -> int:
        return int(self.output_sample_rate * self.frame_duration_ms / 1000)

    @property
    def prebuffer_frames(self) -> int:
        return max(1, self.prebuffer_ms // self.frame_duration_ms)
