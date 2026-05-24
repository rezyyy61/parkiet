from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class SessionState(str, Enum):
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


@dataclass(slots=True)
class RealtimeTextInput:
    session_id: str
    text: str
    is_final: bool = False
    is_delta: bool = False
    received_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class RealtimePhrase:
    session_id: str
    index: int
    text: str
    voice_tag: str
    source_text: str
    is_final: bool = False
    queued_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class PhraseSynthesisMetrics:
    phrase_index: int
    phrase_text: str
    queued_at: datetime
    synthesis_started_at: datetime | None = None
    synthesis_finished_at: datetime | None = None
    generation_ms: float = 0.0
    audio_duration_ms: float = 0.0
    frame_count: int = 0


@dataclass(slots=True)
class SessionMetrics:
    accepted_text_chars: int = 0
    generated_phrase_count: int = 0
    generated_audio_ms: float = 0.0
    frames_read: int = 0
    buffer_level_ms: float = 0.0
    underrun_count: int = 0
    silence_frames_returned: int = 0
    generated_frames_returned: int = 0


@dataclass(slots=True)
class AudioChunk:
    session_id: str
    phrase_index: int
    text: str
    sample_rate: int
    waveform: np.ndarray
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AudioFrame:
    session_id: str
    sequence_number: int
    payload: bytes | np.ndarray
    sample_rate: int
    frame_duration_ms: int
    sample_format: str
    channels: int
    is_silence: bool = False
    is_final: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    buffer_level_ms: float = 0.0
    underrun_count: int = 0
    source_phrase_index: int | None = None


@dataclass(slots=True)
class RealtimeSynthesisResult:
    phrase: RealtimePhrase
    audio_chunk: AudioChunk
    metrics: PhraseSynthesisMetrics
