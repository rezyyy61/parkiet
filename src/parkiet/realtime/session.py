from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from .types import (
    AudioChunk,
    PhraseSynthesisMetrics,
    RealtimePhrase,
    RealtimeTextInput,
    SessionMetrics,
    SessionState,
)


@dataclass(slots=True)
class RealtimeTTSSession:
    session_id: str
    state: SessionState = SessionState.ACTIVE
    pending_text: str = ""
    phrase_queue: deque[RealtimePhrase] = field(default_factory=deque)
    generated_audio_queue: deque[AudioChunk] = field(default_factory=deque)
    phrase_metrics: dict[int, PhraseSynthesisMetrics] = field(default_factory=dict)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    interrupt_requested: bool = False
    is_worker_running: bool = False
    is_synthesizing: bool = False
    current_phrase_id: int | None = None
    next_phrase_index: int = 0
    next_frame_index: int = 0
    generation_epoch: int = 0
    final_input_received: bool = False
    completion_emitted: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_input_at: datetime | None = None
    last_synthesis_at: datetime | None = None
    last_frame_at: datetime | None = None
    last_frame_read_at: datetime | None = None

    @property
    def queued_text(self) -> str:
        return self.pending_text

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()

    def append_input(self, text_input: RealtimeTextInput) -> None:
        normalized_text = text_input.text.strip()
        self.final_input_received = text_input.is_final
        self.completion_emitted = False
        if not normalized_text:
            self.last_input_at = text_input.received_at
            self.touch()
            return
        self.pending_text = (
            f"{self.pending_text} {normalized_text}".strip() if self.pending_text else normalized_text
        )
        self.metrics.accepted_text_chars += len(text_input.text)
        self.last_input_at = text_input.received_at
        self.touch()

    def enqueue_phrases(self, phrases: list[RealtimePhrase], remainder: str) -> None:
        self.phrase_queue.extend(phrases)
        self.pending_text = remainder
        for phrase in phrases:
            self.phrase_metrics[phrase.index] = PhraseSynthesisMetrics(
                phrase_index=phrase.index,
                phrase_text=phrase.text,
                queued_at=phrase.queued_at,
            )
        self.next_phrase_index += len(phrases)
        self.touch()

    def mark_synthesis_started(self, phrase: RealtimePhrase, started_at: datetime) -> None:
        metrics = self.phrase_metrics[phrase.index]
        metrics.synthesis_started_at = started_at
        self.is_synthesizing = True
        self.current_phrase_id = phrase.index
        self.touch()

    def enqueue_audio(self, chunk: AudioChunk, generation_ms: float, frame_count: int) -> None:
        self.generated_audio_queue.append(chunk)
        metrics = self.phrase_metrics[chunk.phrase_index]
        metrics.synthesis_finished_at = chunk.generated_at
        metrics.generation_ms = generation_ms
        metrics.audio_duration_ms = 1000.0 * float(chunk.waveform.shape[0]) / float(chunk.sample_rate)
        metrics.frame_count = frame_count

        self.metrics.generated_phrase_count += 1
        self.metrics.generated_audio_ms += metrics.audio_duration_ms
        self.last_synthesis_at = chunk.generated_at
        self.is_synthesizing = False
        self.current_phrase_id = None
        self.touch()

    def note_buffer_level_ms(self, buffer_level_ms: float) -> None:
        self.metrics.buffer_level_ms = buffer_level_ms
        self.touch()

    def note_frame_read(self, read_at: datetime, buffer_level_ms: float, *, is_silence: bool) -> None:
        self.metrics.frames_read += 1
        self.metrics.buffer_level_ms = buffer_level_ms
        self.last_frame_at = read_at
        self.last_frame_read_at = read_at
        if is_silence:
            self.metrics.silence_frames_returned += 1
        else:
            self.metrics.generated_frames_returned += 1
        self.touch()

    def note_underrun(self) -> None:
        self.metrics.underrun_count += 1
        self.touch()

    def note_synthesis_finished_without_audio(self) -> None:
        self.is_synthesizing = False
        self.current_phrase_id = None
        self.touch()

    def set_worker_running(self, is_running: bool) -> None:
        self.is_worker_running = is_running
        self.touch()

    def has_pending_work(self) -> bool:
        return bool(self.pending_text or self.phrase_queue or self.is_synthesizing or self.current_phrase_id is not None)

    def request_interrupt(self) -> None:
        self.interrupt_requested = True
        self.generation_epoch += 1
        self.state = SessionState.INTERRUPTED
        self.touch()

    def clear_pending(self) -> None:
        self.pending_text = ""
        self.phrase_queue.clear()
        self.generated_audio_queue.clear()
        self.metrics.buffer_level_ms = 0.0
        self.is_synthesizing = False
        self.current_phrase_id = None
        self.final_input_received = False
        self.completion_emitted = False
        self.touch()

    def reactivate(self) -> None:
        self.interrupt_requested = False
        self.state = SessionState.ACTIVE
        self.completion_emitted = False
        self.touch()

    def close(self) -> None:
        self.generation_epoch += 1
        self.clear_pending()
        self.state = SessionState.CLOSED
        self.touch()
