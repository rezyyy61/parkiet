from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from parkiet.realtime import (
    BackendSynthesisError,
    RealtimeTTSConfig,
    RealtimeTTSEngine,
    SessionClosedError,
    SessionNotFoundError,
)
from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.frame_encoder import ensure_fixed_frame_size, waveform_to_pcm16
from parkiet.realtime.types import AudioChunk, PhraseSynthesisMetrics, RealtimePhrase, RealtimeSynthesisResult, SessionState


class StubBackend:
    def __call__(
        self,
        phrase: RealtimePhrase,
        config: RealtimeTTSConfig,
    ) -> RealtimeSynthesisResult:
        duration = 0.06 + 0.01 * phrase.index
        sample_count = int(config.internal_sample_rate * duration)
        waveform = np.linspace(-0.2, 0.2, sample_count, dtype=np.float32)
        chunk = AudioChunk(
            session_id=phrase.session_id,
            phrase_index=phrase.index,
            text=phrase.text,
            sample_rate=config.internal_sample_rate,
            waveform=waveform,
        )
        return RealtimeSynthesisResult(
            phrase=phrase,
            audio_chunk=chunk,
            metrics=PhraseSynthesisMetrics(
                phrase_index=phrase.index,
                phrase_text=phrase.text,
                queued_at=phrase.queued_at,
            ),
        )


class SlowStubBackend:
    def __init__(self, delay_s: float = 0.05):
        self.delay_s = delay_s

    def __call__(
        self,
        phrase: RealtimePhrase,
        config: RealtimeTTSConfig,
    ) -> RealtimeSynthesisResult:
        time.sleep(self.delay_s)
        sample_count = int(config.internal_sample_rate * 0.10)
        waveform = np.full(sample_count, 0.1, dtype=np.float32)
        chunk = AudioChunk(
            session_id=phrase.session_id,
            phrase_index=phrase.index,
            text=phrase.text,
            sample_rate=config.internal_sample_rate,
            waveform=waveform,
        )
        return RealtimeSynthesisResult(
            phrase=phrase,
            audio_chunk=chunk,
            metrics=PhraseSynthesisMetrics(
                phrase_index=phrase.index,
                phrase_text=phrase.text,
                queued_at=phrase.queued_at,
            ),
        )


class FailingBackend:
    def __call__(self, phrase: RealtimePhrase, config: RealtimeTTSConfig) -> RealtimeSynthesisResult:
        raise RuntimeError("backend failed")


def build_engine(**config_overrides) -> RealtimeTTSEngine:
    config = RealtimeTTSConfig(return_silence_when_empty=False, **config_overrides)
    return RealtimeTTSEngine(config=config, backend=StubBackend())


def build_slow_engine(**config_overrides) -> RealtimeTTSEngine:
    config = RealtimeTTSConfig(**config_overrides)
    return RealtimeTTSEngine(config=config, backend=SlowStubBackend())


def test_complete_text_becomes_expected_phrases():
    engine = build_engine(min_phrase_chars=10)
    session = engine.create_session("test")

    phrases = engine.accept_text(
        session.session_id,
        "Ja, dat kan ik voor je controleren. Een momentje alstublieft.",
        is_final=True,
    )

    assert [phrase.text for phrase in phrases] == [
        "[S1] Ja, dat kan ik voor je controleren.",
        "[S1] Een momentje alstublieft.",
    ]
    assert session.pending_text == ""


def test_incremental_text_accumulates_until_final_boundary():
    engine = build_engine(min_phrase_chars=10)
    session = engine.create_session("test")

    phrases_a = engine.accept_text_delta(session.session_id, "Ja, dat kan ik ", is_final=False)
    phrases_b = engine.accept_text_delta(session.session_id, "voor je controleren.", is_final=True)

    assert phrases_a == []
    assert [phrase.text for phrase in phrases_b] == ["[S1] Ja, dat kan ik voor je controleren."]
    assert session.pending_text == ""


def test_audio_frame_metadata_completeness_and_pcm16_output():
    engine = build_engine(output_sample_rate=16000, output_sample_format="pcm16", min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Dit is een contracttest zin.", is_final=True)
    engine.synthesize_pending(session.session_id)

    frame = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)

    assert frame is not None
    assert frame.session_id == session.session_id
    assert frame.sequence_number == 0
    assert isinstance(frame.payload, bytes)
    assert frame.sample_rate == 16000
    assert frame.frame_duration_ms == 20
    assert frame.sample_format == "pcm16"
    assert frame.channels == 1
    assert isinstance(frame.buffer_level_ms, float)
    assert isinstance(frame.underrun_count, int)


def test_float32_output_format_is_supported():
    engine = build_engine(output_sample_rate=16000, output_sample_format="float32", min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Dit is float output.", is_final=True)
    engine.synthesize_pending(session.session_id)

    frame = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)

    assert frame is not None
    assert frame.sample_format == "float32"
    assert isinstance(frame.payload, np.ndarray)
    assert frame.payload.dtype == np.float32
    assert frame.payload.shape == (320,)


def test_sequence_number_is_monotonic_and_continues_after_interrupt():
    engine = build_engine(min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Eerste zin.", is_final=True)
    engine.synthesize_pending(session.session_id)
    first = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)
    assert first is not None

    engine.interrupt(session.session_id)
    engine.accept_text(session.session_id, "Tweede zin.", is_final=True)
    engine.synthesize_pending(session.session_id)
    second = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)

    assert second is not None
    assert second.sequence_number > first.sequence_number


def test_silence_frame_is_not_treated_as_final():
    engine = build_slow_engine(return_silence_when_empty=True, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Trage synthese.", is_final=True)
    engine.start_session_worker(session.session_id)

    frame = engine.read_audio_frame(session.session_id)
    engine.stop_session_worker(session.session_id)

    assert frame is not None
    assert frame.is_silence is True
    assert frame.is_final is False


def test_final_frame_completion_behavior():
    engine = build_engine(output_sample_rate=16000, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Laatste frase.", is_final=True)
    engine.synthesize_pending(session.session_id)

    frames = []
    while True:
        frame = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)
        if frame is None:
            break
        frames.append(frame)

    assert frames
    assert frames[-1].is_final is True
    assert frames[-1].is_silence is False
    assert engine.read_audio_frame(session.session_id, return_silence_when_empty=False) is None


def test_get_session_metrics_snapshot():
    engine = build_engine(min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Eerste stuk.", is_final=True)
    engine.synthesize_pending(session.session_id)
    engine.read_audio_frame(session.session_id, return_silence_when_empty=False)

    metrics = engine.get_session_metrics(session.session_id)

    assert metrics["accepted_text_chars"] > 0
    assert metrics["generated_phrase_count"] == 1
    assert "is_worker_running" in metrics
    assert "queued_phrase_count" in metrics


def test_get_session_state_snapshot():
    engine = build_engine(min_phrase_chars=10)
    session = engine.create_session("test")
    state_before = engine.get_session_state(session.session_id)
    engine.accept_text(session.session_id, "Dit wordt uitgesproken.", is_final=True)
    state_after = engine.get_session_state(session.session_id)

    assert state_before["session_id"] == session.session_id
    assert state_before["is_active"] is True
    assert state_after["has_inflight_work"] is True
    assert "is_ready_for_playback" in state_after


def test_background_worker_processes_phrase_queue():
    engine = build_slow_engine(return_silence_when_empty=False, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Eerste zin. Tweede zin.", is_final=True)

    engine.start_session_worker(session.session_id)
    deadline = time.time() + 1.0
    while time.time() < deadline and engine.get_session_metrics(session.session_id)["generated_phrase_count"] < 2:
        time.sleep(0.01)

    engine.stop_session_worker(session.session_id)

    assert engine.get_session_metrics(session.session_id)["generated_phrase_count"] == 2
    assert session.is_worker_running is False


def test_read_audio_frame_does_not_block_during_synthesis():
    engine = build_slow_engine(return_silence_when_empty=True, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Trage synthese zin.", is_final=True)
    engine.start_session_worker(session.session_id)

    started = time.perf_counter()
    frame = engine.read_audio_frame(session.session_id)
    elapsed = time.perf_counter() - started

    engine.stop_session_worker(session.session_id)

    assert frame is not None
    assert elapsed < 0.03


def test_underrun_count_increments_when_silence_is_returned_while_work_is_pending():
    engine = build_slow_engine(return_silence_when_empty=True, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Dit wordt langzaam gemaakt.", is_final=True)
    engine.start_session_worker(session.session_id)

    frame = engine.read_audio_frame(session.session_id)

    engine.stop_session_worker(session.session_id)

    assert frame is not None
    assert frame.is_silence is True
    assert engine.get_session_metrics(session.session_id)["underrun_count"] >= 1


def test_is_ready_for_playback_respects_prebuffer():
    engine = build_engine(
        output_sample_rate=16000,
        frame_duration_ms=20,
        start_playback_when_buffer_ms=80,
        min_phrase_chars=10,
    )
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Voorbuffer testzin.", is_final=True)

    assert engine.is_ready_for_playback(session.session_id) is False
    engine.synthesize_pending(session.session_id)
    assert engine.is_ready_for_playback(session.session_id) is True


def test_interrupt_clears_runtime_state_during_background_work():
    engine = build_slow_engine(return_silence_when_empty=False, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Eerste zin. Tweede zin.", is_final=True)
    engine.start_session_worker(session.session_id)
    time.sleep(0.01)

    engine.interrupt(session.session_id)
    time.sleep(0.08)
    engine.stop_session_worker(session.session_id)

    assert session.state == SessionState.INTERRUPTED
    assert session.pending_text == ""
    assert not session.phrase_queue
    assert engine.get_buffer_level_ms(session.session_id) == 0.0


def test_close_session_stops_worker_cleanly():
    engine = build_slow_engine(return_silence_when_empty=False, min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Sluit worker netjes af.", is_final=True)
    engine.start_session_worker(session.session_id)

    engine.close_session(session.session_id)

    assert engine.get_session("test") is None


def test_synchronous_synthesize_pending_still_works():
    engine = build_engine(min_phrase_chars=10)
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Nog steeds synchroon.", is_final=True)

    results = engine.synthesize_pending(session.session_id)

    assert len(results) == 1
    assert engine.get_session_metrics(session.session_id)["generated_phrase_count"] == 1


def test_missing_session_errors_are_clear():
    engine = build_engine()
    with pytest.raises(SessionNotFoundError):
        engine.get_session_state("missing")


def test_closed_session_errors_are_clear():
    engine = build_engine()
    engine.create_session("test")
    engine.close_session("test")

    with pytest.raises(SessionNotFoundError):
        engine.read_audio_frame("test")


def test_backend_synthesis_failure_is_wrapped():
    engine = RealtimeTTSEngine(config=RealtimeTTSConfig(return_silence_when_empty=False), backend=FailingBackend())
    session = engine.create_session("test")
    engine.accept_text(session.session_id, "Dit faalt.", is_final=True)

    with pytest.raises(BackendSynthesisError):
        engine.synthesize_pending(session.session_id)


def test_pcm_helpers_produce_expected_shapes():
    waveform = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    pcm = waveform_to_pcm16(waveform)

    assert pcm.dtype == np.int16
    assert pcm.shape == waveform.shape
    assert pcm[0] == -32767
    assert pcm[-1] == 32767

    fixed = ensure_fixed_frame_size(np.array([0.1, 0.2], dtype=np.float32), 5)
    assert fixed.shape == (5,)


def test_dia_backend_uses_complete_phrase_generation():
    class FakeDia:
        def __init__(self):
            self.calls = []

        def generate(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return np.zeros(4410, dtype=np.float32)

    model = FakeDia()
    backend = DiaRealtimeBackend(model)
    phrase = RealtimePhrase(
        session_id="session",
        index=3,
        text="[S1] Volledige frase.",
        voice_tag="[S1]",
        source_text="Volledige frase.",
        is_final=True,
    )

    result = backend(phrase, RealtimeTTSConfig())

    assert result.audio_chunk.sample_rate == 44100
    assert model.calls[0][0] == phrase.text
    assert model.calls[0][1]["stream_callback"] is None


def test_offline_dia_package_has_no_realtime_dependency():
    dia_root = Path("src/parkiet/dia")
    for path in dia_root.glob("*.py"):
        contents = path.read_text()
        assert "parkiet.realtime" not in contents
