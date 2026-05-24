from __future__ import annotations

from pathlib import Path

import numpy as np

from parkiet.realtime import RealtimeTTSConfig, RealtimeTTSEngine
from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.types import AudioChunk, PhraseSynthesisMetrics, RealtimePhrase, RealtimeSynthesisResult


class MockRealtimeBackend:
    def __call__(
        self,
        phrase: RealtimePhrase,
        config: RealtimeTTSConfig,
    ) -> RealtimeSynthesisResult:
        duration_seconds = max(0.30, min(1.50, len(phrase.source_text) / 40.0))
        sample_count = int(config.internal_sample_rate * duration_seconds)
        t = np.linspace(0.0, duration_seconds, sample_count, endpoint=False, dtype=np.float32)
        waveform = (0.08 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32, copy=False)
        chunk = AudioChunk(
            session_id=phrase.session_id,
            phrase_index=phrase.index,
            text=phrase.text,
            sample_rate=config.internal_sample_rate,
            waveform=waveform,
            metadata={"backend": "mock"},
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


def build_backend() -> tuple[object, str]:
    config_path = Path("config.json")
    checkpoint_path = Path("weights/dia-nl-v1.pth")
    if config_path.exists() and checkpoint_path.exists():
        try:
            return (
                DiaRealtimeBackend.from_local_paths(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                ),
                "dia",
            )
        except Exception as exc:
            print(f"Falling back to mock backend: {exc}")
    return MockRealtimeBackend(), "mock"


def print_phrases(label: str, phrases: list[RealtimePhrase]) -> None:
    print(label)
    for phrase in phrases:
        print(f"  phrase[{phrase.index}] {phrase.text}")


def main() -> None:
    backend, backend_name = build_backend()
    config = RealtimeTTSConfig(return_silence_when_empty=False)
    engine = RealtimeTTSEngine(config=config, backend=backend)
    session = engine.create_session("realtime-core-smoke")

    phrases_complete = engine.accept_text(
        session.session_id,
        "Ja, dat kan ik voor je controleren. Een momentje alstublieft.",
        is_final=True,
    )
    print(f"backend={backend_name}")
    print_phrases("complete-input phrases:", phrases_complete)

    phrases_delta_a = engine.accept_text_delta(session.session_id, "[S1] Ik kijk het na ", is_final=False)
    phrases_delta_b = engine.accept_text_delta(session.session_id, "en kom zo bij je terug.", is_final=True)
    print_phrases("delta-input phrases part 1:", phrases_delta_a)
    print_phrases("delta-input phrases part 2:", phrases_delta_b)

    results = engine.synthesize_pending(session.session_id)
    print(f"synthesized={len(results)}")
    for result in results:
        print(
            "  synthesized "
            f"phrase[{result.phrase.index}] generation_ms={result.metrics.generation_ms:.2f} "
            f"audio_ms={result.metrics.audio_duration_ms:.2f} frames={result.metrics.frame_count}"
        )

    frame_sizes: list[int] = []
    frames_read = 0
    while True:
        frame = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)
        if frame is None:
            break
        frames_read += 1
        frame_sizes.append(frame.samples.shape[0])
        print(
            f"frame[{frame.frame_index}] samples={frame.samples.shape[0]} "
            f"silence={frame.is_silence} phrase_index={frame.source_phrase_index} "
            f"buffer_ms={engine.get_buffer_level_ms(session.session_id):.2f}"
        )

    expected = config.samples_per_frame
    stable = bool(frame_sizes) and all(size == expected for size in frame_sizes)
    print(f"frames_read={frames_read} expected_samples_per_frame={expected} stable={stable}")
    print(f"session_metrics={engine.get_session_metrics(session.session_id)}")
    print("phrase_metrics=")
    for metrics in engine.get_phrase_metrics(session.session_id):
        print(
            f"  phrase[{metrics.phrase_index}] queued_at={metrics.queued_at.isoformat()} "
            f"generation_ms={metrics.generation_ms:.2f} audio_ms={metrics.audio_duration_ms:.2f} "
            f"frames={metrics.frame_count}"
        )


if __name__ == "__main__":
    main()
