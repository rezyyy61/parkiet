from __future__ import annotations

from pathlib import Path

import numpy as np

from parkiet.realtime import RealtimeTTSConfig, RealtimeTTSEngine
from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.types import AudioChunk, RealtimePhrase, RealtimeSynthesisResult


class MockRealtimeBackend:
    def __call__(
        self,
        phrase: RealtimePhrase,
        config: RealtimeTTSConfig,
    ) -> RealtimeSynthesisResult:
        duration_seconds = max(0.35, min(1.6, len(phrase.source_text) / 42.0))
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
        return RealtimeSynthesisResult(phrase=phrase, audio_chunk=chunk)


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


def main() -> None:
    backend, backend_name = build_backend()
    config = RealtimeTTSConfig()
    engine = RealtimeTTSEngine(config=config, backend=backend)
    session = engine.create_session("smoke-session")

    inputs = [
        ("[S1] Goedemorgen allemaal. Dit is een eerste testzin voor de realtime laag.", False),
        ("We voegen nog een tweede Nederlandse zin toe, zodat de segmentatie netjes meerdere frases maakt.", False),
        ("En dit is het slot van de smoke test.", True),
    ]

    print(f"backend={backend_name}")
    for text, is_final in inputs:
        phrases = engine.accept_text(session.session_id, text, is_final=is_final)
        for phrase in phrases:
            print(f"phrase[{phrase.index}]={phrase.text}")

    results = engine.synthesize_pending(session.session_id)
    print(f"synthesized={len(results)}")

    frame_sizes: list[int] = []
    while True:
        frame = engine.read_audio_frame(session.session_id)
        if frame is None:
            break
        frame_sizes.append(frame.samples.shape[0])
        print(
            f"frame[{frame.frame_index}] samples={frame.samples.shape[0]} "
            f"phrase_index={frame.source_phrase_index}"
        )

    expected = config.samples_per_frame
    stable = bool(frame_sizes) and all(size == expected for size in frame_sizes)
    print(f"frames={len(frame_sizes)} expected_samples_per_frame={expected} stable={stable}")


if __name__ == "__main__":
    main()
