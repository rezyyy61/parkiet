from __future__ import annotations

from time import sleep

import numpy as np

from parkiet.realtime import RealtimeTTSConfig, RealtimeTTSEngine
from parkiet.realtime.types import AudioChunk, PhraseSynthesisMetrics, RealtimePhrase, RealtimeSynthesisResult


class SlowMockRealtimeBackend:
    def __init__(self, delay_s: float = 0.12):
        self.delay_s = delay_s

    def __call__(
        self,
        phrase: RealtimePhrase,
        config: RealtimeTTSConfig,
    ) -> RealtimeSynthesisResult:
        sleep(self.delay_s)
        duration_seconds = max(0.18, min(0.60, len(phrase.source_text) / 80.0))
        sample_count = int(config.internal_sample_rate * duration_seconds)
        t = np.linspace(0.0, duration_seconds, sample_count, endpoint=False, dtype=np.float32)
        waveform = (0.08 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32, copy=False)
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


def main() -> None:
    engine = RealtimeTTSEngine(
        config=RealtimeTTSConfig(
            output_sample_rate=16000,
            frame_duration_ms=20,
            prebuffer_ms=80,
            start_playback_when_buffer_ms=80,
            return_silence_when_empty=True,
            min_phrase_chars=10,
        ),
        backend=SlowMockRealtimeBackend(),
    )
    session = engine.create_session("background-smoke")

    phrases = engine.accept_text(
        session.session_id,
        "Ja, dat kan ik voor je controleren. Een momentje alstublieft. "
        "Ik kijk het meteen voor je na. Daarna kom ik bij je terug.",
        is_final=True,
    )
    print("queued phrases:")
    for phrase in phrases:
        print(f"  phrase[{phrase.index}] {phrase.text}")

    engine.start_session_worker(session.session_id)
    print(f"ready_for_playback_initial={engine.is_ready_for_playback(session.session_id)}")

    for _ in range(30):
        frame = engine.read_audio_frame(session.session_id)
        metrics = engine.get_session_metrics(session.session_id)
        print(
            f"frame={session.next_frame_index - 1} "
            f"silence={frame.is_silence if frame else None} "
            f"buffer_ms={engine.get_buffer_level_ms(session.session_id):.2f} "
            f"ready={engine.is_ready_for_playback(session.session_id)} "
            f"underruns={metrics.underrun_count} "
            f"generated_frames={metrics.generated_frames_returned} "
            f"silence_frames={metrics.silence_frames_returned}"
        )
        sleep(0.03)

    engine.stop_session_worker(session.session_id)
    print("phrase metrics:")
    for metrics in engine.get_phrase_metrics(session.session_id):
        print(
            f"  phrase[{metrics.phrase_index}] generation_ms={metrics.generation_ms:.2f} "
            f"audio_ms={metrics.audio_duration_ms:.2f} frames={metrics.frame_count}"
        )
    print(f"session_metrics={engine.get_session_metrics(session.session_id)}")


if __name__ == "__main__":
    main()
