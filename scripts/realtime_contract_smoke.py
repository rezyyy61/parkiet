from __future__ import annotations

from time import sleep

import numpy as np

from parkiet.realtime import RealtimeTTSConfig, RealtimeTTSEngine
from parkiet.realtime.types import AudioChunk, PhraseSynthesisMetrics, RealtimePhrase, RealtimeSynthesisResult


class SlowMockRealtimeBackend:
    def __init__(self, delay_s: float = 0.08):
        self.delay_s = delay_s

    def __call__(
        self,
        phrase: RealtimePhrase,
        config: RealtimeTTSConfig,
    ) -> RealtimeSynthesisResult:
        sleep(self.delay_s)
        duration_seconds = max(0.16, min(0.50, len(phrase.source_text) / 90.0))
        sample_count = int(config.internal_sample_rate * duration_seconds)
        t = np.linspace(0.0, duration_seconds, sample_count, endpoint=False, dtype=np.float32)
        waveform = (0.08 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32, copy=False)
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
            start_playback_when_buffer_ms=80,
            output_sample_format="pcm16",
            return_silence_when_empty=True,
            min_phrase_chars=10,
        ),
        backend=SlowMockRealtimeBackend(),
    )
    session = engine.create_session("contract-smoke")

    engine.accept_text(
        session.session_id,
        "Ja, dat kan ik voor je controleren. Een momentje alstublieft. Daarna kom ik bij je terug.",
        is_final=True,
    )
    engine.start_session_worker(session.session_id)

    while not engine.is_ready_for_playback(session.session_id):
        print("waiting for playback readiness...", engine.get_session_state(session.session_id))
        sleep(0.03)

    print("session state at start:", engine.get_session_state(session.session_id))
    while True:
        frame = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)
        if frame is None:
            break
        payload_size = len(frame.payload) if isinstance(frame.payload, bytes) else frame.payload.shape[0]
        print(
            f"frame seq={frame.sequence_number} size={payload_size} "
            f"format={frame.sample_format} silence={frame.is_silence} final={frame.is_final} "
            f"buffer_ms={frame.buffer_level_ms:.2f} underruns={frame.underrun_count}"
        )
        if frame.is_final:
            break

    print("metrics snapshot:", engine.get_session_metrics(session.session_id))
    print("state snapshot:", engine.get_session_state(session.session_id))

    engine.interrupt(session.session_id)
    stale = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)
    print(f"frame after interrupt={stale}")

    engine.stop_session_worker(session.session_id)
    engine.close_session(session.session_id)


if __name__ == "__main__":
    main()
