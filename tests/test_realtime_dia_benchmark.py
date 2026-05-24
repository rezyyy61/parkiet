from __future__ import annotations

import numpy as np

from scripts.realtime_dia_benchmark import (
    build_phrase_benchmark_metrics,
    decode_frame_payload,
    parse_args,
    resolve_phrases,
)
from parkiet.realtime import RealtimeTTSConfig
from parkiet.realtime.types import PhraseSynthesisMetrics


def test_parse_args_defaults():
    args = parse_args([])

    assert args.config_path == "config.json"
    assert args.checkpoint_path == "weights/dia-nl-v1.pth"
    assert args.sample_format == "pcm16"
    assert args.output_sample_rate == 16000


def test_resolve_phrases_adds_voice_tag():
    phrases = resolve_phrases("Hallo daar.", "[S1]")

    assert phrases == ["[S1] Hallo daar."]


def test_decode_frame_payload_pcm16():
    samples = np.array([0, 32767, -32767], dtype=np.int16).tobytes()
    decoded = decode_frame_payload(samples, "pcm16")

    assert decoded.dtype == np.float32
    assert decoded.shape == (3,)


def test_decode_frame_payload_float32():
    payload = np.array([0.1, -0.2], dtype=np.float32)
    decoded = decode_frame_payload(payload, "float32")

    assert decoded.dtype == np.float32
    assert np.allclose(decoded, payload)


def test_build_phrase_benchmark_metrics_computes_realtime_factor():
    config = RealtimeTTSConfig(output_sample_rate=16000, frame_duration_ms=20, output_sample_format="pcm16")
    metrics = PhraseSynthesisMetrics(
        phrase_index=0,
        phrase_text="[S1] Hallo",
        queued_at=__import__("datetime").datetime.utcnow(),
        generation_ms=50.0,
        audio_duration_ms=100.0,
        frame_count=5,
    )
    session_metrics = {
        "buffer_level_ms": 40.0,
        "underrun_count": 1,
        "generated_frames_returned": 3,
        "silence_frames_returned": 1,
    }

    result = build_phrase_benchmark_metrics(
        metrics,
        session_metrics,
        config,
        first_frame_ready_ms=20.0,
        first_non_silence_frame_ms=30.0,
    )

    assert result.realtime_factor == 0.5
    assert result.output_sample_rate == 16000
    assert result.sample_format == "pcm16"
