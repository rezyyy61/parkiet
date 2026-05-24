from __future__ import annotations

import numpy as np

from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.types import RealtimePhrase
from scripts.dia_inference_speed_benchmark import build_presets, parse_args, parse_bool, preset_from_args, resolve_texts, summarize_runs


def test_parse_bool():
    assert parse_bool("true") is True
    assert parse_bool("0") is False


def test_parse_args_defaults():
    args = parse_args([])
    assert args.device == "cuda"
    assert args.compute_dtype == "float32"
    assert args.preset == "both"


def test_build_presets_both():
    args = parse_args([])
    presets = build_presets(args)
    assert [preset.name for preset in presets] == [
        "quality",
        "quality_compile",
        "quality_bfloat16",
        "fast_no_cfg_512",
        "fast_no_cfg_256",
        "fast_no_cfg_limited_duration",
    ]


def test_preset_from_args_uses_overrides():
    args = parse_args(
        [
            "--preset",
            "custom",
            "--max-tokens",
            "1024",
            "--cfg-scale",
            "1.0",
            "--disable-cfg",
            "true",
            "--max-audio-seconds",
            "2.0",
        ]
    )
    preset = preset_from_args(args)
    assert preset.name == "custom"
    assert preset.max_tokens == 1024
    assert preset.cfg_scale == 1.0
    assert preset.disable_cfg is True
    assert preset.max_audio_seconds == 2.0


def test_resolve_texts_defaults():
    texts = resolve_texts(None)
    assert texts
    assert texts[0].startswith("[S1]")


def test_summarize_runs():
    runs = [
        {
            "generation_ms": 100.0,
            "generated_audio_ms": 200.0,
            "decode_ms": 20.0,
            "decoder_loop_ms": 70.0,
            "tokens_generated": 50,
            "decoder_step_calls": 52,
            "ms_per_token": 2.0,
            "realtime_factor": 0.5,
            "eos_detected": [True],
            "eos_step": [45],
            "stop_step": [49],
            "stop_reason": ["eos"],
            "effective_audio_duration_ms": [200.0],
            "max_tokens": 256,
            "disable_cfg": True,
            "timings": {"gpu_peak_memory_bytes": 123},
        },
        {
            "generation_ms": 120.0,
            "generated_audio_ms": 240.0,
            "decode_ms": 24.0,
            "decoder_loop_ms": 84.0,
            "tokens_generated": 60,
            "decoder_step_calls": 61,
            "ms_per_token": 2.0,
            "realtime_factor": 0.5,
            "eos_detected": [False],
            "eos_step": [None],
            "stop_step": [60],
            "stop_reason": ["max_tokens"],
            "effective_audio_duration_ms": [240.0],
            "max_tokens": 256,
            "disable_cfg": True,
            "timings": {"gpu_peak_memory_bytes": 456},
        },
    ]
    summary = summarize_runs(runs, 999.0)
    assert summary["model_load_time_ms"] == 999.0
    assert summary["average_realtime_factor"] == 0.5
    assert summary["max_gpu_memory_bytes"] == 456
    assert summary["average_decoder_loop_ms"] == 77.0
    assert summary["average_decoder_step_calls"] == 56.5


def test_disable_cfg_flag_is_passed_safely():
    class FakeDia:
        def __init__(self):
            self.calls = []
            self.last_generate_metadata = {"stop_reason": ["max_tokens"], "eos_detected": [False]}

        def generate(self, text, **kwargs):
            self.calls.append(kwargs)
            return np.zeros(4410, dtype=np.float32)

    backend = DiaRealtimeBackend(
        FakeDia(),
        disable_cfg=True,
        max_audio_seconds=2.0,
        max_output_tokens_per_char=6.0,
        hard_stop_after_tokens=256,
    )
    phrase = RealtimePhrase(
        session_id="test",
        index=0,
        text="[S1] Hallo",
        voice_tag="[S1]",
        source_text="[S1] Hallo",
        is_final=True,
    )

    result = backend(phrase, None)  # type: ignore[arg-type]

    call = backend.model.calls[0]
    assert call["disable_cfg"] is True
    assert call["max_audio_seconds"] == 2.0
    assert call["max_output_tokens_per_char"] == 6.0
    assert call["hard_stop_after_tokens"] == 256
    assert result.audio_chunk.metadata["timings"]["stop_reason"] == ["max_tokens"]


def test_backend_defaults_preserve_existing_behavior_flags():
    class FakeDia:
        def __init__(self):
            self.calls = []
            self.last_generate_metadata = {}

        def generate(self, text, **kwargs):
            self.calls.append(kwargs)
            return np.zeros(4410, dtype=np.float32)

    backend = DiaRealtimeBackend(FakeDia())
    phrase = RealtimePhrase(
        session_id="test",
        index=0,
        text="[S1] Hallo",
        voice_tag="[S1]",
        source_text="[S1] Hallo",
        is_final=True,
    )

    backend(phrase, None)  # type: ignore[arg-type]

    call = backend.model.calls[0]
    assert call["disable_cfg"] is False
    assert call["max_audio_seconds"] is None
    assert call["max_output_tokens_per_char"] is None
    assert call["hard_stop_after_tokens"] is None
