from __future__ import annotations

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
    assert [preset.name for preset in presets] == ["quality", "quality_compile", "fast"]


def test_preset_from_args_uses_overrides():
    args = parse_args(["--preset", "custom", "--max-tokens", "1024", "--cfg-scale", "1.0"])
    preset = preset_from_args(args)
    assert preset.name == "custom"
    assert preset.max_tokens == 1024
    assert preset.cfg_scale == 1.0


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
            "tokens_generated": 50,
            "ms_per_token": 2.0,
            "realtime_factor": 0.5,
            "timings": {"gpu_peak_memory_bytes": 123},
        },
        {
            "generation_ms": 120.0,
            "generated_audio_ms": 240.0,
            "decode_ms": 24.0,
            "tokens_generated": 60,
            "ms_per_token": 2.0,
            "realtime_factor": 0.5,
            "timings": {"gpu_peak_memory_bytes": 456},
        },
    ]
    summary = summarize_runs(runs, 999.0)
    assert summary["model_load_time_ms"] == 999.0
    assert summary["average_realtime_factor"] == 0.5
    assert summary["max_gpu_memory_bytes"] == 456
