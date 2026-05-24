from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.types import RealtimePhrase


DEFAULT_PHRASES = [
    "[S1] Ja, dat kan ik voor je controleren.",
    "[S1] Een momentje alstublieft.",
    "[S1] Ik heb de gegevens gevonden.",
]


@dataclass(slots=True)
class BenchmarkPreset:
    name: str
    use_torch_compile: bool
    max_tokens: int
    cfg_scale: float
    temperature: float
    top_p: float
    cfg_filter_top_k: int
    disable_cfg: bool = False
    max_audio_seconds: float | None = None
    max_output_tokens_per_char: float | None = None
    hard_stop_after_tokens: int | None = None


QUALITY_PRESET = BenchmarkPreset(
    name="quality",
    use_torch_compile=False,
    max_tokens=3072,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
    disable_cfg=False,
)

QUALITY_BFLOAT16_PRESET = BenchmarkPreset(
    name="quality_bfloat16",
    use_torch_compile=False,
    max_tokens=3072,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
    disable_cfg=False,
)

FAST_NO_CFG_512_PRESET = BenchmarkPreset(
    name="fast_no_cfg_512",
    use_torch_compile=True,
    max_tokens=512,
    cfg_scale=1.0,
    temperature=1.2,
    top_p=0.85,
    cfg_filter_top_k=20,
    disable_cfg=True,
)

QUALITY_COMPILE_PRESET = BenchmarkPreset(
    name="quality_compile",
    use_torch_compile=True,
    max_tokens=QUALITY_PRESET.max_tokens,
    cfg_scale=QUALITY_PRESET.cfg_scale,
    temperature=QUALITY_PRESET.temperature,
    top_p=QUALITY_PRESET.top_p,
    cfg_filter_top_k=QUALITY_PRESET.cfg_filter_top_k,
    disable_cfg=False,
)

FAST_NO_CFG_256_PRESET = BenchmarkPreset(
    name="fast_no_cfg_256",
    use_torch_compile=True,
    max_tokens=256,
    cfg_scale=1.0,
    temperature=1.2,
    top_p=0.85,
    cfg_filter_top_k=20,
    disable_cfg=True,
)

FAST_NO_CFG_LIMITED_DURATION_PRESET = BenchmarkPreset(
    name="fast_no_cfg_limited_duration",
    use_torch_compile=True,
    max_tokens=512,
    cfg_scale=1.0,
    temperature=1.2,
    top_p=0.85,
    cfg_filter_top_k=20,
    disable_cfg=True,
    max_audio_seconds=2.0,
    max_output_tokens_per_char=6.0,
)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Dia inference speed for realtime phrase synthesis.")
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--checkpoint-path", default="weights/dia-nl-v1.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--text", action="append", dest="texts", default=None)
    parser.add_argument("--use-torch-compile", type=parse_bool, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--cfg-filter-top-k", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--disable-cfg", type=parse_bool, default=None)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--max-output-tokens-per-char", type=float, default=None)
    parser.add_argument("--hard-stop-after-tokens", type=int, default=None)
    parser.add_argument(
        "--preset",
        choices=["quality", "quality_bfloat16", "fast_no_cfg_512", "fast_no_cfg_256", "fast_no_cfg_limited_duration", "both", "custom"],
        default="both",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def resolve_texts(texts: list[str] | None) -> list[str]:
    return texts if texts else DEFAULT_PHRASES


def scalarize_singleton(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def preset_from_args(args: argparse.Namespace) -> BenchmarkPreset:
    return BenchmarkPreset(
        name="custom",
        use_torch_compile=args.use_torch_compile if args.use_torch_compile is not None else QUALITY_PRESET.use_torch_compile,
        max_tokens=args.max_tokens if args.max_tokens is not None else QUALITY_PRESET.max_tokens,
        cfg_scale=args.cfg_scale if args.cfg_scale is not None else QUALITY_PRESET.cfg_scale,
        temperature=args.temperature if args.temperature is not None else QUALITY_PRESET.temperature,
        top_p=args.top_p if args.top_p is not None else QUALITY_PRESET.top_p,
        cfg_filter_top_k=args.cfg_filter_top_k if args.cfg_filter_top_k is not None else QUALITY_PRESET.cfg_filter_top_k,
        disable_cfg=args.disable_cfg if args.disable_cfg is not None else False,
        max_audio_seconds=args.max_audio_seconds,
        max_output_tokens_per_char=args.max_output_tokens_per_char,
        hard_stop_after_tokens=args.hard_stop_after_tokens,
    )


def build_presets(args: argparse.Namespace) -> list[BenchmarkPreset]:
    if args.preset == "quality":
        return [QUALITY_PRESET]
    if args.preset == "quality_bfloat16":
        return [QUALITY_BFLOAT16_PRESET]
    if args.preset == "fast_no_cfg_512":
        return [FAST_NO_CFG_512_PRESET]
    if args.preset == "fast_no_cfg_256":
        return [FAST_NO_CFG_256_PRESET]
    if args.preset == "fast_no_cfg_limited_duration":
        return [FAST_NO_CFG_LIMITED_DURATION_PRESET]
    if args.preset == "custom":
        return [preset_from_args(args)]
    return [
        QUALITY_PRESET,
        QUALITY_COMPILE_PRESET,
        QUALITY_BFLOAT16_PRESET,
        FAST_NO_CFG_512_PRESET,
        FAST_NO_CFG_256_PRESET,
        FAST_NO_CFG_LIMITED_DURATION_PRESET,
    ]


def require_files(config_path: Path, checkpoint_path: Path) -> None:
    missing = [str(path) for path in (config_path, checkpoint_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Dia model files: "
            f"{', '.join(missing)}. Provide valid --config-path and --checkpoint-path."
        )


def phrase_item(index: int, text: str, total_phrases: int) -> RealtimePhrase:
    return RealtimePhrase(
        session_id="dia-speed-benchmark",
        index=index,
        text=text,
        voice_tag="[S1]",
        source_text=text,
        is_final=index == total_phrases - 1,
    )


def run_single_backend_benchmark(
    backend: DiaRealtimeBackend,
    texts: list[str],
    *,
    warmup_runs: int,
    repeat: int,
) -> dict[str, object]:
    for warmup_index in range(warmup_runs):
        for phrase_index, text in enumerate(texts):
            backend(phrase_item(phrase_index, text, len(texts)), None)  # type: ignore[arg-type]

    run_metrics: list[dict[str, object]] = []
    for repeat_index in range(repeat):
        for phrase_index, text in enumerate(texts):
            phrase = phrase_item(phrase_index, text, len(texts))
            started = perf_counter()
            result = backend(phrase, None)  # type: ignore[arg-type]
            wall_ms = (perf_counter() - started) * 1000.0
            timings = dict(result.audio_chunk.metadata.get("timings", {}))
            generated_audio_ms = result.metrics.audio_duration_ms
            tokens_generated = int(timings.get("generated_tokens", 0) or 0)
            generation_ms = float(timings.get("total_ms", wall_ms) or wall_ms)
            decode_ms = float(timings.get("dac_decode_ms", 0.0) or 0.0)
            decoder_loop_ms = float(timings.get("decoder_loop_ms", 0.0) or 0.0)
            realtime_factor = generation_ms / generated_audio_ms if generated_audio_ms > 0.0 else float("inf")
            ms_per_token = generation_ms / tokens_generated if tokens_generated > 0 else float("inf")
            run_metrics.append(
                {
                    "repeat_index": repeat_index,
                    "phrase_index": phrase_index,
                    "phrase_text": text,
                    "generated_audio_ms": generated_audio_ms,
                    "generation_ms": generation_ms,
                    "decode_ms": decode_ms,
                    "decoder_loop_ms": decoder_loop_ms,
                    "tokens_generated": tokens_generated,
                    "ms_per_token": ms_per_token,
                    "realtime_factor": realtime_factor,
                    "decoder_step_calls": int(timings.get("decoder_step_calls", 0) or 0),
                    "eos_detected": scalarize_singleton(timings.get("eos_detected")),
                    "eos_step": scalarize_singleton(timings.get("eos_step")),
                    "stop_step": scalarize_singleton(timings.get("stop_step")),
                    "stop_reason": scalarize_singleton(timings.get("stop_reason")),
                    "effective_audio_duration_ms": scalarize_singleton(timings.get("effective_audio_duration_ms")),
                    "max_tokens": int(timings.get("max_tokens", 0) or 0),
                    "disable_cfg": bool(timings.get("disable_cfg", False)),
                    "timings": timings,
                }
            )
    return summarize_runs(run_metrics, backend.model_load_time_ms)


def summarize_runs(run_metrics: list[dict[str, object]], model_load_time_ms: float | None) -> dict[str, object]:
    generation_values = [float(item["generation_ms"]) for item in run_metrics]
    audio_values = [float(item["generated_audio_ms"]) for item in run_metrics]
    rtf_values = [float(item["realtime_factor"]) for item in run_metrics]
    decode_values = [float(item["decode_ms"]) for item in run_metrics]
    decoder_loop_values = [float(item["decoder_loop_ms"]) for item in run_metrics]
    token_values = [int(item["tokens_generated"]) for item in run_metrics]
    decoder_step_calls = [int(item["decoder_step_calls"]) for item in run_metrics]
    ms_per_token_values = [float(item["ms_per_token"]) for item in run_metrics if float(item["ms_per_token"]) != float("inf")]
    peak_memory_values = [
        int(item["timings"].get("gpu_peak_memory_bytes") or 0)  # type: ignore[index]
        for item in run_metrics
    ]
    stop_reasons = [item["stop_reason"] for item in run_metrics]
    eos_detected_values = [item["eos_detected"] for item in run_metrics]

    return {
        "model_load_time_ms": model_load_time_ms,
        "run_count": len(run_metrics),
        "generated_audio_ms_total": sum(audio_values),
        "generation_ms_total": sum(generation_values),
        "average_generated_audio_ms": mean(audio_values) if audio_values else 0.0,
        "average_generation_ms": mean(generation_values) if generation_values else 0.0,
        "average_realtime_factor": mean(rtf_values) if rtf_values else float("inf"),
        "max_realtime_factor": max(rtf_values) if rtf_values else float("inf"),
        "average_decode_ms": mean(decode_values) if decode_values else 0.0,
        "average_decoder_loop_ms": mean(decoder_loop_values) if decoder_loop_values else 0.0,
        "total_tokens_generated": sum(token_values),
        "average_tokens_generated": mean(token_values) if token_values else 0.0,
        "average_decoder_step_calls": mean(decoder_step_calls) if decoder_step_calls else 0.0,
        "average_ms_per_token": mean(ms_per_token_values) if ms_per_token_values else float("inf"),
        "max_gpu_memory_bytes": max(peak_memory_values) if peak_memory_values else 0,
        "stop_reasons": stop_reasons,
        "eos_detected": eos_detected_values,
        "per_run": run_metrics,
    }


def build_backend(
    args: argparse.Namespace,
    preset: BenchmarkPreset,
) -> DiaRealtimeBackend:
    require_files(Path(args.config_path), Path(args.checkpoint_path))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return DiaRealtimeBackend.from_local_paths(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        compute_dtype=args.compute_dtype,
        device=device,
        use_torch_compile=preset.use_torch_compile,
        max_tokens=preset.max_tokens,
        cfg_scale=preset.cfg_scale,
        temperature=preset.temperature,
        top_p=preset.top_p,
        cfg_filter_top_k=preset.cfg_filter_top_k,
        disable_cfg=preset.disable_cfg,
        max_audio_seconds=preset.max_audio_seconds,
        max_output_tokens_per_char=preset.max_output_tokens_per_char,
        hard_stop_after_tokens=preset.hard_stop_after_tokens,
        collect_timings=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    texts = resolve_texts(args.texts)
    results: dict[str, object] = {
        "device": args.device,
        "compute_dtype": args.compute_dtype,
        "texts": texts,
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "presets": {},
    }

    for preset in build_presets(args):
        print(f"Running preset: {preset.name}")
        backend = build_backend(args, preset)
        summary = run_single_backend_benchmark(
            backend,
            texts,
            warmup_runs=args.warmup_runs,
            repeat=args.repeat,
        )
        results["presets"][preset.name] = {
            "config": asdict(preset),
            "summary": summary,
        }
        print(json.dumps({"preset": preset.name, "summary": summary}, ensure_ascii=False, indent=2))

    if "quality" in results["presets"] and "quality_compile" in results["presets"]:
        quality = results["presets"]["quality"]["summary"]  # type: ignore[index]
        quality_compile = results["presets"]["quality_compile"]["summary"]  # type: ignore[index]
        compile_helped = float(quality_compile["average_generation_ms"]) < float(quality["average_generation_ms"])  # type: ignore[index]
        results["comparison"] = {
            "compile_helped": compile_helped,
            "quality_average_rtf": quality["average_realtime_factor"],
            "quality_compile_average_rtf": quality_compile["average_realtime_factor"],
            "quality_average_generation_ms": quality["average_generation_ms"],
            "quality_compile_average_generation_ms": quality_compile["average_generation_ms"],
        }
    if "quality_bfloat16" in results["presets"] and "fast_no_cfg_512" in results["presets"]:
        quality = results["presets"]["quality_bfloat16"]["summary"]  # type: ignore[index]
        fast = results["presets"]["fast_no_cfg_512"]["summary"]  # type: ignore[index]
        results["fast_preset_comparison"] = {
            "quality_average_rtf": quality["average_realtime_factor"],
            "fast_average_rtf": fast["average_realtime_factor"],
            "quality_average_generation_ms": quality["average_generation_ms"],
            "fast_average_generation_ms": fast["average_generation_ms"],
            "disable_cfg_reduced_decoder_time": float(fast["average_decoder_loop_ms"]) < float(quality["average_decoder_loop_ms"]),
        }
        print(json.dumps(results["comparison"], ensure_ascii=False, indent=2))
        print(json.dumps(results["fast_preset_comparison"], ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
