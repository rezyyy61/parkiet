from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np
import soundfile as sf
import torch

from parkiet.realtime import RealtimeTTSConfig, RealtimeTTSEngine, SessionNotFoundError
from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.types import PhraseSynthesisMetrics


DEFAULT_PHRASES = [
    "Ja, dat kan ik voor je controleren.",
    "Een momentje alstublieft.",
    "Ik heb de gegevens gevonden.",
]


@dataclass(slots=True)
class PhraseBenchmarkMetrics:
    phrase_text: str
    synthesis_started_at: str | None
    synthesis_finished_at: str | None
    generation_ms: float
    audio_duration_ms: float
    realtime_factor: float
    first_frame_ready_ms: float | None
    first_non_silence_frame_ms: float | None
    buffer_level_ms: float
    underrun_count: int
    generated_frames_returned: int
    silence_frames_returned: int
    output_sample_rate: int
    frame_duration_ms: int
    sample_format: str


@dataclass(slots=True)
class BenchmarkPreset:
    name: str
    compute_dtype: str = "float32"
    use_torch_compile: bool = False
    disable_cfg: bool = False
    max_tokens: int = 3072
    cfg_scale: float = 3.0
    temperature: float = 1.8
    top_p: float = 0.90
    cfg_filter_top_k: int = 50
    max_audio_seconds: float | None = None
    max_output_tokens_per_char: float | None = None
    hard_stop_after_tokens: int | None = None


REALTIME_FAST_PRESET = BenchmarkPreset(
    name="realtime_fast",
    compute_dtype="bfloat16",
    use_torch_compile=True,
    disable_cfg=True,
    max_tokens=512,
    cfg_scale=1.0,
    temperature=1.2,
    top_p=0.85,
    cfg_filter_top_k=20,
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
    parser = argparse.ArgumentParser(description="Benchmark Parkiet realtime TTS with the actual Dia backend.")
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--checkpoint-path", default="weights/dia-nl-v1.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--preset", choices=["default", "realtime_fast"], default="default")
    parser.add_argument("--use-torch-compile", type=parse_bool, default=None)
    parser.add_argument("--disable-cfg", type=parse_bool, default=None)
    parser.add_argument("--compute-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--cfg-filter-top-k", type=int, default=None)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--max-output-tokens-per-char", type=float, default=None)
    parser.add_argument("--hard-stop-after-tokens", type=int, default=None)
    parser.add_argument("--output-sample-rate", type=int, choices=[44100, 16000, 8000], default=16000)
    parser.add_argument("--sample-format", choices=["pcm16", "float32"], default="pcm16")
    parser.add_argument("--frame-duration-ms", type=int, default=20)
    parser.add_argument("--text", default=None)
    parser.add_argument("--voice-tag", default="[S1]")
    parser.add_argument("--output-dir", default="debug_realtime_dia")
    parser.add_argument("--realtime-sleep", action="store_true")
    return parser.parse_args(argv)


def resolve_phrases(text: str | None, voice_tag: str) -> list[str]:
    phrases = [text] if text else DEFAULT_PHRASES
    return [phrase if phrase.strip().startswith("[") else f"{voice_tag} {phrase}" for phrase in phrases]


def require_model_files(config_path: Path, checkpoint_path: Path) -> None:
    missing = [str(path) for path in (config_path, checkpoint_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Dia model files for realtime benchmark: "
            f"{', '.join(missing)}. Provide valid --config-path and --checkpoint-path."
        )


def build_backend_options(args: argparse.Namespace) -> dict[str, object]:
    preset = REALTIME_FAST_PRESET if args.preset == "realtime_fast" else BenchmarkPreset(name="default")
    return {
        "compute_dtype": args.compute_dtype if args.compute_dtype is not None else preset.compute_dtype,
        "use_torch_compile": args.use_torch_compile if args.use_torch_compile is not None else preset.use_torch_compile,
        "disable_cfg": args.disable_cfg if args.disable_cfg is not None else preset.disable_cfg,
        "max_tokens": args.max_tokens if args.max_tokens is not None else preset.max_tokens,
        "cfg_scale": args.cfg_scale if args.cfg_scale is not None else preset.cfg_scale,
        "temperature": args.temperature if args.temperature is not None else preset.temperature,
        "top_p": args.top_p if args.top_p is not None else preset.top_p,
        "cfg_filter_top_k": args.cfg_filter_top_k if args.cfg_filter_top_k is not None else preset.cfg_filter_top_k,
        "max_audio_seconds": args.max_audio_seconds if args.max_audio_seconds is not None else preset.max_audio_seconds,
        "max_output_tokens_per_char": (
            args.max_output_tokens_per_char
            if args.max_output_tokens_per_char is not None
            else preset.max_output_tokens_per_char
        ),
        "hard_stop_after_tokens": (
            args.hard_stop_after_tokens
            if args.hard_stop_after_tokens is not None
            else preset.hard_stop_after_tokens
        ),
    }


def load_backend(
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
    backend_options: dict[str, object],
) -> DiaRealtimeBackend:
    require_model_files(config_path, checkpoint_path)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return DiaRealtimeBackend.from_local_paths(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        compute_dtype=str(backend_options["compute_dtype"]),
        device=torch.device(device_name),
        use_torch_compile=bool(backend_options["use_torch_compile"]),
        max_tokens=int(backend_options["max_tokens"]),
        cfg_scale=float(backend_options["cfg_scale"]),
        temperature=float(backend_options["temperature"]),
        top_p=float(backend_options["top_p"]),
        cfg_filter_top_k=int(backend_options["cfg_filter_top_k"]),
        disable_cfg=bool(backend_options["disable_cfg"]),
        max_audio_seconds=backend_options["max_audio_seconds"],
        max_output_tokens_per_char=backend_options["max_output_tokens_per_char"],
        hard_stop_after_tokens=backend_options["hard_stop_after_tokens"],
        collect_timings=True,
    )


def decode_frame_payload(payload: bytes | np.ndarray, sample_format: str) -> np.ndarray:
    if sample_format == "pcm16":
        if not isinstance(payload, bytes):
            raise TypeError("Expected PCM16 frame payload as bytes")
        if len(payload) == 0:
            return np.zeros(0, dtype=np.float32)
        pcm = np.frombuffer(payload, dtype=np.int16)
        return (pcm.astype(np.float32) / 32767.0).copy()
    if sample_format == "float32":
        if not isinstance(payload, np.ndarray):
            raise TypeError("Expected float32 frame payload as numpy array")
        return np.asarray(payload, dtype=np.float32).copy()
    raise ValueError(f"Unsupported sample format: {sample_format}")


def serialize_phrase_metrics(metrics: PhraseSynthesisMetrics) -> dict[str, Any]:
    return {
        "phrase_index": metrics.phrase_index,
        "phrase_text": metrics.phrase_text,
        "queued_at": metrics.queued_at.isoformat(),
        "synthesis_started_at": metrics.synthesis_started_at.isoformat() if metrics.synthesis_started_at else None,
        "synthesis_finished_at": metrics.synthesis_finished_at.isoformat() if metrics.synthesis_finished_at else None,
        "generation_ms": metrics.generation_ms,
        "audio_duration_ms": metrics.audio_duration_ms,
        "frame_count": metrics.frame_count,
    }


def build_phrase_benchmark_metrics(
    phrase_metrics: PhraseSynthesisMetrics,
    session_metrics: dict[str, Any],
    config: RealtimeTTSConfig,
    first_frame_ready_ms: float | None,
    first_non_silence_frame_ms: float | None,
) -> PhraseBenchmarkMetrics:
    realtime_factor = (
        phrase_metrics.generation_ms / phrase_metrics.audio_duration_ms
        if phrase_metrics.audio_duration_ms > 0.0
        else float("inf")
    )
    return PhraseBenchmarkMetrics(
        phrase_text=phrase_metrics.phrase_text,
        synthesis_started_at=phrase_metrics.synthesis_started_at.isoformat()
        if phrase_metrics.synthesis_started_at
        else None,
        synthesis_finished_at=phrase_metrics.synthesis_finished_at.isoformat()
        if phrase_metrics.synthesis_finished_at
        else None,
        generation_ms=phrase_metrics.generation_ms,
        audio_duration_ms=phrase_metrics.audio_duration_ms,
        realtime_factor=realtime_factor,
        first_frame_ready_ms=first_frame_ready_ms,
        first_non_silence_frame_ms=first_non_silence_frame_ms,
        buffer_level_ms=float(session_metrics["buffer_level_ms"]),
        underrun_count=int(session_metrics["underrun_count"]),
        generated_frames_returned=int(session_metrics["generated_frames_returned"]),
        silence_frames_returned=int(session_metrics["silence_frames_returned"]),
        output_sample_rate=config.output_sample_rate,
        frame_duration_ms=config.frame_duration_ms,
        sample_format=config.output_sample_format,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_options = build_backend_options(args)
    backend = load_backend(
        Path(args.config_path),
        Path(args.checkpoint_path),
        args.device,
        backend_options,
    )
    config = RealtimeTTSConfig(
        output_sample_rate=args.output_sample_rate,
        frame_duration_ms=args.frame_duration_ms,
        output_sample_format=args.sample_format,
        return_silence_when_empty=True,
        min_phrase_chars=10,
        prebuffer_ms=max(args.frame_duration_ms * 2, 40),
        start_playback_when_buffer_ms=max(args.frame_duration_ms * 2, 40),
    )
    engine = RealtimeTTSEngine(config=config, backend=backend)
    session = engine.create_session("realtime-dia-benchmark")

    phrases = resolve_phrases(args.text, args.voice_tag)
    for index, phrase in enumerate(phrases):
        engine.accept_text(session.session_id, phrase, is_final=index == len(phrases) - 1)

    engine.start_session_worker(session.session_id)
    benchmark_start = perf_counter()

    first_frame_ready_ms: float | None = None
    first_non_silence_frame_ms: float | None = None
    final_frame_count = 0
    sequence_numbers: list[int] = []
    silence_final_violation = False
    reconstructed_frames: list[np.ndarray] = []
    simulated_clock_ms = 0.0

    while True:
        frame = engine.read_audio_frame(session.session_id)
        now_ms = (perf_counter() - benchmark_start) * 1000.0
        if frame is not None:
            sequence_numbers.append(frame.sequence_number)
            if first_frame_ready_ms is None:
                first_frame_ready_ms = now_ms
            if not frame.is_silence and first_non_silence_frame_ms is None:
                first_non_silence_frame_ms = now_ms
            if frame.is_silence and frame.is_final:
                silence_final_violation = True
            if frame.is_final:
                final_frame_count += 1

            decoded = decode_frame_payload(frame.payload, frame.sample_format)
            if decoded.size > 0:
                reconstructed_frames.append(decoded)
            simulated_clock_ms += frame.frame_duration_ms
            if frame.is_final:
                break
        else:
            state = engine.get_session_state(session.session_id)
            if not state["has_inflight_work"] and not state["has_buffered_audio"]:
                break

        if args.realtime_sleep:
            sleep(config.frame_duration_ms / 1000.0)

    session_metrics = engine.get_session_metrics(session.session_id)
    phrase_metrics = engine.get_phrase_metrics(session.session_id)
    state_snapshot_before_close = engine.get_session_state(session.session_id)

    stale_before_close = engine.read_audio_frame(session.session_id, return_silence_when_empty=False)
    engine.stop_session_worker(session.session_id)

    full_audio = (
        np.concatenate(reconstructed_frames).astype(np.float32, copy=False)
        if reconstructed_frames
        else np.zeros(0, dtype=np.float32)
    )
    output_wav_path = output_dir / "full_generated_realtime_output.wav"
    sf.write(output_wav_path, full_audio, config.output_sample_rate)

    for metrics in phrase_metrics:
        matching_chunks = [chunk for chunk in session.generated_audio_queue if chunk.phrase_index == metrics.phrase_index]
        if matching_chunks:
            chunk = matching_chunks[-1]
            phrase_path = output_dir / f"per_phrase_{metrics.phrase_index + 1:03d}.wav"
            sf.write(phrase_path, np.asarray(chunk.waveform, dtype=np.float32), chunk.sample_rate)

    total_generation_ms = sum(metric.generation_ms for metric in phrase_metrics)
    total_generated_audio_ms = sum(metric.audio_duration_ms for metric in phrase_metrics)
    realtime_factors = [
        metric.generation_ms / metric.audio_duration_ms
        for metric in phrase_metrics
        if metric.audio_duration_ms > 0.0
    ]
    average_realtime_factor = sum(realtime_factors) / len(realtime_factors) if realtime_factors else float("inf")
    max_realtime_factor = max(realtime_factors) if realtime_factors else float("inf")
    monotonic_sequences = sequence_numbers == sorted(sequence_numbers) and len(sequence_numbers) == len(set(sequence_numbers))

    phrase_benchmark_metrics = [
        asdict(
            build_phrase_benchmark_metrics(
                metrics,
                session_metrics,
                config,
                first_frame_ready_ms,
                first_non_silence_frame_ms,
            )
        )
        for metrics in phrase_metrics
    ]

    engine.close_session(session.session_id)
    try:
        engine.get_session_state(session.session_id)
        stale_after_close_ok = False
    except SessionNotFoundError:
        stale_after_close_ok = True

    passed = all(
        [
            final_frame_count == 1,
            not silence_final_violation,
            monotonic_sequences,
            stale_before_close is None,
            stale_after_close_ok,
        ]
    )

    summary = {
        "status": "PASS" if passed else "FAIL",
        "preset": args.preset,
        "backend_options": backend_options,
        "total_generated_audio_ms": total_generated_audio_ms,
        "total_generation_ms": total_generation_ms,
        "average_realtime_factor": average_realtime_factor,
        "max_realtime_factor": max_realtime_factor,
        "first_audio_latency_ms": first_non_silence_frame_ms,
        "first_frame_ready_ms": first_frame_ready_ms,
        "total_underruns": session_metrics["underrun_count"],
        "total_frames": len(sequence_numbers),
        "output_wav_path": str(output_wav_path),
        "metadata_json_path": str(output_dir / "metadata.json"),
        "final_frame_count": final_frame_count,
        "sequence_numbers_monotonic": monotonic_sequences,
        "simulated_clock_ms": simulated_clock_ms,
    }

    metadata = {
        "args": vars(args),
        "summary": summary,
        "session_metrics": session_metrics,
        "session_state_before_close": state_snapshot_before_close,
        "phrase_metrics": [serialize_phrase_metrics(metric) for metric in phrase_metrics],
        "phrase_benchmark_metrics": phrase_benchmark_metrics,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Benchmark status: {summary['status']}")
    print(
        f"RTF={summary['average_realtime_factor']:.4f} "
        f"first_audio_latency_ms={summary['first_audio_latency_ms']}"
    )
    for metric in phrase_benchmark_metrics:
        print(json.dumps(metric, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_benchmark(args)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
