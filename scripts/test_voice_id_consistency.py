from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from parkiet.realtime import RealtimeTTSConfig, RealtimeTTSEngine
from parkiet.realtime.engine import DiaRealtimeBackend


DEFAULT_PHRASES = [
    "[S1] Ja, dat kan ik voor je controleren.",
    "[S1] Een momentje alstublieft.",
    "[S1] Ik heb de gegevens gevonden.",
    "[S1] Zal ik het nog even samenvatten?",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare phrase-based realtime generation with and without voice_id-backed audio prompts."
    )
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--checkpoint-path", default="weights/dia-nl-v1.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--voice-registry-path", default="voice_prompts")
    parser.add_argument("--output-dir", default="debug_voice_id")
    parser.add_argument("--output-sample-rate", type=int, default=44100)
    parser.add_argument("--sample-format", choices=["pcm16", "float32"], default="pcm16")
    parser.add_argument("--frame-duration-ms", type=int, default=20)
    parser.add_argument("--text", action="append", default=None)
    return parser.parse_args(argv)


def decode_payload(payload: bytes | np.ndarray, sample_format: str) -> np.ndarray:
    if sample_format == "pcm16":
        pcm = np.frombuffer(payload, dtype=np.int16)
        return (pcm.astype(np.float32) / 32767.0).copy()
    return np.asarray(payload, dtype=np.float32).copy()


def get_latest_phrase_metric(session):
    metrics = session.phrase_metrics
    if not metrics:
        return None
    if isinstance(metrics, list):
        return metrics[-1]
    if isinstance(metrics, dict):
        return list(metrics.values())[-1]
    return None


def build_backend(args: argparse.Namespace, *, voice_id: str | None) -> DiaRealtimeBackend:
    return DiaRealtimeBackend.from_local_paths(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        compute_dtype="bfloat16",
        device=torch.device(args.device),
        use_torch_compile=True,
        disable_cfg=False,
        cfg_scale=3.0,
        temperature=1.0,
        top_p=0.80,
        cfg_filter_top_k=50,
        voice_id=voice_id,
        voice_registry_path=args.voice_registry_path,
        collect_timings=True,
    )


def synthesize_run(
    run_name: str,
    phrases: list[str],
    args: argparse.Namespace,
    *,
    voice_id: str | None,
) -> dict[str, Any]:
    backend = build_backend(args, voice_id=voice_id)
    config = RealtimeTTSConfig(
        output_sample_rate=args.output_sample_rate,
        output_sample_format=args.sample_format,
        frame_duration_ms=args.frame_duration_ms,
        return_silence_when_empty=False,
    )
    engine = RealtimeTTSEngine(config=config, backend=backend)
    session = engine.create_session(run_name)

    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    phrase_results: list[dict[str, Any]] = []
    full_audio_segments: list[np.ndarray] = []

    for phrase_index, phrase_text in enumerate(phrases):
        engine.accept_text(session.session_id, phrase_text, is_final=True)
        engine.synthesize_pending(session.session_id)

        phrase_audio_segments: list[np.ndarray] = []
        while True:
            frame = engine.read_audio_frame(
                session.session_id,
                return_silence_when_empty=False,
            )
            if frame is None:
                break
            if frame.is_silence:
                continue
            decoded = decode_payload(frame.payload, frame.sample_format)
            if decoded.size:
                phrase_audio_segments.append(decoded)
                full_audio_segments.append(decoded)
            if frame.is_final:
                break

        phrase_audio = (
            np.concatenate(phrase_audio_segments).astype(np.float32)
            if phrase_audio_segments
            else np.zeros(0, dtype=np.float32)
        )
        phrase_path = run_dir / f"per_phrase_{phrase_index + 1:03d}.wav"
        sf.write(phrase_path, phrase_audio, args.output_sample_rate)

        phrase_metric = get_latest_phrase_metric(session)
        phrase_results.append(
            {
                "phrase_text": phrase_text,
                "audio_path": str(phrase_path),
                "sample_count": int(phrase_audio.shape[0]),
                "duration_ms": 1000.0
                * float(phrase_audio.shape[0])
                / float(args.output_sample_rate),
                "timings": dict(backend.last_timing_breakdown),
                "phrase_metrics": {
                    "generation_ms": phrase_metric.generation_ms if phrase_metric else None,
                    "audio_duration_ms": phrase_metric.audio_duration_ms if phrase_metric else None,
                },
            }
        )

    full_audio = (
        np.concatenate(full_audio_segments).astype(np.float32)
        if full_audio_segments
        else np.zeros(0, dtype=np.float32)
    )
    full_audio_path = run_dir / "full_generated_realtime_output.wav"
    sf.write(full_audio_path, full_audio, args.output_sample_rate)

    metrics_snapshot = engine.get_session_metrics(session.session_id)
    state_snapshot = engine.get_session_state(session.session_id)
    engine.close_session(session.session_id)

    return {
        "run_name": run_name,
        "voice_id": voice_id,
        "output_dir": str(run_dir),
        "full_audio_path": str(full_audio_path),
        "session_metrics": metrics_snapshot,
        "session_state": state_snapshot,
        "phrases": phrase_results,
        "backend_config": {
            "compute_dtype": "bfloat16",
            "use_torch_compile": True,
            "disable_cfg": False,
            "cfg_scale": 3.0,
            "temperature": 1.0,
            "top_p": 0.80,
            "cfg_filter_top_k": 50,
            "voice_id": voice_id,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phrases = args.text if args.text else DEFAULT_PHRASES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    no_prompt = synthesize_run("no_prompt", phrases, args, voice_id=None)
    with_voice_id = synthesize_run(
        "with_voice_id",
        phrases,
        args,
        voice_id=args.voice_id,
    )

    metadata = {
        "args": vars(args),
        "phrases": phrases,
        "runs": {
            "no_prompt": no_prompt,
            "with_voice_id": with_voice_id,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("config:")
    print(json.dumps(with_voice_id["backend_config"], indent=2))
    for run in (no_prompt, with_voice_id):
        print(f"run={run['run_name']}")
        print(f"  full_audio_path={run['full_audio_path']}")
        print(f"  generated_phrase_count={run['session_metrics']['generated_phrase_count']}")
        print(f"  generated_audio_ms={run['session_metrics']['generated_audio_ms']:.2f}")
        for phrase in run["phrases"]:
            print(f"  phrase={phrase['phrase_text']}")
            print(f"    generation_ms={phrase['phrase_metrics']['generation_ms']}")
            print(f"    audio_path={phrase['audio_path']}")
    print(f"metadata_path={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
