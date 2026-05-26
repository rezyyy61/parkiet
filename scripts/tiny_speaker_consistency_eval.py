from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia


DEFAULT_OUTPUT_SAMPLE_RATE = 44100
DEFAULT_PROMPTS = [
    "[S1] Hallo, dit is een korte test.",
    "[S1] Ik bel u even over uw afspraak.",
    "[S1] Kunt u mij goed verstaan?",
    "[S1] Dank u wel, ik help u graag verder.",
    "[S1] We plannen samen een nieuwe afspraak.",
]


def resolve_output_sample_rate(config: DiaConfig) -> int:
    sample_rate = getattr(config, "output_sample_rate", None)
    if sample_rate is None:
        return DEFAULT_OUTPUT_SAMPLE_RATE
    return int(sample_rate)


def ensure_waveform(audio: Any) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        waveform = audio
    elif torch.is_tensor(audio):
        waveform = audio.detach().cpu().numpy()
    elif isinstance(audio, list) and len(audio) == 1:
        return ensure_waveform(audio[0])
    else:
        raise TypeError(f"Expected waveform output, got {type(audio)!r}")

    if waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.reshape(-1)
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")
    return np.asarray(waveform)


def compute_waveform_stats(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    waveform_f32 = waveform.astype(np.float32, copy=False)
    num_samples = int(waveform_f32.shape[0])
    if num_samples == 0:
        return {
            "duration_sec": 0.0,
            "sample_rate": int(sample_rate),
            "num_samples": 0,
            "dtype": str(waveform.dtype),
            "rms": 0.0,
            "peak_abs": 0.0,
            "near_silence_ratio": 1.0,
        }
    abs_waveform = np.abs(waveform_f32)
    return {
        "duration_sec": float(num_samples) / float(sample_rate),
        "sample_rate": int(sample_rate),
        "num_samples": num_samples,
        "dtype": str(waveform.dtype),
        "rms": float(np.sqrt(np.mean(np.square(waveform_f32)))),
        "peak_abs": float(np.max(abs_waveform)),
        "near_silence_ratio": float(np.mean(abs_waveform < 1e-4)),
    }


def generate_sample(
    *,
    model: Dia,
    sample_rate: int,
    prompt: str,
    speaker_id: int,
    output_path: Path,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    cfg_scale: float,
    cfg_filter_top_k: int,
) -> dict[str, Any]:
    start_time = time.time()
    audio = model.generate(
        prompt,
        speaker_id=speaker_id,
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        cfg_scale=cfg_scale,
        cfg_filter_top_k=cfg_filter_top_k,
        use_torch_compile=False,
        verbose=False,
    )
    generation_duration_sec = time.time() - start_time
    waveform = ensure_waveform(audio)
    model.save_audio(str(output_path), waveform)
    stats = compute_waveform_stats(waveform, sample_rate)
    return {
        "prompt": prompt,
        "speaker_id": int(speaker_id),
        "output_path": str(output_path),
        "wav_created": output_path.exists(),
        "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "generation_duration_sec": float(generation_duration_sec),
        **stats,
        "generation_metadata": dict(model.last_generate_metadata),
    }


def run_eval(
    *,
    config_path: str | Path,
    speaker_checkpoint_path: str | Path,
    output_dir: str | Path,
    speaker_id: int,
    seed: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    cfg_scale: float,
    cfg_filter_top_k: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config.speaker_conditioning_enabled:
        raise ValueError("Expected speaker_conditioning_enabled=True")

    sample_rate = resolve_output_sample_rate(config)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(speaker_checkpoint_path),
        compute_dtype="float32",
        load_dac=True,
    )

    outputs: list[dict[str, Any]] = []
    for current_speaker_id in (speaker_id, 0):
        for index, prompt in enumerate(DEFAULT_PROMPTS, start=1):
            output_path = output_dir_path / (
                f"speaker_{current_speaker_id}_sample_{index:02d}.wav"
            )
            print(
                f"speaker_id={current_speaker_id} sample={index:02d} prompt={prompt}"
            )
            outputs.append(
                generate_sample(
                    model=model,
                    sample_rate=sample_rate,
                    prompt=prompt,
                    speaker_id=current_speaker_id,
                    output_path=output_path,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_scale=cfg_scale,
                    cfg_filter_top_k=cfg_filter_top_k,
                )
            )

    report = {
        "config_path": str(config_path),
        "speaker_checkpoint_path": str(speaker_checkpoint_path),
        "output_dir": str(output_dir_path),
        "speaker_id": int(speaker_id),
        "seed": int(seed),
        "sample_rate": int(sample_rate),
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "cfg_scale": float(cfg_scale),
        "cfg_filter_top_k": int(cfg_filter_top_k),
        "prompts": list(DEFAULT_PROMPTS),
        "outputs": outputs,
    }
    report_path = output_dir_path / "consistency_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple speaker-conditioned samples for consistency listening"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--speaker-checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speaker-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--cfg-filter-top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        config_path=args.config_path,
        speaker_checkpoint_path=args.speaker_checkpoint_path,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        cfg_filter_top_k=args.cfg_filter_top_k,
    )
    print(f"report_path={report['report_path']}")
    print(f"generated_files={len(report['outputs'])}")


if __name__ == "__main__":
    main()
