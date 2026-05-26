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
DEFAULT_TEXT = (
    "[S1] denk je dat je een open source model kan trainen met weinig geld en middelen? "
    "[S2] ja, ik denk het wel. [S1] oh ja, hoe dan? "
    "[S2] nou kijk maar in de repo op Git Hub of Hugging Face."
)
DEFAULT_CFG_SCALE = 3.0
DEFAULT_TEMPERATURE = 1.8
DEFAULT_TOP_P = 0.90
DEFAULT_CFG_FILTER_TOP_K = 50
DEFAULT_MAX_TOKENS = 3072


def resolve_output_sample_rate(config: DiaConfig) -> int:
    sample_rate = getattr(config, "output_sample_rate", None)
    if sample_rate is None:
        return DEFAULT_OUTPUT_SAMPLE_RATE
    return int(sample_rate)


def load_original_model(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[Dia, DiaConfig]:
    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        compute_dtype="float32",
        load_dac=True,
    )
    return model, config


def load_conditioned_model(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    allow_missing_speaker_modules: bool,
) -> tuple[Dia, DiaConfig]:
    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config.speaker_conditioning_enabled:
        raise ValueError("Expected speaker_conditioning_enabled=True")

    dia = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        compute_dtype="float32",
        load_dac=True,
        allow_missing_speaker_modules=allow_missing_speaker_modules,
    )
    return dia, config


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
        if waveform.ndim == 2 and waveform.shape[-1] == 9 and np.issubdtype(
            waveform.dtype, np.integer
        ):
            raise ValueError(
                "Conditioned generation returned codec codes; DAC decoder was not loaded or decode path was skipped."
            )
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")
    if not np.issubdtype(waveform.dtype, np.number):
        raise TypeError(f"Expected numeric waveform dtype, got {waveform.dtype}")
    return np.asarray(waveform)


def compute_waveform_stats(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    waveform_f32 = waveform.astype(np.float32, copy=False)
    num_samples = int(waveform_f32.shape[0])
    if num_samples == 0:
        return {
            "duration_sec": 0.0,
            "sample_rate": int(sample_rate),
            "num_samples": 0,
            "dtype": str(waveform.dtype),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "rms": 0.0,
            "peak_abs": 0.0,
            "zero_ratio": 1.0,
            "near_silence_ratio": 1.0,
        }
    abs_waveform = np.abs(waveform_f32)
    return {
        "duration_sec": float(num_samples) / float(sample_rate),
        "sample_rate": int(sample_rate),
        "num_samples": num_samples,
        "dtype": str(waveform.dtype),
        "min": float(np.min(waveform_f32)),
        "max": float(np.max(waveform_f32)),
        "mean": float(np.mean(waveform_f32)),
        "rms": float(np.sqrt(np.mean(np.square(waveform_f32)))),
        "peak_abs": float(np.max(abs_waveform)),
        "zero_ratio": float(np.mean(waveform_f32 == 0.0)),
        "near_silence_ratio": float(np.mean(abs_waveform < 1e-4)),
    }


def describe_audio_object(audio: Any) -> dict[str, Any]:
    if isinstance(audio, np.ndarray):
        return {
            "python_type": type(audio).__name__,
            "dtype": str(audio.dtype),
            "shape": list(audio.shape),
        }
    if torch.is_tensor(audio):
        return {
            "python_type": type(audio).__name__,
            "dtype": str(audio.dtype),
            "shape": list(audio.shape),
        }
    if isinstance(audio, list):
        return {
            "python_type": type(audio).__name__,
            "length": len(audio),
        }
    return {"python_type": type(audio).__name__}


def generate_and_save(
    *,
    label: str,
    dia: Dia,
    checkpoint_path: str | Path,
    config_path: str | Path,
    text: str,
    speaker_id: int | None,
    output_path: str | Path,
    sample_rate: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    print(f"[{label}] checkpoint={checkpoint_path}")
    print(f"[{label}] config_path={config_path}")
    print(f"[{label}] speaker_id={speaker_id}")
    print(f"[{label}] text={text}")

    start_time = time.time()
    generate_kwargs = {
        "use_torch_compile": False,
        "verbose": True,
        "cfg_scale": DEFAULT_CFG_SCALE,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "cfg_filter_top_k": DEFAULT_CFG_FILTER_TOP_K,
        "max_tokens": int(max_output_tokens),
    }
    if speaker_id is not None:
        generate_kwargs["speaker_id"] = int(speaker_id)
    audio = dia.generate(text, **generate_kwargs)
    generation_duration_sec = time.time() - start_time

    audio_description = describe_audio_object(audio)
    print(f"[{label}] returned_audio={audio_description}")

    waveform = ensure_waveform(audio)
    stats = compute_waveform_stats(waveform, sample_rate)
    print(
        f"[{label}] waveform dtype={stats['dtype']} shape={[stats['num_samples']]} "
        f"min={stats['min']:.6f} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6f} rms={stats['rms']:.6f} "
        f"sample_rate={sample_rate}"
    )

    output_path = Path(output_path)
    dia.save_audio(str(output_path), waveform)
    file_size_bytes = output_path.stat().st_size if output_path.exists() else 0
    print(f"[{label}] output_file={output_path}")
    print(f"[{label}] output_file_size_bytes={file_size_bytes}")

    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "speaker_id": None if speaker_id is None else int(speaker_id),
        "text": text,
        "returned_audio": audio_description,
        "wav_created": output_path.exists(),
        "output_path": str(output_path),
        "file_size_bytes": int(file_size_bytes),
        "generation_duration_sec": float(generation_duration_sec),
        "sample_rate": int(sample_rate),
        **stats,
        "generate_metadata": dict(dia.last_generate_metadata),
    }


def run_compare(
    *,
    config_path: str | Path,
    original_config_path: str | Path,
    base_checkpoint_path: str | Path,
    speaker_checkpoint_path: str | Path,
    output_dir: str | Path,
    speaker_id: int,
    default_speaker_id: int,
    seed: int,
    text: str,
    skip_base: bool,
    max_output_tokens: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}

    original_model, original_config = load_original_model(
        config_path=original_config_path,
        checkpoint_path=base_checkpoint_path,
    )
    outputs["original_base"] = generate_and_save(
        label="original_base",
        dia=original_model,
        checkpoint_path=base_checkpoint_path,
        config_path=original_config_path,
        text=text,
        speaker_id=None,
        output_path=output_dir_path / "original_base.wav",
        sample_rate=resolve_output_sample_rate(original_config),
        max_output_tokens=max_output_tokens,
    )

    if not skip_base:
        conditioned_base_model, conditioned_config = load_conditioned_model(
            config_path=config_path,
            checkpoint_path=base_checkpoint_path,
            allow_missing_speaker_modules=True,
        )
        outputs["base_speaker_0"] = generate_and_save(
            label="base_speaker_0",
            dia=conditioned_base_model,
            checkpoint_path=base_checkpoint_path,
            config_path=config_path,
            text=text,
            speaker_id=default_speaker_id,
            output_path=output_dir_path / "base_speaker_0.wav",
            sample_rate=resolve_output_sample_rate(conditioned_config),
            max_output_tokens=max_output_tokens,
        )

    tiny_model, conditioned_config = load_conditioned_model(
        config_path=config_path,
        checkpoint_path=speaker_checkpoint_path,
        allow_missing_speaker_modules=False,
    )
    outputs["tiny_speaker_1"] = generate_and_save(
        label="tiny_speaker_1",
        dia=tiny_model,
        checkpoint_path=speaker_checkpoint_path,
        config_path=config_path,
        text=text,
        speaker_id=speaker_id,
        output_path=output_dir_path / "tiny_speaker_1.wav",
        sample_rate=resolve_output_sample_rate(conditioned_config),
        max_output_tokens=max_output_tokens,
    )
    outputs["tiny_speaker_0"] = generate_and_save(
        label="tiny_speaker_0",
        dia=tiny_model,
        checkpoint_path=speaker_checkpoint_path,
        config_path=config_path,
        text=text,
        speaker_id=default_speaker_id,
        output_path=output_dir_path / "tiny_speaker_0.wav",
        sample_rate=resolve_output_sample_rate(conditioned_config),
        max_output_tokens=max_output_tokens,
    )

    report = {
        "config_path": str(config_path),
        "original_config_path": str(original_config_path),
        "base_checkpoint_path": str(base_checkpoint_path),
        "speaker_checkpoint_path": str(speaker_checkpoint_path),
        "text": text,
        "speaker_id": int(speaker_id),
        "default_speaker_id": int(default_speaker_id),
        "skip_base": bool(skip_base),
        "max_output_tokens": int(max_output_tokens),
        "outputs": outputs,
    }
    report_path = output_dir_path / "inference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original and speaker-conditioned Parkiet inference outputs"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--original-config-path", default="config.json")
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--speaker-checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--default-speaker-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_compare(
        config_path=args.config_path,
        original_config_path=args.original_config_path,
        base_checkpoint_path=args.base_checkpoint_path,
        speaker_checkpoint_path=args.speaker_checkpoint_path,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id,
        default_speaker_id=args.default_speaker_id,
        seed=args.seed,
        text=args.text,
        skip_base=args.skip_base,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"report_path={report['report_path']}")
    created_files = [
        item["output_path"]
        for item in report["outputs"].values()
        if item["wav_created"]
    ]
    print(f"created_files={created_files}")


if __name__ == "__main__":
    main()
