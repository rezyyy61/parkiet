from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from parkiet.dia.audio import DEFAULT_SAMPLE_RATE
from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia, load_state_dict_allowing_missing_speaker_modules


def load_dia_with_checkpoint(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    allow_missing_speaker_modules: bool,
) -> Dia:
    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config.speaker_conditioning_enabled:
        raise ValueError("Expected speaker_conditioning_enabled=True")

    dia = Dia(config=config, compute_dtype="float32", load_dac=True)
    state_dict = torch.load(checkpoint_path, map_location=dia.device)
    load_state_dict_allowing_missing_speaker_modules(
        dia.model,
        state_dict,
        allow_missing_speaker_modules=allow_missing_speaker_modules,
    )
    dia.model.to(dia.device)
    dia.model.eval()
    dia._load_dac_model()
    return dia


def generate_and_save(
    *,
    dia: Dia,
    text: str,
    speaker_id: int,
    output_path: str | Path,
) -> dict[str, Any]:
    start_time = time.time()
    audio = dia.generate(
        text,
        speaker_id=speaker_id,
        use_torch_compile=False,
        verbose=False,
    )
    generation_duration_sec = time.time() - start_time
    dia.save_audio(str(output_path), audio)
    metadata = dict(dia.last_generate_metadata)
    output_path = Path(output_path)
    return {
        "speaker_id": int(speaker_id),
        "output_path": str(output_path),
        "wav_created": output_path.exists(),
        "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "generation_duration_sec": generation_duration_sec,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "generate_metadata": metadata,
    }


def run_compare(
    *,
    config_path: str | Path,
    base_checkpoint_path: str | Path,
    speaker_checkpoint_path: str | Path,
    output_dir: str | Path,
    speaker_id: int,
    default_speaker_id: int,
    seed: int,
    text: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    base_dia = load_dia_with_checkpoint(
        config_path=config_path,
        checkpoint_path=base_checkpoint_path,
        allow_missing_speaker_modules=True,
    )
    tiny_dia = load_dia_with_checkpoint(
        config_path=config_path,
        checkpoint_path=speaker_checkpoint_path,
        allow_missing_speaker_modules=False,
    )

    base_speaker_0 = generate_and_save(
        dia=base_dia,
        text=text,
        speaker_id=default_speaker_id,
        output_path=output_dir_path / "base_speaker_0.wav",
    )
    tiny_speaker_1 = generate_and_save(
        dia=tiny_dia,
        text=text,
        speaker_id=speaker_id,
        output_path=output_dir_path / "tiny_speaker_1.wav",
    )
    tiny_speaker_0 = generate_and_save(
        dia=tiny_dia,
        text=text,
        speaker_id=default_speaker_id,
        output_path=output_dir_path / "tiny_speaker_0.wav",
    )

    report = {
        "config_path": str(config_path),
        "base_checkpoint_path": str(base_checkpoint_path),
        "speaker_checkpoint_path": str(speaker_checkpoint_path),
        "text": text,
        "speaker_id": int(speaker_id),
        "default_speaker_id": int(default_speaker_id),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "outputs": {
            "base_speaker_0": base_speaker_0,
            "tiny_speaker_1": tiny_speaker_1,
            "tiny_speaker_0": tiny_speaker_0,
        },
    }
    report_path = output_dir_path / "inference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and tiny speaker-conditioned inference outputs"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--speaker-checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--default-speaker-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_compare(
        config_path=args.config_path,
        base_checkpoint_path=args.base_checkpoint_path,
        speaker_checkpoint_path=args.speaker_checkpoint_path,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id,
        default_speaker_id=args.default_speaker_id,
        seed=args.seed,
        text=args.text,
    )
    print(f"report_path={report['report_path']}")
    print(
        "created_files="
        f"{[item['output_path'] for item in report['outputs'].values() if item['wav_created']]}"
    )


if __name__ == "__main__":
    main()
