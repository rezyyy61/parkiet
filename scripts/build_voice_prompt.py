from __future__ import annotations

import argparse
from pathlib import Path

import torch

from parkiet.dia.model import Dia
from parkiet.realtime.voice_registry import VoicePromptRegistry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a voice_id bundle backed by saved audio_prompt codes."
    )
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--checkpoint-path", default="weights/dia-nl-v1.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="float32",
    )
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--output-dir", default="voice_prompts")
    parser.add_argument("--notes", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_path)
    checkpoint_path = Path(args.checkpoint_path)
    reference_wav = Path(args.reference_wav)

    missing = [
        str(path)
        for path in (config_path, checkpoint_path, reference_wav)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        compute_dtype=args.compute_dtype,
        device=torch.device(args.device),
        load_dac=True,
    )
    prompt_codes = model.load_audio(str(reference_wav))

    registry = VoicePromptRegistry(args.output_dir)
    metadata = registry.register_voice(
        args.voice_id,
        reference_wav,
        prompt_codes,
        notes=args.notes,
    )

    print(f"voice_id={metadata.voice_id}")
    print(f"reference_wav_path={metadata.reference_wav_path}")
    print(f"sample_rate={metadata.sample_rate}")
    print(f"duration_sec={metadata.duration_sec:.3f}")
    print(f"voice_dir={Path(args.output_dir) / args.voice_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
