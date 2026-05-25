from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from parkiet.dia.model import DEFAULT_SAMPLE_RATE, Dia


TEST_PHRASES = [
    "[S1] Goedemiddag, waarmee kan ik u helpen?",
    "[S1] Natuurlijk, ik kan een afspraak voor u inplannen.",
    "[S1] Welke dag en tijd komt u het beste uit?",
    "[S1] Ik controleer meteen de beschikbaarheid voor u.",
    "[S1] Dat is gelukt, uw afspraak staat nu genoteerd.",
    "[S1] Kan ik verder nog iets voor u doen?",
]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple independent phrases with the same audio_prompt to inspect voice consistency."
    )
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--checkpoint-path", default="weights/dia-nl-v1.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--prompt-transcript", required=True)
    parser.add_argument("--prompt-codes-path", default=None)
    parser.add_argument("--output-dir", default="debug_audio_prompt_long")
    parser.add_argument("--use-torch-compile", type=parse_bool, default=True)
    parser.add_argument("--trim-audio-prompt", type=parse_bool, default=True)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.80)
    parser.add_argument("--cfg-filter-top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args(argv)


def ensure_files_exist(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {"min": 0.0, "max": 0.0, "rms": 0.0, "duration_sec": 0.0}
    return {
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "duration_sec": float(audio.shape[0]) / float(DEFAULT_SAMPLE_RATE),
    }


def build_full_text(prompt_transcript: str | None, phrase: str) -> str:
    if prompt_transcript and prompt_transcript.strip():
        return f"{prompt_transcript.strip()} {phrase.strip()}"
    return phrase.strip()


def run_set(
    model: Dia,
    *,
    output_dir: Path,
    phrases: list[str],
    prompt_transcript: str,
    audio_prompt: str | torch.Tensor | None,
    args: argparse.Namespace,
    with_prompt: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_results: list[dict[str, Any]] = []

    for index, phrase in enumerate(phrases, start=1):
        full_text = build_full_text(prompt_transcript, phrase) if with_prompt else phrase
        audio = model.generate(
            full_text,
            audio_prompt=audio_prompt if with_prompt else None,
            use_torch_compile=args.use_torch_compile,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_filter_top_k=args.cfg_filter_top_k,
            max_tokens=args.max_tokens,
            trim_audio_prompt_from_output=args.trim_audio_prompt if with_prompt else False,
            verbose=False,
        )
        audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
        wav_path = output_dir / f"per_phrase_{index:03d}.wav"
        sf.write(wav_path, audio_np, DEFAULT_SAMPLE_RATE)
        generation_metadata = dict(getattr(model, "last_generate_metadata", {}))
        stats = audio_stats(audio_np)
        phrase_results.append(
            {
                "phrase_index": index,
                "text": phrase,
                "full_text": full_text,
                "wav_path": str(wav_path),
                "audio_stats": stats,
                "generation_metadata": generation_metadata,
            }
        )
        print(f"run={'with_prompt' if with_prompt else 'no_prompt'} phrase={index}")
        print(f"  wav_path={wav_path}")
        print(
            f"  rms={stats['rms']:.5f} duration_sec={stats['duration_sec']:.3f} "
            f"min={stats['min']:.5f} max={stats['max']:.5f}"
        )
        print(f"  generation_metadata={json.dumps(generation_metadata, ensure_ascii=False)}")

    return {
        "with_prompt": with_prompt,
        "output_dir": str(output_dir),
        "phrases": phrase_results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_path)
    checkpoint_path = Path(args.checkpoint_path)
    reference_wav = Path(args.reference_wav)
    ensure_files_exist(config_path, checkpoint_path, reference_wav)

    prompt_codes_path = Path(args.prompt_codes_path) if args.prompt_codes_path else None
    if prompt_codes_path is not None:
        ensure_files_exist(prompt_codes_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        compute_dtype=args.compute_dtype,
        device=torch.device(args.device),
        load_dac=True,
    )

    prompt_codes = (
        torch.load(prompt_codes_path, map_location="cpu")
        if prompt_codes_path is not None
        else model.load_audio(str(reference_wav))
    )
    if not isinstance(prompt_codes, torch.Tensor):
        raise TypeError("Prompt codes must be a torch.Tensor")

    output_dir = Path(args.output_dir)
    no_prompt = run_set(
        model,
        output_dir=output_dir / "no_prompt",
        phrases=TEST_PHRASES,
        prompt_transcript=args.prompt_transcript,
        audio_prompt=None,
        args=args,
        with_prompt=False,
    )
    with_prompt = run_set(
        model,
        output_dir=output_dir / "with_prompt",
        phrases=TEST_PHRASES,
        prompt_transcript=args.prompt_transcript,
        audio_prompt=prompt_codes,
        args=args,
        with_prompt=True,
    )

    metadata = {
        "args": vars(args),
        "reference_wav": str(reference_wav),
        "prompt_codes_path": str(prompt_codes_path) if prompt_codes_path is not None else None,
        "prompt_codes_shape": list(prompt_codes.shape),
        "prompt_codes_dtype": str(prompt_codes.dtype),
        "runs": {
            "no_prompt": no_prompt,
            "with_prompt": with_prompt,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata_path={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
