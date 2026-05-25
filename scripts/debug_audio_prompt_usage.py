from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from parkiet.dia.model import DEFAULT_SAMPLE_RATE, Dia


DEFAULT_TEXT = "[S1] Goedemiddag, waarmee kan ik u helpen?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug Dia audio_prompt behavior and compare direct prompt usage modes."
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
    parser.add_argument("--prompt-codes-path", default=None)
    parser.add_argument("--prompt-transcript", default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--crop-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", default="debug_audio_prompt")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.80)
    parser.add_argument("--cfg-filter-top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args(argv)


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {"min": 0.0, "max": 0.0, "rms": 0.0, "duration_sec": 0.0}
    return {
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "duration_sec": float(audio.shape[0]) / float(DEFAULT_SAMPLE_RATE),
    }


def ensure_files_exist(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def crop_reference_wav(reference_wav: Path, crop_seconds: float) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    tmp_dir = tempfile.TemporaryDirectory(prefix="parkiet_audio_prompt_")
    cropped_path = Path(tmp_dir.name) / f"{reference_wav.stem}_crop.wav"
    audio, sr = torchaudio.load(str(reference_wav), channels_first=True)
    target_frames = max(1, int(sr * crop_seconds))
    cropped = audio[:, :target_frames]
    torchaudio.save(str(cropped_path), cropped, sr)
    return cropped_path, tmp_dir


def build_full_text(prompt_transcript: str | None, target_text: str) -> str:
    if prompt_transcript and prompt_transcript.strip():
        return f"{prompt_transcript.strip()} {target_text.strip()}"
    return target_text


def run_case(
    model: Dia,
    *,
    case_name: str,
    text: str,
    output_dir: Path,
    use_torch_compile: bool,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    max_tokens: int,
    audio_prompt: str | torch.Tensor | None = None,
    prompt_tensor_shape: list[int] | None = None,
    prompt_tensor_dtype: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = model.generate(
        text,
        audio_prompt=audio_prompt,
        use_torch_compile=use_torch_compile,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        cfg_filter_top_k=cfg_filter_top_k,
        max_tokens=max_tokens,
        verbose=False,
    )
    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav_path = output_dir / f"{case_name}.wav"
    sf.write(wav_path, audio_np, DEFAULT_SAMPLE_RATE)
    stats = audio_stats(audio_np)
    metadata = {
        "case_name": case_name,
        "text": text,
        "wav_path": str(wav_path),
        "audio_stats": stats,
        "generation_metadata": dict(getattr(model, "last_generate_metadata", {})),
        "audio_prompt_kind": (
            "tensor"
            if isinstance(audio_prompt, torch.Tensor)
            else "path"
            if isinstance(audio_prompt, str)
            else "none"
        ),
        "audio_prompt_tensor_shape": prompt_tensor_shape,
        "audio_prompt_tensor_dtype": prompt_tensor_dtype,
    }
    print(f"case={case_name}")
    print(f"  wav_path={wav_path}")
    print(
        "  audio_stats="
        f"min={stats['min']:.5f} max={stats['max']:.5f} rms={stats['rms']:.5f} duration_sec={stats['duration_sec']:.3f}"
    )
    if prompt_tensor_shape is not None:
        print(f"  audio_prompt_tensor_shape={prompt_tensor_shape} dtype={prompt_tensor_dtype}")
    print(f"  generation_metadata={json.dumps(metadata['generation_metadata'], ensure_ascii=False)}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_path)
    checkpoint_path = Path(args.checkpoint_path)
    reference_wav = Path(args.reference_wav)
    ensure_files_exist(config_path, checkpoint_path, reference_wav)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        compute_dtype=args.compute_dtype,
        device=torch.device(args.device),
        load_dac=True,
    )

    prompt_codes_path = Path(args.prompt_codes_path) if args.prompt_codes_path else None
    prompt_codes_tensor = None
    if prompt_codes_path is not None:
        ensure_files_exist(prompt_codes_path)
        prompt_codes_tensor = torch.load(prompt_codes_path, map_location="cpu")
        if not isinstance(prompt_codes_tensor, torch.Tensor):
            raise TypeError(f"Prompt codes file does not contain a torch.Tensor: {prompt_codes_path}")

    cropped_wav_path, temp_dir = crop_reference_wav(reference_wav, args.crop_seconds)
    try:
        cropped_prompt_codes = model.load_audio(str(cropped_wav_path))

        base_text = args.text.strip()
        anchored_text = build_full_text(args.prompt_transcript, base_text)

        results = [
            run_case(
                model,
                case_name="A_no_audio_prompt",
                text=base_text,
                output_dir=output_dir,
                use_torch_compile=args.use_torch_compile,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_p=args.top_p,
                cfg_filter_top_k=args.cfg_filter_top_k,
                max_tokens=args.max_tokens,
            ),
            run_case(
                model,
                case_name="B_audio_prompt_path_direct",
                text=anchored_text,
                output_dir=output_dir,
                use_torch_compile=args.use_torch_compile,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_p=args.top_p,
                cfg_filter_top_k=args.cfg_filter_top_k,
                max_tokens=args.max_tokens,
                audio_prompt=str(reference_wav),
            ),
        ]

        if prompt_codes_tensor is None:
            prompt_codes_tensor = model.load_audio(str(reference_wav))

        results.append(
            run_case(
                model,
                case_name="C_prompt_codes_tensor",
                text=anchored_text,
                output_dir=output_dir,
                use_torch_compile=args.use_torch_compile,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_p=args.top_p,
                cfg_filter_top_k=args.cfg_filter_top_k,
                max_tokens=args.max_tokens,
                audio_prompt=prompt_codes_tensor,
                prompt_tensor_shape=list(prompt_codes_tensor.shape),
                prompt_tensor_dtype=str(prompt_codes_tensor.dtype),
            )
        )
        results.append(
            run_case(
                model,
                case_name=f"D_short_{int(args.crop_seconds)}s_audio_prompt_path",
                text=anchored_text,
                output_dir=output_dir,
                use_torch_compile=args.use_torch_compile,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_p=args.top_p,
                cfg_filter_top_k=args.cfg_filter_top_k,
                max_tokens=args.max_tokens,
                audio_prompt=str(cropped_wav_path),
            )
        )
        results.append(
            run_case(
                model,
                case_name=f"E_short_{int(args.crop_seconds)}s_prompt_codes",
                text=anchored_text,
                output_dir=output_dir,
                use_torch_compile=args.use_torch_compile,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_p=args.top_p,
                cfg_filter_top_k=args.cfg_filter_top_k,
                max_tokens=args.max_tokens,
                audio_prompt=cropped_prompt_codes,
                prompt_tensor_shape=list(cropped_prompt_codes.shape),
                prompt_tensor_dtype=str(cropped_prompt_codes.dtype),
            )
        )

        metadata = {
            "args": vars(args),
            "reference_wav": str(reference_wav),
            "prompt_codes_path": str(prompt_codes_path) if prompt_codes_path is not None else None,
            "prompt_transcript_used": anchored_text != base_text,
            "cases": results,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"metadata_path={metadata_path}")
    finally:
        temp_dir.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
