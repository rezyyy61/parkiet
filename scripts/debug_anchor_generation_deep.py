from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from parkiet.dia.model import DEFAULT_PROMPT_TRIM_CONTEXT_FRAMES, DEFAULT_SAMPLE_RATE, SAMPLE_RATE_RATIO, Dia


DEFAULT_REFERENCE_WAV = "debug_reference_voice/nl_test_reference_3s.wav"
DEFAULT_PROMPT_TRANSCRIPT = "[S1] Goedemiddag, u spreekt met de assistent van de salon."
DEFAULT_CONTINUATION_TEXT = "[S1] Natuurlijk, ik kan een afspraak voor u inplannen."
STREAMER_CONTEXTS = [32, 64, 128, 256]
ALLOWED_CASES = {
    "all",
    "A_no_prompt",
    "B_prompt_path_current",
    "C_prompt_codes_current",
    "D_context_32",
    "D_context_64",
    "D_context_128",
    "D_context_256",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep diagnostics for Dia anchored generation using audio_prompt."
    )
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--checkpoint-path", default="weights/dia-nl-v1.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--reference-wav", default=DEFAULT_REFERENCE_WAV)
    parser.add_argument("--prompt-transcript", default=DEFAULT_PROMPT_TRANSCRIPT)
    parser.add_argument("--continuation-text", default=DEFAULT_CONTINUATION_TEXT)
    parser.add_argument("--case", choices=sorted(ALLOWED_CASES), default="all")
    parser.add_argument("--output-dir", default="debug_anchor_generation_deep")
    parser.add_argument("--use-torch-compile", type=parse_bool, default=False)
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


def build_full_text(prompt_transcript: str, continuation_text: str) -> str:
    return f"{prompt_transcript.strip()} {continuation_text.strip()}"


def write_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), DEFAULT_SAMPLE_RATE)


def prepare_model(args: argparse.Namespace) -> Dia:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return Dia.from_local(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        compute_dtype=args.compute_dtype,
        device=torch.device(args.device),
        load_dac=True,
    )


def maybe_compile_model(model: Dia) -> None:
    if not hasattr(model, "_compiled"):
        model._prepare_generation = torch.compile(
            model._prepare_generation,
            dynamic=True,
            fullgraph=True,
        )
        model._decoder_step = torch.compile(
            model._decoder_step,
            fullgraph=True,
            mode="max-autotune",
        )
        model._compiled = True


def cleanup_torch_memory(device: torch.device | None) -> None:
    gc.collect()
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(device)


def run_direct_case(
    model: Dia,
    *,
    case_name: str,
    text: str,
    audio_prompt: str | torch.Tensor | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    audio = model.generate(
        text,
        audio_prompt=audio_prompt,
        use_torch_compile=args.use_torch_compile,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_filter_top_k=args.cfg_filter_top_k,
        max_tokens=args.max_tokens,
        trim_audio_prompt_from_output=False,
        verbose=False,
    )
    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav_path = output_dir / f"{case_name}.wav"
    write_audio(wav_path, audio_np)
    metadata = {
        "case_name": case_name,
        "wav_path": str(wav_path),
        "audio_stats": audio_stats(audio_np),
        "last_generate_metadata": dict(model.last_generate_metadata),
        "audio_prompt_shape": list(audio_prompt.shape)
        if isinstance(audio_prompt, torch.Tensor)
        else None,
        "audio_prompt_dtype": str(audio_prompt.dtype)
        if isinstance(audio_prompt, torch.Tensor)
        else None,
    }
    print_case_metadata(metadata)
    return metadata


def manual_old_streamer_style_case(
    model: Dia,
    *,
    case_name: str,
    full_text: str,
    prompt_codes: torch.Tensor,
    args: argparse.Namespace,
    context_frames: int,
    output_dir: Path,
) -> dict[str, Any]:
    if args.use_torch_compile:
        maybe_compile_model(model)

    text_tokens = [model._encode_text(full_text)]
    text_tensor = model._pad_text_input(text_tokens)
    prefill_preview, prefill_steps_preview = model._prepare_audio_prompt([prompt_codes])

    effective_max_tokens = args.max_tokens
    dec_state, dec_output = model._prepare_generation(
        text_tensor,
        [prompt_codes],
        max_tokens=effective_max_tokens,
    )
    prefill_step = dec_output.prefill_steps[0]
    dec_step = min(dec_output.prefill_steps) - 1
    current_idx = torch.tensor([dec_step], device=model.device)

    audio_eos_value = model.config.eos_token_id
    audio_pad_value = model.config.pad_token_id
    delay_pattern = model.config.delay_pattern
    max_delay_pattern = max(delay_pattern)
    delay_pattern_Cx = torch.tensor(delay_pattern, device=model.device, dtype=torch.long)

    eos_detected_Bx = torch.zeros((1,), dtype=torch.bool, device=model.device)
    eos_countdown_Bx = torch.full((1,), -1, dtype=torch.long, device=model.device)
    finished_step_Bx = torch.full((1,), -1, dtype=torch.long, device=model.device)
    bos_over = False

    while dec_step < effective_max_tokens:
        if (eos_countdown_Bx == 0).all():
            break

        current_step_idx = dec_step + 1
        torch.compiler.cudagraph_mark_step_begin()
        dec_state.prepare_step(dec_step)
        tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)

        pred_BxC = model._decoder_step(
            tokens_Bx1xC,
            dec_state,
            args.cfg_scale,
            args.temperature,
            args.top_p,
            args.cfg_filter_top_k,
            current_idx,
        )
        current_idx += 1

        active_mask_Bx = eos_countdown_Bx != 0
        eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
        if active_mask_Bx.any():
            is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (
                pred_BxC[active_mask_Bx, 0] == audio_eos_value
            )
            is_max_len = current_step_idx >= effective_max_tokens - max_delay_pattern
            eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len

        eos_detected_Bx |= eos_trigger_Bx
        start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
        if start_countdown_mask_Bx.any():
            eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
            finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

        padding_mask_Bx = eos_countdown_Bx > 0
        if padding_mask_Bx.any():
            pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
            countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
            step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
            step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
            delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
            eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
            pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
            pred_active_BxC[eos_mask_NxC] = audio_eos_value
            pred_active_BxC[pad_mask_NxC] = audio_pad_value
            pred_BxC[padding_mask_Bx] = pred_active_BxC
            eos_countdown_Bx[padding_mask_Bx] -= 1

        if not bos_over:
            bos_over = all(
                dec_step - current_prefill_step > max_delay_pattern
                for current_prefill_step in dec_output.prefill_steps
            )

        dec_output.update_one(pred_BxC, current_step_idx, not bos_over)
        dec_step += 1

    final_step = dec_step + 1
    finished_step_Bx[finished_step_Bx == -1] = final_step - max_delay_pattern
    generated_token_count = int(torch.clamp(finished_step_Bx[0] - prefill_step, min=0).item())

    # Method 1: decode generated slice only.
    generated_codes_len = generated_token_count + max_delay_pattern
    generated_codes = torch.full(
        (1, generated_codes_len, model.config.decoder_config.num_channels),
        fill_value=model.config.pad_token_id,
        dtype=torch.long,
        device=model.device,
    )
    if generated_codes_len > 0:
        generated_codes[0, :generated_codes_len, :] = dec_output.generated_tokens[
            0,
            prefill_step : prefill_step + generated_codes_len,
            :,
        ]
    direct_audio = model._generate_output(
        generated_codes,
        torch.tensor([generated_token_count], dtype=torch.long, device=model.device),
    )[0]

    # Method 2: decode from prefill start and trim full prompt waveform.
    full_prefill_valid_frames = prefill_step + generated_token_count
    full_prefill_delayed_frames = full_prefill_valid_frames + max_delay_pattern
    full_prefill_codes = dec_output.generated_tokens[
        0:1,
        0 : full_prefill_delayed_frames,
        :,
    ]
    full_prefill_audio = model._generate_output(
        full_prefill_codes,
        torch.tensor([full_prefill_valid_frames], dtype=torch.long, device=model.device),
    )[0]
    waveform_trim_samples = prefill_step * SAMPLE_RATE_RATIO
    full_prefill_trimmed_audio = full_prefill_audio[
        waveform_trim_samples : waveform_trim_samples + generated_token_count * SAMPLE_RATE_RATIO
    ]

    # Method 3: old streamer-style context decode and trim context waveform.
    decode_start = max(0, prefill_step - context_frames)
    actual_prompt_context_frames = prefill_step - decode_start
    total_valid_frames = actual_prompt_context_frames + generated_token_count
    total_delayed_frames = total_valid_frames + max_delay_pattern
    delayed_codes = dec_output.generated_tokens[
        0:1,
        decode_start : decode_start + total_delayed_frames,
        :,
    ]
    context_audio = model._generate_output(
        delayed_codes,
        torch.tensor([total_valid_frames], dtype=torch.long, device=model.device),
    )[0]
    context_trim_samples = actual_prompt_context_frames * SAMPLE_RATE_RATIO
    context_end_sample = context_trim_samples + generated_token_count * SAMPLE_RATE_RATIO
    continuation_audio = context_audio[context_trim_samples:context_end_sample]

    wav_path = output_dir / f"{case_name}.wav"
    write_audio(wav_path, continuation_audio)

    metadata = {
        "case_name": case_name,
        "wav_path": str(wav_path),
        "audio_prompt_shape": list(prompt_codes.shape),
        "audio_prompt_dtype": str(prompt_codes.dtype),
        "prefill_tensor_shape": list(prefill_preview.shape),
        "prefill_steps": list(prefill_steps_preview),
        "generated_tokens_tensor_shape": list(dec_output.generated_tokens.shape),
        "requested_max_tokens": int(args.max_tokens),
        "effective_max_tokens": int(effective_max_tokens),
        "prompt_code_steps": int(prompt_codes.shape[0]),
        "prompt_duration_ms": float(prompt_codes.shape[0]) * SAMPLE_RATE_RATIO / DEFAULT_SAMPLE_RATE * 1000.0,
        "generated_token_count": int(generated_token_count),
        "decode_start": int(decode_start),
        "decode_end": int(decode_start + total_delayed_frames),
        "decoded_code_steps": int(total_valid_frames),
        "trimmed_code_steps": int(generated_token_count),
        "output_samples_before_trim": int(len(context_audio)),
        "output_samples_after_trim": int(len(continuation_audio)),
        "context_frames_requested": int(context_frames),
        "context_frames_used": int(actual_prompt_context_frames),
        "audio_stats": audio_stats(continuation_audio),
        "comparison": {
            "method_decode_generated_slice_only": {
                "samples": int(len(direct_audio)),
                "audio_stats": audio_stats(np.asarray(direct_audio, dtype=np.float32)),
            },
            "method_decode_full_prefill_then_trim_waveform": {
                "samples": int(len(full_prefill_trimmed_audio)),
                "trimmed_samples": int(waveform_trim_samples),
                "audio_stats": audio_stats(np.asarray(full_prefill_trimmed_audio, dtype=np.float32)),
            },
            "method_streamer_style_context_trim": {
                "samples": int(len(continuation_audio)),
                "trimmed_samples": int(context_trim_samples),
                "audio_stats": audio_stats(np.asarray(continuation_audio, dtype=np.float32)),
            },
        },
    }
    print_case_metadata(metadata)
    return metadata


def print_case_metadata(metadata: dict[str, Any]) -> None:
    print(f"case={metadata['case_name']}")
    print(f"  wav_path={metadata['wav_path']}")
    if metadata.get("audio_prompt_shape") is not None:
        print(
            f"  audio_prompt_shape={metadata['audio_prompt_shape']} "
            f"dtype={metadata.get('audio_prompt_dtype')}"
        )
    if metadata.get("prefill_tensor_shape") is not None:
        print(
            f"  prefill_tensor_shape={metadata['prefill_tensor_shape']} "
            f"prefill_steps={metadata.get('prefill_steps')}"
        )
    print(
        "  generation="
        f"requested_max_tokens={metadata.get('requested_max_tokens')} "
        f"effective_max_tokens={metadata.get('effective_max_tokens')} "
        f"generated_token_count={metadata.get('generated_token_count')}"
    )
    if "decode_start" in metadata:
        print(
            "  decode="
            f"decode_start={metadata.get('decode_start')} "
            f"decode_end={metadata.get('decode_end')} "
            f"decoded_code_steps={metadata.get('decoded_code_steps')} "
            f"trimmed_code_steps={metadata.get('trimmed_code_steps')}"
        )
        print(
            "  output="
            f"output_samples_before_trim={metadata.get('output_samples_before_trim')} "
            f"output_samples_after_trim={metadata.get('output_samples_after_trim')}"
        )
    stats = metadata["audio_stats"]
    print(
        f"  audio_stats=min={stats['min']:.5f} max={stats['max']:.5f} "
        f"rms={stats['rms']:.5f} duration_sec={stats['duration_sec']:.3f}"
    )
    if "last_generate_metadata" in metadata:
        print(
            f"  last_generate_metadata={json.dumps(metadata['last_generate_metadata'], ensure_ascii=False)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_path)
    checkpoint_path = Path(args.checkpoint_path)
    reference_wav = Path(args.reference_wav)
    ensure_files_exist(config_path, checkpoint_path, reference_wav)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = prepare_model(args)
    prompt_codes = model.load_audio(str(reference_wav))
    full_text = build_full_text(args.prompt_transcript, args.continuation_text)

    metadata = {
        "args": vars(args),
        "reference_wav": str(reference_wav),
        "prompt_transcript": args.prompt_transcript,
        "continuation_text": args.continuation_text,
        "prompt_codes_shape": list(prompt_codes.shape),
        "prompt_codes_dtype": str(prompt_codes.dtype),
        "default_prompt_context_frames": DEFAULT_PROMPT_TRIM_CONTEXT_FRAMES,
        "cases": [],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    requested_cases: list[str]
    if args.case == "all":
        requested_cases = [
            "A_no_prompt",
            "B_prompt_path_current",
            "C_prompt_codes_current",
            "D_context_32",
            "D_context_64",
            "D_context_128",
            "D_context_256",
        ]
    else:
        requested_cases = [args.case]

    try:
        for requested_case in requested_cases:
            if requested_case == "A_no_prompt":
                case_result = run_direct_case(
                    model,
                    case_name="A_no_prompt",
                    text=args.continuation_text,
                    audio_prompt=None,
                    args=args,
                    output_dir=output_dir,
                )
            elif requested_case == "B_prompt_path_current":
                case_result = run_direct_case(
                    model,
                    case_name="B_prompt_path_current",
                    text=full_text,
                    audio_prompt=str(reference_wav),
                    args=args,
                    output_dir=output_dir,
                )
            elif requested_case == "C_prompt_codes_current":
                case_result = run_direct_case(
                    model,
                    case_name="C_prompt_codes_current",
                    text=full_text,
                    audio_prompt=prompt_codes,
                    args=args,
                    output_dir=output_dir,
                )
            else:
                context_frames = int(requested_case.rsplit("_", 1)[-1])
                case_result = manual_old_streamer_style_case(
                    model,
                    case_name=f"D_streamer_style_context_{context_frames}",
                    full_text=full_text,
                    prompt_codes=prompt_codes,
                    args=args,
                    context_frames=context_frames,
                    output_dir=output_dir,
                )

            metadata["cases"].append(case_result)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            cleanup_torch_memory(getattr(model, "device", None))
    finally:
        del prompt_codes
        cleanup_torch_memory(getattr(model, "device", None))

    print(f"metadata_path={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
