from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from parkiet.dia.model import DEFAULT_SAMPLE_RATE, SAMPLE_RATE_RATIO, Dia


DEFAULT_REFERENCE_WAV = "debug_reference_voice/nl_test_reference_3s.wav"
DEFAULT_PROMPT_TRANSCRIPT = "[S1] Goedemiddag, u spreekt met de assistent van de salon."


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
    parser.add_argument("--reference-wav", default=DEFAULT_REFERENCE_WAV)
    parser.add_argument("--prompt-transcript", default=DEFAULT_PROMPT_TRANSCRIPT)
    parser.add_argument("--prompt-codes-path", default=None)
    parser.add_argument("--output-dir", default="debug_audio_prompt_long")
    parser.add_argument("--use-torch-compile", type=parse_bool, default=True)
    parser.add_argument("--trim-audio-prompt", type=parse_bool, default=True)
    parser.add_argument("--use-streamer-style-anchor", type=parse_bool, default=True)
    parser.add_argument("--audio-prompt-context-frames", type=int, default=32)
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


def cleanup_torch_memory(device: torch.device | None) -> None:
    gc.collect()
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_full_text(prompt_transcript: str | None, phrase: str) -> str:
    if prompt_transcript and prompt_transcript.strip():
        return f"{prompt_transcript.strip()} {phrase.strip()}"
    return phrase.strip()


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


def set_eval_mode(model: Dia) -> None:
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    if hasattr(model, "eval"):
        model.eval()


def generate_streamer_style_continuation(
    model: Dia,
    *,
    full_text: str,
    prompt_codes: torch.Tensor,
    args: argparse.Namespace,
    context_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_eval_mode(model)
    if args.use_torch_compile:
        maybe_compile_model(model)

    with torch.inference_mode():
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
        delay_pattern_Cx = torch.tensor(
            delay_pattern,
            device=model.device,
            dtype=torch.long,
        )

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
            ).detach()
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
        if args.trim_audio_prompt:
            selected_audio = context_audio[context_trim_samples:context_end_sample]
            trimmed_samples = context_trim_samples
        else:
            selected_audio = context_audio[:context_end_sample]
            trimmed_samples = 0
        continuation_audio = np.asarray(selected_audio, dtype=np.float32).reshape(-1)

        metadata = {
            "context_frames": int(context_frames),
            "prompt_code_steps": int(prompt_codes.shape[0]),
            "prefill_steps": list(prefill_steps_preview),
            "decode_start": int(decode_start),
            "decode_end": int(decode_start + total_delayed_frames),
            "generated_token_count": int(generated_token_count),
            "trimmed_samples": int(trimmed_samples),
            "output_duration_ms": 1000.0 * float(len(continuation_audio)) / float(DEFAULT_SAMPLE_RATE),
            "prompt_duration_ms": float(prompt_codes.shape[0]) * SAMPLE_RATE_RATIO / DEFAULT_SAMPLE_RATE * 1000.0,
            "requested_max_tokens": int(args.max_tokens),
            "effective_max_tokens": int(effective_max_tokens),
        }

    del text_tokens
    del text_tensor
    del prefill_preview
    del dec_state
    del dec_output
    del current_idx
    del eos_detected_Bx
    del eos_countdown_Bx
    del finished_step_Bx
    del delay_pattern_Cx
    del delayed_codes
    cleanup_torch_memory(getattr(model, "device", None))
    return continuation_audio, metadata


def run_set(
    model: Dia,
    *,
    output_dir: Path,
    phrases: list[str],
    prompt_transcript: str,
    audio_prompt: str | torch.Tensor | None,
    args: argparse.Namespace,
    with_prompt: bool,
    run_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_results: list[dict[str, Any]] = []

    for index, phrase in enumerate(phrases, start=1):
        full_text = build_full_text(prompt_transcript, phrase) if with_prompt else phrase
        if with_prompt and args.use_streamer_style_anchor:
            audio_np, generation_metadata = generate_streamer_style_continuation(
                model,
                full_text=full_text,
                prompt_codes=audio_prompt,
                args=args,
                context_frames=max(0, int(args.audio_prompt_context_frames)),
            )
        else:
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
                audio_prompt_context_frames=max(0, int(args.audio_prompt_context_frames)),
                verbose=False,
            )
            audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
            generation_metadata = dict(getattr(model, "last_generate_metadata", {}))

        wav_path = output_dir / f"per_phrase_{index:03d}.wav"
        sf.write(wav_path, audio_np, DEFAULT_SAMPLE_RATE)
        stats = audio_stats(audio_np)
        phrase_results.append(
            {
                "phrase_index": index,
                "text": phrase,
                "full_text": full_text,
                "wav_path": str(wav_path),
                "name": run_name,
                "context_frames": generation_metadata.get("context_frames"),
                "prompt_code_steps": generation_metadata.get("prompt_code_steps"),
                "prefill_steps": generation_metadata.get("prefill_steps"),
                "decode_start": generation_metadata.get("decode_start"),
                "decode_end": generation_metadata.get("decode_end"),
                "generated_token_count": generation_metadata.get("generated_token_count"),
                "output_duration_ms": generation_metadata.get(
                    "output_duration_ms",
                    1000.0 * float(len(audio_np)) / float(DEFAULT_SAMPLE_RATE),
                ),
                "min": stats["min"],
                "max": stats["max"],
                "rms": stats["rms"],
                "audio_stats": stats,
                "generation_metadata": generation_metadata,
                "generation_params": {
                    "use_torch_compile": bool(args.use_torch_compile),
                    "use_streamer_style_anchor": bool(args.use_streamer_style_anchor),
                    "trim_audio_prompt": bool(args.trim_audio_prompt),
                    "audio_prompt_context_frames": int(args.audio_prompt_context_frames),
                    "cfg_scale": float(args.cfg_scale),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "cfg_filter_top_k": int(args.cfg_filter_top_k),
                    "max_tokens": int(args.max_tokens),
                },
            }
        )
        print(f"run={run_name} phrase={index}")
        print(f"  wav_path={wav_path}")
        print(
            f"  rms={stats['rms']:.5f} duration_sec={stats['duration_sec']:.3f} "
            f"min={stats['min']:.5f} max={stats['max']:.5f}"
        )
        print(f"  generation_metadata={json.dumps(generation_metadata, ensure_ascii=False)}")
        cleanup_torch_memory(getattr(model, "device", None))

    return {
        "name": run_name,
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
        run_name="no_prompt",
    )
    with_prompt_dir_name = f"with_prompt_context{max(0, int(args.audio_prompt_context_frames))}"
    with_prompt = run_set(
        model,
        output_dir=output_dir / with_prompt_dir_name,
        phrases=TEST_PHRASES,
        prompt_transcript=args.prompt_transcript,
        audio_prompt=prompt_codes,
        args=args,
        with_prompt=True,
        run_name=with_prompt_dir_name,
    )

    metadata = {
        "args": vars(args),
        "reference_wav": str(reference_wav),
        "prompt_codes_path": str(prompt_codes_path) if prompt_codes_path is not None else None,
        "prompt_codes_shape": list(prompt_codes.shape),
        "prompt_codes_dtype": str(prompt_codes.dtype),
        "runs": {
            "no_prompt": no_prompt,
            with_prompt_dir_name: with_prompt,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata_path={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
