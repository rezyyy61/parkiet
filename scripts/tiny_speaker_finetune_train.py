from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from parkiet.dia.audio import apply_audio_delay, build_delay_indices
from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia, load_state_dict_allowing_missing_speaker_modules
from parkiet.dia.state import DecoderInferenceState, EncoderInferenceState
from parkiet.jax.dataset import create_dataset
from parkiet.speaker_dataset_summary import (
    summarize_parquet_speaker_ids,
    summarize_rows_mapped_speaker_ids,
)
from parkiet.speaker_vocab import load_speaker_vocab


def summarize_batch_speaker_ids(batch: dict[str, torch.Tensor]) -> list[int]:
    speaker_ids = batch.get("speaker_id")
    if speaker_ids is None:
        return []
    return sorted({int(v) for v in speaker_ids.detach().cpu().reshape(-1).tolist()})


def prepare_input_target_pair(
    audio_sequence: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio_input = audio_sequence
    pad_tail = torch.full(
        (audio_sequence.shape[0], 1, audio_sequence.shape[2]),
        fill_value=pad_token_id,
        dtype=audio_sequence.dtype,
        device=audio_sequence.device,
    )
    audio_target = torch.cat([audio_sequence[:, 1:, :], pad_tail], dim=1)
    return audio_input, audio_target


def build_training_batch(
    raw_batch: dict[str, Any],
    config: DiaConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text = torch.as_tensor(np.asarray(raw_batch["text"]), dtype=torch.long, device=device)
    audio = torch.as_tensor(np.asarray(raw_batch["audio"]), dtype=torch.long, device=device)
    batch_size, seq_len, num_channels = audio.shape
    delay_precomp = build_delay_indices(
        B=batch_size,
        T=seq_len,
        C=num_channels,
        delay_pattern=list(config.delay_pattern),
    )
    delayed_audio = apply_audio_delay(
        audio_BxTxC=audio,
        pad_value=config.pad_token_id,
        bos_value=config.bos_token_id,
        precomp=delay_precomp,
    )
    audio_input, audio_target = prepare_input_target_pair(
        delayed_audio,
        config.pad_token_id,
    )
    batch = {
        "text": text,
        "audio_input": audio_input,
        "audio_target": audio_target,
    }
    if "speaker_id" in raw_batch:
        batch["speaker_id"] = torch.as_tensor(
            np.asarray(raw_batch["speaker_id"]),
            dtype=torch.long,
            device=device,
        )
    return batch


def compute_training_loss(
    dia: Dia,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    text_tokens = batch["text"]
    audio_input = batch["audio_input"]
    audio_target = batch["audio_target"]
    speaker_ids = batch.get("speaker_id")
    config = dia.config
    batch_size = text_tokens.shape[0]

    encoder_state = EncoderInferenceState.new(
        config,
        text_tokens,
        batch_multiplier=1,
    )
    encoder_speaker_condition = None
    decoder_speaker_condition = None
    if config.speaker_conditioning_enabled:
        if speaker_ids is None:
            raise ValueError(
                "speaker_id is required when speaker_conditioning_enabled=True"
            )
        encoder_speaker_condition, decoder_speaker_condition = (
            dia.model.get_speaker_condition(speaker_ids)
        )

    encoder_outputs = dia.model.encoder(
        text_tokens,
        encoder_state,
        speaker_condition=encoder_speaker_condition,
    )
    cross_attn_cache = dia.model.decoder.precompute_cross_attn_cache(encoder_outputs)
    decoder_state = DecoderInferenceState.new(
        config,
        encoder_state,
        encoder_outputs,
        cross_attn_cache,
        dia.compute_dtype,
        max_generation_length=audio_input.shape[1],
        batch_multiplier=1,
    )
    decoder_state.prepare_step(0, audio_input.shape[1])
    decoder_outputs = dia.model.decoder(
        audio_input,
        decoder_state,
        speaker_condition=decoder_speaker_condition,
    )

    vocab_size = decoder_outputs.shape[-1]
    logits = decoder_outputs.reshape(-1, vocab_size)
    targets = audio_target.reshape(-1)

    num_channels = config.decoder_config.num_channels
    channel_weights = torch.ones(
        (batch_size, audio_target.shape[1], num_channels),
        dtype=torch.float32,
        device=audio_target.device,
    )
    channel_weights[:, :, 0] = 4.0
    channel_weights_flat = channel_weights.reshape(-1)

    meaningful_mask = targets != config.pad_token_id
    loss_per_token = F.cross_entropy(logits, targets, reduction="none")
    valid_loss = loss_per_token * meaningful_mask.to(loss_per_token.dtype) * channel_weights_flat
    total_valid_weight = torch.sum(
        meaningful_mask.to(channel_weights_flat.dtype) * channel_weights_flat
    )
    loss = torch.sum(valid_loss) / torch.clamp(total_valid_weight, min=1e-8)

    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets) & meaningful_mask
    accuracy = (
        correct.to(torch.float32).sum()
        / torch.clamp(meaningful_mask.to(torch.float32).sum(), min=1.0)
    )

    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "accuracy": float(accuracy.detach().cpu().item()),
        "non_pad_tokens": float(meaningful_mask.to(torch.float32).sum().detach().cpu().item()),
    }
    return loss, metrics


def load_conditioned_model_for_training(
    config: DiaConfig,
    checkpoint_path: str | Path,
) -> tuple[Dia, dict[str, Any]]:
    dia = Dia(config=config, compute_dtype="float32", load_dac=False)
    state_dict = torch.load(checkpoint_path, map_location=dia.device)
    missing_keys, unexpected_keys = load_state_dict_allowing_missing_speaker_modules(
        dia.model,
        state_dict,
        allow_missing_speaker_modules=config.speaker_conditioning_enabled,
    )
    dia.model.to(dia.device)
    dia.model.train()
    return dia, {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "initialized_speaker_module_keys": [
            key for key in missing_keys if key.startswith("speaker_")
        ],
    }


def _create_batch_iterator(dataset, batch_size: int, seed: int):
    return dataset.batch_iterator(
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        use_sample_prob=False,
    )


def run_tiny_train(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    parquet_path: str | Path,
    speaker_vocab_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    seed: int = 1234,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config.speaker_conditioning_enabled:
        raise ValueError("Expected speaker_conditioning_enabled=True")

    speaker_vocab = load_speaker_vocab(speaker_vocab_path)
    parquet_summary = summarize_parquet_speaker_ids(parquet_path, speaker_vocab)
    if 1 not in parquet_summary["speaker_id_unique"]:
        raise RuntimeError(
            "Parquet speaker mapping does not include speaker_id 1: "
            f"{parquet_summary['speaker_id_unique']}"
        )

    dataset = create_dataset(
        config=config,
        parquet_path=str(parquet_path),
        transcription_clean_prob=0.0,
        text_dropout_prob=0.0,
        speaker_vocab=speaker_vocab,
    )
    if parquet_summary["row_count"] <= 0:
        raise RuntimeError("No parquet rows found for tiny speaker-conditioned training")
    if batch_size > parquet_summary["row_count"]:
        raise ValueError(
            f"batch_size={batch_size} exceeds available parquet rows={parquet_summary['row_count']}"
        )
    dataset.rng = np.random.RandomState(seed)
    batch_iterator = _create_batch_iterator(dataset, batch_size, seed)

    dia, checkpoint_summary = load_conditioned_model_for_training(
        config,
        checkpoint_path,
    )
    optimizer = torch.optim.AdamW(dia.model.parameters(), lr=learning_rate)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    loss_history: list[float] = []
    speaker_id_unique_seen: set[int] = set()
    initial_loss: float | None = None
    final_loss: float | None = None
    completed_steps = 0
    progress_every = max(1, min(5, max_steps))

    while completed_steps < max_steps:
        try:
            raw_batch = next(batch_iterator)
        except StopIteration:
            dataset.reset()
            batch_iterator = _create_batch_iterator(
                dataset,
                batch_size,
                seed + completed_steps + 1,
            )
            continue

        batch = build_training_batch(raw_batch, config, dia.device)
        batch_speaker_ids = summarize_batch_speaker_ids(batch)
        speaker_id_unique_seen.update(batch_speaker_ids)

        if batch_speaker_ids == [0]:
            raise RuntimeError("Tiny train saw only default speaker_id 0 in a batch")

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_training_loss(dia, batch)
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at step {completed_steps + 1}: {loss_value}")

        loss.backward()
        optimizer.step()

        loss_history.append(loss_value)
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        completed_steps += 1

        if completed_steps == 1 or completed_steps % progress_every == 0 or completed_steps == max_steps:
            print(
                f"step={completed_steps} "
                f"loss={loss_value:.6f} "
                f"speaker_id_unique={batch_speaker_ids}"
            )

    speaker_id_unique_seen_sorted = sorted(speaker_id_unique_seen)
    if 1 not in speaker_id_unique_seen_sorted:
        raise RuntimeError(
            "speaker_id_unique_seen does not include 1: "
            f"{speaker_id_unique_seen_sorted}"
        )
    if speaker_id_unique_seen_sorted == [0]:
        raise RuntimeError("speaker_id_unique_seen only contains default speaker_id 0")

    output_checkpoint_path = output_dir_path / "checkpoint_tiny_speaker_conditioned.pt"
    torch.save(dia.model.state_dict(), output_checkpoint_path)

    report = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "parquet_path": str(parquet_path),
        "speaker_vocab_path": str(speaker_vocab_path),
        "output_checkpoint_path": str(output_checkpoint_path),
        "max_steps": int(max_steps),
        "completed_steps": int(completed_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(dia.device),
        "speaker_conditioning_enabled": True,
        "num_speakers": int(config.num_speakers),
        "speaker_id_unique_seen": speaker_id_unique_seen_sorted,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "parquet_speaker_id_unique": parquet_summary["speaker_id_unique"],
        "initialized_speaker_module_keys": checkpoint_summary[
            "initialized_speaker_module_keys"
        ],
    }
    report_path = output_dir_path / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny real speaker-conditioned fine-tune smoke run"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--speaker-vocab-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_tiny_train(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        parquet_path=args.parquet_path,
        speaker_vocab_path=args.speaker_vocab_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(f"completed_steps={report['completed_steps']}")
    print(f"speaker_id_unique_seen={report['speaker_id_unique_seen']}")
    print(f"initial_loss={report['initial_loss']}")
    print(f"final_loss={report['final_loss']}")
    print(f"output_checkpoint_path={report['output_checkpoint_path']}")


if __name__ == "__main__":
    main()
