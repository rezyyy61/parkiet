from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from parkiet.dia.audio import apply_audio_delay, build_delay_indices
from parkiet.dia.config import DiaConfig
from parkiet.dia.model import Dia, load_state_dict_allowing_missing_speaker_modules
from parkiet.dia.state import DecoderInferenceState, EncoderInferenceState
from parkiet.jax.dataset import create_dataset, discover_parquet_shards
from parkiet.speaker_checkpoint import extract_speaker_module_state_dict
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


def resolve_precision(requested_precision: str | None, device: torch.device) -> str:
    if requested_precision is not None:
        return requested_precision
    if device.type == "cuda":
        return "bfloat16"
    return "float32"


def get_autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_cuda_memory_stats(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    allocated_gb = float(torch.cuda.memory_allocated(device)) / (1024.0 ** 3)
    reserved_gb = float(torch.cuda.memory_reserved(device)) / (1024.0 ** 3)
    return allocated_gb, reserved_gb


def filter_rows_by_duration_ms(
    rows: list[dict[str, Any]],
    max_duration_ms: float,
) -> list[dict[str, Any]]:
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        duration_ms = row.get("duration_ms")
        if duration_ms is None or not pd.notna(duration_ms):
            continue
        if float(duration_ms) <= max_duration_ms:
            filtered_rows.append(row)
    return filtered_rows


def load_filtered_training_frame(
    parquet_path: str | Path,
    max_duration_sec: float,
) -> pd.DataFrame:
    max_duration_ms = float(max_duration_sec) * 1000.0
    frames: list[pd.DataFrame] = []
    for parquet_file in discover_parquet_shards(str(parquet_path)):
        frame = pd.read_parquet(parquet_file)
        if "duration_ms" not in frame.columns:
            raise ValueError(
                f"Parquet shard {parquet_file} is missing required column 'duration_ms'"
            )
        filtered_frame = frame[frame["duration_ms"].notna() & (frame["duration_ms"] <= max_duration_ms)]
        if not filtered_frame.empty:
            frames.append(filtered_frame.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_filtered_dataset(
    config: DiaConfig,
    parquet_path: str | Path,
    speaker_vocab: dict[str, Any],
    max_duration_sec: float,
    seed: int,
):
    dataset = create_dataset(
        config=config,
        parquet_path=str(parquet_path),
        transcription_clean_prob=0.0,
        text_dropout_prob=0.0,
        speaker_vocab=speaker_vocab,
    )
    filtered_df = load_filtered_training_frame(parquet_path, max_duration_sec)
    if filtered_df.empty:
        raise RuntimeError(
            "No parquet rows remain after duration filter: "
            f"max_duration_sec={max_duration_sec}"
        )
    dataset.df = filtered_df.reset_index(drop=True)
    dataset.parquet_files = ["filtered_in_memory.parquet"]
    dataset.current_shard_idx = 0
    dataset.rng = np.random.RandomState(seed)
    return dataset


def iter_training_batches(dataset, batch_size: int, seed: int):
    rng = np.random.RandomState(seed)
    while True:
        num_samples = len(dataset.df)
        if num_samples <= 0:
            raise RuntimeError("Filtered training dataset is empty")
        indices = np.arange(num_samples)
        if num_samples > 1:
            indices = rng.permutation(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            if len(batch_indices) < batch_size:
                continue
            batch_data: dict[str, list[Any]] = {"text": [], "audio": [], "cb_weight": []}
            for idx in batch_indices:
                sample = dataset[int(idx)]
                batch_data["text"].append(sample["text"])
                batch_data["audio"].append(sample["audio"])
                batch_data["cb_weight"].append(sample["cb_weight"])
                if "speaker_id" in sample:
                    batch_data.setdefault("speaker_id", []).append(sample["speaker_id"])
            yield batch_data


def configure_trainable_parameters(
    dia: Dia,
    train_speaker_modules_only: bool,
) -> list[torch.nn.Parameter]:
    trainable_parameters: list[torch.nn.Parameter] = []
    for name, parameter in dia.model.named_parameters():
        should_train = True
        if train_speaker_modules_only:
            should_train = name.startswith("speaker_embedding") or name.startswith(
                "speaker_to_encoder"
            ) or name.startswith("speaker_to_decoder")
        parameter.requires_grad = should_train
        if should_train:
            trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters selected for tiny speaker training")
    return trainable_parameters


def save_training_checkpoints(
    dia: Dia,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    full_checkpoint_path = output_dir_path / "checkpoint_tiny_speaker_conditioned.pt"
    full_state_dict = dia.model.state_dict()
    torch.save(full_state_dict, full_checkpoint_path)

    speaker_modules_only_path = output_dir_path / "speaker_modules_only.pt"
    speaker_only_state_dict = extract_speaker_module_state_dict(full_state_dict)
    torch.save(speaker_only_state_dict, speaker_modules_only_path)

    return {
        "output_checkpoint_path": str(full_checkpoint_path),
        "speaker_modules_only_checkpoint_path": str(speaker_modules_only_path),
        "full_checkpoint_size_bytes": full_checkpoint_path.stat().st_size,
        "speaker_modules_only_checkpoint_size_bytes": speaker_modules_only_path.stat().st_size,
        "speaker_modules_only_state_keys": sorted(speaker_only_state_dict.keys()),
    }


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
    return iter_training_batches(dataset, batch_size=batch_size, seed=seed)


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
    max_duration_sec: float = 3.0,
    precision: str | None = None,
    clear_cache_each_step: bool = False,
    train_speaker_modules_only: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = DiaConfig.load(str(config_path))
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config.speaker_conditioning_enabled:
        raise ValueError("Expected speaker_conditioning_enabled=True")

    speaker_vocab = load_speaker_vocab(speaker_vocab_path)
    full_parquet_summary = summarize_parquet_speaker_ids(parquet_path, speaker_vocab)
    filtered_df = load_filtered_training_frame(parquet_path, max_duration_sec)
    filtered_rows = filtered_df.to_dict(orient="records")
    parquet_summary = summarize_rows_mapped_speaker_ids(filtered_rows, speaker_vocab)
    parquet_summary["parquet_files"] = len(discover_parquet_shards(str(parquet_path)))
    parquet_summary["max_duration_sec"] = float(max_duration_sec)
    if 1 not in parquet_summary["speaker_id_unique"]:
        raise RuntimeError(
            "Parquet speaker mapping does not include speaker_id 1: "
            f"{parquet_summary['speaker_id_unique']}"
        )
    if parquet_summary["row_count"] <= 0:
        raise RuntimeError(
            "No parquet rows remain after duration filter: "
            f"max_duration_sec={max_duration_sec}"
        )

    if batch_size > parquet_summary["row_count"]:
        raise ValueError(
            f"batch_size={batch_size} exceeds available parquet rows={parquet_summary['row_count']}"
        )
    dataset = build_filtered_dataset(
        config=config,
        parquet_path=parquet_path,
        speaker_vocab=speaker_vocab,
        max_duration_sec=max_duration_sec,
        seed=seed,
    )
    batch_iterator = _create_batch_iterator(dataset, batch_size, seed)

    dia, checkpoint_summary = load_conditioned_model_for_training(
        config,
        checkpoint_path,
    )
    resolved_precision = resolve_precision(precision, dia.device)
    trainable_parameters = configure_trainable_parameters(
        dia,
        train_speaker_modules_only=train_speaker_modules_only,
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    loss_history: list[float] = []
    speaker_id_unique_seen: set[int] = set()
    initial_loss: float | None = None
    final_loss: float | None = None
    completed_steps = 0
    progress_every = max(1, min(5, max_steps))
    cuda_memory_allocated_gb_last: float | None = None
    cuda_memory_reserved_gb_last: float | None = None

    while completed_steps < max_steps:
        raw_batch = next(batch_iterator)

        batch = build_training_batch(raw_batch, config, dia.device)
        batch_speaker_ids = summarize_batch_speaker_ids(batch)
        speaker_id_unique_seen.update(batch_speaker_ids)

        if batch_speaker_ids == [0]:
            raise RuntimeError("Tiny train saw only default speaker_id 0 in a batch")

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(dia.device, resolved_precision):
            loss, metrics = compute_training_loss(dia, batch)
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at step {completed_steps + 1}: {loss_value}")

        loss.backward()
        optimizer.step()
        if clear_cache_each_step and dia.device.type == "cuda":
            torch.cuda.empty_cache()

        loss_history.append(loss_value)
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        completed_steps += 1
        cuda_memory_allocated_gb_last, cuda_memory_reserved_gb_last = (
            get_cuda_memory_stats(dia.device)
        )

        if completed_steps == 1 or completed_steps % progress_every == 0 or completed_steps == max_steps:
            memory_suffix = ""
            if cuda_memory_allocated_gb_last is not None and cuda_memory_reserved_gb_last is not None:
                memory_suffix = (
                    f" cuda_allocated_gb={cuda_memory_allocated_gb_last:.2f}"
                    f" cuda_reserved_gb={cuda_memory_reserved_gb_last:.2f}"
                )
            print(
                f"step={completed_steps} "
                f"loss={loss_value:.6f} "
                f"speaker_id_unique={batch_speaker_ids}"
                f"{memory_suffix}"
            )

        del raw_batch
        del batch
        del loss
        del metrics
        if dia.device.type == "cuda":
            del batch_speaker_ids

    speaker_id_unique_seen_sorted = sorted(speaker_id_unique_seen)
    if 1 not in speaker_id_unique_seen_sorted:
        raise RuntimeError(
            "speaker_id_unique_seen does not include 1: "
            f"{speaker_id_unique_seen_sorted}"
        )
    if speaker_id_unique_seen_sorted == [0]:
        raise RuntimeError("speaker_id_unique_seen only contains default speaker_id 0")

    checkpoint_artifacts = save_training_checkpoints(dia, output_dir_path)

    report = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "parquet_path": str(parquet_path),
        "speaker_vocab_path": str(speaker_vocab_path),
        "output_checkpoint_path": checkpoint_artifacts["output_checkpoint_path"],
        "speaker_modules_only_checkpoint_path": checkpoint_artifacts[
            "speaker_modules_only_checkpoint_path"
        ],
        "max_steps": int(max_steps),
        "completed_steps": int(completed_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(dia.device),
        "max_duration_sec": float(max_duration_sec),
        "precision": resolved_precision,
        "train_speaker_modules_only": bool(train_speaker_modules_only),
        "speaker_conditioning_enabled": True,
        "num_speakers": int(config.num_speakers),
        "speaker_id_unique_seen": speaker_id_unique_seen_sorted,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "parquet_speaker_id_unique": parquet_summary["speaker_id_unique"],
        "full_parquet_speaker_id_unique": full_parquet_summary["speaker_id_unique"],
        "initialized_speaker_module_keys": checkpoint_summary[
            "initialized_speaker_module_keys"
        ],
        "speaker_modules_only_state_keys": checkpoint_artifacts[
            "speaker_modules_only_state_keys"
        ],
        "speaker_modules_only_checkpoint_size_bytes": checkpoint_artifacts[
            "speaker_modules_only_checkpoint_size_bytes"
        ],
        "full_checkpoint_size_bytes": checkpoint_artifacts[
            "full_checkpoint_size_bytes"
        ],
        "cuda_memory_allocated_gb_last": cuda_memory_allocated_gb_last,
        "cuda_memory_reserved_gb_last": cuda_memory_reserved_gb_last,
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
    parser.add_argument("--max-duration-sec", type=float, default=3.0)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--clear-cache-each-step", action="store_true")
    parser.add_argument(
        "--train-speaker-modules-only",
        dest="train_speaker_modules_only",
        action="store_true",
    )
    parser.add_argument(
        "--train-all-modules",
        dest="train_speaker_modules_only",
        action="store_false",
    )
    parser.set_defaults(train_speaker_modules_only=True)
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
        max_duration_sec=args.max_duration_sec,
        precision=args.precision,
        clear_cache_each_step=args.clear_cache_each_step,
        train_speaker_modules_only=args.train_speaker_modules_only,
    )
    print(f"completed_steps={report['completed_steps']}")
    print(f"speaker_id_unique_seen={report['speaker_id_unique_seen']}")
    print(f"initial_loss={report['initial_loss']}")
    print(f"final_loss={report['final_loss']}")
    print(f"output_checkpoint_path={report['output_checkpoint_path']}")


if __name__ == "__main__":
    main()
