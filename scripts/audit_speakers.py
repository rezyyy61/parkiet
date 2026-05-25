from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from parkiet.jax.dataset import discover_parquet_shards


DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_source_speaker_id(chunk_owner: Any) -> str | None:
    if chunk_owner is None or pd.isna(chunk_owner):
        return None
    try:
        value = int(chunk_owner)
    except (TypeError, ValueError):
        return str(chunk_owner)
    if value < 0:
        return None
    return str(value)


def derive_duration_sec_from_row(row: pd.Series) -> float | None:
    if "duration_sec" in row and pd.notna(row["duration_sec"]):
        return float(row["duration_sec"])
    if "audio_duration_sec" in row and pd.notna(row["audio_duration_sec"]):
        return float(row["audio_duration_sec"])
    if "duration_ms" in row and pd.notna(row["duration_ms"]):
        return float(row["duration_ms"]) / 1000.0
    if "audio_duration_ms" in row and pd.notna(row["audio_duration_ms"]):
        return float(row["audio_duration_ms"]) / 1000.0
    if "encoded_audio_shape" in row and row["encoded_audio_shape"] is not None:
        shape = row["encoded_audio_shape"]
        if hasattr(shape, "tolist"):
            shape = shape.tolist()
        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
            try:
                time_steps = int(shape[0])
            except (TypeError, ValueError):
                return None
            return float(time_steps * SAMPLE_RATE_RATIO) / float(DEFAULT_SAMPLE_RATE)
    return None


def _maybe_float_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _maybe_int_summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": int(min(values)), "max": int(max(values))}


def _is_missing_text(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip() == ""


def summarize_speaker_rows(
    rows: list[pd.Series],
    *,
    chunk_owner: str,
    min_duration_sec: float,
    min_sample_count: int,
) -> dict[str, Any]:
    sample_count = len(rows)
    durations = [duration for row in rows if (duration := derive_duration_sec_from_row(row)) is not None]
    total_duration_sec = float(sum(durations)) if durations else None
    avg_duration_sec = (float(sum(durations)) / len(durations)) if durations else None

    clean_count = 0
    missing_transcription_count = 0
    cb_weights: list[float] = []
    encoded_audio_lengths: list[int] = []
    encoded_audio_channels: list[int] = []

    for row in rows:
        if "transcription_clean" in row and not _is_missing_text(row["transcription_clean"]):
            clean_count += 1
        if "transcription" not in row or _is_missing_text(row["transcription"]):
            missing_transcription_count += 1
        if "cb_weight" in row and pd.notna(row["cb_weight"]):
            cb_weights.append(float(row["cb_weight"]))
        if "encoded_audio_shape" in row and row["encoded_audio_shape"] is not None:
            shape = row["encoded_audio_shape"]
            if hasattr(shape, "tolist"):
                shape = shape.tolist()
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                try:
                    encoded_audio_lengths.append(int(shape[0]))
                    encoded_audio_channels.append(int(shape[1]))
                except (TypeError, ValueError):
                    pass

    candidate = sample_count >= min_sample_count and (
        total_duration_sec is None or total_duration_sec >= min_duration_sec
    )
    quality_flags: list[str] = []
    if sample_count < min_sample_count:
        quality_flags.append("low_sample_count")
    if total_duration_sec is not None and total_duration_sec < min_duration_sec:
        quality_flags.append("low_duration")
    if missing_transcription_count > 0:
        quality_flags.append("missing_transcription")

    return {
        "chunk_owner": chunk_owner,
        "source_speaker_id": chunk_owner,
        "sample_count": sample_count,
        "total_duration_sec": total_duration_sec,
        "avg_duration_sec": avg_duration_sec,
        "has_transcription_clean_count": clean_count,
        "missing_transcription_count": missing_transcription_count,
        "cb_weight_summary": _maybe_float_summary(cb_weights),
        "encoded_audio_shape_summary": {
            "time_steps": _maybe_int_summary(encoded_audio_lengths),
            "channels": sorted(set(encoded_audio_channels)),
        },
        "candidate": candidate,
        "quality_flags": quality_flags,
    }


def audit_parquet_speakers(
    parquet_path: str | Path,
    *,
    min_duration_sec: float = 300.0,
    min_sample_count: int = 50,
) -> dict[str, Any]:
    shards = discover_parquet_shards(str(parquet_path))
    grouped_rows: dict[str, list[pd.Series]] = defaultdict(list)

    for shard_path in shards:
        frame = pd.read_parquet(shard_path)
        for _, row in frame.iterrows():
            source_speaker_id = normalize_source_speaker_id(row.get("chunk_owner"))
            if source_speaker_id is None:
                continue
            grouped_rows[source_speaker_id].append(row)

    speakers = [
        summarize_speaker_rows(
            rows,
            chunk_owner=chunk_owner,
            min_duration_sec=min_duration_sec,
            min_sample_count=min_sample_count,
        )
        for chunk_owner, rows in sorted(grouped_rows.items(), key=lambda item: int(item[0]))
    ]
    candidates = [speaker for speaker in speakers if speaker["candidate"]]
    return {
        "generated_at": iso_now(),
        "parquet_path": str(parquet_path),
        "min_duration_sec": float(min_duration_sec),
        "min_sample_count": int(min_sample_count),
        "speaker_count": len(speakers),
        "candidate_count": len(candidates),
        "speakers": speakers,
        "candidates": candidates,
    }


def write_audit_outputs(audit: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    speaker_audit_json = output_path / "speaker_audit.json"
    speaker_audit_csv = output_path / "speaker_audit.csv"
    candidate_json = output_path / "candidate_speakers.json"

    speaker_audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    candidate_json.write_text(
        json.dumps(
            {
                "generated_at": audit["generated_at"],
                "parquet_path": audit["parquet_path"],
                "min_duration_sec": audit["min_duration_sec"],
                "min_sample_count": audit["min_sample_count"],
                "candidates": audit["candidates"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "chunk_owner",
        "sample_count",
        "total_duration_sec",
        "avg_duration_sec",
        "has_transcription_clean_count",
        "missing_transcription_count",
        "cb_weight_min",
        "cb_weight_max",
        "cb_weight_mean",
        "encoded_audio_time_steps_min",
        "encoded_audio_time_steps_max",
        "encoded_audio_channels",
        "candidate",
        "quality_flags",
    ]
    with speaker_audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for speaker in audit["speakers"]:
            writer.writerow(
                {
                    "chunk_owner": speaker["chunk_owner"],
                    "sample_count": speaker["sample_count"],
                    "total_duration_sec": speaker["total_duration_sec"],
                    "avg_duration_sec": speaker["avg_duration_sec"],
                    "has_transcription_clean_count": speaker["has_transcription_clean_count"],
                    "missing_transcription_count": speaker["missing_transcription_count"],
                    "cb_weight_min": speaker["cb_weight_summary"]["min"],
                    "cb_weight_max": speaker["cb_weight_summary"]["max"],
                    "cb_weight_mean": speaker["cb_weight_summary"]["mean"],
                    "encoded_audio_time_steps_min": speaker["encoded_audio_shape_summary"]["time_steps"]["min"],
                    "encoded_audio_time_steps_max": speaker["encoded_audio_shape_summary"]["time_steps"]["max"],
                    "encoded_audio_channels": ",".join(
                        str(value) for value in speaker["encoded_audio_shape_summary"]["channels"]
                    ),
                    "candidate": speaker["candidate"],
                    "quality_flags": ",".join(speaker["quality_flags"]),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit parquet speakers for conditioning")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-duration-sec", type=float, default=300.0)
    parser.add_argument("--min-sample-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_parquet_speakers(
        args.parquet_path,
        min_duration_sec=args.min_duration_sec,
        min_sample_count=args.min_sample_count,
    )
    write_audit_outputs(audit, args.output_dir)
    print(f"speaker_count={audit['speaker_count']}")
    print(f"candidate_count={audit['candidate_count']}")


if __name__ == "__main__":
    main()
