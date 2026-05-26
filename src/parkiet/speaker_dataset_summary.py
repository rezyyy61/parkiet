from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from parkiet.jax.dataset import discover_parquet_shards
from parkiet.speaker_vocab import map_chunk_owner_to_speaker_id


def summarize_rows_mapped_speaker_ids(
    rows: list[dict[str, Any]],
    speaker_vocab: dict[str, Any],
) -> dict[str, Any]:
    default_speaker_id = int(speaker_vocab.get("default_speaker_id", 0))
    speaker_id_values: list[int] = []
    for row in rows:
        chunk_owner = row.get("chunk_owner", None)
        if chunk_owner is not None and not pd.notna(chunk_owner):
            chunk_owner = None
        speaker_id_values.append(
            map_chunk_owner_to_speaker_id(
                chunk_owner,
                speaker_vocab,
                default_speaker_id=default_speaker_id,
            )
        )
    return {
        "row_count": len(rows),
        "speaker_id_values": speaker_id_values,
        "speaker_id_unique": sorted(set(speaker_id_values)),
    }


def summarize_parquet_speaker_ids(
    parquet_path: str | Path,
    speaker_vocab: dict[str, Any],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    parquet_files = discover_parquet_shards(str(parquet_path))
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        all_rows.extend(df.to_dict(orient="records"))
    summary = summarize_rows_mapped_speaker_ids(all_rows, speaker_vocab)
    summary["parquet_files"] = len(parquet_files)
    return summary
