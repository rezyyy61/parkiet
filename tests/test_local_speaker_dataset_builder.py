from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from scripts.build_local_speaker_parquet import (
    build_local_speaker_dataset,
    write_outputs,
)


def _write_fake_wav(path: Path) -> None:
    path.write_bytes(b"fake")


def test_folder_scanning_and_stable_speaker_ids(tmp_path: Path, monkeypatch):
    root = tmp_path / "local_speakers"
    (root / "nl_female_01").mkdir(parents=True)
    (root / "nl_male_01").mkdir(parents=True)
    _write_fake_wav(root / "nl_female_01" / "001.wav")
    (root / "nl_female_01" / "001.txt").write_text("[S1] hallo", encoding="utf-8")
    _write_fake_wav(root / "nl_male_01" / "001.wav")
    (root / "nl_male_01" / "001.txt").write_text("[S1] daar", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.build_local_speaker_parquet.get_audio_duration_sec",
        lambda audio_path: 3.0,
    )
    rows, source_speaker_map, report = build_local_speaker_dataset(
        input_dir=root,
        audio_encoder=lambda audio_path: torch.arange(18, dtype=torch.int32).reshape(2, 9),
        min_duration_sec=1.0,
        max_duration_sec=30.0,
    )
    assert source_speaker_map == {"nl_female_01": 1, "nl_male_01": 2}
    assert len(rows) == 2
    assert report["row_count"] == 2


def test_rows_match_expected_schema_fields(tmp_path: Path, monkeypatch):
    root = tmp_path / "local_speakers"
    (root / "nl_female_01").mkdir(parents=True)
    _write_fake_wav(root / "nl_female_01" / "001.wav")
    (root / "nl_female_01" / "001.txt").write_text("[S1] hallo", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.build_local_speaker_parquet.get_audio_duration_sec",
        lambda audio_path: 2.5,
    )

    rows, source_speaker_map, report = build_local_speaker_dataset(
        input_dir=root,
        audio_encoder=lambda audio_path: torch.arange(18, dtype=torch.int32).reshape(2, 9),
    )
    row = rows[0]
    assert set(row.keys()) == {
        "source_file",
        "chunk_id",
        "start_ms",
        "end_ms",
        "duration_ms",
        "transcription",
        "transcription_clean",
        "file_path",
        "chunk_owner",
        "sample_prob",
        "cb_weight",
        "encoded_audio_shape",
        "encoded_audio",
    }
    assert source_speaker_map == {"nl_female_01": 1}
    assert report["errors"] == []


def test_missing_txt_is_reported(tmp_path: Path, monkeypatch):
    root = tmp_path / "local_speakers"
    (root / "nl_female_01").mkdir(parents=True)
    _write_fake_wav(root / "nl_female_01" / "001.wav")
    monkeypatch.setattr(
        "scripts.build_local_speaker_parquet.get_audio_duration_sec",
        lambda audio_path: 2.0,
    )
    rows, source_speaker_map, report = build_local_speaker_dataset(
        input_dir=root,
        audio_encoder=lambda audio_path: torch.arange(18, dtype=torch.int32).reshape(2, 9),
    )
    assert rows == []
    assert source_speaker_map == {"nl_female_01": 1}
    assert report["errors"][0]["type"] == "missing_transcript"
    assert report["skipped"]["missing_transcript"] == [str(root / "nl_female_01" / "001.wav")]


def test_output_report_and_parquet_are_written(tmp_path: Path, monkeypatch):
    root = tmp_path / "local_speakers"
    (root / "nl_female_01").mkdir(parents=True)
    _write_fake_wav(root / "nl_female_01" / "001.wav")
    (root / "nl_female_01" / "001.txt").write_text("[S1] hallo", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.build_local_speaker_parquet.get_audio_duration_sec",
        lambda audio_path: 2.0,
    )
    rows, source_speaker_map, report = build_local_speaker_dataset(
        input_dir=root,
        audio_encoder=lambda audio_path: torch.arange(18, dtype=torch.int32).reshape(2, 9),
    )
    output_path = tmp_path / "parquet" / "local_speakers.parquet"
    write_outputs(
        rows=rows,
        source_speaker_map=source_speaker_map,
        report=report,
        output_path=output_path,
    )
    assert output_path.exists()
    assert (output_path.parent / "source_speaker_map.json").exists()
    assert (output_path.parent / "build_report.json").exists()

    table = pq.read_table(output_path)
    assert table.num_rows == 1
    saved_map = json.loads((output_path.parent / "source_speaker_map.json").read_text(encoding="utf-8"))
    assert saved_map == {"nl_female_01": 1}
