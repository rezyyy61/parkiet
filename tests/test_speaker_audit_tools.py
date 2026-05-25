from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from parkiet.voice_identity import build_speaker_vocab_from_registry, load_voice_registry
from scripts.audit_speakers import audit_parquet_speakers, write_audit_outputs
from scripts.build_voice_registry_from_audit import (
    build_registry_from_candidates,
    load_candidates,
    write_registry_outputs,
)


def _fake_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_owner": 17,
                "transcription": "[S1] hallo",
                "transcription_clean": "[S1] hallo",
                "duration_ms": 200_000,
                "cb_weight": 1.0,
                "encoded_audio_shape": [100, 9],
            },
            {
                "chunk_owner": 17,
                "transcription": "[S1] daar",
                "transcription_clean": "[S1] daar",
                "duration_ms": 150_000,
                "cb_weight": 1.5,
                "encoded_audio_shape": [120, 9],
            },
            {
                "chunk_owner": 42,
                "transcription": "",
                "transcription_clean": "",
                "duration_ms": 50_000,
                "cb_weight": 0.9,
                "encoded_audio_shape": [80, 9],
            },
        ]
    )


def test_audit_groups_chunk_owner_correctly(monkeypatch):
    monkeypatch.setattr(
        "scripts.audit_speakers.discover_parquet_shards",
        lambda parquet_path: ["fake.parquet"],
    )
    monkeypatch.setattr(pd, "read_parquet", lambda parquet_file: _fake_frame().copy())

    audit = audit_parquet_speakers(
        "unused",
        min_duration_sec=300.0,
        min_sample_count=2,
    )
    assert audit["speaker_count"] == 2
    speaker17 = next(s for s in audit["speakers"] if s["chunk_owner"] == "17")
    assert speaker17["sample_count"] == 2
    assert speaker17["total_duration_sec"] == 350.0
    assert speaker17["candidate"] is True
    speaker42 = next(s for s in audit["speakers"] if s["chunk_owner"] == "42")
    assert speaker42["missing_transcription_count"] == 1
    assert speaker42["candidate"] is False


def test_candidate_threshold_and_output_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "scripts.audit_speakers.discover_parquet_shards",
        lambda parquet_path: ["fake.parquet"],
    )
    monkeypatch.setattr(pd, "read_parquet", lambda parquet_file: _fake_frame().copy())

    audit = audit_parquet_speakers(
        "unused",
        min_duration_sec=400.0,
        min_sample_count=2,
    )
    assert audit["candidate_count"] == 0
    write_audit_outputs(audit, tmp_path)
    assert (tmp_path / "speaker_audit.json").exists()
    assert (tmp_path / "speaker_audit.csv").exists()
    assert (tmp_path / "candidate_speakers.json").exists()


def test_registry_output_validates_and_builds_speaker_vocab(tmp_path: Path):
    candidates = [
        {
            "source_speaker_id": "17",
            "sample_count": 100,
            "total_duration_sec": 720.0,
            "candidate": True,
        },
        {
            "source_speaker_id": "42",
            "sample_count": 80,
            "total_duration_sec": 540.0,
            "candidate": True,
        },
    ]
    registry = build_registry_from_candidates(
        candidates,
        registry_version="v1",
        default_voice_id="nl_default_01",
        source_dataset="parkiet_v1",
        locale="nl-NL",
        activate_candidates=True,
    )
    write_registry_outputs(registry, tmp_path, force=True)

    loaded = load_voice_registry(tmp_path / "voices.json")
    vocab = build_speaker_vocab_from_registry(loaded)
    assert vocab["default_speaker_id"] == 0
    assert vocab["source_speaker_id_to_model_speaker_id"]["17"] == 1
    assert vocab["source_speaker_id_to_model_speaker_id"]["42"] == 2


def test_existing_registry_append_only_behavior_is_respected(tmp_path: Path):
    old_registry = build_registry_from_candidates(
        [
            {
                "source_speaker_id": "17",
                "sample_count": 100,
                "total_duration_sec": 720.0,
                "candidate": True,
            }
        ],
        registry_version="v1",
        default_voice_id="nl_default_01",
        source_dataset="parkiet_v1",
        locale="nl-NL",
        activate_candidates=True,
    )
    write_registry_outputs(old_registry, tmp_path, force=True)

    new_registry = build_registry_from_candidates(
        [
            {
                "source_speaker_id": "17",
                "sample_count": 100,
                "total_duration_sec": 720.0,
                "candidate": True,
            },
            {
                "source_speaker_id": "42",
                "sample_count": 80,
                "total_duration_sec": 540.0,
                "candidate": True,
            },
        ],
        registry_version="v1",
        default_voice_id="nl_default_01",
        source_dataset="parkiet_v1",
        locale="nl-NL",
        existing_registry=old_registry,
        activate_candidates=True,
    )
    write_registry_outputs(new_registry, tmp_path, existing_registry=old_registry, force=True)

    loaded = json.loads((tmp_path / "voices.json").read_text(encoding="utf-8"))
    voice_ids = {voice["voice_id"] for voice in loaded["voices"]}
    assert "nl_default_01" in voice_ids
    assert len(loaded["voices"]) == 3


def test_load_candidates_supports_candidate_file(tmp_path: Path):
    candidate_payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "candidates": [
            {
                "source_speaker_id": "17",
                "sample_count": 100,
                "total_duration_sec": 720.0,
                "candidate": True,
            }
        ],
    }
    candidate_path = tmp_path / "candidate_speakers.json"
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
    candidates = load_candidates(candidate_path)
    assert len(candidates) == 1
    assert candidates[0]["source_speaker_id"] == "17"
