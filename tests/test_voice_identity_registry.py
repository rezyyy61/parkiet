from __future__ import annotations

import copy

from parkiet.voice_identity import (
    build_speaker_vocab_from_registry,
    get_model_speaker_id,
    get_voice_by_id,
    validate_append_only_ids,
    validate_voice_registry,
)


def _valid_registry():
    return {
        "registry_version": "v1",
        "default_voice_id": "nl_default_01",
        "voices": [
            {
                "voice_id": "nl_default_01",
                "model_speaker_id": 0,
                "display_name": "Dutch Default",
                "language": "Dutch",
                "locale": "nl-NL",
                "source_dataset": "parkiet_v1",
                "source_speaker_ids": ["dataset_a:11"],
                "duration_sec": 3600.0,
                "sample_count": 700,
                "quality_status": "approved",
                "status": "active",
                "created_at": "2026-05-25T00:00:00Z",
                "updated_at": "2026-05-25T00:00:00Z",
                "notes": None,
            },
            {
                "voice_id": "nl_female_salon_01",
                "model_speaker_id": 1,
                "display_name": "Dutch Female Salon",
                "language": "Dutch",
                "locale": "nl-NL",
                "source_dataset": "parkiet_v1",
                "source_speaker_ids": ["dataset_a:17"],
                "duration_sec": 7200.0,
                "sample_count": 1500,
                "quality_status": "approved",
                "status": "active",
                "created_at": "2026-05-25T00:00:00Z",
                "updated_at": "2026-05-25T00:00:00Z",
                "notes": None,
            },
        ],
    }


def test_valid_registry_passes():
    validate_voice_registry(_valid_registry())


def test_duplicate_voice_id_fails():
    registry = _valid_registry()
    registry["voices"][1]["voice_id"] = registry["voices"][0]["voice_id"]
    try:
        validate_voice_registry(registry)
    except ValueError as exc:
        assert "duplicate voice_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate voice_id failure")


def test_duplicate_model_speaker_id_fails():
    registry = _valid_registry()
    registry["voices"][1]["model_speaker_id"] = registry["voices"][0]["model_speaker_id"]
    try:
        validate_voice_registry(registry)
    except ValueError as exc:
        assert "duplicate model_speaker_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate model_speaker_id failure")


def test_missing_required_fields_fail():
    registry = _valid_registry()
    del registry["voices"][0]["display_name"]
    try:
        validate_voice_registry(registry)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("Expected missing field failure")


def test_inactive_voice_can_exist_but_not_selected_by_default():
    registry = _valid_registry()
    registry["voices"][1]["status"] = "inactive"
    validate_voice_registry(registry)
    try:
        get_model_speaker_id(registry, "nl_female_salon_01")
    except ValueError as exc:
        assert "not active" in str(exc)
    else:
        raise AssertionError("Expected inactive voice selection failure")
    assert get_model_speaker_id(registry, "nl_female_salon_01", allow_inactive=True) == 1


def test_get_model_speaker_id_works():
    registry = _valid_registry()
    assert get_model_speaker_id(registry, "nl_default_01") == 0
    assert get_model_speaker_id(registry, "nl_female_salon_01") == 1
    voice = get_voice_by_id(registry, "nl_female_salon_01")
    assert voice["display_name"] == "Dutch Female Salon"


def test_build_speaker_vocab_from_registry_works():
    registry = _valid_registry()
    vocab = build_speaker_vocab_from_registry(registry)
    assert vocab["speaker_vocab_version"] == "v1"
    assert vocab["default_speaker_id"] == 0
    assert vocab["voice_id_to_model_speaker_id"] == {
        "nl_default_01": 0,
        "nl_female_salon_01": 1,
    }
    assert vocab["source_speaker_id_to_model_speaker_id"] == {
        "dataset_a:11": 0,
        "dataset_a:17": 1,
    }


def test_append_only_validation_rejects_changed_existing_ids():
    old_registry = _valid_registry()
    new_registry = copy.deepcopy(old_registry)
    new_registry["voices"][1]["model_speaker_id"] = 7
    try:
        validate_append_only_ids(old_registry, new_registry)
    except ValueError as exc:
        assert "changed model_speaker_id" in str(exc)
    else:
        raise AssertionError("Expected append-only validation failure")


def test_append_only_validation_allows_adding_new_voice_ids():
    old_registry = _valid_registry()
    new_registry = copy.deepcopy(old_registry)
    new_registry["voices"].append(
        {
            "voice_id": "nl_male_agent_01",
            "model_speaker_id": 2,
            "display_name": "Dutch Male Agent",
            "language": "Dutch",
            "locale": "nl-NL",
            "source_dataset": "parkiet_v1",
            "source_speaker_ids": ["dataset_b:22"],
            "duration_sec": 5400.0,
            "sample_count": 1100,
            "quality_status": "candidate",
            "status": "inactive",
            "created_at": "2026-05-25T00:00:00Z",
            "updated_at": "2026-05-25T00:00:00Z",
            "notes": None,
        }
    )
    validate_append_only_ids(old_registry, new_registry)

