from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_VOICE_FIELDS = {
    "voice_id",
    "model_speaker_id",
    "display_name",
    "language",
    "locale",
    "source_dataset",
    "source_speaker_ids",
    "duration_sec",
    "sample_count",
    "quality_status",
    "status",
    "created_at",
    "updated_at",
    "notes",
}


def load_voice_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_voice_registry(registry)
    return registry


def save_voice_registry(path: str | Path, registry: dict[str, Any]) -> None:
    validate_voice_registry(registry)
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def validate_voice_registry(registry: dict[str, Any]) -> None:
    if not isinstance(registry, dict):
        raise ValueError("voice registry must be an object")
    if "registry_version" not in registry:
        raise ValueError("voice registry is missing 'registry_version'")
    if "default_voice_id" not in registry:
        raise ValueError("voice registry is missing 'default_voice_id'")
    if "voices" not in registry:
        raise ValueError("voice registry is missing 'voices'")
    voices = registry["voices"]
    if not isinstance(voices, list):
        raise ValueError("'voices' must be a list")

    seen_voice_ids: set[str] = set()
    seen_model_speaker_ids: set[int] = set()
    known_voice_ids: set[str] = set()
    for voice in voices:
        if not isinstance(voice, dict):
            raise ValueError("each voice entry must be an object")
        missing = REQUIRED_VOICE_FIELDS - set(voice.keys())
        if missing:
            raise ValueError(f"voice entry is missing required fields: {sorted(missing)}")

        voice_id = str(voice["voice_id"])
        model_speaker_id = int(voice["model_speaker_id"])
        source_speaker_ids = voice["source_speaker_ids"]

        if voice_id in seen_voice_ids:
            raise ValueError(f"duplicate voice_id: {voice_id}")
        if model_speaker_id in seen_model_speaker_ids:
            raise ValueError(f"duplicate model_speaker_id: {model_speaker_id}")
        if not isinstance(source_speaker_ids, list):
            raise ValueError("'source_speaker_ids' must be a list")

        seen_voice_ids.add(voice_id)
        seen_model_speaker_ids.add(model_speaker_id)
        known_voice_ids.add(voice_id)

    default_voice_id = str(registry["default_voice_id"])
    if default_voice_id not in known_voice_ids:
        raise ValueError("default_voice_id must refer to an existing voice")


def get_voice_by_id(registry: dict[str, Any], voice_id: str) -> dict[str, Any]:
    validate_voice_registry(registry)
    for voice in registry["voices"]:
        if voice["voice_id"] == voice_id:
            return voice
    raise KeyError(voice_id)


def get_model_speaker_id(
    registry: dict[str, Any],
    voice_id: str,
    *,
    allow_inactive: bool = False,
) -> int:
    voice = get_voice_by_id(registry, voice_id)
    if not allow_inactive and voice["status"] != "active":
        raise ValueError(f"voice_id {voice_id} is not active")
    return int(voice["model_speaker_id"])


def build_speaker_vocab_from_registry(registry: dict[str, Any]) -> dict[str, Any]:
    validate_voice_registry(registry)
    voice_map: dict[str, int] = {}
    source_map: dict[str, int] = {}

    for voice in registry["voices"]:
        model_speaker_id = int(voice["model_speaker_id"])
        voice_id = str(voice["voice_id"])
        voice_map[voice_id] = model_speaker_id
        for source_speaker_id in voice["source_speaker_ids"]:
            source_key = str(source_speaker_id)
            if source_key in source_map and source_map[source_key] != model_speaker_id:
                raise ValueError(
                    f"source_speaker_id {source_key} maps to multiple model_speaker_id values"
                )
            source_map[source_key] = model_speaker_id

    default_voice_id = str(registry["default_voice_id"])
    default_speaker_id = int(voice_map[default_voice_id])
    return {
        "speaker_vocab_version": registry["registry_version"],
        "default_speaker_id": default_speaker_id,
        "voice_id_to_model_speaker_id": voice_map,
        "source_speaker_id_to_model_speaker_id": source_map,
    }


def validate_append_only_ids(old_registry: dict[str, Any], new_registry: dict[str, Any]) -> None:
    validate_voice_registry(old_registry)
    validate_voice_registry(new_registry)
    old_by_id = {
        str(voice["voice_id"]): int(voice["model_speaker_id"])
        for voice in old_registry["voices"]
    }
    new_by_id = {
        str(voice["voice_id"]): int(voice["model_speaker_id"])
        for voice in new_registry["voices"]
    }
    old_by_speaker = {
        int(voice["model_speaker_id"]): str(voice["voice_id"])
        for voice in old_registry["voices"]
    }
    new_by_speaker = {
        int(voice["model_speaker_id"]): str(voice["voice_id"])
        for voice in new_registry["voices"]
    }

    for voice_id, old_model_speaker_id in old_by_id.items():
        if voice_id not in new_by_id:
            raise ValueError(f"existing voice_id removed: {voice_id}")
        if new_by_id[voice_id] != old_model_speaker_id:
            raise ValueError(
                f"existing voice_id changed model_speaker_id: {voice_id}"
            )

    for model_speaker_id, old_voice_id in old_by_speaker.items():
        if model_speaker_id not in new_by_speaker:
            raise ValueError(
                f"existing model_speaker_id removed: {model_speaker_id}"
            )
        if new_by_speaker[model_speaker_id] != old_voice_id:
            raise ValueError(
                f"existing model_speaker_id reassigned: {model_speaker_id}"
            )

