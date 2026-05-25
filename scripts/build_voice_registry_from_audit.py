from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parkiet.voice_identity import (
    build_speaker_vocab_from_registry,
    save_voice_registry,
    validate_append_only_ids,
    validate_voice_registry,
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_candidates(audit_json_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(audit_json_path).read_text(encoding="utf-8"))
    if "candidates" in payload:
        return list(payload["candidates"])
    if "speakers" in payload:
        return [speaker for speaker in payload["speakers"] if speaker.get("candidate")]
    raise ValueError("audit json must contain 'candidates' or 'speakers'")


def _next_voice_id(existing_voice_ids: set[str], next_index: int) -> str:
    while True:
        candidate = f"nl_speaker_{next_index:04d}"
        if candidate not in existing_voice_ids:
            return candidate
        next_index += 1


def build_registry_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    registry_version: str,
    default_voice_id: str,
    source_dataset: str,
    locale: str,
    existing_registry: dict[str, Any] | None = None,
    activate_candidates: bool = False,
) -> dict[str, Any]:
    now = iso_now()
    registry = {
        "registry_version": registry_version,
        "default_voice_id": default_voice_id,
        "voices": [],
    }

    if existing_registry is not None:
        registry["voices"].extend(existing_registry["voices"])
        registry["default_voice_id"] = existing_registry["default_voice_id"]

    existing_voice_ids = {str(voice["voice_id"]) for voice in registry["voices"]}
    existing_source_map = {
        str(source_speaker_id): voice
        for voice in registry["voices"]
        for source_speaker_id in voice["source_speaker_ids"]
    }
    used_model_ids = {int(voice["model_speaker_id"]) for voice in registry["voices"]}
    next_model_speaker_id = max(used_model_ids, default=0) + 1
    next_voice_index = 1

    if not registry["voices"]:
        registry["voices"].append(
            {
                "voice_id": default_voice_id,
                "model_speaker_id": 0,
                "display_name": "Default Voice",
                "language": "Dutch",
                "locale": locale,
                "source_dataset": source_dataset,
                "source_speaker_ids": [],
                "duration_sec": 0.0,
                "sample_count": 0,
                "quality_status": "baseline",
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "notes": "Default fallback voice entry",
            }
        )
        existing_voice_ids.add(default_voice_id)

    for candidate in candidates:
        source_speaker_id = str(candidate["source_speaker_id"])
        if source_speaker_id in existing_source_map:
            continue
        voice_id = _next_voice_id(existing_voice_ids, next_voice_index)
        while int(voice_id.split("_")[-1]) in used_model_ids:
            next_voice_index += 1
            voice_id = _next_voice_id(existing_voice_ids, next_voice_index)
        status = "active" if activate_candidates else "inactive"
        quality_status = "approved" if activate_candidates else "candidate"
        registry["voices"].append(
            {
                "voice_id": voice_id,
                "model_speaker_id": next_model_speaker_id,
                "display_name": f"Dutch Speaker {next_model_speaker_id:04d}",
                "language": "Dutch",
                "locale": locale,
                "source_dataset": source_dataset,
                "source_speaker_ids": [source_speaker_id],
                "duration_sec": float(candidate["total_duration_sec"] or 0.0),
                "sample_count": int(candidate["sample_count"]),
                "quality_status": quality_status,
                "status": status,
                "created_at": now,
                "updated_at": now,
                "notes": None,
            }
        )
        existing_voice_ids.add(voice_id)
        used_model_ids.add(next_model_speaker_id)
        next_model_speaker_id += 1
        next_voice_index += 1

    validate_voice_registry(registry)
    return registry


def write_registry_outputs(
    registry: dict[str, Any],
    output_dir: str | Path,
    *,
    existing_registry: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    output_path = Path(output_dir)
    voices_path = output_path / "voices.json"
    speaker_vocab_path = output_path / "speaker_vocab.json"

    if (voices_path.exists() or speaker_vocab_path.exists()) and not force:
        raise FileExistsError(
            f"registry already exists at {output_path}; pass --force to overwrite"
        )
    if existing_registry is not None:
        validate_append_only_ids(existing_registry, registry)

    output_path.mkdir(parents=True, exist_ok=True)
    save_voice_registry(voices_path, registry)
    speaker_vocab = build_speaker_vocab_from_registry(registry)
    speaker_vocab_path.write_text(json.dumps(speaker_vocab, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate voice registry from audit")
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output-dir", default="voice_registry")
    parser.add_argument("--registry-version", default="v1")
    parser.add_argument("--default-voice-id", default="nl_default_01")
    parser.add_argument("--source-dataset", default="parkiet_v1")
    parser.add_argument("--locale", default="nl-NL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--activate-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    existing_registry = None
    voices_path = output_dir / "voices.json"
    if voices_path.exists():
        existing_registry = json.loads(voices_path.read_text(encoding="utf-8"))

    candidates = load_candidates(args.audit_json)
    registry = build_registry_from_candidates(
        candidates,
        registry_version=args.registry_version,
        default_voice_id=args.default_voice_id,
        source_dataset=args.source_dataset,
        locale=args.locale,
        existing_registry=existing_registry,
        activate_candidates=args.activate_candidates,
    )
    write_registry_outputs(
        registry,
        output_dir,
        existing_registry=existing_registry,
        force=args.force,
    )
    print(f"voices={len(registry['voices'])}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
