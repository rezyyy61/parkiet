from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import torch
import torchaudio


VOICE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


@dataclass(slots=True)
class VoicePromptMetadata:
    voice_id: str
    reference_wav_path: str
    sample_rate: int
    duration_sec: float
    created_at: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "reference_wav_path": self.reference_wav_path,
            "sample_rate": self.sample_rate,
            "duration_sec": self.duration_sec,
            "created_at": self.created_at,
            "notes": self.notes,
        }


class VoicePromptRegistry:
    def __init__(self, root_dir: str | Path = "voice_prompts"):
        self.root_dir = Path(root_dir)

    def validate_voice_id(self, voice_id: str) -> bool:
        return bool(VOICE_ID_PATTERN.fullmatch(voice_id))

    def register_voice(
        self,
        voice_id: str,
        reference_wav_path: str | Path,
        prompt_codes: torch.Tensor,
        *,
        notes: str | None = None,
    ) -> VoicePromptMetadata:
        if not self.validate_voice_id(voice_id):
            raise ValueError(
                "Invalid voice_id. Use only letters, numbers, underscores, and hyphens."
            )

        reference_path = Path(reference_wav_path)
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference WAV file not found for voice_id '{voice_id}': {reference_path}"
            )

        info = torchaudio.info(str(reference_path))
        duration_sec = (
            float(info.num_frames) / float(info.sample_rate)
            if info.sample_rate > 0
            else 0.0
        )

        voice_dir = self.root_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        torch.save(prompt_codes.detach().to("cpu"), voice_dir / "prompt_codes.pt")

        metadata = VoicePromptMetadata(
            voice_id=voice_id,
            reference_wav_path=str(reference_path),
            sample_rate=int(info.sample_rate),
            duration_sec=duration_sec,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        (voice_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2),
            encoding="utf-8",
        )
        return metadata

    def load_voice_prompt(self, voice_id: str) -> torch.Tensor:
        self._require_existing_voice(voice_id)
        prompt_codes = torch.load(
            self.root_dir / voice_id / "prompt_codes.pt",
            map_location="cpu",
        )
        if not isinstance(prompt_codes, torch.Tensor):
            raise TypeError(
                f"Stored prompt codes for voice_id '{voice_id}' are not a torch.Tensor"
            )
        return prompt_codes

    def load_voice_metadata(self, voice_id: str) -> dict[str, Any]:
        self._require_existing_voice(voice_id)
        return json.loads(
            (self.root_dir / voice_id / "metadata.json").read_text(encoding="utf-8")
        )

    def list_voices(self) -> list[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            item.name
            for item in self.root_dir.iterdir()
            if item.is_dir()
            and (item / "metadata.json").exists()
            and (item / "prompt_codes.pt").exists()
        )

    def _require_existing_voice(self, voice_id: str) -> None:
        if not self.validate_voice_id(voice_id):
            raise ValueError(
                f"Invalid voice_id '{voice_id}'. Use only letters, numbers, underscores, and hyphens."
            )
        voice_dir = self.root_dir / voice_id
        if not (
            voice_dir.exists()
            and (voice_dir / "metadata.json").exists()
            and (voice_dir / "prompt_codes.pt").exists()
        ):
            raise FileNotFoundError(
                f"Voice prompt not found for voice_id '{voice_id}' in {self.root_dir}"
            )
