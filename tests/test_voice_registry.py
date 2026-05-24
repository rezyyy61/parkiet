from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from parkiet.realtime import RealtimeTTSConfig
from parkiet.realtime.engine import DiaRealtimeBackend
from parkiet.realtime.types import RealtimePhrase
from parkiet.realtime.voice_registry import VoicePromptRegistry


class CaptureGenerateModel:
    def __init__(self):
        self.calls: list[dict] = []
        self.last_generate_metadata = {}

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return np.zeros(128, dtype=np.float32)


def write_reference_wav(path: Path, sample_rate: int = 44100) -> None:
    waveform = np.zeros(sample_rate // 4, dtype=np.float32)
    sf.write(path, waveform, sample_rate)


def test_voice_registry_loads_metadata(tmp_path: Path):
    registry = VoicePromptRegistry(tmp_path / "voice_prompts")
    reference_wav = tmp_path / "reference.wav"
    write_reference_wav(reference_wav)

    metadata = registry.register_voice(
        "nl_female_salon_01",
        reference_wav,
        torch.ones((10, 9), dtype=torch.int32),
        notes="test voice",
    )

    loaded_metadata = registry.load_voice_metadata("nl_female_salon_01")
    loaded_prompt = registry.load_voice_prompt("nl_female_salon_01")

    assert metadata.voice_id == "nl_female_salon_01"
    assert loaded_metadata["voice_id"] == "nl_female_salon_01"
    assert loaded_metadata["reference_wav_path"] == str(reference_wav)
    assert tuple(loaded_prompt.shape) == (10, 9)
    assert registry.list_voices() == ["nl_female_salon_01"]


def test_missing_voice_id_raises_clear_error(tmp_path: Path):
    registry = VoicePromptRegistry(tmp_path / "voice_prompts")
    try:
        registry.load_voice_prompt("missing_voice")
    except FileNotFoundError as exc:
        assert "missing_voice" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing voice_id")


def test_dia_realtime_backend_passes_audio_prompt_when_voice_id_is_used(tmp_path: Path):
    registry = VoicePromptRegistry(tmp_path / "voice_prompts")
    reference_wav = tmp_path / "reference.wav"
    write_reference_wav(reference_wav)
    expected_codes = torch.arange(18, dtype=torch.int32).view(2, 9)
    registry.register_voice("nl_voice_01", reference_wav, expected_codes)

    model = CaptureGenerateModel()
    backend = DiaRealtimeBackend(
        model,
        voice_id="nl_voice_01",
        voice_registry_path=str(registry.root_dir),
    )
    backend(
        RealtimePhrase(
            session_id="session",
            index=0,
            text="[S1] Hallo daar.",
            voice_tag="[S1]",
            source_text="Hallo daar.",
            is_final=True,
        ),
        RealtimeTTSConfig(),
    )

    assert len(model.calls) == 1
    passed_prompt = model.calls[0]["audio_prompt"]
    assert isinstance(passed_prompt, torch.Tensor)
    assert torch.equal(passed_prompt, expected_codes)


def test_dia_realtime_backend_default_behavior_is_unchanged_when_voice_id_is_none():
    model = CaptureGenerateModel()
    backend = DiaRealtimeBackend(model)
    backend(
        RealtimePhrase(
            session_id="session",
            index=0,
            text="[S1] Hallo daar.",
            voice_tag="[S1]",
            source_text="Hallo daar.",
            is_final=True,
        ),
        RealtimeTTSConfig(),
    )

    assert len(model.calls) == 1
    assert model.calls[0]["audio_prompt"] is None
