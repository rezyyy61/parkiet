from __future__ import annotations

import numpy as np

from scripts.debug_audio_prompt_usage import get_prompt_debug_info, trim_waveform_prefix
from parkiet.realtime.engine import DiaRealtimeBackend


class CaptureGenerateModel:
    def __init__(self):
        self.calls: list[dict] = []
        self.last_generate_metadata = {}

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return np.zeros(64, dtype=np.float32)


def test_trim_waveform_prefix_removes_expected_number_of_samples():
    audio = np.arange(20, dtype=np.float32)
    trimmed = trim_waveform_prefix(audio, 6)
    assert trimmed.dtype == np.float32
    assert np.array_equal(trimmed, np.arange(6, 20, dtype=np.float32))


def test_backend_default_behavior_keeps_trim_flag_false():
    model = CaptureGenerateModel()
    backend = DiaRealtimeBackend(model)
    backend._resolve_audio_prompt = lambda: None  # type: ignore[method-assign]
    backend(
        phrase=type(
            "Phrase",
            (),
            {
                "session_id": "s",
                "index": 0,
                "text": "[S1] Hallo",
                "queued_at": __import__("datetime").datetime.utcnow(),
            },
        )(),
        config=None,  # type: ignore[arg-type]
    )
    assert model.calls[0]["trim_audio_prompt_from_output"] is False


def test_audio_prompt_metadata_includes_prompt_duration_and_trim_info():
    metadata = {
        "prompt_duration_ms": [1234.5],
        "prompt_code_steps": [106],
        "trimmed_samples": [65536],
        "output_duration_before_trim_ms": [2400.0],
        "output_duration_after_trim_ms": [1165.0],
    }
    prompt_duration_ms, prompt_code_steps, trimmed_samples, before_ms, after_ms = (
        get_prompt_debug_info(metadata)
    )
    assert prompt_duration_ms == 1234.5
    assert prompt_code_steps == 106
    assert trimmed_samples == 65536
    assert before_ms == 2400.0
    assert after_ms == 1165.0
