from __future__ import annotations

import torch

from parkiet.dia.model import SPEAKER_MODULE_STATE_KEYS


def extract_speaker_module_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    extracted: dict[str, torch.Tensor] = {}
    for key in sorted(SPEAKER_MODULE_STATE_KEYS):
        if key in state_dict:
            extracted[key] = state_dict[key].detach().cpu().clone()
    return extracted
