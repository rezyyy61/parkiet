from pathlib import Path

import torch
import torchaudio

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE


OUTPUT_DIR = Path("speaker_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="bfloat16",
)

tests = [
    {
        "name": "s1.wav",
        "text": "[S1] Goedemiddag Jan, met Bart van Circle en Borne. Mag ik je even kort storen?",
    },
    {
        "name": "s2.wav",
        "text": "[S2] Goedemiddag Jan, met Bart van Circle en Borne. Mag ik je even kort storen?",
    },
]


def save_wav(path: Path, audio):
    tensor = torch.from_numpy(audio).float()

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    torchaudio.save(str(path), tensor.cpu(), DEFAULT_SAMPLE_RATE)


print("warming up...")
model.generate(
    "[S2] Dit is een korte test.",
    cfg_scale=3.0,
    temperature=1.0,
    top_p=0.95,
    cfg_filter_top_k=45,
    use_torch_compile=True,
    verbose=True,
)

for item in tests:
    print(f"generating {item['name']}")

    audio = model.generate(
        item["text"],
        cfg_scale=3.0,
        temperature=1.0,
        top_p=0.95,
        cfg_filter_top_k=45,
        use_torch_compile=True,
        verbose=True,
    )

    save_wav(OUTPUT_DIR / item["name"], audio)
    print(f"saved {OUTPUT_DIR / item['name']}")

print("done")
