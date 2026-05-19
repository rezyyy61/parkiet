import os
import torch

from parkiet.dia.model import Dia

torch.manual_seed(2202)

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="bfloat16",
)

anchor = model.load_audio("voices/s2_plain_t10/anchor.wav")

texts = [
    "[S2] Goedemiddag Jan, met Bart van Circle en Borne. Mag ik je even kort storen?",
    "[S2] Top, ik hou het kort. We helpen bedrijven in de regio Cuijk om meer grip te krijgen op klantopvolging.",
    "[S2] Snap ik helemaal, Jan. Als je het ooit toch eens wilt bekijken, laat het gerust weten. Fijne dag nog!",
]

os.makedirs("anchor_test", exist_ok=True)

for i, text in enumerate(texts, start=1):
    torch.manual_seed(2202)

    audio = model.generate(
        text,
        audio_prompt=anchor,
        use_torch_compile=False,
        verbose=True,
        cfg_scale=3.0,
        temperature=1.0,
        top_p=0.95,
        cfg_filter_top_k=45,
    )

    path = f"anchor_test/anchored_{i}.wav"
    model.save_audio(path, audio)
    print(f"saved {path}")
