from pathlib import Path

import soundfile as sf
import torch

from parkiet.dia.model import Dia

config_path = "config.json"
checkpoint_path = "weights/dia-nl-v1.pth"
output_path = Path("debug_reference_voice/nl_test_reference.wav")

text = (
    "[S1] Goedemiddag, u spreekt met de assistent van de salon. "
    "Ik help u graag met het maken of wijzigen van een afspraak. "
    "Ook kan ik korte vragen beantwoorden over behandelingen, openingstijden en beschikbaarheid. "
    "Vertel mij gerust waarmee ik u vandaag kan helpen."
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Dia.from_local(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    device=device,
    compute_dtype="bfloat16",
)

audio = model.generate(
    text,
    cfg_scale=3.0,
    temperature=1.2,
    top_p=0.85,
    cfg_filter_top_k=50,
    max_tokens=2048,
    use_torch_compile=True,
)

output_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(output_path, audio, 44100)

print(f"wrote: {output_path}")
