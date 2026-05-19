import time
import torch
import torchaudio

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="bfloat16",
)

text = "[S1] Goedemiddag meneer. U spreekt met Walter Meijer. Ik bel u kort terug naar aanleiding van ons vorige gesprek."

start = time.time()

audio = model.generate(
    text,
    cfg_scale=3.0,
    temperature=1.0,
    top_p=0.95,
    cfg_filter_top_k=45,
    use_torch_compile=True,
    verbose=True,
)

duration = time.time() - start

audio_tensor = torch.from_numpy(audio).float()

if audio_tensor.ndim == 1:
    audio_tensor = audio_tensor.unsqueeze(0)

torchaudio.save("test_native.wav", audio_tensor.cpu(), DEFAULT_SAMPLE_RATE)

print("saved test_native.wav")
print("duration:", round(duration, 2), "sec")
