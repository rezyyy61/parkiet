import torch
import torchaudio

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="bfloat16",
)

print("=== warmup ===")
model.generate(
    "[S1] Dit is een korte test.",
    cfg_scale=3.0,
    temperature=1.0,
    top_p=0.95,
    cfg_filter_top_k=45,
    use_torch_compile=True,
    verbose=True,
)

print("=== streaming probe ===")
audio = model.generate(
    "[S1] Goedemiddag meneer. U spreekt met Walter Meijer. Ik bel u kort terug naar aanleiding van ons vorige gesprek. "
    "Ik wilde u rustig uitleggen waarom ik bel. Support Squads helpt bedrijven met klantenservice via AI technologie. "
    "Veel klantvragen kunnen automatisch worden afgehandeld, waardoor klanten sneller geholpen worden. "
    "Het leek mij interessant om eens vrijblijvend kennis te maken.",
    cfg_scale=3.0,
    temperature=1.0,
    top_p=0.95,
    cfg_filter_top_k=45,
    use_torch_compile=True,
    verbose=True,
    stream_probe_dir="stream_probe",
    stream_probe_every_tokens=86,
)

audio_tensor = torch.from_numpy(audio).float()

if audio_tensor.ndim == 1:
    audio_tensor = audio_tensor.unsqueeze(0)

torchaudio.save(
    "stream_probe/final_output.wav",
    audio_tensor.cpu(),
    DEFAULT_SAMPLE_RATE,
)

print("saved stream_probe/final_output.wav")
