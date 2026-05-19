from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE, SAMPLE_RATE_RATIO


OUT_DIR = Path("dac_stream_test")
CHUNK_DIR = OUT_DIR / "chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


TEXT = "[S2] Goedemiddag Jan, met Bart van Circle en Borne. Mag ik je even kort storen?"

CFG_SCALE = 3.0
TEMPERATURE = 1.0
TOP_P = 0.95
CFG_FILTER_TOP_K = 45
SEED = 2202

# 20 frames ~= 232ms, 10 frames ~= 116ms right/left context
CHUNK_FRAMES = 20
OVERLAP_FRAMES = 10


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(str(path), audio.astype(np.float32), DEFAULT_SAMPLE_RATE)


class OfflineDacStreamer:
    def __init__(self, model: Dia, chunk_frames: int, overlap_frames: int) -> None:
        self.model = model
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.samples_per_frame = SAMPLE_RATE_RATIO

        self.codes: list[torch.Tensor] = []
        self.emitted_frames = 0
        self.prev_tail: torch.Tensor | None = None
        self.chunk_index = 0

    def push(self, frame: torch.Tensor) -> list[np.ndarray]:
        self.codes.append(frame.detach().cpu())
        chunks = []

        while True:
            chunk = self.maybe_emit()
            if chunk is None:
                break
            chunks.append(chunk)

        return chunks

    def maybe_emit(self) -> np.ndarray | None:
        needed = self.emitted_frames + self.chunk_frames + self.overlap_frames

        if len(self.codes) < needed:
            return None

        left = max(0, self.emitted_frames - self.overlap_frames)
        right = self.emitted_frames + self.chunk_frames + self.overlap_frames

        window = torch.stack(self.codes[left:right], dim=0).to(
            device=self.model.device,
            dtype=torch.long,
        )

        with torch.no_grad(), torch.inference_mode():
            audio = self.model._decode(window).float().cpu()

        start_sample = (self.emitted_frames - left) * self.samples_per_frame
        end_sample = start_sample + self.chunk_frames * self.samples_per_frame

        chunk = audio[start_sample:end_sample].clone()

        if self.prev_tail is not None:
            crossfade_samples = min(
                self.overlap_frames * self.samples_per_frame // 2,
                len(self.prev_tail),
                len(chunk),
            )

            if crossfade_samples > 0:
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
                fade_out = 1.0 - fade_in

                chunk[:crossfade_samples] = (
                    chunk[:crossfade_samples] * fade_in
                    + self.prev_tail[-crossfade_samples:] * fade_out
                )

        self.prev_tail = audio[
            end_sample : end_sample + self.overlap_frames * self.samples_per_frame // 2
        ].clone()

        self.emitted_frames += self.chunk_frames
        self.chunk_index += 1

        return chunk.numpy()

    def flush(self) -> np.ndarray | None:
        if len(self.codes) <= self.emitted_frames:
            return None

        left = max(0, self.emitted_frames - self.overlap_frames)

        window = torch.stack(self.codes[left:], dim=0).to(
            device=self.model.device,
            dtype=torch.long,
        )

        with torch.no_grad(), torch.inference_mode():
            audio = self.model._decode(window).float().cpu()

        start_sample = (self.emitted_frames - left) * self.samples_per_frame
        tail = audio[start_sample:].clone()

        if self.prev_tail is not None:
            crossfade_samples = min(
                self.overlap_frames * self.samples_per_frame // 2,
                len(self.prev_tail),
                len(tail),
            )

            if crossfade_samples > 0:
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
                fade_out = 1.0 - fade_in

                tail[:crossfade_samples] = (
                    tail[:crossfade_samples] * fade_in
                    + self.prev_tail[-crossfade_samples:] * fade_out
                )

        # final tiny fade-out to avoid end click
        fade_samples = min(DEFAULT_SAMPLE_RATE // 100, len(tail))

        if fade_samples > 0:
            tail[-fade_samples:] *= torch.linspace(1.0, 0.0, fade_samples)

        return tail.numpy()


print("loading model...")
model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="bfloat16",
)

print("generating aligned codebook with DAC disabled...")
set_seed(SEED)

model.load_dac = False

codes_np = model.generate(
    TEXT,
    cfg_scale=CFG_SCALE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    cfg_filter_top_k=CFG_FILTER_TOP_K,
    use_torch_compile=True,
    verbose=True,
)

model.load_dac = True

codes = torch.from_numpy(codes_np).long()

print(f"codes shape={tuple(codes.shape)}")

print("decoding full reference from same codebook...")
with torch.no_grad(), torch.inference_mode():
    full_audio = model._decode(codes.to(model.device)).float().cpu().numpy()

full_path = OUT_DIR / "full_reference.wav"
save_wav(full_path, full_audio)
print(f"saved {full_path}")

print("streaming DAC windows offline...")
streamer = OfflineDacStreamer(
    model=model,
    chunk_frames=CHUNK_FRAMES,
    overlap_frames=OVERLAP_FRAMES,
)

stream_chunks = []

for frame in codes:
    for chunk in streamer.push(frame):
        stream_chunks.append(chunk)
        save_wav(CHUNK_DIR / f"chunk_{len(stream_chunks):03d}.wav", chunk)

tail = streamer.flush()

if tail is not None and len(tail) > 0:
    stream_chunks.append(tail)
    save_wav(CHUNK_DIR / f"chunk_{len(stream_chunks):03d}_final.wav", tail)

stream_audio = np.concatenate(stream_chunks) if stream_chunks else np.array([], dtype=np.float32)

stream_path = OUT_DIR / "stream_concat.wav"
save_wav(stream_path, stream_audio)

print(f"saved {stream_path}")
print(f"chunks={len(stream_chunks)}")
print(f"full_duration_sec={len(full_audio) / DEFAULT_SAMPLE_RATE:.3f}")
print(f"stream_duration_sec={len(stream_audio) / DEFAULT_SAMPLE_RATE:.3f}")
print("done")
