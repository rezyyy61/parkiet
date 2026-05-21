import asyncio
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE
from parkiet_streaming import generate_stream_pcm


OUTPUT_SAMPLE_RATE: int = int(os.getenv("PARKIET_OUTPUT_SAMPLE_RATE", "44100"))
USE_TORCH_COMPILE: bool = os.getenv("PARKIET_USE_TORCH_COMPILE", "1") == "1"


# Each preset may carry an anchor_path (path to a real human recording) and
# anchor_text (exact transcript of that recording, with leading speaker tag).
# anchor_path=None means no voice anchoring; voice will be sampled fresh.
VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "s2_plain_t14": {
        "speaker": "S2",
        "temperature": 1.4,
        "top_p": 0.95,
        "cfg_scale": 3.0,
        "cfg_filter_top_k": 45,
        "anchor_path": None,
        "anchor_text": None,
    },
    "s2_plain_t10": {
        "speaker": "S2",
        "temperature": 1.0,
        "top_p": 0.95,
        "cfg_scale": 3.0,
        "cfg_filter_top_k": 45,
        "seed": 2202,
        "anchor_path": None,
        "anchor_text": None,
    },
    "s2_stable_t09": {
        "speaker": "S2",
        "temperature": 0.9,
        "top_p": 0.92,
        "cfg_scale": 3.0,
        "cfg_filter_top_k": 45,
        "seed": 2202,
        "anchor_path": None,
        "anchor_text": None,
    },
    "s2_stable_t07": {
        "speaker": "S2",
        "temperature": 0.7,
        "top_p": 0.85,
        "cfg_scale": 3.0,
        "cfg_filter_top_k": 35,
        "seed": 2202,
        "anchor_path": None,
        "anchor_text": None,
    },
    # Recommended cold-call preset. Requires a real human anchor recording at
    # voices/voxora_voice_01/anchor.wav and the exact WhisperD-NL transcript
    # in anchor_text below. Falls back gracefully if the file is absent.
    "voxora_voice_01": {
        "speaker": "S2",
        "temperature": 0.7,
        "top_p": 0.85,
        "cfg_scale": 3.5,
        "cfg_filter_top_k": 35,
        "seed": 2202,
        "anchor_path": "voices/voxora_voice_01/anchor.wav",
        "anchor_text": "[S2] <transcript here>",
    },
    "voxora_s2_t14": {
        "speaker": "S1",
        "temperature": 1.4,
        "top_p": 0.95,
        "cfg_scale": 3.0,
        "cfg_filter_top_k": 45,
        "seed": 2202,
        "anchor_path": "voices/voxora_s2_t14/anchor.wav",
        "anchor_text": (
            "[S1] goedemiddag meneer, u spreekt met walter meijer, "
            "ik bel u kort terug naar aanleiding van ons vorige gesprek."
        ),
    },
}


class _StatefulResampler:
    """Per-job stateful resampler. Instantiate once per generation; never reuse across jobs.

    Prefers the `samplerate` package (libsamplerate sinc_best) when installed.
    Falls back to torchaudio with a ~1-chunk lookahead to reduce boundary artefacts.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self._source_rate = source_rate
        self._target_rate = target_rate
        self._ratio = target_rate / source_rate
        self._converter = None
        self._use_samplerate = False
        self._pending: np.ndarray | None = None  # torchaudio fallback lookahead

        try:
            import samplerate as _sr  # type: ignore[import]
            self._converter = _sr.Resampler("sinc_best", channels=1)
            self._use_samplerate = True
        except ImportError:
            pass

    def process(self, chunk: np.ndarray) -> np.ndarray:
        data = np.asarray(chunk, dtype=np.float32)
        if self._use_samplerate:
            return self._converter.process(data, self._ratio, end_of_input=False)
        # torchaudio fallback: hold current chunk, emit previous one.
        out: np.ndarray = np.array([], dtype=np.float32)
        if self._pending is not None:
            out = self._ta_resample(self._pending)
        self._pending = data
        return out

    def flush(self, last_chunk: np.ndarray | None = None) -> np.ndarray:
        if self._use_samplerate:
            data = (
                np.asarray(last_chunk, dtype=np.float32)
                if last_chunk is not None
                else np.array([], dtype=np.float32)
            )
            return self._converter.process(data, self._ratio, end_of_input=True)
        # torchaudio fallback: resample everything still buffered.
        parts: list[np.ndarray] = []
        if self._pending is not None:
            parts.append(self._pending)
        if last_chunk is not None:
            parts.append(np.asarray(last_chunk, dtype=np.float32))
        self._pending = None
        if not parts:
            return np.array([], dtype=np.float32)
        return self._ta_resample(np.concatenate(parts))

    def _ta_resample(self, data: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(data).unsqueeze(0)
        with torch.no_grad():
            out = torchaudio.functional.resample(tensor, self._source_rate, self._target_rate)
        return out.squeeze(0).numpy().astype(np.float32)


class TtsRequest(BaseModel):
    text: str
    voice: str = "s2_plain_t10"


@dataclass
class GenerationJob:
    text: str
    voice: str
    output_queue: queue.Queue[dict[str, Any] | None]
    cancelled: threading.Event = field(default_factory=threading.Event)


class ParkietWorker:
    def __init__(self) -> None:
        self.jobs: queue.Queue[GenerationJob] = queue.Queue()
        self.ready = threading.Event()
        self.failed: Exception | None = None
        self.model: Dia | None = None
        self._anchor_cache: dict[str, torch.Tensor] = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, job: GenerationJob) -> None:
        self.jobs.put(job)

    def _run(self) -> None:
        try:
            print("loading Parkiet model inside generation worker...", flush=True)

            self.model = Dia.from_local(
                config_path="config.json",
                checkpoint_path="weights/dia-nl-v1.pth",
                compute_dtype="bfloat16",
            )

            # Pre-encode any configured anchor WAVs so we only pay the DAC encode
            # cost once rather than on every request.
            for preset_name, preset in VOICE_PRESETS.items():
                anchor_path = preset.get("anchor_path")
                if not anchor_path:
                    continue
                if not os.path.exists(anchor_path):
                    print(
                        f"Warning: anchor file not found for preset {preset_name!r}: {anchor_path}",
                        flush=True,
                    )
                    continue
                try:
                    self._anchor_cache[preset_name] = self.model.load_audio(anchor_path)
                    print(f"Anchor loaded for preset {preset_name!r}", flush=True)
                except Exception as exc:
                    print(
                        f"Warning: failed to load anchor for {preset_name!r}: {exc}",
                        flush=True,
                    )

            # Warm up torch.compile in this same worker thread (CUDA graph safety).
            # Use anchor-style settings so the compiled graph matches production.
            print("warming up torch.compile inside same worker thread...", flush=True)

            def _noop_chunk(chunk: np.ndarray, sr: int, idx: int, is_final: bool) -> None:
                pass

            generate_stream_pcm(
                self.model,
                "[S2] Dit is een korte warmup test.",
                on_pcm_chunk=_noop_chunk,
                cfg_scale=3.5,
                temperature=0.7,
                top_p=0.85,
                cfg_filter_top_k=35,
                use_torch_compile=USE_TORCH_COMPILE,
                verbose=True,
            )

            print("Parkiet worker ready", flush=True)
            self.ready.set()

            while True:
                job = self.jobs.get()
                self._handle_job(job)

        except Exception as exc:
            import traceback

            self.failed = exc
            print("Parkiet worker failed:", repr(exc), flush=True)
            traceback.print_exc()
            self.ready.set()

    def _handle_job(self, job: GenerationJob) -> None:
        assert self.model is not None

        try:
            if job.cancelled.is_set():
                job.output_queue.put(None)
                return

            preset = VOICE_PRESETS[job.voice]
            speaker = preset.get("speaker", "S2")
            text = normalize_text(job.text, speaker)

            seed = int(preset.get("seed", 2202))

            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Anchor handling: prepend transcript so the text encoder's cross-attention
            # aligns with the DAC-prefix on the decoder side (see root-cause analysis).
            anchor: torch.Tensor | None = self._anchor_cache.get(job.voice)
            anchor_text: str = preset.get("anchor_text") or ""
            if anchor is not None and anchor_text:
                full_text = f"{anchor_text} {text}"
            else:
                full_text = text
                if preset.get("anchor_path") and anchor is None:
                    print(
                        f"Warning: anchor not cached for {job.voice!r}, generating without anchor",
                        flush=True,
                    )

            print(
                f"Parkiet generation start voice={job.voice} seed={seed} anchored={anchor is not None} text={full_text[:120]!r}",
                flush=True,
            )

            # Build a per-job stateful resampler only when the output rate differs.
            resampler: _StatefulResampler | None = None
            if OUTPUT_SAMPLE_RATE != DEFAULT_SAMPLE_RATE:
                resampler = _StatefulResampler(DEFAULT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

            def on_chunk(
                audio_chunk: np.ndarray,
                sample_rate: int,
                chunk_index: int,
                is_final: bool,
            ) -> None:
                if job.cancelled.is_set():
                    return

                if resampler is not None:
                    audio_out = (
                        resampler.flush(audio_chunk) if is_final else resampler.process(audio_chunk)
                    )
                else:
                    audio_out = np.asarray(audio_chunk, dtype=np.float32)

                if len(audio_out) == 0:
                    return

                pcm_bytes = float_audio_to_pcm16_bytes(audio_out)

                job.output_queue.put(
                    {
                        "type": "chunk",
                        "index": chunk_index,
                        "is_final": is_final,
                        "sample_rate": OUTPUT_SAMPLE_RATE,
                        "bytes": pcm_bytes,
                    }
                )

            generate_stream_pcm(
                self.model,
                full_text,
                on_pcm_chunk=on_chunk,
                audio_prompt=anchor,
                cancel_event=job.cancelled,
                cfg_scale=preset["cfg_scale"],
                temperature=preset["temperature"],
                top_p=preset["top_p"],
                cfg_filter_top_k=preset["cfg_filter_top_k"],
                use_torch_compile=USE_TORCH_COMPILE,
                verbose=True,
                chunk_frames=20,
                overlap_frames=10,
            )

            print("Parkiet generation done", flush=True)

        except Exception as exc:
            import traceback

            print("Parkiet generation error:", repr(exc), flush=True)
            traceback.print_exc()
            job.output_queue.put({"type": "error", "message": str(exc) or repr(exc)})
        finally:
            job.output_queue.put(None)


app = FastAPI()
WORKER = ParkietWorker()


def normalize_text(text: str, speaker: str) -> str:
    text = text.strip()

    for tag in ("[S1]", "[S2]"):
        if text.startswith(tag):
            text = text[len(tag):].strip()

    return f"[{speaker}] {text}"


def float_audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")
    return pcm16.tobytes()


# Kept for offline/debug use but no longer used in the realtime path.
def resample_audio_chunk(
    audio_chunk: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return audio_chunk.astype(np.float32, copy=False)

    tensor = torch.from_numpy(audio_chunk.astype(np.float32, copy=False))

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        resampled = torchaudio.functional.resample(
            tensor,
            orig_freq=source_rate,
            new_freq=target_rate,
        )

    return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


@app.on_event("startup")
def startup() -> None:
    WORKER.start()
    WORKER.ready.wait()

    if WORKER.failed is not None:
        raise WORKER.failed

    print("Parkiet realtime TTS server ready", flush=True)


@app.get("/health")
def health() -> dict[str, Any]:
    anchored_voices = list(WORKER._anchor_cache.keys())
    return {
        "ok": WORKER.ready.is_set() and WORKER.failed is None,
        "model_loaded": WORKER.model is not None,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "voices": list(VOICE_PRESETS.keys()),
        "anchored_voices": anchored_voices,
        "format": "pcm_s16le",
        "channels": 1,
        "compile": USE_TORCH_COMPILE,
        "worker_thread": True,
    }


@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket) -> None:
    await websocket.accept()

    if WORKER.failed is not None or WORKER.model is None:
        await websocket.send_json({"type": "error", "message": "model not loaded"})
        await websocket.close()
        return

    job: GenerationJob | None = None

    try:
        payload = await websocket.receive_json()
        request = TtsRequest(**payload)

        if request.voice not in VOICE_PRESETS:
            await websocket.send_json(
                {"type": "error", "message": f"unknown voice: {request.voice}"}
            )
            await websocket.close()
            return

        output_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        job = GenerationJob(
            text=request.text,
            voice=request.voice,
            output_queue=output_queue,
        )

        await websocket.send_json(
            {
                "type": "start",
                "sample_rate": OUTPUT_SAMPLE_RATE,
                "format": "pcm_s16le",
                "channels": 1,
                "voice": request.voice,
            }
        )

        WORKER.submit(job)

        while True:
            item = await asyncio.to_thread(output_queue.get)

            if item is None:
                await websocket.send_json({"type": "done"})
                break

            if item.get("type") == "error":
                await websocket.send_json(
                    {"type": "error", "message": item.get("message", "unknown error")}
                )
                break

            await websocket.send_json(
                {
                    "type": "chunk",
                    "index": item["index"],
                    "is_final": item["is_final"],
                    "sample_rate": item["sample_rate"],
                    "format": "pcm_s16le",
                    "channels": 1,
                    "bytes": len(item["bytes"]),
                }
            )
            await websocket.send_bytes(item["bytes"])

    except WebSocketDisconnect:
        if job is not None:
            job.cancelled.set()
        return
    except Exception as exc:
        if job is not None:
            job.cancelled.set()

        import traceback

        print("Parkiet WS error:", repr(exc), flush=True)
        traceback.print_exc()

        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if job is not None:
            job.cancelled.set()

        try:
            await websocket.close()
        except Exception:
            pass
