"""Benchmark voice quality for the Parkiet realtime TTS pipeline.

Generates 5 cold-call sentences × 3 runs each for 3 presets using
generate_stream_pcm (the same path as production).  Writes:
  benchmark_out/<preset>/<sentence_id>_run<N>.wav
  benchmark_out/results.csv            — TTFA, total time, audio duration, RTF
  benchmark_out/consistency_summary.csv — pairwise speaker cosine sim (resemblyzer)

Run:
    uv run python benchmark_voice_quality.py
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE
from parkiet_streaming import generate_stream_pcm


SENTENCES = [
    ("s01_short",  "[S2] Goedemiddag, u spreekt met Bart. Bel ik gelegen?"),
    ("s02_short",  "[S2] Top, ik zal het kort houden."),
    ("s03_medium", "[S2] We helpen bedrijven in uw regio om meer grip te krijgen op klantopvolging en salesprocessen."),
    ("s04_medium", "[S2] Snap ik helemaal, wanneer zou het voor u wel een goed moment zijn om kort te spreken?"),
    ("s05_long",   "[S2] Fijn, dan stel ik voor dat ik u volgende week dinsdag even terugbel, zodat we rustig kunnen kijken of er een mogelijkheid is om samen iets te bereiken."),
]

N_RUNS = 3

# Three presets that span the quality spectrum.
# anchored_t07 requires voices/voxora_voice_01/anchor.wav + correct transcript.
PRESETS: dict[str, dict] = {
    "baseline_no_anchor_t10": {
        "temperature": 1.0,
        "top_p": 0.95,
        "cfg_scale": 3.0,
        "cfg_filter_top_k": 45,
        "seed": 2202,
        "anchor_path": None,
        "anchor_text": None,
    },
    "stable_no_anchor_t07": {
        "temperature": 0.7,
        "top_p": 0.85,
        "cfg_scale": 3.5,
        "cfg_filter_top_k": 35,
        "seed": 2202,
        "anchor_path": None,
        "anchor_text": None,
    },
    "anchored_t07": {
        "temperature": 0.7,
        "top_p": 0.85,
        "cfg_scale": 3.5,
        "cfg_filter_top_k": 35,
        "seed": 2202,
        "anchor_path": "voices/voxora_voice_01/anchor.wav",
        "anchor_text": "[S2] <transcript here>",
    },
}

OUT_DIR = Path("benchmark_out")


def _load_anchors(model: Dia) -> dict[str, torch.Tensor | None]:
    cache: dict[str, torch.Tensor | None] = {}
    for preset_name, preset in PRESETS.items():
        anchor_path = preset.get("anchor_path")
        if not anchor_path:
            cache[preset_name] = None
            continue
        if not os.path.exists(anchor_path):
            print(f"Warning: anchor not found for {preset_name!r}: {anchor_path}", flush=True)
            cache[preset_name] = None
            continue
        try:
            cache[preset_name] = model.load_audio(anchor_path)
            print(f"Loaded anchor for {preset_name!r}", flush=True)
        except Exception as exc:
            print(f"Warning: failed to load anchor for {preset_name!r}: {exc}", flush=True)
            cache[preset_name] = None
    return cache


def _run_generation(
    model: Dia,
    full_text: str,
    preset: dict,
    anchor: "torch.Tensor | None",
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Returns (audio, ttfa_seconds, total_seconds)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    chunks: list[np.ndarray] = []
    first_chunk_time: float | None = None
    start = time.perf_counter()

    def on_chunk(chunk: np.ndarray, sr: int, idx: int, is_final: bool) -> None:
        nonlocal first_chunk_time
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start
        chunks.append(chunk.copy())

    generate_stream_pcm(
        model,
        full_text,
        on_pcm_chunk=on_chunk,
        audio_prompt=anchor,
        temperature=preset["temperature"],
        top_p=preset["top_p"],
        cfg_scale=preset["cfg_scale"],
        cfg_filter_top_k=preset["cfg_filter_top_k"],
        use_torch_compile=False,  # disabled for reproducible benchmark timing
        verbose=False,
    )

    total = time.perf_counter() - start
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    ttfa = first_chunk_time if first_chunk_time is not None else total
    return audio, ttfa, total


def run_benchmark() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading model...", flush=True)
    model = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-nl-v1.pth",
        compute_dtype="bfloat16",
    )

    anchor_cache = _load_anchors(model)

    results: list[dict] = []

    for preset_name, preset in PRESETS.items():
        preset_dir = OUT_DIR / preset_name
        preset_dir.mkdir(exist_ok=True)

        anchor = anchor_cache[preset_name]
        anchor_text: str = preset.get("anchor_text") or ""

        for sent_id, sent_text in SENTENCES:
            for run in range(1, N_RUNS + 1):
                run_seed = int(preset.get("seed", 2202)) + run - 1

                # Build full text (anchor transcript prepended when anchor is present).
                if anchor is not None and anchor_text:
                    full_text = f"{anchor_text} {sent_text}"
                else:
                    full_text = sent_text

                audio, ttfa, total = _run_generation(
                    model, full_text, preset, anchor, run_seed
                )

                audio_dur_s = len(audio) / DEFAULT_SAMPLE_RATE if len(audio) > 0 else 0.0
                rtf = total / audio_dur_s if audio_dur_s > 0 else float("inf")

                wav_path = preset_dir / f"{sent_id}_run{run}.wav"
                if len(audio) > 0:
                    sf.write(str(wav_path), audio, DEFAULT_SAMPLE_RATE)

                row = {
                    "preset": preset_name,
                    "sentence": sent_id,
                    "run": run,
                    "ttfa_ms": f"{ttfa * 1000:.1f}",
                    "total_ms": f"{total * 1000:.1f}",
                    "audio_ms": f"{audio_dur_s * 1000:.1f}",
                    "rtf": f"{rtf:.3f}",
                }
                results.append(row)

                print(
                    f"{preset_name}/{sent_id}/run{run}: "
                    f"TTFA={ttfa*1000:.0f}ms  total={total*1000:.0f}ms  "
                    f"audio={audio_dur_s*1000:.0f}ms  RTF={rtf:.2f}x",
                    flush=True,
                )

    csv_path = OUT_DIR / "results.csv"
    fieldnames = ["preset", "sentence", "run", "ttfa_ms", "total_ms", "audio_ms", "rtf"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults → {csv_path}", flush=True)

    _maybe_run_consistency(anchor_cache)


def _maybe_run_consistency(anchor_cache: dict) -> None:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore[import]
    except ImportError:
        print("resemblyzer not installed — skipping speaker consistency analysis", flush=True)
        return

    print("\nComputing speaker consistency (resemblyzer)...", flush=True)
    encoder = VoiceEncoder()
    rows: list[dict] = []

    for preset_name in PRESETS:
        preset_dir = OUT_DIR / preset_name
        embeddings: list[np.ndarray] = []

        for sent_id, _ in SENTENCES:
            for run in range(1, N_RUNS + 1):
                wav_path = preset_dir / f"{sent_id}_run{run}.wav"
                if not wav_path.exists():
                    continue
                try:
                    wav = preprocess_wav(str(wav_path))
                    emb = encoder.embed_utterance(wav)
                    embeddings.append(emb)
                except Exception as exc:
                    print(f"  Warning: could not embed {wav_path}: {exc}", flush=True)

        if len(embeddings) < 2:
            print(f"  {preset_name}: not enough files to compute consistency", flush=True)
            continue

        mat = np.stack(embeddings)
        n = len(mat)
        sims: list[float] = [
            float(np.dot(mat[i], mat[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        mean_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))

        if mean_sim > 0.85:
            interpretation = "very_consistent"
        elif mean_sim > 0.75:
            interpretation = "consistent"
        else:
            interpretation = "unstable"

        rows.append({
            "preset": preset_name,
            "n_files": n,
            "mean_cosine": f"{mean_sim:.4f}",
            "min_cosine": f"{min_sim:.4f}",
            "interpretation": interpretation,
        })
        print(
            f"  {preset_name}: mean={mean_sim:.4f}  min={min_sim:.4f}  → {interpretation}",
            flush=True,
        )

    if rows:
        cons_path = OUT_DIR / "consistency_summary.csv"
        fields = ["preset", "n_files", "mean_cosine", "min_cosine", "interpretation"]
        with open(cons_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Consistency summary → {cons_path}", flush=True)


if __name__ == "__main__":
    run_benchmark()
