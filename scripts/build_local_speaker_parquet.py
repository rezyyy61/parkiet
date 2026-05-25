from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from parkiet.dia.config import DiaConfig
from parkiet.dia.model import ComputeDtype, Dia


SAFE_SPEAKER_FOLDER_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass
class LocalSpeakerEncoder:
    dia: Dia

    def encode(self, audio_path: Path):
        return self.dia.load_audio(str(audio_path))


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def define_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_file", pa.string()),
            ("chunk_id", pa.string()),
            ("start_ms", pa.float64()),
            ("end_ms", pa.float64()),
            ("duration_ms", pa.float64()),
            ("transcription", pa.string()),
            ("transcription_clean", pa.string()),
            ("file_path", pa.string()),
            ("chunk_owner", pa.int64()),
            ("sample_prob", pa.float64()),
            ("cb_weight", pa.float64()),
            ("encoded_audio_shape", pa.list_(pa.int64())),
            ("encoded_audio", pa.list_(pa.int64())),
        ]
    )


def validate_speaker_folder_name(name: str) -> None:
    if not SAFE_SPEAKER_FOLDER_RE.fullmatch(name):
        raise ValueError(
            f"Unsafe speaker folder name: {name}. Use lowercase letters, digits, and underscores only."
        )


def discover_speaker_folders(input_dir: str | Path) -> list[Path]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory not found: {base}")
    folders = [path for path in base.iterdir() if path.is_dir()]
    for folder in folders:
        validate_speaker_folder_name(folder.name)
    return sorted(folders, key=lambda path: path.name)


def read_transcript(transcript_path: Path) -> str:
    return transcript_path.read_text(encoding="utf-8").strip()


def get_audio_duration_sec(audio_path: Path) -> float:
    import torchaudio

    info = torchaudio.info(str(audio_path))
    if info.sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for audio file: {audio_path}")
    return float(info.num_frames) / float(info.sample_rate)


def make_chunk_id(speaker_name: str, stem: str) -> str:
    return f"{speaker_name}_{stem}_{uuid4().hex[:12]}"


def build_row(
    *,
    speaker_folder: Path,
    speaker_id: int,
    audio_path: Path,
    transcript: str,
    duration_sec: float,
    encoded_audio,
) -> dict[str, Any]:
    encoded_audio_shape = list(encoded_audio.shape)
    encoded_audio_flat = encoded_audio.flatten().tolist()
    duration_ms = float(duration_sec * 1000.0)
    return {
        "source_file": str(audio_path),
        "chunk_id": make_chunk_id(speaker_folder.name, audio_path.stem),
        "start_ms": 0.0,
        "end_ms": duration_ms,
        "duration_ms": duration_ms,
        "transcription": transcript,
        "transcription_clean": transcript,
        "file_path": str(audio_path),
        "chunk_owner": int(speaker_id),
        "sample_prob": 1.0,
        "cb_weight": 1.0,
        "encoded_audio_shape": encoded_audio_shape,
        "encoded_audio": encoded_audio_flat,
    }


def build_local_speaker_dataset(
    *,
    input_dir: str | Path,
    audio_encoder: Callable[[Path], Any],
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    speaker_folders = discover_speaker_folders(input_dir)
    source_speaker_map = {
        folder.name: index for index, folder in enumerate(speaker_folders, start=1)
    }

    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "input_dir": str(input_dir),
        "speaker_count": len(speaker_folders),
        "row_count": 0,
        "speakers": [],
        "errors": [],
        "skipped": {
            "missing_transcript": [],
            "unsupported_audio": [],
            "empty_transcript": [],
            "duration_filtered": [],
        },
    }

    for folder in speaker_folders:
        speaker_id = source_speaker_map[folder.name]
        speaker_report = {
            "speaker_folder": folder.name,
            "speaker_id": speaker_id,
            "accepted_files": 0,
            "skipped_files": 0,
        }
        audio_files = sorted(
            [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}
            ],
            key=lambda path: path.name,
        )
        for audio_path in audio_files:
            transcript_path = audio_path.with_suffix(".txt")
            if not transcript_path.exists():
                report["errors"].append(
                    {
                        "type": "missing_transcript",
                        "speaker_folder": folder.name,
                        "audio_path": str(audio_path),
                        "transcript_path": str(transcript_path),
                    }
                )
                speaker_report["skipped_files"] += 1
                report["skipped"]["missing_transcript"].append(str(audio_path))
                continue

            transcript = read_transcript(transcript_path)
            if not transcript:
                report["skipped"]["empty_transcript"].append(str(audio_path))
                speaker_report["skipped_files"] += 1
                continue

            try:
                duration_sec = get_audio_duration_sec(audio_path)
            except Exception as exc:  # pragma: no cover - defensive path
                report["errors"].append(
                    {
                        "type": "unsupported_audio",
                        "speaker_folder": folder.name,
                        "audio_path": str(audio_path),
                        "error": str(exc),
                    }
                )
                report["skipped"]["unsupported_audio"].append(str(audio_path))
                speaker_report["skipped_files"] += 1
                continue

            if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
                report["skipped"]["duration_filtered"].append(
                    {
                        "audio_path": str(audio_path),
                        "duration_sec": duration_sec,
                    }
                )
                speaker_report["skipped_files"] += 1
                continue

            try:
                encoded_audio = audio_encoder(audio_path)
            except Exception as exc:
                report["errors"].append(
                    {
                        "type": "audio_encode_failed",
                        "speaker_folder": folder.name,
                        "audio_path": str(audio_path),
                        "error": str(exc),
                    }
                )
                report["skipped"]["unsupported_audio"].append(str(audio_path))
                speaker_report["skipped_files"] += 1
                continue

            rows.append(
                build_row(
                    speaker_folder=folder,
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    transcript=transcript,
                    duration_sec=duration_sec,
                    encoded_audio=encoded_audio,
                )
            )
            speaker_report["accepted_files"] += 1
        report["speakers"].append(speaker_report)

    report["row_count"] = len(rows)
    return rows, source_speaker_map, report


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    source_speaker_map: dict[str, int],
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    schema = define_arrow_schema()
    column_data = {field.name: [] for field in schema}
    for row in rows:
        for column_name in column_data:
            column_data[column_name].append(row[column_name])
    table = pa.table(column_data, schema=schema)
    pq.write_table(table, output_file, compression="zstd")

    map_path = output_file.parent / "source_speaker_map.json"
    report_path = output_file.parent / "build_report.json"
    map_path.write_text(json.dumps(source_speaker_map, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_encoder(config_path: str, sample_rate: int) -> LocalSpeakerEncoder:
    if sample_rate != 44100:
        raise ValueError("Only sample_rate=44100 is currently supported by the DAC path")
    config = DiaConfig.load(config_path)
    if config is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    dia = Dia(config=config, compute_dtype=ComputeDtype.FLOAT32, load_dac=True)
    dia._load_dac_model()
    return LocalSpeakerEncoder(dia)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local speaker parquet dataset")
    parser.add_argument("--input-dir", default="data/local_speakers")
    parser.add_argument("--output-path", default="data/parquet/local_speakers.parquet")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = build_encoder(args.config_path, args.sample_rate)
    rows, source_speaker_map, report = build_local_speaker_dataset(
        input_dir=args.input_dir,
        audio_encoder=encoder.encode,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    write_outputs(
        rows=rows,
        source_speaker_map=source_speaker_map,
        report=report,
        output_path=args.output_path,
    )
    print(f"rows={len(rows)}")
    print(f"speakers={len(source_speaker_map)}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
