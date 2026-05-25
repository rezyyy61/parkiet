# Local Speaker Dataset Builder

## Why this exists

The original Parkiet preprocessing path is built for large-scale production ingestion:

- GCS
- Redis
- PostgreSQL
- Whisper-based transcription stages

That is the right pipeline for bulk data preparation, but it is too heavy for a small speaker-conditioned fine-tune experiment.

This local builder creates a much smaller path:

1. curated local speaker folders
2. matching `.wav` + `.txt`
3. DAC encoding through the existing `Dia.load_audio(...)` path
4. parquet output compatible with `src/parkiet/jax/dataset.py`

## Expected folder format

```text
data/local_speakers/
  nl_female_01/
    metadata.json
    001.wav
    001.txt
    002.wav
    002.txt
  nl_male_01/
    metadata.json
    001.wav
    001.txt
```

Each speaker folder is treated as one stable source speaker.

## Speaker mapping

Each speaker folder gets a stable integer `chunk_owner`.

The mapping is deterministic and sorted by folder name, for example:

```json
{
  "nl_female_01": 1,
  "nl_male_01": 2
}
```

This local mapping is written to `source_speaker_map.json`.

## Output parquet schema

The output rows match the fields consumed by the existing training dataset and speaker audit tools:

- `source_file`
- `chunk_id`
- `start_ms`
- `end_ms`
- `duration_ms`
- `transcription`
- `transcription_clean`
- `file_path`
- `chunk_owner`
- `sample_prob`
- `cb_weight`
- `encoded_audio_shape`
- `encoded_audio`

This is intentionally aligned with the Arrow schema used in `src/parkiet/audioprep/arrow_writer.py`.

## How transcripts are handled

The local builder assumes the `.txt` file already contains the desired transcript.

It writes:

- `transcription`
- `transcription_clean`

using the same local text content.

No additional normalization or Whisper stage is added here.

## Limitations

- no speaker diarization
- no automatic transcript cleaning
- no sample probability balancing beyond a simple default
- no database/GCS integration
- no multi-speaker chunk splitting

This tool is for **curated local speaker data only**.

It is suitable for preparing a small clean fine-tune dataset, not for replacing the original large-scale audioprep pipeline.
