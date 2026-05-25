# Speaker Conditioning Phase 4 Dataset Audit

## Why this phase exists

Before any speaker-conditioned fine-tune, the dataset must be audited at the speaker level.

`chunk_owner` values in parquet shards are raw source identifiers. They are not suitable as product-facing IDs, and they are not automatically good training speakers.

We need a curated intermediate layer:

1. inspect available source speakers
2. reject noisy or weak candidates
3. build a candidate `voice_registry`
4. derive a stable `speaker_vocab`

## Generated files

### `scripts/audit_speakers.py`

Outputs:

- `speaker_audit.json`
- `speaker_audit.csv`
- `candidate_speakers.json`

These files summarize:

- `chunk_owner`
- `sample_count`
- `total_duration_sec`
- `avg_duration_sec`
- transcription quality indicators
- optional `cb_weight` summary
- optional `encoded_audio_shape` summary
- whether the speaker passes threshold-based candidacy

### `scripts/build_voice_registry_from_audit.py`

Outputs:

- `voice_registry/voices.json`
- `voice_registry/speaker_vocab.json`

This step converts approved source speakers into:

- public `voice_id`
- contiguous `model_speaker_id`

## How to select speakers

The initial thresholds are intentionally simple:

- minimum total duration
- minimum sample count

That only produces *candidates*.

Human review is still needed for:

- noisy speakers
- mixed-speaker IDs
- wrong language or locale
- poor transcript quality
- unstable acoustic conditions

By default, generated candidate voices are kept `inactive` unless explicitly activated.

## Mapping path

The mapping is:

`source_speaker_id / chunk_owner -> candidate audit -> voice_registry -> speaker_vocab -> model_speaker_id`

Training should consume only:

- `speaker_vocab.json`
- curated parquet filters

The model should never see raw public `voice_id` strings.

## Phase 5

Phase 5 should use this audit output to run a tiny conditioned fine-tune with:

- a cleaned subset of approved speakers
- a frozen registry version
- a frozen `speaker_vocab_version`
- explicit evaluation of same-speaker / different-speaker phrase consistency
