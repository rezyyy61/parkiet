# Voice Identity Registry Phase

## What this phase adds

This phase adds the registry foundation only:

1. a design contract for public `voice_id` and internal `model_speaker_id`
2. a small utility module for loading, saving, validating, and querying registry data
3. tests for registry integrity and append-only ID rules

## What this phase does not do

This phase does **not**:

- change model behavior
- change training behavior
- change realtime behavior
- inject voice registry lookups into `Dia.generate(...)`
- create or train speaker-conditioned checkpoints

## Why this layer is needed before training

Future speaker-conditioned training needs a stable, versioned identity system.

Without this layer:

- raw `chunk_owner` values leak into product APIs
- model speaker IDs become hard to reproduce across checkpoints
- curated public voice selection becomes unstable

## What the next phase will do

The next phase should use this registry to:

1. build training `speaker_vocab.json`
2. define dataset filtering / quality gates
3. map curated voices to contiguous `model_speaker_id`
4. prepare speaker-conditioned training inputs

