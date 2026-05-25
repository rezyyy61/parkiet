# Speaker Conditioning Phase 3

## Scope

This phase adds **training plumbing only** for speaker conditioning in the JAX path.

It does **not** start fine-tuning, does **not** change realtime behavior, and does **not** change offline behavior when `speaker_conditioning_enabled=False`.

## What Was Added

### 1. `speaker_vocab` reaches the training dataset

`src/parkiet/jax/dataset.py`

- `AudioTextDataset` already supported optional `speaker_vocab`.
- `batch_iterator(...)` now carries `speaker_id` into the batch when it exists in individual samples.
- Without `speaker_vocab`, the batch fields remain unchanged:
  - `text`
  - `audio`
  - `cb_weight`

With `speaker_vocab`, the batch may also contain:
  - `speaker_id`

### 2. Training config now supports `speaker_vocab_path`

`src/parkiet/jax/train.py`
`src/parkiet/jax/train_distributed.py`

`TrainingConfig` now accepts:

- `speaker_vocab_path: str | None`

Behavior:

- omitted: old path unchanged
- provided: passed to `create_dataset(...)` as `speaker_vocab`

### 3. Prepared batches preserve `speaker_id`

`load_and_prepare_batch(...)` in both training files now forwards `speaker_id` if it exists.

The prepared batch still always includes:

- `text`
- `audio_input`
- `audio_target`

And conditionally includes:

- `speaker_id`

### 4. Loss path now accepts `speaker_id`

`compute_loss(...)` in both training files now accepts:

- `speaker_id: jnp.ndarray | None = None`

Rules:

- if `speaker_conditioning_enabled=False`, `speaker_id` is ignored
- if `speaker_conditioning_enabled=True`, missing `speaker_id` raises a clear `ValueError`

### 5. JAX model-side hooks are now usable from training

`src/parkiet/jax/layers.py`

The JAX `DiaModel` now mirrors the Phase 2 PyTorch scaffolding:

- conditional `speaker_embedding`
- `speaker_to_encoder`
- `speaker_to_decoder`
- `get_speaker_condition(...)`
- additive conditioning entry points on:
  - `Encoder.__call__(..., speaker_condition=None)`
  - `Decoder.__call__(..., speaker_condition=None)`
  - `Decoder.decode_step(..., speaker_condition=None)`

This remains fully gated behind `speaker_conditioning_enabled`.

## What Is Still Not Trained

No conditioned checkpoint exists yet.

This phase does **not** prove that stable voices work. It only ensures the training path can now carry:

- `voice_id` -> `speaker_vocab`
- `speaker_vocab` -> `model_speaker_id`
- `model_speaker_id` -> `speaker_id` batch field
- `speaker_id` -> model conditioning tensors

## How `speaker_vocab` connects `voice_id` to `model_speaker_id`

The intended mapping remains:

1. `voice_registry/voices.json` defines public `voice_id`
2. `voice_registry/speaker_vocab.json` maps:
   - `voice_id -> model_speaker_id`
   - `source_speaker_id -> model_speaker_id`
3. dataset rows with `chunk_owner` map to `speaker_id`
4. training sees only contiguous `model_speaker_id`

## CFG Rule

The dataset still applies text dropout for CFG-style training.

That dropout only affects text.

It does **not** remove `speaker_id`.

This is intentional: speaker identity should remain conditioned while semantic text condition may be dropped.

## Phase 4

Phase 4 should do the first small fine-tune with:

- a curated `speaker_vocab`
- filtered, speaker-clean training data
- `speaker_conditioning_enabled=True`
- a new conditioned checkpoint

Evaluation should focus on:

- same `speaker_id`, different phrases => stable voice
- different `speaker_id`, same phrase => distinct voices
- no `speaker_id` / conditioning disabled => old behavior unchanged
