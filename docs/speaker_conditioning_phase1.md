# Speaker Conditioning Phase 1

## What was added

Phase 1 adds safe scaffolding only:

1. `DiaConfig` schema support for speaker conditioning fields
2. `src/parkiet/speaker_vocab.py` for speaker vocab persistence and mapping
3. optional `speaker_vocab` plumbing in `src/parkiet/jax/dataset.py`
4. `speaker_id` argument plumbing in `Dia.generate(...)`
5. validation helpers that remain inactive when speaker conditioning is disabled

## What is still not functional

This phase does **not** add actual speaker conditioning to the model.

Specifically, it does not yet:

- add a speaker embedding table
- inject speaker conditioning into encoder or decoder
- change training forward passes
- change checkpoint structure
- change realtime behavior

`speaker_id` is currently only:

- accepted by `Dia.generate(...)`
- validated if speaker conditioning is enabled
- ignored when speaker conditioning is disabled

## Why no behavior changed

Defaults are backward-compatible:

- `speaker_conditioning_enabled=False`
- `speaker_embedding_dim=256`
- `num_speakers=0`
- `default_speaker_id=0`
- `speaker_conditioning_mode="encoder_decoder_additive"`

Because conditioning is disabled by default:

- existing configs still load
- existing checkpoints still load
- existing offline inference behavior is unchanged
- existing realtime behavior is unchanged

## What Phase 2 will implement

Phase 2 should implement model-side conditioning:

1. add speaker embedding modules
2. inject conditioning into encoder and decoder
3. plumb `speaker_id` through training and inference
4. keep old behavior when no `speaker_id` is provided

After that, the model can be fine-tuned for actual speaker consistency evaluation.

