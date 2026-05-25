# Speaker Conditioning Phase 2

## What is now wired

Phase 2 adds model-side scaffolding for explicit speaker conditioning:

1. `DiaModel` can now create speaker conditioning modules when `speaker_conditioning_enabled=True`
   - `speaker_embedding`
   - `speaker_to_encoder`
   - `speaker_to_decoder`

2. `Encoder.forward(...)` now accepts:
   - `speaker_condition: torch.Tensor | None`

3. `Decoder.forward(...)` and `Decoder.decode_step(...)` now accept:
   - `speaker_condition: torch.Tensor | None`

4. `Dia._prepare_generation(...)` now wires validated `speaker_id` values into model-side speaker bias tensors when conditioning is enabled.

5. Under CFG, the unconditional text branch keeps the same speaker identity.

## Why behavior still does not change by default

All speaker conditioning remains disabled by default:

- `speaker_conditioning_enabled=False`

When disabled:

- no speaker modules are constructed
- `speaker_id` is ignored
- encoder and decoder run exactly as before
- old checkpoints still load unchanged

## Why this still does not produce stable voices

This phase only wires the model to *accept* speaker conditioning.

It does **not** mean the current checkpoint suddenly understands speaker identity.

The current Dutch checkpoint was not trained with:

- `speaker_id`
- speaker embedding injection
- conditioned speaker-consistency objectives

So even though the code path exists now, stable speaker behavior requires a new conditioned checkpoint produced by fine-tuning or retraining.

## Checkpoint behavior

- old checkpoints still load when `speaker_conditioning_enabled=False`
- when `speaker_conditioning_enabled=True`, a checkpoint with speaker-conditioning weights is required

This is intentional. Silent partial loading would hide configuration mistakes.

## What Phase 3 must do

Phase 3 should add the actual training path:

1. batch `speaker_id` through training
2. pass `speaker_id` into model forward during training
3. initialize a conditioned checkpoint from the current Dutch model
4. fine-tune on speaker-labeled data
5. evaluate same-speaker consistency across independent phrases

