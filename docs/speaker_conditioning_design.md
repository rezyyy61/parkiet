# Speaker Conditioning Design

## Scope

This document turns the findings in [docs/realtime_voice_identity_audit.md](/home/rezy/PhpstormProjects/parkiet/docs/realtime_voice_identity_audit.md) into a concrete implementation design for adding explicit speaker conditioning to Parkiet/Dia.

This design does **not** implement changes. It defines the model, dataset, training, inference, and rollout plan required to add reliable speaker identity control.

## Design Goal

Enable stable speaker identity across independent phrase generations by adding explicit speaker conditioning to the model.

Target capability:

- `Dia.generate(text, speaker_id=...)` should keep one speaker/accent across independent calls
- old behavior must remain available when `speaker_id` is omitted

---

## 1. Current Architecture Summary

## 1.1 Encoder input shape

Current text path:

- raw input text is byte-tokenized in `Dia._encode_text()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:261)
- padded shape is `[B, 1, T_text]` in `_pad_text_input()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:281)
- `_prepare_generation()` flattens this into `[B, T_text]` for conditional mode or `[2B, T_text]` for CFG in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:373)
- `Encoder.embedding` maps token IDs to hidden states in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:631)

Effective encoder input:

- token IDs: `[B_or_2B, T_text]`
- embedded text hidden states: `[B_or_2B, T_text, encoder_hidden_size]`

## 1.2 Decoder input shape

Current decoder path:

- `_prepare_audio_prompt()` prepares delayed DAC tokens in shape `[B, T_audio_prefill, C]` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:304)
- decoder input per step is `[B_or_2B, 1, C]` in `_decoder_step()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:432)
- decoder embeddings are one table per codebook channel in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:776)

Effective decoder input:

- token IDs: `[B_or_2B, T_audio, C]`
- embedded hidden states: `[B_or_2B, T_audio, decoder_hidden_size]`

## 1.3 DAC code structure

Current audio token structure:

- `decoder_config.num_channels = 9` in [config.json](/home/rezy/PhpstormProjects/parkiet/config.json:9)
- DAC code prompt/output layout is `[T, C]` where `C = 9`
- prompt waveform is encoded by `_encode()` / `load_audio()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:590) and [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:614)

## 1.4 Where text conditioning enters

Text conditioning enters only through encoder-decoder cross-attention:

- `DecoderLayer.cross_attention` in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:703)
- `Decoder.precompute_cross_attn_cache()` builds cached K/V from encoder output in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:806)

## 1.5 Why current model has no `voice_id`

The current model has no explicit identity input because:

- no `speaker_id` is accepted by model forward/generate
- no speaker embedding table exists in model config or modules
- no speaker embedding vector is injected into encoder or decoder
- training dataset does not pass `chunk_owner` / `speaker_id` to the model

Result:

- speaker identity is latent and underdetermined
- text-only generation can change speaker/accent across calls

---

## 2. Proposed Conditioning Interface

## 2.1 Initial API

Add explicit model inputs:

- `speaker_id: int | None`
- `speaker_embedding: torch.Tensor | None` later

Recommended first implementation:

- implement `speaker_id`
- reserve architecture for future `speaker_embedding`

## 2.2 Config additions

Extend `DiaConfig` / `config.json` schema with:

- `speaker_conditioning_enabled: bool = False`
- `speaker_embedding_dim: int = 256`
- `num_speakers: int = 0`
- `default_speaker_id: int = 0`

Optional later:

- `speaker_conditioning_mode: "encoder_decoder_additive" | "encoder_only" | "decoder_only" | "film" | "prefix"`

## 2.3 Speaker embedding representation

Add learned table:

- `nn.Embedding(num_speakers, speaker_embedding_dim)` in PyTorch
- equivalent module in JAX

Projection layers:

- `speaker_to_encoder: Linear(speaker_embedding_dim -> encoder_hidden_size)`
- `speaker_to_decoder: Linear(speaker_embedding_dim -> decoder_hidden_size)`

## 2.4 Unknown/default speaker behavior

Recommended behavior:

- `speaker_id is None`:
  - if `speaker_conditioning_enabled=False`: old behavior
  - if `speaker_conditioning_enabled=True`: use `default_speaker_id`
- invalid speaker IDs:
  - fail explicitly in training
  - fail explicitly in inference unless a caller deliberately maps unknown IDs to `default_speaker_id`

## 2.5 Backward compatibility

Backward-compatible behavior must be explicit:

- old checkpoints with no speaker conditioning must still load if `speaker_conditioning_enabled=False`
- inference with no `speaker_id` should preserve current model behavior

Recommended checkpoint policy:

- support old checkpoints only when conditioning is disabled
- speaker-conditioned checkpoints are new checkpoints with new config fields

---

## 3. Where To Inject Speaker Conditioning

This section compares five options.

## A) Encoder-only additive conditioning

Mechanism:

- add projected speaker vector to all encoder token embeddings

Example:

```python
x = self.embedding(x_ids)
x = x + speaker_bias[:, None, :]
```

Pros:

- simple
- low parameter cost
- strongly influences text interpretation

Cons:

- acoustic realization is still only indirectly influenced through cross-attention
- may be weaker for stable timbre than decoder-side conditioning

## B) Decoder-only additive conditioning

Mechanism:

- add projected speaker vector to decoder input hidden states

Example:

```python
x = sum(channel_embeddings)
x = x + speaker_bias[:, None, :]
```

Pros:

- directly influences acoustic token generation

Cons:

- lexical/prosodic interpretation remains speaker-agnostic upstream
- may be less stable than encoder+decoder combined

## C) Encoder + decoder additive conditioning

Mechanism:

- add one projected speaker vector to encoder inputs
- add another projected speaker vector to decoder inputs

Pros:

- simplest strong conditioning design
- explicit on both semantic and acoustic sides
- small code change relative to more complex adapters

Cons:

- slightly more parameters than A or B

## D) Adaptive layer norm / FiLM

Mechanism:

- generate per-layer scale/shift from speaker embedding
- apply to encoder/decoder blocks or norms

Pros:

- more expressive conditioning
- likely stronger speaker control

Cons:

- more invasive
- larger implementation surface
- harder first rollout

## E) Learned prefix token

Mechanism:

- convert speaker embedding into one or more learned conditioning tokens
- prepend to encoder input or decoder prefix

Pros:

- natural Transformer conditioning pattern

Cons:

- interacts with positional structure
- less direct than additive bias
- more moving pieces for first implementation

## Recommendation

### Minimal first implementation

Use **Option C: encoder + decoder additive conditioning**.

Reason:

- smallest robust model-level change
- explicit conditioning on both sides
- easy to implement in both PyTorch and JAX
- easier to debug than FiLM/prefix methods

### Future upgrade path

If additive conditioning is not strong enough:

- move to FiLM / adaptive norm in a second phase

---

## 4. Exact Code Areas To Modify

## 4.1 `src/parkiet/dia/layers.py`

Primary targets:

- `class Encoder`
- `class Decoder`
- `class DiaModel`

Required changes:

1. Add speaker embedding module(s) to `DiaModel`
   - `speaker_embedding`
   - `speaker_to_encoder`
   - `speaker_to_decoder`

2. Update `Encoder.forward(...)`
   - accept optional `speaker_condition: torch.Tensor | None`
   - add it to embedded token states

3. Update `Decoder.forward(...)`
   - accept optional `speaker_condition: torch.Tensor | None`
   - add it to decoder input embeddings before decoder layers

4. Update `Decoder.decode_step(...)`
   - same speaker conditioning path as `Decoder.forward(...)`

Recommended interface shape:

- encoder speaker bias: `[B_or_2B, encoder_hidden_size]`
- decoder speaker bias: `[B_or_2B, decoder_hidden_size]`

## 4.2 `src/parkiet/dia/model.py`

Primary targets:

- `Dia._prepare_generation(...)`
- `Dia._decoder_step(...)` only as argument plumbing if needed
- `Dia.generate(...)`

Required changes:

1. Add `speaker_id: int | list[int] | None = None`
2. Add optional future `speaker_embedding: torch.Tensor | None = None`
3. Resolve batch speaker inputs
4. If CFG is enabled:
   - conditional branch gets the real speaker condition
   - unconditional branch should still keep speaker condition

### Important design choice: CFG unconditional text should not imply unconditional speaker

Recommended rule:

- CFG removes text, not speaker identity
- unconditional branch uses zero text but the **same speaker condition**

Reason:

- we want CFG to anchor semantics, not erase identity

Implementation implication in `_prepare_generation(...)`:

- duplicate speaker condition across unconditional/conditional branches
- do **not** zero it for the unconditional branch

## 4.3 `src/parkiet/jax/model.py`

Modify the JAX path in parallel with PyTorch if training/fine-tune remains JAX-first.

Targets:

- `class Dia`
- `Encoder`
- `Decoder`
- generation/state plumbing equivalent to PyTorch

Reason:

- current training path is JAX
- speaker-conditioned fine-tuning must train the same architecture that inference will use

## 4.4 `src/parkiet/jax/dataset.py`

Modify:

- `AudioTextDataset.__getitem__()`

Add output:

- `speaker_id`

Behavior:

- read `chunk_owner` from parquet
- map invalid / missing owner to `default_speaker_id`

## 4.5 `src/parkiet/jax/train.py`

Modify:

- `compute_loss(...)`
- any batch preparation path that builds model inputs

Changes:

- pass `speaker_id` into model forward path
- loss itself remains unchanged

## 4.6 `src/parkiet/jax/train_distributed.py`

Same changes as `train.py`:

- batch must include `speaker_id`
- model forward must receive it

## 4.7 `config.json` schema

Add the new fields described above.

If config parsing is defined elsewhere, update:

- `src/parkiet/dia/config.py`

to include:

- `speaker_conditioning_enabled`
- `speaker_embedding_dim`
- `num_speakers`
- `default_speaker_id`

---

## 5. Dataset Changes

## 5.1 Existing usable fields

Existing data pipeline already has:

- `chunk_owner`
- speaker embeddings in prep/database
- speaker IDs from diarization/database

Evidence from current audit:

- parquet schema includes `chunk_owner`
- training dataset currently ignores it

## 5.2 New training dataset output

Recommended dataset sample:

```python
{
    "text": ...,
    "audio": ...,
    "cb_weight": ...,
    "speaker_id": ...,
}
```

## 5.3 Mapping rule

Recommended mapping:

- if `chunk_owner >= 0`: use it as `speaker_id`
- else use `default_speaker_id`

## 5.4 Speaker ID space

Need one consistent mapping layer:

- dataset speaker IDs from DB may be sparse
- training should remap them into contiguous `0..num_speakers-1`

Recommended preprocessing artifact:

- `speaker_vocab.json`

Contents:

```json
{
  "default_speaker_id": 0,
  "db_speaker_id_to_model_speaker_id": {
    "17": 1,
    "42": 2
  }
}
```

This avoids tying model embedding table indices directly to database primary keys.

---

## 6. Training Changes

## 6.1 Batch structure

Batches must include:

- `text`
- `audio`
- `cb_weight`
- `speaker_id`

## 6.2 Forward path

Current training flow:

- encode text
- build decoder state
- teacher-force audio tokens
- compute token loss

New flow:

- encode text with `speaker_id`
- decode audio with same `speaker_id`
- compute the same loss as before

## 6.3 Loss function

Loss stays unchanged.

Reason:

- speaker identity is an input condition
- target remains next audio token prediction

No new speaker classification loss is required for the first implementation.

## 6.4 Old checkpoint preservation

Recommended compatibility policy:

1. If `speaker_conditioning_enabled=False`
   - old checkpoints load normally
2. If `speaker_conditioning_enabled=True`
   - use new checkpoints only

This avoids fragile partial loading behavior.

Optional later:

- allow partial initialization where new speaker modules are randomly initialized and old weights are loaded into shared layers

That is useful for fine-tuning from current checkpoints, but should be implemented deliberately and tested, not assumed implicitly.

---

## 7. Inference Changes

## 7.1 Generation args

Add to `Dia.generate(...)`:

- `speaker_id: int | list[int] | None = None`

Reserve for later:

- `speaker_embedding: torch.Tensor | list[torch.Tensor] | None = None`

## 7.2 Batch semantics

Rules:

- single `speaker_id` with single `text` => scalar behavior
- list of `speaker_id` with list of `text` => batch behavior
- lengths must match

## 7.3 Default behavior

When `speaker_id is None`:

- if speaker conditioning is disabled, preserve current behavior
- if speaker conditioning is enabled, use `default_speaker_id`

## 7.4 Realtime call shape

Realtime code does not change in this design phase, but target usage becomes:

```python
Dia.generate(text, speaker_id=...)
```

This is the correct long-term replacement for trying to infer identity from `[S1]` or `audio_prompt`.

---

## 8. Fine-Tuning Plan

## 8.1 Minimal first phase

Recommended first fine-tune:

- initialize from current Dutch checkpoint
- add speaker embedding table + additive conditioning projections
- fine-tune all layers

### Why not freeze most of the model first?

Because:

- current model was not trained to consume explicit speaker identity
- if only the new embedding table is trained while core layers are frozen, the network may not learn to use the new condition strongly enough

### Recommendation

Start with:

- full-model fine-tune
- small learning rate
- possibly smaller LR for backbone and slightly larger LR for new speaker modules

Example policy:

- backbone LR: low
- new speaker modules LR: 3x to 10x backbone LR

## 8.2 Data requirements

Need:

- multiple utterances per speaker
- enough variation per speaker across text/prosody

If labels are weak or sparse, the model may overfit or entangle identity with chunk-specific acoustics.

## 8.3 Evaluation target

Primary success criterion:

- same `speaker_id`, different phrases => stable voice across independent generations

---

## 9. Evaluation Plan

## 9.1 Functional tests

1. **Same speaker consistency**
   - fixed `speaker_id`
   - 10 independent Dutch phrases
   - expected: same speaker/accent/timbre

2. **Different speaker separability**
   - same phrase
   - multiple `speaker_id` values
   - expected: clearly different voices

3. **Backward-compatible behavior**
   - no `speaker_id`
   - expected: old behavior preserved

4. **Dutch pronunciation quality**
   - compare baseline text quality vs speaker-conditioned model
   - expected: no regression in pronunciation/naturalness

5. **Realtime suitability**
   - measure `RTF`
   - compare conditioned vs unconditioned generation
   - expected: minimal overhead from speaker embedding lookup/addition

## 9.2 Objective measurements

Recommended metrics:

- speaker embedding cosine similarity for same-speaker generations
- speaker embedding separation for different-speaker generations
- MOS-style listening evaluation
- WER / ASR transcript quality as regression guard

## 9.3 Expected performance impact

Additive speaker conditioning should have negligible inference overhead relative to the decoder itself.

Main added costs:

- embedding lookup
- two small projection layers
- bias addition

These are insignificant compared to Transformer decoding.

---

## 10. Risks And Open Questions

## 10.1 Noisy speaker labels

Risk:

- `chunk_owner` may be wrong
- diarization may mix speakers

Impact:

- embedding table learns blurred identity clusters

Mitigation:

- filter low-confidence chunks
- require minimum duration per speaker
- build contiguous model speaker vocab only from reliable speakers

## 10.2 Speaker / accent / style entanglement

Risk:

- current data may entangle:
  - speaker
  - accent
  - recording condition
  - speaking style

Impact:

- `speaker_id` may partly act as “speaker + acoustic condition + style”

Mitigation:

- collect more balanced data
- later extend conditioning with separate style/accent metadata if needed

## 10.3 Checkpoint compatibility

Risk:

- adding new modules changes checkpoint structure

Mitigation:

- explicit config versioning
- separate load path for conditioned checkpoints
- optional partial init utility for fine-tuning from current checkpoint

## 10.4 Amount of data needed

Unknown:

- how many speakers and utterances are enough for stable conditioning in this architecture

Likely:

- a small number of speakers with enough utterances can already prove the mechanism
- production generalization needs broader coverage

## 10.5 CFG interaction

Open question:

- should unconditional branch keep speaker conditioning?

Recommendation:

- yes, keep speaker conditioning in both branches

Reason:

- CFG should remove text content, not erase identity

This should still be validated experimentally.

---

## 11. Final Recommended Implementation Order

## Phase 1: Dataset plumbing

1. Extend parquet-to-dataset path to expose `speaker_id`
2. Add speaker vocab remapping
3. Make JAX dataset return `speaker_id`

## Phase 2: Model config + embedding

1. Extend `DiaConfig`
2. Add speaker embedding table + projection layers
3. Add additive conditioning to encoder and decoder

## Phase 3: PyTorch inference path

1. Add `speaker_id` argument to `Dia.generate(...)`
2. Add `speaker_id` plumbing through `_prepare_generation(...)`
3. Preserve old behavior when conditioning is absent

## Phase 4: Training / fine-tune path

1. Pass `speaker_id` through JAX train path
2. Fine-tune from current checkpoint with new modules
3. Validate same-speaker / different-speaker behavior

## Phase 5: Realtime integration

After the model proves stable:

1. add `speaker_id` support to realtime backend
2. map session voice selection to model `speaker_id`
3. leave `audio_prompt` as optional continuation/reference tool, not the main `voice_id`

---

## Final Recommendation

The first implementation should be:

1. **speaker-conditioned checkpoint fine-tune**
2. **explicit `speaker_id` embedding**
3. **encoder + decoder additive conditioning**
4. **backward-compatible fallback when `speaker_id` is omitted**

This is the smallest design that directly solves the actual missing architectural variable: explicit speaker identity.

