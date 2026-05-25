# Realtime Voice Identity Audit

## Scope

This report audits the current Parkiet/Dia architecture to answer one narrow question:

Can the current model support a stable `voice_id` / speaker identity across independent realtime phrase generations, and if not, what is the correct model-level fix?

This report is evidence-based. It does **not** propose quick inference-only patches as a final solution, and it does **not** change behavior.

## Executive Summary

### Confirmed conclusions

1. **The current model has no explicit speaker identity input.**
   There is no `speaker_id`, `voice_id`, `accent_id`, speaker embedding, style embedding, or reference encoder in the PyTorch or JAX model architecture. The exposed conditioning inputs are:
   - byte-level text tokens into the encoder
   - optional `audio_prompt` DAC codes into the decoder prefill

2. **`[S1]`, `[S2]`, `[S3]`, `[S4]` are role/turn markers, not stable voice IDs.**
   In the PyTorch path, `[S1]` and `[S2]` are replaced with byte values in `_encode_text()`. In the JAX path and dataset preprocessing, `[S1]..[S4]` are also byte substitutions. There is no mapping from these tags to any learned speaker embedding table.

3. **The model appears to be trained as a multi-speaker text+audio next-token model without explicit speaker labels.**
   The JAX dataset loader feeds only:
   - `text`
   - `audio`
   - `cb_weight`
   into training. The parquet dataset schema stores `chunk_owner`, but the actual training dataset does not consume it.

4. **Therefore speaker identity is underdetermined at inference time for text-only generation.**
   If the same text style can be spoken by multiple speakers in the training set and no explicit speaker identity is provided, the model can produce different valid speakers/accent realizations across independent calls.

5. **`audio_prompt` is a decoder continuation mechanism, not an explicit reusable voice embedding system.**
   It pre-fills delayed DAC token streams into the decoder. That can help local continuation, but it is not equivalent to a speaker encoder or a stable reference embedding. This explains why it can remove some artifacts yet still fail to lock speaker identity across separate phrases.

6. **A reliable product-level `voice_id` cannot be obtained from the current architecture without model-level conditioning changes or fine-tuning.**
   The best long-term fix is to add explicit speaker conditioning to the model and train or fine-tune with speaker identity labels or reference-derived speaker embeddings.

### Go / No-Go

- **Quick inference-only fix:** No-Go
- **`audio_prompt`-based `voice_id` as a reliable product feature:** No-Go
- **Model-level speaker conditioning:** Go
- **Fine-tuning / retraining with speaker conditioning:** Go

---

## 1. Confirmed Facts From Code

### 1.1 Encoder architecture

The text encoder is a plain byte-token Transformer encoder:

- `Encoder.embedding` is `nn.Embedding(enc_config.vocab_size, enc_config.hidden_size)` in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:631)
- `encoder_config.vocab_size` is `256` in [config.json](/home/rezy/PhpstormProjects/parkiet/config.json:18)
- `Dia._encode_text()` encodes UTF-8 bytes and replaces special tags with single-byte values in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:261)

There is no additional speaker/style input path into the encoder.

### 1.2 Decoder architecture

The decoder is a multi-channel autoregressive Transformer over DAC token codebooks:

- `decoder_config.num_channels = 9` in [config.json](/home/rezy/PhpstormProjects/parkiet/config.json:9)
- `Decoder.embeddings` contains one embedding table per audio channel in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:776)
- `Decoder.logits_dense` outputs `[num_channels, vocab_size]` logits in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:796)
- Text conditioning enters the decoder only through cross-attention to encoder outputs in `DecoderLayer.cross_attention` at [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:703)

This is an encoder-decoder next-token model over audio token streams, not a model with explicit speaker conditioning.

### 1.3 Audio codec usage

The audio representation is DAC code indices:

- `_encode()` converts waveform to DAC codes in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:590)
- `_decode()` converts DAC codes back to waveform in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:602)
- `load_audio()` loads waveform, resamples to `44100`, folds to mono, and returns DAC codes in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:614)

The prompt format used by `audio_prompt` is therefore `[T, C]` DAC code indices, not waveform embeddings.

### 1.4 Where `audio_prompt` enters

`audio_prompt` is inserted into the decoder prefill stream:

- `_prepare_audio_prompt()` expects each prompt as `[T, C]` and writes it into `prefill[i, 1:prompt.shape[0]+1, :]` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:304)
- `_prepare_generation()` calls `_prepare_audio_prompt()` and then performs decoder prefill in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:373)

This is decoder-prefix conditioning. It is not a separate speaker encoder.

### 1.5 Generation uses stochastic sampling

The PyTorch path samples with `torch.multinomial()` in `_sample_next_token()`:

- [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:26)

It uses:

- `temperature`
- `top_p`
- `top_k`

There is no `generator` argument in the PyTorch path. There is no library-level `seed` argument in `Dia.generate()`.

The JAX path also samples stochastically with `random.categorical()` and a rolling RNG key in [src/parkiet/jax/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/model.py:151) and [src/parkiet/jax/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/model.py:760).

---

## 2. What `[S1]`, `[S2]`, `[S3]`, `[S4]` Actually Mean

### Confirmed facts

In PyTorch:

- `[S1] -> 0x01`
- `[S2] -> 0x02`

via `Dia._encode_text()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:261)

In JAX dataset preprocessing:

- `[S1] -> 0x01`
- `[S2] -> 0x02`
- `[S3] -> 0x03`
- `[S4] -> 0x04`

via `AudioTextDataset._encode_text()` in [src/parkiet/jax/dataset.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/dataset.py:118)

In JAX inference:

- multi-speaker examples explicitly use `[S1]..[S4]` in [src/parkiet/jax/inference.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/inference.py:26)

### Conclusion

`[S1]..[S4]` are **special text tokens / turn markers**. They are not speaker IDs in the architecture.

### PyTorch vs JAX mismatch

- JAX path recognizes `[S1]..[S4]`
- PyTorch path currently only recognizes `[S1]` and `[S2]`

Evidence:

- PyTorch `_encode_text()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:261)
- JAX `_encode_text()` in [src/parkiet/jax/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/model.py:369)

### Impact of this mismatch

Confirmed:

- It is a real mismatch.
- It can affect semantics for prompts containing `[S3]` and `[S4]`.

Unknown:

- The exact quantitative impact on single-speaker Dutch TTS drift.

Assessment:

- This is a correctness bug for multi-speaker text formatting.
- It is **not** the core explanation for single-speaker voice instability across separate calls.

---

## 3. Speaker / Voice / Style Conditioning: What Exists and What Does Not

### What exists

1. Byte-level text tokens
2. Decoder `audio_prompt` DAC prefix
3. Classifier-free guidance via text dropout training

Evidence:

- CFG training text dropout in [src/parkiet/jax/dataset.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/dataset.py:99)
- CFG unconditional/conditional batching in `_prepare_generation()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:373)

### What does not exist

No explicit:

- `speaker_id`
- `voice_id`
- `accent_id`
- `speaker_embedding`
- `style_embedding`
- reference encoder
- speaker classifier head
- style adapter

Evidence:

- No such fields in [config.json](/home/rezy/PhpstormProjects/parkiet/config.json:1)
- No such modules in [src/parkiet/dia/layers.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/layers.py:1)
- No such arguments in `Dia.generate()` other than `audio_prompt` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:661)

### Conclusion

The current architecture has **no explicit speaker control surface**.

---

## 4. Training / Dataset Path

### 4.1 What the parquet dataset stores

The Arrow schema includes:

- `transcription`
- `transcription_clean`
- `chunk_owner`
- `sample_prob`
- `cb_weight`
- `encoded_audio_shape`
- `encoded_audio`

in [src/parkiet/audioprep/arrow_writer.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/audioprep/arrow_writer.py:228)

### 4.2 What the actual training dataset uses

`AudioTextDataset.__getitem__()` returns only:

- `text`
- `audio`
- `cb_weight`

in [src/parkiet/jax/dataset.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/dataset.py:78)

It does **not** return `chunk_owner`, `speaker_id`, or any speaker label.

### 4.3 What the training loss consumes

The training loss consumes:

- `text_tokens`
- `audio_input`
- `audio_target`

in [src/parkiet/jax/train.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/train.py:201) and [src/parkiet/jax/train_distributed.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/train_distributed.py:230)

No speaker labels are used in loss computation.

### 4.4 Speaker information exists upstream in prep, but is discarded before model training

The audio prep/database pipeline clearly tracks speaker embeddings and speaker IDs:

- `SpeakerExtractor.extract_speaker_events()` returns `speaker_embeddings` in [src/parkiet/audioprep/speaker_extractor.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/audioprep/speaker_extractor.py:22)
- `AudioStore.store_speaker_embeddings()` persists speaker IDs in [src/parkiet/database/audio_store.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/database/audio_store.py:103)
- `chunk_owner` is written into parquet in [src/parkiet/audioprep/arrow_writer.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/audioprep/arrow_writer.py:268)

But the model training path does not use it.

### Conclusion

**Confirmed:** speaker identity exists in the data engineering pipeline, but is not consumed by the model training path.

That is the core architectural reason there is no stable `voice_id` at inference time.

---

## 5. Why Text-Only Phrase Generation Changes Voice

### Confirmed facts

1. The model is multi-speaker in practice:
   - prompts and comments explicitly discuss multiple voices and cloning in [src/parkiet/dia/inference.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/inference.py:13) and [src/parkiet/jax/inference.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/inference.py:26)
2. The model is not given an explicit speaker label during generation.
3. The decoder samples stochastically from a conditional token distribution.

### Architectural implication

If multiple speakers in training can realize similar text and no speaker label is part of the conditioning state, the decoder must implicitly choose one valid acoustic realization.

That means the model can vary:

- timbre
- accent
- prosody
- speaker identity

across independent calls, especially at the start of generation.

### Confirmed vs hypothesis

Confirmed:

- no explicit speaker conditioning
- stochastic generation
- multi-speaker training examples

Hypothesis:

- the initial speaker/accent choice is effectively resolved as a latent mode of the decoder distribution at generation start

This hypothesis is strongly supported by the architecture and observed behavior, but it is still an inference from code and behavior rather than an explicit comment in the repository.

---

## 6. Why `audio_prompt` Did Not Solve Stable Voice Identity

### 6.1 What `audio_prompt` is designed to do

`audio_prompt` is inserted as a decoder token prefix in `_prepare_audio_prompt()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:304).

This is much closer to:

- continuation conditioning
- priming the decoder state

than to:

- extracting a stable speaker embedding
- binding speaker identity as an explicit control variable

### 6.2 Evidence that transcript alignment is expected

Both old anchored server logic and JAX inference comments expect transcript prepending:

- `parkiet_realtime_server.py` explicitly prepends `anchor_text` so text encoder cross-attention aligns with the DAC prefix in [parkiet_realtime_server.py](/home/rezy/PhpstormProjects/parkiet/parkiet_realtime_server.py:269)
- JAX inference comments say to prefix the prompt with the spoken audio transcript in [src/parkiet/jax/inference.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/inference.py:38)

### 6.3 Evidence that old streamer-style anchor logic was continuation-oriented

`parkiet_streaming.py`:

- computes `decode_start = max(0, prefill_step - stream_prompt_context_frames)` in [parkiet_streaming.py](/home/rezy/PhpstormProjects/parkiet/parkiet_streaming.py:402)
- emits only after the prompt-prefill and delay window in [parkiet_streaming.py](/home/rezy/PhpstormProjects/parkiet/parkiet_streaming.py:547)
- final continuation length is derived from `finished_step - prefill_step` in [parkiet_streaming.py](/home/rezy/PhpstormProjects/parkiet/parkiet_streaming.py:577)

This is continuation decoding, not a reference-embedding cloning system.

### 6.4 Why it still fails for stable speaker identity across separate phrases

Confirmed constraints:

1. `audio_prompt` consumes decoder prefix budget because prefill steps are part of the same generation timeline in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:427) and [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:840)
2. `audio_prompt` is token-prefix conditioning, not a persistent speaker latent
3. separate phrase calls re-run the entire generation from scratch

Most likely explanation:

- `audio_prompt` provides local acoustic continuation context
- it does **not** force a strong global identity manifold across independent prompts

This makes it plausible that:

- beep/corruption could be fixed by better anchored decode
- but stable speaker identity still fails across separate phrase generations

### Conclusion

`audio_prompt` is likely **meant for continuation / local prompting**, not as a full product-grade `voice_id` abstraction for stable cross-utterance identity.

---

## 7. Sampling, CFG, and Voice Variability

### Confirmed facts

PyTorch sampling path:

- `_sample_next_token()` uses `torch.multinomial()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:26)
- `Dia.generate()` accepts `temperature`, `top_p`, `cfg_scale`, `cfg_filter_top_k` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:661)

JAX path:

- uses RNG splitting and `random.categorical()` in [src/parkiet/jax/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/model.py:760)

### What this means

Lower temperature and lower `top_p` can reduce variance, but they do not introduce missing speaker information.

### Why `disable_cfg` can degrade quality

Confirmed from code:

- normal path computes unconditional and conditional branches and then uses CFG logits to derive a filtered conditional distribution in `_decoder_step()` in [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:432)

Architectural implication:

- CFG helps anchor the decoder to textual conditioning
- removing it weakens text anchoring

Hypothesis:

- weaker text anchoring can degrade prosody, pronunciation, or stability

This is consistent with observations, but the exact causal contribution to speaker drift is still an inference.

### Can deterministic sampling solve this?

No, not structurally.

Even if a deterministic `generator` were added, it would only make sampling reproducible for the same input and internal state. It would not create a missing speaker identity control variable.

---

## 8. PyTorch vs JAX Mismatch Audit

### Confirmed mismatch

PyTorch `_encode_text()`:

- supports `[S1]`, `[S2]` only

JAX `_encode_text()` and dataset:

- support `[S1]..[S4]`

Evidence:

- [src/parkiet/dia/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/model.py:261)
- [src/parkiet/jax/model.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/model.py:369)
- [src/parkiet/jax/dataset.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/dataset.py:118)

### Does it explain missing stable `voice_id`?

No.

It explains:

- a possible mismatch in multi-speaker prompt formatting
- possible quality regressions for prompts using `[S3]` and `[S4]`

It does **not** explain why single-speaker Dutch TTS has no stable identity across independent calls.

That root cause is still the lack of explicit speaker conditioning in the trained model.

---

## 9. Upstream / Original Dia Assumptions Found in Repo

### Evidence of intended voice cloning usage

The repo contains comments and examples that assume:

1. `audio_prompt` can be used for voice cloning
2. the spoken prompt transcript should be prepended to text

Evidence:

- PyTorch demo comments in [src/parkiet/dia/inference.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/dia/inference.py:19)
- JAX demo comments in [src/parkiet/jax/inference.py](/home/rezy/PhpstormProjects/parkiet/src/parkiet/jax/inference.py:38)
- anchored server logic in [parkiet_realtime_server.py](/home/rezy/PhpstormProjects/parkiet/parkiet_realtime_server.py:269)

### What is missing

There is no evidence in the codebase of:

- a reference encoder that extracts a speaker embedding from prompt audio
- a persistent voice latent carried across calls
- explicit upstream speaker-ID conditioning logic that Parkiet accidentally dropped from inference only

### Conclusion

The repo does not support the hypothesis that Parkiet merely missed a small inference-side speaker-control hook.

The deeper limitation appears architectural and training-related.

---

## 10. Why Current `[S1]` and Current `audio_prompt` Approaches Fail

### Why `[S1]` fails

Confirmed:

- `[S1]` is only a special byte token in text encoding
- there is no embedding table keyed by `S1` as a stable voice identity

Therefore:

- `[S1]` can indicate a turn or speaker role within text
- it cannot reliably select one specific speaker/accent across independent generations

### Why current `audio_prompt` likely fails

Confirmed:

- it is decoder prefill, not speaker embedding
- it needs transcript alignment
- it shares generation budget with new output
- it was historically used in a continuation-style anchored stream path

Most likely failure mode:

- `audio_prompt` conditions the decoder locally but not strongly enough to define an explicit reusable speaker identity across fresh phrase calls

Unknown:

- whether stronger training data, longer anchors, or different conditioning schedules could partially improve it without architectural change

But current code gives no evidence of a robust speaker identity mechanism.

---

## 11. What a Real `voice_id` Would Require

## Option A: Use `audio_prompt` correctly as voice reference

### What it would require

- short clean reference audio
- exact prompt transcript prepended
- anchored decode path
- possibly phrase grouping instead of tiny independent calls

### Benefits

- no architecture change

### Limits

- still no explicit speaker variable
- likely brittle across different phrases and sessions
- likely continuation-biased rather than identity-biased

### Verdict

Useful as a diagnostic or best-effort anchor, but **not** a reliable product-grade `voice_id`.

**Go/No-Go:** No-Go as the final solution.

---

## Option B: Add explicit speaker embedding / voice embedding

### Recommended architecture

Add a learned speaker-conditioning path and feed it into the model explicitly.

The cleanest architecture options are:

1. **Encoder-side conditioning**
   - add a `speaker_embedding` or `voice_embedding`
   - inject it into encoder hidden states, for example by:
     - additive bias to all encoder token embeddings
     - prepended special conditioning token
     - FiLM / adaptive layer norm style conditioning

2. **Decoder-side conditioning**
   - inject the same embedding into decoder hidden states as:
     - additive conditioning on decoder input embeddings
     - per-layer conditioning bias or adaptive norm
     - learned prefix tokens in decoder space

3. **Both encoder and decoder**
   - strongest and most explicit option
   - recommended if realtime `voice_id` is a product requirement

### Best-practice recommendation

Use **both encoder and decoder conditioning**, with one shared `voice_embedding`.

Why:

- encoder side anchors lexical/prosodic interpretation
- decoder side anchors acoustic realization

### Verdict

This is the most direct model-level fix.

**Go/No-Go:** Go

---

## Option C: Fine-tune with speaker IDs

### Required dataset format

Per sample:

- `speaker_id` or `voice_id`
- transcript
- encoded DAC audio tokens
- optional accent/style metadata

### What to change

- training dataset must return `speaker_id`
- model forward/generation must accept `speaker_id`
- loss stays sequence modeling, but conditioning now includes identity

### Why this is strong

The repo already has speaker information upstream in prep/database:

- `speaker_embeddings`
- `speaker_id`
- `chunk_owner`

But that information is discarded before training.

### Verdict

This is a practical path if speaker-labeled data exists or can be derived reliably.

**Go/No-Go:** Go

---

## Option D: Train a small speaker adapter / LoRA / conditioning module

### Feasibility

Moderately feasible.

Possible approach:

- freeze most of the base model
- add a small adapter that consumes speaker embedding or learned speaker ID
- inject at encoder and/or decoder blocks

### Benefit

- less compute than full retraining
- can preserve current Dutch quality base model

### Risk

- if the base model never learned to represent speaker identity cleanly, a very small adapter may not be enough

### Verdict

Good as a fine-tuning strategy after a speaker-conditioning interface exists.

**Go/No-Go:** Go, but as a second-stage engineering path, not the first inference-only fix.

---

## Option E: Derive speaker embedding from reference audio

### What this would mean

Add a reference encoder or external speaker encoder that maps reference audio to a fixed-dimensional speaker vector, then inject that vector into the model.

### Does the current architecture have a native place for it?

Not currently.

There is no speaker encoder module in `DiaModel`, no input slot for such an embedding, and no conditioning adapter path in the current Transformer blocks.

### Could it be added?

Yes. The clean integration points are the same as Option B:

- encoder token conditioning
- decoder token conditioning
- adaptive norms / prefix conditioning

### Verdict

Potentially the best product-oriented `voice_id` path if you want reference-audio-based voice selection instead of fixed speaker IDs.

**Go/No-Go:** Go, but requires model architecture change and fine-tuning.

---

## 12. Recommended Architectural Fix

## Recommendation

The correct fix is:

1. **Add explicit speaker conditioning to the model**
2. **Train or fine-tune using speaker identity labels or reference-derived speaker embeddings**

### Minimal viable architecture

1. Add a `speaker_embedding_dim`
2. Add one conditioning input to `Dia.generate()` / forward path:
   - either `speaker_id`
   - or `speaker_embedding`
3. Inject conditioning into:
   - encoder input embeddings
   - decoder input embeddings

### Stronger production architecture

Preferred path:

- learn a `speaker_embedding` table for known speakers **or**
- derive `speaker_embedding` from reference audio using a speaker encoder
- inject into both encoder and decoder via adaptive layer norm or additive conditioning

### Why this is the correct fix

Because the current model is missing the variable you want to control.

Inference tricks can reshape sampling, but they cannot invent a speaker identity signal that the model was never trained to consume explicitly.

---

## 13. Minimal Implementation Plan

This is an architectural plan, not an implementation patch.

### Phase 1: Training data plumbing

1. Extend parquet/training dataset to return:
   - `speaker_id`
   - or `speaker_embedding`
2. Preserve existing:
   - `text`
   - `audio`
   - `cb_weight`

### Phase 2: Model conditioning interface

1. Extend `DiaModel` / `Encoder` / `Decoder` to accept speaker conditioning
2. Add conditioning injection at:
   - encoder token input
   - decoder token input

### Phase 3: Generation interface

1. Add explicit generation args:
   - `speaker_id`
   - or `speaker_embedding`
2. Keep `audio_prompt` optional for local continuation only, not as the main `voice_id`

### Phase 4: Fine-tuning

1. Start with a small speaker-conditioned fine-tune
2. Evaluate:
   - same-speaker consistency across independent phrases
   - accent stability
   - cross-text robustness

### Phase 5: Optional reference-audio voice path

1. Add speaker encoder or offline speaker embedding extraction
2. map reference audio -> `speaker_embedding`
3. use that embedding as stable conditioning for all phrases in a session

---

## 14. Risks

1. **Dataset label quality**
   - if speaker diarization / `chunk_owner` labels are noisy, the model may learn unstable speaker conditioning

2. **Speaker-vs-style entanglement**
   - accent, timbre, recording conditions, and prosody may be entangled in current data

3. **Need for fine-tuning**
   - adding a conditioning input without retraining will not solve the problem

4. **Backward compatibility**
   - new conditioning path should preserve current offline default behavior when no speaker condition is provided

---

## 15. Confirmed Facts vs Hypotheses

## Confirmed from code

- No explicit speaker embedding exists in current model architecture
- `audio_prompt` is DAC-code decoder prefill
- `[S1]..[S4]` are byte-level special tokens
- PyTorch only maps `[S1]`, `[S2]`; JAX maps `[S1]..[S4]`
- training dataset does not feed speaker IDs into the model
- stochastic sampling is used in generation
- anchored generation historically required transcript prepending and context decode

## Hypotheses

- speaker identity is effectively selected as a latent mode of the decoder at generation start
- `audio_prompt` is too weak / too continuation-oriented to define a stable cross-utterance speaker identity
- the correct long-term solution requires explicit model conditioning plus fine-tuning

These hypotheses are strongly supported by the code, but they are still hypotheses until validated by controlled model experiments.

---

## 16. Final Decision Matrix

| Path | Decision | Reason |
|---|---|---|
| Quick inference-only fix | No-Go | Missing explicit speaker control variable |
| `audio_prompt`-based `voice_id` | No-Go | Decoder-prefix continuation is not a robust identity mechanism |
| Model-level speaker conditioning | Go | Directly solves the missing control surface |
| Fine-tuning / retraining | Go | Required to teach the model to use explicit speaker identity |

---

## Final Answer

### Can the current model support reliable `voice_id` without retraining?

**No, not reliably.**

It can produce good Dutch TTS quality, and `audio_prompt` can act as a local anchor, but the architecture has no explicit stable speaker identity representation. That makes robust cross-phrase `voice_id` control structurally weak.

### Why does the current `[S1]` approach fail?

Because `[S1]` is only a special text token. It is not connected to a persistent speaker embedding or voice identity table.

### Why does the current `audio_prompt` approach likely fail?

Because it is a decoder continuation prefix, not a stable reference embedding system. It can help local continuation, but it does not provide a reusable explicit identity state for independent phrase generations.

### What is the correct architectural fix?

Add explicit speaker conditioning to the model and fine-tune with speaker-labeled or reference-embedded data. The cleanest design is a `speaker_embedding` injected into both encoder and decoder.

