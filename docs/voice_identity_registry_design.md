# Voice Identity Registry Design

## Scope

This document defines the stable identity layer that should sit between:

- raw dataset speaker labels
- internal model speaker IDs
- product-facing `voice_id` selection

The goal is to make future speaker-conditioned training and inference reproducible, versioned, and safe.

This design does **not** change model behavior. It defines the registry contract that later phases will use.

---

## 1. Identity Types

## 1.1 `voice_id`

`voice_id` is the public, product-facing identifier.

Examples:

- `nl_default_01`
- `nl_female_salon_01`

Properties:

- stable over time
- human-readable
- safe to expose to product code and users
- should not depend on database primary keys or transient dataset labels

## 1.2 `model_speaker_id`

`model_speaker_id` is the contiguous integer ID used by the model.

Examples:

- `0`
- `1`
- `2`

Properties:

- compact contiguous integer space
- used for `nn.Embedding(num_speakers, ...)`
- tied to a specific checkpoint family / speaker vocabulary version
- not intended to be directly user-facing

## 1.3 `source_speaker_id` / `chunk_owner`

This is the raw speaker identity coming from data prep or source datasets.

Examples:

- database `chunk_owner`
- `dataset_a:17`
- `parkiet_v1:42`

Properties:

- may be noisy
- may change between exports
- may be sparse
- may be dataset-specific
- must not be treated as a stable public identity

## 1.4 `speaker_embedding`

`speaker_embedding` is a vector representation of a speaker identity.

Potential sources:

- learned embedding table index by `model_speaker_id`
- future reference-audio encoder output

Properties:

- model-internal or future conditioning artifact
- not a stable public identifier

## 1.5 Voice metadata

Voice metadata is descriptive and governance-oriented information about a `voice_id`.

Examples:

- display name
- locale
- quality status
- data origin
- duration

This metadata is required to:

- curate voices
- audit training inputs
- keep product-facing voice selection sane

---

## 2. Why Raw `chunk_owner` Must Not Be Used Directly As Public `voice_id`

Raw `chunk_owner` or source speaker IDs are unsuitable as public identity because they are:

1. **dataset-local**
   - `17` only means something inside one source dataset or DB

2. **unstable**
   - re-imports, deduplication, diarization updates, or new DBs can change IDs

3. **not curated**
   - a `chunk_owner` may correspond to poor-quality, mixed, or underrepresented data

4. **not product-safe**
   - internal numeric speaker IDs are not meaningful to users or application code

5. **not versioned for checkpoints**
   - the model needs stable contiguous IDs tied to a known training vocabulary

Conclusion:

- `chunk_owner` is a source label
- `voice_id` is a curated public contract
- `model_speaker_id` is the model-facing stable integer contract

---

## 3. Recommended Registry Files

Recommended layout:

```text
voice_registry/
  voices.json
  speaker_vocab.json
  speakers/
    nl_female_salon_01/
      metadata.json
      reference.wav            # optional
      samples.json             # optional
```

## 3.1 `voice_registry/voices.json`

Primary public registry.

Contains:

- registry version
- default voice
- list of voice metadata entries

## 3.2 `voice_registry/speaker_vocab.json`

Model-facing mapping artifact.

Contains:

- speaker vocab version
- default speaker ID
- `voice_id -> model_speaker_id`
- `source_speaker_id -> model_speaker_id`

## 3.3 `voice_registry/speakers/<voice_id>/metadata.json`

Per-voice detail file.

Useful for:

- extended audit metadata
- per-voice history
- richer notes and curation state

## 3.4 Optional files

- `reference.wav`
  - optional reference sample for listening or future reference-based cloning
- `samples.json`
  - optional per-voice sample inventory

---

## 4. Required Metadata Fields

Each voice entry should contain:

- `voice_id`
- `model_speaker_id`
- `display_name`
- `language`
- `locale`
- `source_dataset`
- `source_speaker_ids`
- `duration_sec`
- `sample_count`
- `quality_status`
- `status`
- `created_at`
- `updated_at`
- `notes`

Recommended meanings:

- `voice_id`: stable public ID
- `model_speaker_id`: stable contiguous model ID
- `display_name`: human-friendly label
- `language`: broad language label, e.g. `Dutch`
- `locale`: locale tag, e.g. `nl-NL`
- `source_dataset`: canonical dataset family used to build this voice
- `source_speaker_ids`: list of raw source speaker IDs contributing to this voice
- `duration_sec`: total usable duration included
- `sample_count`: number of usable training samples
- `quality_status`: `approved`, `candidate`, `rejected`
- `status`: `active`, `inactive`, `deprecated`, `private`
- `created_at`: creation timestamp
- `updated_at`: last update timestamp
- `notes`: free-form curator notes or `null`

---

## 5. Versioning Rules

## 5.1 Registry version

`voices.json` should contain:

- `registry_version`

This versions the public voice registry contract.

## 5.2 Speaker vocab version

`speaker_vocab.json` should contain:

- `speaker_vocab_version`

This versions the model-facing integer mapping.

## 5.3 Checkpoint linkage

Every speaker-conditioned checkpoint should record:

- `speaker_vocab_version`

Reason:

- a checkpoint is only meaningful relative to the exact model speaker ID assignment it was trained with

## 5.4 Append-only rule

Once a `voice_id -> model_speaker_id` mapping is used for training:

- existing IDs must be append-only
- existing `voice_id` must never change its `model_speaker_id`
- existing `model_speaker_id` must never be reassigned to a different `voice_id`

New voices may be added only by appending new `model_speaker_id` values.

---

## 6. Data Quality Rules

Before a voice is included in the active registry, it should satisfy minimum quality rules.

Recommended rules:

1. **minimum duration per speaker**
   - require a minimum amount of usable audio

2. **minimum sample count**
   - require enough distinct utterances

3. **exclude noisy speakers**
   - heavy noise, music bleed, clipping, severe codec artifacts

4. **exclude mixed speakers**
   - diarization-confused or multi-speaker chunks should not define one voice

5. **require transcript quality**
   - poor transcripts degrade alignment and conditioning

6. **require language / locale consistency**
   - do not mix unrelated locales or strong accent clusters into one voice entry unless explicitly intended

These rules should gate:

- `quality_status`
- `status`

---

## 7. Training Mapping

The training mapping must be:

```text
public voice_id -> model_speaker_id
source chunk_owner/source_speaker_id -> model_speaker_id
model uses only model_speaker_id
```

Meaning:

- product code should never feed raw `chunk_owner` directly into the model
- training dataset may start from raw source speaker IDs
- dataset preparation must map them into stable `model_speaker_id`

The model should only see:

- `speaker_id = model_speaker_id`

---

## 8. Product Mapping

Product-facing flow should be:

```text
voice_id = "nl_female_salon_01"
-> registry lookup
-> model_speaker_id = 1
-> Dia.generate(text, speaker_id=1)
```

This keeps:

- public identity stable
- model identity compact and checkpoint-compatible

---

## 9. Future Support

The registry design should support the following future cases.

## 9.1 Custom user voices

Possible future path:

- create new `voice_id`
- attach user-owned metadata
- optionally map to future `speaker_embedding` or custom fine-tuned model

## 9.2 Private voices

Need `status` or policy fields that allow:

- not publicly selectable
- restricted internal usage

## 9.3 Inactive / deprecated voices

Registry must support:

- `inactive`
- `deprecated`

These voices may remain for compatibility but should not be default selections.

## 9.4 Multiple models / checkpoints

Future expansion may require:

- one registry to support multiple conditioned checkpoints
- different `speaker_vocab_version` per checkpoint family

That means registry metadata may later need:

- checkpoint compatibility lists
- model family tags

## 9.5 `speaker_embedding`-based voice cloning later

This registry should not block future support for:

- reference-based voice cloning
- per-voice stored speaker embeddings

In that future:

- `voice_id` remains public
- `model_speaker_id` may remain for trained voices
- `speaker_embedding` may become an alternate conditioning path

---

## 10. Recommended JSON Examples

## 10.1 `voice_registry/voices.json`

```json
{
  "registry_version": "v1",
  "default_voice_id": "nl_default_01",
  "voices": [
    {
      "voice_id": "nl_female_salon_01",
      "model_speaker_id": 1,
      "display_name": "Dutch Female Salon",
      "language": "Dutch",
      "locale": "nl-NL",
      "source_dataset": "parkiet_v1",
      "source_speaker_ids": ["dataset_a:17"],
      "duration_sec": 7200.0,
      "sample_count": 1500,
      "quality_status": "approved",
      "status": "active",
      "created_at": "...",
      "updated_at": "...",
      "notes": null
    }
  ]
}
```

## 10.2 `voice_registry/speaker_vocab.json`

```json
{
  "speaker_vocab_version": "v1",
  "default_speaker_id": 0,
  "voice_id_to_model_speaker_id": {
    "nl_default_01": 0,
    "nl_female_salon_01": 1
  },
  "source_speaker_id_to_model_speaker_id": {
    "dataset_a:17": 1
  }
}
```

---

## 11. Recommended Implementation Contract

The first registry utility layer should provide:

- load / save
- validation
- lookup by `voice_id`
- `voice_id -> model_speaker_id`
- `source_speaker_id -> model_speaker_id`
- append-only ID validation

This layer must not modify model behavior. It is a governance and mapping foundation for later phases.

---

## 12. Final Recommendation

The system should treat identity as three distinct layers:

1. **source identity**
   - raw dataset speaker labels
2. **model identity**
   - stable contiguous `model_speaker_id`
3. **public identity**
   - curated `voice_id`

This separation is required to keep future speaker-conditioned checkpoints reproducible and product-safe.

