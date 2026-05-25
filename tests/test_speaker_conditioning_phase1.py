from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from parkiet.dia.config import DecoderConfig, DiaConfig, EncoderConfig
from parkiet.dia.layers import DiaModel
from parkiet.dia.model import Dia
from parkiet.dia.state import DecoderInferenceState, EncoderInferenceState
from parkiet.jax.dataset import AudioTextDataset
from parkiet.speaker_vocab import (
    build_speaker_vocab_from_chunk_owners,
    load_speaker_vocab,
    map_chunk_owner_to_speaker_id,
    save_speaker_vocab,
)


def _minimal_config(enabled: bool = False, num_speakers: int = 0) -> DiaConfig:
    return DiaConfig(
        encoder_config=EncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        ),
        decoder_config=DecoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            cross_hidden_size=16,
            cross_num_attention_heads=2,
            cross_num_key_value_heads=2,
            cross_head_dim=8,
            max_position_embeddings=32,
            num_channels=9,
        ),
        delay_pattern=(0, 8, 9, 10, 11, 12, 13, 14, 15),
        speaker_conditioning_enabled=enabled,
        num_speakers=num_speakers,
    )


def test_old_config_without_speaker_fields_loads(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["DiaForConditionalGeneration"],
                "bos_token_id": 1026,
                "decoder_config": {"num_channels": 9},
                "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
                "encoder_config": {},
                "eos_token_id": 1024,
                "is_encoder_decoder": True,
                "model_type": "dia",
                "pad_token_id": 1025,
            }
        ),
        encoding="utf-8",
    )
    config = DiaConfig.load(str(config_path))
    assert config is not None
    assert config.speaker_conditioning_enabled is False
    assert config.speaker_embedding_dim == 256
    assert config.num_speakers == 0
    assert config.default_speaker_id == 0
    assert config.speaker_conditioning_mode == "encoder_decoder_additive"


def test_config_with_speaker_fields_loads(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["DiaForConditionalGeneration"],
                "bos_token_id": 1026,
                "decoder_config": {"num_channels": 9},
                "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
                "encoder_config": {},
                "eos_token_id": 1024,
                "is_encoder_decoder": True,
                "model_type": "dia",
                "pad_token_id": 1025,
                "speaker_conditioning_enabled": True,
                "speaker_embedding_dim": 192,
                "num_speakers": 7,
                "default_speaker_id": 2,
                "speaker_conditioning_mode": "encoder_decoder_additive",
            }
        ),
        encoding="utf-8",
    )
    config = DiaConfig.load(str(config_path))
    assert config is not None
    assert config.speaker_conditioning_enabled is True
    assert config.speaker_embedding_dim == 192
    assert config.num_speakers == 7
    assert config.default_speaker_id == 2


def test_speaker_vocab_mapping_roundtrip(tmp_path: Path):
    vocab = build_speaker_vocab_from_chunk_owners([17, 42, 17, None, -1, 42, 99], min_count=2)
    assert vocab["default_speaker_id"] == 0
    assert vocab["db_speaker_id_to_model_speaker_id"] == {"17": 1, "42": 2}
    vocab_path = tmp_path / "speaker_vocab.json"
    save_speaker_vocab(vocab_path, vocab)
    loaded = load_speaker_vocab(vocab_path)
    assert loaded == vocab
    assert map_chunk_owner_to_speaker_id(17, loaded) == 1
    assert map_chunk_owner_to_speaker_id(42, loaded) == 2
    assert map_chunk_owner_to_speaker_id(99, loaded) == 0
    assert map_chunk_owner_to_speaker_id(None, loaded) == 0


def test_dataset_without_speaker_vocab_behaves_as_before(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "transcription": "[S1] hallo",
                "transcription_clean": "[S1] hallo",
                "encoded_audio_shape": np.array([2, 9], dtype=np.int64),
                "encoded_audio": list(range(18)),
                "cb_weight": 1.25,
                "chunk_owner": 17,
            }
        ]
    )
    monkeypatch.setattr(
        "parkiet.jax.dataset.discover_parquet_shards",
        lambda parquet_path: ["fake.parquet"],
    )
    monkeypatch.setattr(pd, "read_parquet", lambda parquet_file: frame.copy())
    dataset = AudioTextDataset("unused", _minimal_config())
    sample = dataset[0]
    assert set(sample.keys()) == {"text", "audio", "cb_weight"}
    assert sample["cb_weight"] == np.float32(1.25)


def test_dataset_with_speaker_vocab_maps_chunk_owner(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "transcription": "[S1] hallo",
                "transcription_clean": "[S1] hallo",
                "encoded_audio_shape": np.array([2, 9], dtype=np.int64),
                "encoded_audio": list(range(18)),
                "cb_weight": 1.25,
                "chunk_owner": 42,
            }
        ]
    )
    monkeypatch.setattr(
        "parkiet.jax.dataset.discover_parquet_shards",
        lambda parquet_path: ["fake.parquet"],
    )
    monkeypatch.setattr(pd, "read_parquet", lambda parquet_file: frame.copy())
    vocab = {
        "default_speaker_id": 0,
        "db_speaker_id_to_model_speaker_id": {"42": 3},
    }
    dataset = AudioTextDataset("unused", _minimal_config(), speaker_vocab=vocab)
    sample = dataset[0]
    assert sample["speaker_id"] == np.int32(3)
    assert set(sample.keys()) == {"text", "audio", "cb_weight", "speaker_id"}


class _DummyState:
    def prepare_step(self, step_from: int, step_to: int | None = None) -> None:
        return None


class _DummyOutput:
    def __init__(self, num_channels: int):
        self.prefill_steps = [1]
        self.generated_tokens = torch.full((1, 32, num_channels), fill_value=1025, dtype=torch.int)

    def get_tokens_at(self, step_from: int, step_to: int | None = None) -> torch.Tensor:
        if step_to is None:
            step_to = step_from + 1
        return self.generated_tokens[:, step_from:step_to, :]

    def update_one(self, dec_out: torch.Tensor, step: int, apply_mask: bool = False):
        self.generated_tokens[:, step : step + 1, :] = dec_out.unsqueeze(1).to(self.generated_tokens.dtype)


def _stub_dia(enabled: bool = False, num_speakers: int = 0) -> Dia:
    dia = Dia.__new__(Dia)
    dia.config = _minimal_config(enabled=enabled, num_speakers=num_speakers)
    dia.device = torch.device("cpu")
    dia.model = type("DummyModel", (), {"eval": lambda self: None})()
    dia.last_generate_metadata = {}
    dia._encode_text = lambda text: torch.tensor([1], dtype=torch.long)
    dia._pad_text_input = lambda text_tokens: torch.zeros((1, 1, 4), dtype=torch.long)
    dia._prepare_generation = lambda text, audio_prompts, max_tokens=None, disable_cfg=False, speaker_ids=None: (_DummyState(), _DummyOutput(dia.config.decoder_config.num_channels))
    dia._decoder_step = lambda tokens_Bx1xC, dec_state, cfg_scale, temperature, top_p, top_k, current_idx, disable_cfg=False: torch.full((1, dia.config.decoder_config.num_channels), dia.config.eos_token_id, dtype=torch.long)
    dia._generate_output = lambda generated_codes, lengths_Bx: [np.zeros(16, dtype=np.float32)]
    return dia


def test_generate_accepts_speaker_id_and_ignores_when_conditioning_disabled():
    dia = _stub_dia(enabled=False)
    output = dia.generate("[S1] hallo", speaker_id=7, max_tokens=20)
    assert isinstance(output, np.ndarray)
    assert output.shape == (16,)


def test_invalid_speaker_id_raises_when_conditioning_enabled():
    dia = _stub_dia(enabled=True, num_speakers=2)
    try:
        dia.generate("[S1] hallo", speaker_id=5, max_tokens=20)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid speaker_id")


def test_old_unconditioned_config_constructs_model():
    model = DiaModel(_minimal_config(enabled=False), torch.float32)
    assert model.speaker_embedding is None
    assert model.speaker_to_encoder is None
    assert model.speaker_to_decoder is None


def test_speaker_conditioning_enabled_creates_modules():
    model = DiaModel(_minimal_config(enabled=True, num_speakers=4), torch.float32)
    assert model.speaker_embedding is not None
    assert model.speaker_to_encoder is not None
    assert model.speaker_to_decoder is not None


def test_invalid_num_speakers_raises_clear_error():
    try:
        DiaModel(_minimal_config(enabled=True, num_speakers=0), torch.float32)
    except ValueError as exc:
        assert "num_speakers" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid num_speakers")


def test_speaker_id_batch_is_duplicated_correctly_under_cfg():
    config = _minimal_config(enabled=True, num_speakers=8)
    dia = Dia(config=config, compute_dtype="float32", device=torch.device("cpu"), load_dac=False)
    text = dia._pad_text_input([dia._encode_text("[S1] hallo"), dia._encode_text("[S1] daar")])
    captured: dict[str, torch.Tensor] = {}

    def capture_get_speaker_condition(speaker_ids: torch.Tensor):
        captured["speaker_ids"] = speaker_ids.detach().cpu()
        batch = speaker_ids.shape[0]
        return (
            torch.zeros((batch, config.encoder_config.hidden_size), dtype=torch.float32),
            torch.zeros((batch, config.decoder_config.hidden_size), dtype=torch.float32),
        )

    dia.model.get_speaker_condition = capture_get_speaker_condition  # type: ignore[method-assign]
    dia._prepare_generation(
        text,
        [None, None],
        max_tokens=8,
        disable_cfg=False,
        speaker_ids=[1, 3],
    )
    assert captured["speaker_ids"].tolist() == [1, 1, 3, 3]


def test_encoder_forward_accepts_speaker_condition_without_shape_change():
    config = _minimal_config(enabled=True, num_speakers=4)
    model = DiaModel(config, torch.float32)
    x_ids = torch.randint(
        0,
        config.encoder_config.vocab_size,
        (2, config.encoder_config.max_position_embeddings),
        dtype=torch.long,
    )
    cond_src = x_ids.unsqueeze(1).clone()
    state = EncoderInferenceState.new(config, cond_src, batch_multiplier=1)
    speaker_ids = torch.tensor([1, 2], dtype=torch.long)
    encoder_bias, _ = model.get_speaker_condition(speaker_ids)
    output = model.encoder(x_ids, state, speaker_condition=encoder_bias)
    assert output.shape == (2, config.encoder_config.max_position_embeddings, config.encoder_config.hidden_size)


def test_decoder_decode_step_accepts_speaker_condition():
    config = _minimal_config(enabled=True, num_speakers=4)
    model = DiaModel(config, torch.float32)
    batch_size = 2
    cond_src = torch.zeros((batch_size, 1, config.encoder_config.max_position_embeddings), dtype=torch.long)
    enc_state = EncoderInferenceState.new(config, cond_src, batch_multiplier=1)
    enc_input = torch.zeros((batch_size, config.encoder_config.max_position_embeddings), dtype=torch.long)
    enc_out = model.encoder(enc_input, enc_state)
    dec_cross_attn_cache = model.decoder.precompute_cross_attn_cache(enc_out)
    dec_state = DecoderInferenceState.new(
        config,
        enc_state,
        enc_out,
        dec_cross_attn_cache,
        torch.float32,
        max_generation_length=8,
        batch_multiplier=1,
    )
    dec_state.prepare_step(0)
    tgt_ids = torch.zeros((batch_size, 1, config.decoder_config.num_channels), dtype=torch.long)
    speaker_ids = torch.tensor([1, 2], dtype=torch.long)
    _, decoder_bias = model.get_speaker_condition(speaker_ids)
    logits = model.decoder.decode_step(
        tgt_ids,
        dec_state,
        current_idx=torch.tensor([0], dtype=torch.long),
        speaker_condition=decoder_bias,
    )
    assert logits.shape == (
        batch_size,
        1,
        config.decoder_config.num_channels,
        config.decoder_config.vocab_size,
    )


def test_generate_accepts_valid_speaker_id_when_conditioning_enabled():
    dia = _stub_dia(enabled=True, num_speakers=8)
    output = dia.generate("[S1] hallo", speaker_id=3, max_tokens=20)
    assert isinstance(output, np.ndarray)
    assert output.shape == (16,)
