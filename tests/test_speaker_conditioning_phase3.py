from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from parkiet.dia.config import DecoderConfig, DiaConfig, EncoderConfig
from parkiet.jax.dataset import AudioTextDataset
from parkiet.jax.train import (
    TrainingConfig as SingleTrainingConfig,
    compute_loss_impl as compute_loss_single_impl,
    load_and_prepare_batch as load_and_prepare_batch_single,
)


def _minimal_config(enabled: bool = False) -> DiaConfig:
    return DiaConfig(
        encoder_config=EncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8,
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
            max_position_embeddings=8,
            num_channels=9,
            vocab_size=32,
        ),
        delay_pattern=(0, 8, 9, 10, 11, 12, 13, 14, 15),
        bos_token_id=30,
        eos_token_id=31,
        pad_token_id=29,
        speaker_conditioning_enabled=enabled,
        num_speakers=4 if enabled else 0,
    )


class _FakeSpeakerAwareModel:
    def __init__(self, config: DiaConfig):
        self.config = config
        self.compute_dtype = jnp.float32
        self.encoder_calls: list[jnp.ndarray | None] = []
        self.decoder_calls: list[jnp.ndarray | None] = []

    def get_speaker_condition(self, speaker_id: jnp.ndarray):
        batch = speaker_id.shape[0]
        return (
            jnp.zeros((batch, self.config.encoder_config.hidden_size), dtype=jnp.float32),
            jnp.zeros((batch, self.config.decoder_config.hidden_size), dtype=jnp.float32),
        )

    def encoder(self, text_tokens, enc_state, speaker_condition=None):
        self.encoder_calls.append(speaker_condition)
        batch = text_tokens.shape[0]
        seq_len = text_tokens.shape[1]
        return jnp.zeros((batch, seq_len, self.config.encoder_config.hidden_size), dtype=jnp.float32)

    def decoder(self, audio_input, dec_state, speaker_condition=None):
        self.decoder_calls.append(speaker_condition)
        batch, seq_len, channels = audio_input.shape
        return jnp.zeros(
            (batch, seq_len, channels, self.config.decoder_config.vocab_size),
            dtype=jnp.float32,
        )


def test_training_config_speaker_vocab_path_is_optional():
    cfg = SingleTrainingConfig()
    assert cfg.speaker_vocab_path is None
    cfg2 = SingleTrainingConfig(speaker_vocab_path="voice_registry/speaker_vocab.json")
    assert cfg2.speaker_vocab_path == "voice_registry/speaker_vocab.json"


def test_load_and_prepare_batch_without_speaker_id_keeps_old_fields():
    config = _minimal_config(enabled=False)
    raw_batch = iter(
        [
            {
                "text": np.zeros((2, config.encoder_config.max_position_embeddings), dtype=np.int32),
                "audio": np.zeros((2, config.decoder_config.max_position_embeddings, config.decoder_config.num_channels), dtype=np.int32),
                "cb_weight": np.ones((2,), dtype=np.float32),
            }
        ]
    )
    batch = load_and_prepare_batch_single(raw_batch, config)
    assert set(batch.keys()) == {"text", "audio_input", "audio_target"}


def test_load_and_prepare_batch_carries_speaker_id():
    config = _minimal_config(enabled=True)
    raw_batch = iter(
        [
            {
                "text": np.zeros((2, config.encoder_config.max_position_embeddings), dtype=np.int32),
                "audio": np.zeros((2, config.decoder_config.max_position_embeddings, config.decoder_config.num_channels), dtype=np.int32),
                "cb_weight": np.ones((2,), dtype=np.float32),
                "speaker_id": np.array([1, 2], dtype=np.int32),
            }
        ]
    )
    batch = load_and_prepare_batch_single(raw_batch, config)
    assert "speaker_id" in batch
    assert batch["speaker_id"].shape == (2,)


def test_compute_loss_accepts_speaker_id_when_enabled():
    config = _minimal_config(enabled=True)
    model = _FakeSpeakerAwareModel(config)
    batch_size = 2
    text = jnp.zeros((batch_size, config.encoder_config.max_position_embeddings), dtype=jnp.int32)
    audio_input = jnp.zeros((batch_size, config.decoder_config.max_position_embeddings, config.decoder_config.num_channels), dtype=jnp.int32)
    audio_target = jnp.zeros_like(audio_input)
    speaker_id = jnp.array([1, 2], dtype=jnp.int32)
    loss, metrics = compute_loss_single_impl(
        model,
        text,
        audio_input,
        audio_target,
        config,
        speaker_id=speaker_id,
    )
    assert loss.shape == ()
    assert "loss" in metrics
    assert model.encoder_calls[-1] is not None
    assert model.decoder_calls[-1] is not None


def test_missing_speaker_id_raises_only_when_enabled():
    config = _minimal_config(enabled=True)
    model = _FakeSpeakerAwareModel(config)
    batch_size = 1
    text = jnp.zeros((batch_size, config.encoder_config.max_position_embeddings), dtype=jnp.int32)
    audio_input = jnp.zeros((batch_size, config.decoder_config.max_position_embeddings, config.decoder_config.num_channels), dtype=jnp.int32)
    audio_target = jnp.zeros_like(audio_input)
    try:
        compute_loss_single_impl(
            model, text, audio_input, audio_target, config, speaker_id=None
        )
    except ValueError as exc:
        assert "speaker_id is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when speaker conditioning is enabled")

    config_disabled = _minimal_config(enabled=False)
    model_disabled = _FakeSpeakerAwareModel(config_disabled)
    loss, metrics = compute_loss_single_impl(
        model_disabled,
        text,
        audio_input,
        audio_target,
        config_disabled,
        speaker_id=None,
    )
    assert loss.shape == ()
    assert "loss" in metrics
    assert model_disabled.encoder_calls[-1] is None
    assert model_disabled.decoder_calls[-1] is None
