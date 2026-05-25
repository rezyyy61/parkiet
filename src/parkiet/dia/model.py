import time
from math import ceil
from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .audio import (
    apply_audio_delay,
    build_delay_indices,
    build_revert_indices,
    revert_audio_delay,
)
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512
DEFAULT_PROMPT_TRIM_CONTEXT_FRAMES = 128


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int | None,
    audio_eos_value: int,
) -> torch.Tensor:
    if temperature == 0.0:
        argmax_tokens = torch.argmax(logits_BCxV, dim=-1)
        return argmax_tokens

    logits_BCxV = logits_BCxV / temperature

    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
        mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)
        eos_highest_mask_BC = top_logit_indices_BC == audio_eos_value
        mask_eos_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_highest_BCxV[eos_highest_mask_BC, :audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_highest_BCxV, -torch.inf)

    if top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(
            probs_BCxV, dim=-1, descending=True
        )
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV = torch.roll(
            sorted_indices_to_remove_BCxV, shifts=1, dims=-1
        )
        sorted_indices_to_remove_BCxV[..., 0] = torch.zeros_like(
            sorted_indices_to_remove_BCxV[..., 0]
        )

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)

    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model: DiaModel = DiaModel(config, self.compute_dtype)
        self.dac_model = None
        self._compiled_step = None
        self.load_dac = load_dac
        self.last_generate_metadata: dict = {}

        if not self.load_dac:
            print("Warning: DAC model will not be loaded. This is not recommended.")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device, load_dac)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error loading checkpoint from {checkpoint_path}"
            ) from e

        dia.model.to(dia.device)
        dia.model.eval()
        if load_dac:
            dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B-0626",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "nari-labs/Dia-1.6B-0626").
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)

        # Load model directly using DiaModel's from_pretrained which handles HF download
        try:
            loaded_model = DiaModel.from_pretrained(
                model_name, compute_dtype=compute_dtype.to_dtype()
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading model from Hugging Face Hub ({model_name})"
            ) from e

        config = loaded_model.config  # Get config from the loaded model
        dia = cls(config, compute_dtype, device, load_dac)

        dia.model = loaded_model  # Assign the already loaded model
        dia.model.to(dia.device)
        dia.model.eval()
        if load_dac:
            dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        """Loads the Descript Audio Codec (DAC) model.

        Downloads the DAC model if necessary and loads it onto the specified device.
        Sets the DAC model to evaluation mode.

        Raises:
            RuntimeError: If downloading or loading the DAC model fails.
        """
        import dac

        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
            dac_model.eval()  # Ensure DAC is in eval mode
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encodes the input text string into a tensor of token IDs using byte-level encoding.

        Special tokens [S1] and [S2] are replaced by their byte values. The resulting
        sequence is truncated to the maximum configured text length.

        Args:
            text: The input text string.

        Returns:
            A tensor containing the encoded byte token IDs.
        """
        max_len = self.config.encoder_config.max_position_embeddings

        byte_text = text.encode("utf-8")
        # Replace special tokens with their byte values if needed by the specific tokenizer/config
        # Assuming byte values 1 and 2 are correct placeholders based on original code
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)
        return torch.tensor(
            text_tokens[:max_len],
            dtype=torch.long,
            device=self.device,
        )

    def _pad_text_input(self, text_tokens: list[torch.Tensor]) -> torch.Tensor:
        """Pads the text input to the maximum length."""
        text_pad_value = 0
        max_len = self.config.encoder_config.max_position_embeddings
        batch_size = len(text_tokens)

        src_tokens = torch.full(
            (batch_size, 1, max_len),
            fill_value=text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        for i in range(batch_size):
            current_len = len(text_tokens[i])
            src_tokens[i, 0, :current_len] = text_tokens[i]
        return src_tokens

    def _prepare_audio_prompt(
        self, audio_prompts: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[int]]:
        """Prepares the audio prompt tensor for the decoder.

        Handles padding, adds the beginning-of-sequence (BOS) token, applies the
        delay pattern, and determines the number of prefill steps for each item
        in the batch.

        Args:
            audio_prompts: A list of audio prompt tensors (encoded DAC frames) or None.
                           Each tensor should have shape [T, C].

        Returns:
            A tuple containing:
                - delayed_batch (torch.Tensor): The prepared audio prompt tensor with
                  delays applied, shape [B, T_max_padded, C].
                - prefill_steps (list[int]): A list containing the number of valid
                  tokens (including BOS) for each prompt in the batch.
        """
        num_channels = self.config.decoder_config.num_channels
        audio_bos_value = self.config.bos_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts)

        max_len = (
            max(p.shape[0] if p is not None else 0 for p in audio_prompts)
            + max_delay_pattern
        )
        prefill_steps = []

        prefill = torch.full(
            (batch_size, max_len, num_channels),
            fill_value=-1,
            dtype=torch.int,
            device=self.device,
        )

        prefill[:, 0, :] = audio_bos_value

        for i in range(batch_size):
            prompt = audio_prompts[i]
            if prompt is not None:
                prompt = prompt.to(device=self.device, dtype=torch.int)
                prefill[i, 1 : prompt.shape[0] + 1, :] = prompt
                prefill_steps.append(prompt.shape[0] + 1)
            else:
                prefill_steps.append(1)

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=max_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        return delayed_batch, prefill_steps

    def _prepare_generation(
        self,
        text: torch.Tensor,
        audio_prompts: list[torch.Tensor | None],
        max_tokens: int | None = None,
        disable_cfg: bool = False,
        attn_fn: Callable = F.scaled_dot_product_attention,
    ):
        """Initializes the model state for generation.

        Encodes the text input (conditional and unconditional), prepares the
        encoder and decoder states (including KV caches and cross-attention),
        prepares the audio prompt, and performs the initial decoder prefill steps
        based on the audio prompts.

        Args:
            text: The padded text input tensor, shape [B, 1, T_text].
            audio_prompts: A list of prepared audio prompt tensors or None.

        Returns:
            A tuple containing:
                - dec_state (DecoderInferenceState): The initialized decoder state.
                - dec_output (DecoderOutput): The initialized decoder output manager,
                  containing the prefilled audio tokens.
        """
        batch_size = text.shape[0]

        enc_input_cond = text
        if disable_cfg:
            enc_input = enc_input_cond.view(batch_size, -1)
            batch_multiplier = 1
        else:
            enc_input_uncond = torch.zeros_like(text)
            stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
            enc_input = stacked_inputs.view(2 * batch_size, -1)
            batch_multiplier = 2

        enc_state = EncoderInferenceState.new(
            self.config,
            enc_input_cond,
            batch_multiplier=batch_multiplier,
        )
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(
            encoder_out
        )
        dec_state = DecoderInferenceState.new(
            self.config,
            enc_state,
            encoder_out,
            dec_cross_attn_cache,
            self.compute_dtype,
            max_generation_length=max_tokens,
            batch_multiplier=batch_multiplier,
        )

        prefill, prefill_steps = self._prepare_audio_prompt(audio_prompts)

        dec_output = DecoderOutput.new(batch_size, self.config, self.device)
        dec_output.prefill(prefill, prefill_steps)

        dec_step = min(prefill_steps) - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step)
            if not disable_cfg:
                tokens_BxTxC = tokens_BxTxC.repeat_interleave(2, dim=0)
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        current_idx: int,
        disable_cfg: bool = False,
    ) -> torch.Tensor:
        """Performs a single step of the decoder inference.

        Takes the tokens from the previous step, runs them through the decoder
        (for both conditional and unconditional paths), applies classifier-free
        guidance (CFG), samples the next token using temperature, top-p, and top-k
        sampling, and applies constraints (e.g., preventing EOS in certain channels).

        Args:
            tokens_Bx1xC: The input tokens for the current step, shape [2*B, 1, C].
                         Repeated for CFG (unconditional and conditional).
            dec_state: The current state of the decoder (KV caches, etc.).
            cfg_scale: The scale factor for classifier-free guidance.
            temperature: The temperature for sampling.
            top_p: The cumulative probability threshold for top-p sampling.
            top_k: The number of top logits to consider for top-k sampling.
            current_idx: The current generation step index.

        Returns:
            torch.Tensor: The sampled next tokens for each item in the batch,
                          shape [B, C].
        """
        audio_eos_value = self.config.eos_token_id
        logits_Bx1xCxV = self.model.decoder.decode_step(
            tokens_Bx1xC, dec_state, current_idx
        )
        if disable_cfg:
            B = tokens_Bx1xC.shape[0]
            logits_BxCxV = logits_Bx1xCxV[:, -1]
            if top_k is not None:
                _, top_k_indices_BxCxk = torch.topk(logits_BxCxV, k=top_k, dim=-1)
                mask_BxCxV = torch.ones_like(logits_BxCxV, dtype=torch.bool)
                mask_BxCxV = mask_BxCxV.scatter(
                    dim=-1, index=top_k_indices_BxCxk, value=False
                )
                logits_BxCxV = logits_BxCxV.masked_fill(mask_BxCxV, -torch.inf)
        else:
            B = tokens_Bx1xC.shape[0] // 2
            logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
            logits_last_Bx2xCxV = logits_last_2BxCxV.view(
                B, 2, *logits_last_2BxCxV.shape[1:]
            )

            uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]
            cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]
            cfg_logits_BxCxV = cond_logits_BxCxV + cfg_scale * (
                cond_logits_BxCxV - uncond_logits_BxCxV
            )

            _, top_k_indices_BxCxk = torch.topk(cfg_logits_BxCxV, k=top_k, dim=-1)
            mask_BxCxV = torch.ones_like(cond_logits_BxCxV, dtype=torch.bool)
            mask_BxCxV = mask_BxCxV.scatter(
                dim=-1, index=top_k_indices_BxCxk, value=False
            )
            logits_BxCxV = cond_logits_BxCxV.masked_fill(mask_BxCxV, -torch.inf)

        logits_BxCxV[:, :, audio_eos_value + 1 :] = torch.full_like(
            logits_BxCxV[:, :, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )
        logits_BxCxV[:, 1:, audio_eos_value:] = torch.full_like(
            logits_BxCxV[:, 1:, audio_eos_value:],
            fill_value=-torch.inf,
        )

        flat_logits_BCxV = logits_BxCxV.view(
            B * self.config.decoder_config.num_channels, -1
        )

        pred_BC = _sample_next_token(
            flat_logits_BCxV.float(),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_eos_value=audio_eos_value,
        )

        pred_BxC = pred_BC.view(B, self.config.decoder_config.num_channels)
        return pred_BxC

    def _generate_output(
        self, generated_codes: torch.Tensor, lengths_Bx: torch.Tensor
    ) -> list[np.ndarray]:
        """Converts generated delayed codes into audio waveforms.

        Reverts the delay pattern applied during generation, decodes the resulting
        codebook using the DAC model (if loaded), and returns a list of audio
        waveforms as NumPy arrays. If DAC is not loaded, returns the raw codebook indices.

        Args:
            generated_codes: The tensor of generated audio codes with delays,
                             shape [B, T_gen, C].
            lengths_Bx: A tensor containing the valid length of generated codes
                        (excluding padding and BOS/EOS markers) for each item
                        in the batch, shape [B].

        Returns:
            A list of NumPy arrays, where each array represents the generated audio
            waveform for one item in the batch. If DAC is not loaded, returns the
            raw, reverted codebook indices as NumPy arrays.
        """
        num_channels = self.config.decoder_config.num_channels
        batch_size = generated_codes.shape[0]
        seq_length = generated_codes.shape[1]
        delay_pattern = self.config.delay_pattern
        audio_pad_value = self.config.pad_token_id
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=batch_size,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes,
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audios = []

        if self.load_dac:
            for i in range(batch_size):
                audio = self._decode(codebook[i, : lengths_Bx[i], :])
                audio_np = audio.cpu().numpy()
                audios.append(audio_np)
        else:
            for i in range(batch_size):
                audios.append(codebook[i, : lengths_Bx[i], :].cpu().numpy())
        return audios

    @torch.no_grad()
    @torch.inference_mode()
    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encodes the given audio waveform into a tensor of DAC codebook indices
        """
        audio = audio.unsqueeze(0)
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)
        encoded_frame: torch.Tensor
        return encoded_frame.squeeze(0).transpose(0, 1)

    @torch.no_grad()
    @torch.inference_mode()
    def _decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes the given frames into an output audio waveform
        """
        audio_codes = audio_codes.unsqueeze(0).transpose(1, 2)
        audio_values, _, _ = self.dac_model.quantizer.from_codes(audio_codes)
        audio_values = self.dac_model.decode(audio_values)
        audio_values: torch.Tensor
        return audio_values.squeeze()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and preprocesses an audio file for use as a prompt.

        Loads the audio file, resamples it to the target sample rate if necessary,
        preprocesses it using the DAC model's preprocessing, and encodes it into
        DAC codebook indices.

        Args:
            audio_path: Path to the audio file.

        Returns:
            torch.Tensor: The encoded audio prompt as DAC codebook indices,
                          shape [T, C].

        Raises:
            RuntimeError: If the DAC model is not loaded (`load_dac=False` during init).
            FileNotFoundError: If the audio file cannot be found.
            Exception: If there's an error during loading or processing.
        """
        if self.dac_model is None:
            raise RuntimeError(
                "DAC model is required for loading audio prompts but was not loaded."
            )
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(
                audio, dim=0, keepdim=True
            )  # Average channels to get mono
        return self._encode(audio.to(self.device))

    def save_audio(self, path: str, audio: np.ndarray):
        """Saves the generated audio waveform to a file.

        Uses the soundfile library to write the NumPy audio array to the specified
        path with the default sample rate.

        Args:
            path: The path where the audio file will be saved.
            audio: The audio waveform as a NumPy array.
        """
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    @torch.inference_mode()
    def generate(
        self,
        text: str | list[str],
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 45,
        audio_prompt: list[str | torch.Tensor | None]
        | str
        | torch.Tensor
        | None = None,
        audio_prompt_path: list[str | torch.Tensor | None]
        | str
        | torch.Tensor
        | None = None,
        use_cfg_filter: bool | None = None,
        verbose: bool = False,
        stream_probe_dir: str | None = None,
        stream_probe_every_tokens: int = 86,
        disable_cfg: bool = False,
        max_audio_seconds: float | None = None,
        max_output_tokens_per_char: float | None = None,
        hard_stop_after_tokens: int | None = None,
        trim_audio_prompt_from_output: bool = False,
        audio_prompt_context_frames: int = DEFAULT_PROMPT_TRIM_CONTEXT_FRAMES,
        stream_callback = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Generates audio corresponding to the input text.

        Args:
            text: The input text prompt, or a list of text prompts for batch generation.
            max_tokens: The maximum number of audio tokens to generate per prompt.
                        Defaults to the model's configured audio length if None.
            cfg_scale: The scale factor for classifier-free guidance (CFG). Higher values
                       lead to stronger guidance towards the text prompt.
            temperature: The temperature for sampling. Higher values increase randomness.
            top_p: The cumulative probability threshold for nucleus (top-p) sampling.
            use_torch_compile: Whether to compile the generation steps using torch.compile.
                               Can significantly speed up generation after the initial
                               compilation overhead. Defaults to False.
            cfg_filter_top_k: The number of top logits to consider during CFG filtering.
                              (Note: This parameter name might be slightly misleading based
                              on the code; it's used in the `_sample_next_token` function.)
            audio_prompt: An audio prompt or list of prompts to condition the generation.
                          Can be a file path (str), a pre-loaded tensor (DAC codes), or None.
                          If a list, its length must match the batch size of the text input.
            audio_prompt_path: (Deprecated) Use `audio_prompt` instead.
            use_cfg_filter: (Deprecated) This parameter is no longer used.
            verbose: If True, prints progress information during generation, including
                     speed metrics.

        Returns:
            If a single text prompt was provided, returns a NumPy array containing the
            generated audio waveform.
            If a list of text prompts was provided, returns a list of NumPy arrays,
            each corresponding to a prompt in the input list. Returns None for a
            sequence if no audio was generated for it.
        """
        batch_size = len(text) if isinstance(text, list) else 1
        audio_eos_value = self.config.eos_token_id
        audio_pad_value = self.config.pad_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = torch.tensor(
            delay_pattern, device=self.device, dtype=torch.long
        )
        requested_max_tokens = max_tokens
        self.model.eval()
        self.last_generate_metadata = {}

        if audio_prompt_path:
            print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path
        if use_cfg_filter is not None:
            print("Warning: use_cfg_filter is deprecated.")

        if verbose:
            total_start_time = time.time()

        if use_torch_compile and not hasattr(self, "_compiled"):
            # Compilation can take about a minute.
            self._prepare_generation = torch.compile(
                self._prepare_generation, dynamic=True, fullgraph=True
            )
            self._decoder_step = torch.compile(
                self._decoder_step, fullgraph=True, mode="max-autotune"
            )
            self._compiled = True

        if isinstance(audio_prompt, list):
            audio_prompt = [
                self.load_audio(p) if isinstance(p, str) else p for p in audio_prompt
            ]
        elif isinstance(audio_prompt, str):
            audio_prompt = [self.load_audio(audio_prompt)]
        elif isinstance(audio_prompt, torch.Tensor):
            audio_prompt = [audio_prompt]
        elif audio_prompt is None:
            audio_prompt = [None] * batch_size

        assert len(audio_prompt) == batch_size, (
            "Number of audio prompts must match batch size"
        )
        prompt_code_steps = [
            int(prompt.shape[0]) if isinstance(prompt, torch.Tensor) else 0
            for prompt in audio_prompt
        ]
        prompt_duration_ms = [
            float(step) * SAMPLE_RATE_RATIO / DEFAULT_SAMPLE_RATE * 1000.0
            for step in prompt_code_steps
        ]

        if isinstance(text, list):
            text_inputs = text
            text = [self._encode_text(t) for t in text]
        else:
            text_inputs = [text]
            text = [self._encode_text(text)]
        text = self._pad_text_input(text)

        effective_max_tokens = max_tokens
        stop_constraints: dict[str, int | float] = {}
        if hard_stop_after_tokens is not None:
            effective_max_tokens = min(effective_max_tokens, hard_stop_after_tokens)
            stop_constraints["hard_stop_after_tokens"] = hard_stop_after_tokens
        if max_audio_seconds is not None and max_audio_seconds > 0:
            tokens_for_audio_seconds = max(
                1,
                ceil(
                    max_audio_seconds * DEFAULT_SAMPLE_RATE / SAMPLE_RATE_RATIO
                ),
            ) + max_delay_pattern
            effective_max_tokens = min(effective_max_tokens, tokens_for_audio_seconds)
            stop_constraints["max_audio_seconds"] = max_audio_seconds
        if max_output_tokens_per_char is not None and max_output_tokens_per_char > 0:
            max_chars = max(len(item) for item in text_inputs)
            tokens_for_chars = max(
                1, ceil(max_chars * max_output_tokens_per_char)
            ) + max_delay_pattern
            effective_max_tokens = min(effective_max_tokens, tokens_for_chars)
            stop_constraints["max_output_tokens_per_char"] = max_output_tokens_per_char
        effective_max_tokens = max(effective_max_tokens, max_delay_pattern + 1)

        dec_state, dec_output = self._prepare_generation(
            text,
            audio_prompt,
            max_tokens=effective_max_tokens,
            disable_cfg=disable_cfg,
        )
        dec_step = min(dec_output.prefill_steps) - 1
        current_idx = torch.tensor([dec_step], device=self.device)

        eos_detected_Bx = torch.zeros(
            (batch_size,), dtype=torch.bool, device=self.device
        )
        eos_countdown_Bx = torch.full(
            (batch_size,), -1, dtype=torch.long, device=self.device
        )
        finished_step_Bx = torch.full(
            (batch_size,), -1, dtype=torch.long, device=self.device
        )
        eos_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        stop_reason_code_Bx = torch.zeros(
            (batch_size,), dtype=torch.long, device=self.device
        )

        bos_over = False

        # --- Stream probe init ---
        _probe_prefill = dec_output.prefill_steps[0]
        _probe_sample_count = 0
        _probe_chunk_idx = 0

        if stream_probe_dir is not None:
            from pathlib import Path

            Path(stream_probe_dir).mkdir(parents=True, exist_ok=True)

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print(
                    "generate: using use_torch_compile=True, the first step may be slow"
                )
            start_time = time.time()

        # --- Generation Loop ---
        while dec_step < effective_max_tokens:
            if (eos_countdown_Bx == 0).all():
                break

            current_step_idx = dec_step + 1
            torch.compiler.cudagraph_mark_step_begin()
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step)
            if not disable_cfg:
                tokens_Bx1xC = tokens_Bx1xC.repeat_interleave(2, dim=0)

            pred_BxC = self._decoder_step(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                current_idx,
                disable_cfg=disable_cfg,
            )

            current_idx += 1

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
            if active_mask_Bx.any():
                is_eos_token_active = (~eos_detected_Bx[active_mask_Bx]) & (
                    pred_BxC[active_mask_Bx, 0] == audio_eos_value
                )
                active_indices = torch.nonzero(active_mask_Bx, as_tuple=False).squeeze(-1)
                is_eos_token = torch.zeros_like(active_mask_Bx)
                is_eos_token[active_indices] = is_eos_token_active
                is_max_len = current_step_idx >= effective_max_tokens - max_delay_pattern
                is_max_len_mask = active_mask_Bx & is_max_len
                eos_trigger_Bx = is_eos_token | is_max_len_mask
            eos_detected_Bx |= eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
            if start_countdown_mask_Bx.any():
                eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                finished_step_Bx[start_countdown_mask_Bx] = current_step_idx
                eos_step_Bx[start_countdown_mask_Bx] = current_step_idx
                eos_only_mask_Bx = start_countdown_mask_Bx & is_eos_token
                max_len_mask_Bx = start_countdown_mask_Bx & (~is_eos_token)
                stop_reason_code_Bx[eos_only_mask_Bx] = 1
                stop_reason_code_Bx[max_len_mask_Bx] = 2

            padding_mask_Bx = eos_countdown_Bx > 0
            if padding_mask_Bx.any():
                pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
                delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC[eos_mask_NxC] = audio_eos_value
                pred_active_BxC[pad_mask_NxC] = audio_pad_value
                pred_BxC[padding_mask_Bx] = pred_active_BxC
                eos_countdown_Bx[padding_mask_Bx] -= 1

            # --- Update BOS flag (Original) ---
            if not bos_over:
                bos_over = all(
                    dec_step - prefill_step > max_delay_pattern
                    for prefill_step in dec_output.prefill_steps
                )

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

            dec_step += 1

            # --- Stream probe ---
            if (
                (stream_probe_dir is not None or stream_callback is not None)
                and self.load_dac
                and dec_step > 0
                and dec_step % stream_probe_every_tokens == 0
            ):
                _n_tokens = dec_step - _probe_prefill

                if _n_tokens > max_delay_pattern:
                    _window = dec_output.generated_tokens[
                        0,
                        _probe_prefill : _probe_prefill + _n_tokens,
                        :,
                    ].unsqueeze(0)

                    _n_valid = max(0, _n_tokens - max_delay_pattern)

                    if _n_valid > 0:
                        _probe_lengths = torch.tensor(
                            [_n_valid],
                            dtype=torch.long,
                            device=self.device,
                        )

                        _audio_so_far = self._generate_output(
                            _window,
                            _probe_lengths,
                        )[0]

                        _chunk = _audio_so_far[_probe_sample_count:]
                        _probe_sample_count = len(_audio_so_far)

                        if len(_chunk) > 0:
                            _probe_chunk_idx += 1

                            _chunk_tensor = torch.from_numpy(_chunk).float().cpu()

                            if _chunk_tensor.ndim == 1:
                                _chunk_tensor = _chunk_tensor.unsqueeze(0)

                            if stream_callback is not None:
                                stream_callback(
                                    _chunk,
                                    DEFAULT_SAMPLE_RATE,
                                    _probe_chunk_idx,
                                    False,
                                )

                            if stream_probe_dir is not None:
                                torchaudio.save(
                                    f"{stream_probe_dir}/chunk_{_probe_chunk_idx:03d}.wav",
                                    _chunk_tensor,
                                    DEFAULT_SAMPLE_RATE,
                                )

                            if verbose:
                                print(
                                    f"stream probe: saved chunk {_probe_chunk_idx:03d}, samples={len(_chunk)}"
                                )

            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                if duration > 0:
                    print(
                        f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x"
                    )
                start_time = time.time()

        # --- Finalize and Extract Output ---
        final_step = dec_step + 1

        unfinished_mask_Bx = finished_step_Bx == -1
        finished_step_Bx[unfinished_mask_Bx] = final_step - max_delay_pattern
        stop_reason_code_Bx[unfinished_mask_Bx] = 2

        prefill_steps_tensor = torch.tensor(
            dec_output.prefill_steps, device=self.device
        )
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = torch.clamp(lengths_Bx, min=0)
        effective_audio_duration_ms_Bx = (
            lengths_Bx.to(torch.float32) * SAMPLE_RATE_RATIO / DEFAULT_SAMPLE_RATE * 1000.0
        )

        max_len = lengths_Bx.max().item() + max_delay_pattern
        outputs = []

        if max_len > 0:
            num_channels = self.config.decoder_config.num_channels
            audio_pad_value = self.config.pad_token_id
            generated_codes = torch.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=torch.long,
                device=self.device,
            )

            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = lengths_Bx[i].item() + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[
                        i, start_step : start_step + actual_len, :
                    ]
                    generated_codes[i, :actual_len, :] = tokens_to_copy

            if verbose:
                avg_steps = lengths_Bx.float().mean().item()
                total_duration = time.time() - total_start_time
                print(
                    f"generate: avg steps={avg_steps:.1f}, total duration={total_duration:.3f}s"
                )

            del dec_state

            if trim_audio_prompt_from_output and any(prompt_code_steps):
                outputs = []
                trimmed_samples: list[int] = []
                prompt_context_frames_per_item: list[int] = []
                output_duration_before_trim_ms: list[float] = []
                output_duration_after_trim_ms: list[float] = []
                decode_start_steps: list[int] = []
                decode_end_steps: list[int] = []
                decoded_code_steps: list[int] = []
                trimmed_code_steps: list[int] = []

                for i in range(batch_size):
                    prefill_step = dec_output.prefill_steps[i]
                    prompt_context_frames = min(
                        max(0, prefill_step),
                        max(0, audio_prompt_context_frames),
                    )
                    prompt_context_frames_per_item.append(prompt_context_frames)

                    decode_start = max(0, prefill_step - prompt_context_frames)
                    decode_start_steps.append(int(decode_start))
                    total_valid_frames = prompt_context_frames + lengths_Bx[i].item()
                    decoded_code_steps.append(int(total_valid_frames))
                    trimmed_code_steps.append(int(lengths_Bx[i].item()))
                    total_delayed_frames = total_valid_frames + max_delay_pattern
                    decode_end_steps.append(int(decode_start + total_delayed_frames))
                    delayed_codes = dec_output.generated_tokens[
                        i : i + 1,
                        decode_start : decode_start + total_delayed_frames,
                        :,
                    ]
                    decode_lengths = torch.tensor(
                        [total_valid_frames],
                        dtype=torch.long,
                        device=self.device,
                    )
                    decoded_audio = self._generate_output(delayed_codes, decode_lengths)[0]
                    before_trim_ms = (
                        1000.0 * float(len(decoded_audio)) / float(DEFAULT_SAMPLE_RATE)
                        if decoded_audio is not None
                        else 0.0
                    )
                    discard_samples = prompt_context_frames * SAMPLE_RATE_RATIO
                    end_sample = discard_samples + lengths_Bx[i].item() * SAMPLE_RATE_RATIO
                    trimmed_samples.append(discard_samples)
                    output_duration_before_trim_ms.append(before_trim_ms)
                    if decoded_audio is None:
                        outputs.append(None)
                        output_duration_after_trim_ms.append(0.0)
                    else:
                        trimmed_audio = decoded_audio[discard_samples:end_sample]
                        outputs.append(trimmed_audio)
                        output_duration_after_trim_ms.append(
                            1000.0
                            * float(len(trimmed_audio))
                            / float(DEFAULT_SAMPLE_RATE)
                        )
            else:
                outputs = self._generate_output(generated_codes, lengths_Bx)
                trimmed_samples = [0] * batch_size
                prompt_context_frames_per_item = [0] * batch_size
                output_duration_before_trim_ms = [
                    1000.0 * float(len(audio)) / float(DEFAULT_SAMPLE_RATE)
                    if audio is not None
                    else 0.0
                    for audio in outputs
                ]
                output_duration_after_trim_ms = list(output_duration_before_trim_ms)
                decode_start_steps = [0] * batch_size
                decode_end_steps = [0] * batch_size
                decoded_code_steps = [int(v) for v in lengths_Bx.detach().cpu().tolist()]
                trimmed_code_steps = [int(v) for v in lengths_Bx.detach().cpu().tolist()]

            # --- Stream probe final tail ---
            if (
                (stream_probe_dir is not None or stream_callback is not None)
                and self.load_dac
                and outputs
                and outputs[0] is not None
            ):
                _final_audio = outputs[0]
                _final_tail = _final_audio[_probe_sample_count:]

                if len(_final_tail) > 0:
                    _probe_chunk_idx += 1

                    _final_tail_tensor = torch.from_numpy(_final_tail).float().cpu()

                    if _final_tail_tensor.ndim == 1:
                        _final_tail_tensor = _final_tail_tensor.unsqueeze(0)

                    if stream_callback is not None:
                        stream_callback(
                            _final_tail,
                            DEFAULT_SAMPLE_RATE,
                            _probe_chunk_idx,
                            True,
                        )

                    if stream_probe_dir is not None:
                        torchaudio.save(
                            f"{stream_probe_dir}/chunk_{_probe_chunk_idx:03d}_final.wav",
                            _final_tail_tensor,
                            DEFAULT_SAMPLE_RATE,
                        )

                    if verbose:
                        print(
                            f"stream probe: saved final tail chunk {_probe_chunk_idx:03d}, samples={len(_final_tail)}"
                        )
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            outputs = [None] * batch_size

        stop_reason_lookup = {
            0: "other",
            1: "eos",
            2: "max_tokens",
        }
        self.last_generate_metadata = {
            "disable_cfg": disable_cfg,
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "stop_constraints": stop_constraints,
            "trim_audio_prompt_from_output": trim_audio_prompt_from_output,
            "prompt_code_steps": prompt_code_steps,
            "prompt_duration_ms": prompt_duration_ms,
            "audio_prompt_context_frames_requested": int(audio_prompt_context_frames),
            "prompt_context_frames": prompt_context_frames_per_item,
            "trimmed_samples": trimmed_samples,
            "decode_start": decode_start_steps,
            "decode_end": decode_end_steps,
            "decoded_code_steps": decoded_code_steps,
            "trimmed_code_steps": trimmed_code_steps,
            "output_duration_before_trim_ms": output_duration_before_trim_ms,
            "output_duration_after_trim_ms": output_duration_after_trim_ms,
            "eos_detected": [bool(v) for v in eos_detected_Bx.detach().cpu().tolist()],
            "eos_step": [
                None if int(v) < 0 else int(v) for v in eos_step_Bx.detach().cpu().tolist()
            ],
            "stop_step": [int(v) for v in finished_step_Bx.detach().cpu().tolist()],
            "stop_reason": [
                stop_reason_lookup[int(v)]
                for v in stop_reason_code_Bx.detach().cpu().tolist()
            ],
            "effective_audio_duration_ms": [
                float(v) for v in effective_audio_duration_ms_Bx.detach().cpu().tolist()
            ],
            "generated_token_count": [int(v) for v in lengths_Bx.detach().cpu().tolist()],
            "max_tokens": requested_max_tokens,
        }

        return outputs if batch_size > 1 else outputs[0]
