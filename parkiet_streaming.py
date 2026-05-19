import time
from typing import Callable

import numpy as np
import torch

from parkiet.dia.model import Dia, DEFAULT_SAMPLE_RATE, SAMPLE_RATE_RATIO


PcmCallback = Callable[[np.ndarray, int, int, bool], None]


class DelayResolver:
    def __init__(self, delay_pattern: tuple[int, ...]) -> None:
        self.delays = list(delay_pattern)
        self.max_delay = max(self.delays)
        self.num_channels = len(self.delays)
        self.buffer: list[torch.Tensor] = []
        self.next_aligned_index = 0

    def push(self, delayed_frame: torch.Tensor) -> None:
        self.buffer.append(delayed_frame.detach().cpu().long())

    def has_aligned(self) -> bool:
        return len(self.buffer) > self.next_aligned_index + self.max_delay

    def pop(self) -> torch.Tensor:
        t = self.next_aligned_index

        aligned = torch.stack(
            [
                self.buffer[t + self.delays[channel]][channel]
                for channel in range(self.num_channels)
            ],
            dim=0,
        ).long()

        # DAC codebook valid range is 0..1023. BOS/EOS/PAD/negative values
        # are not decodable audio codes, so map them to 0 like offline output path.
        invalid = (aligned < 0) | (aligned > 1023)
        aligned[invalid] = 0

        self.next_aligned_index += 1

        return aligned


class StreamingDacWindow:
    def __init__(
        self,
        model: Dia,
        chunk_frames: int = 20,
        overlap_frames: int = 10,
    ) -> None:
        self.model = model
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.samples_per_frame = SAMPLE_RATE_RATIO

        self.aligned_frames: list[torch.Tensor] = []
        self.emitted_frames = 0
        self.prev_tail: torch.Tensor | None = None

    def push(self, aligned_frame: torch.Tensor) -> list[np.ndarray]:
        self.aligned_frames.append(aligned_frame.detach().cpu().long())

        chunks: list[np.ndarray] = []

        while True:
            chunk = self.maybe_emit()

            if chunk is None:
                break

            chunks.append(chunk)

        return chunks

    def maybe_emit(self) -> np.ndarray | None:
        needed = self.emitted_frames + self.chunk_frames + self.overlap_frames

        if len(self.aligned_frames) < needed:
            return None

        left = max(0, self.emitted_frames - self.overlap_frames)
        right = self.emitted_frames + self.chunk_frames + self.overlap_frames

        window = torch.stack(self.aligned_frames[left:right], dim=0).to(
            device=self.model.device,
            dtype=torch.long,
        )

        with torch.no_grad(), torch.inference_mode():
            audio = self.model._decode(window).float().cpu()

        start_sample = (self.emitted_frames - left) * self.samples_per_frame
        end_sample = start_sample + self.chunk_frames * self.samples_per_frame

        chunk = audio[start_sample:end_sample].clone()

        if self.prev_tail is not None:
            crossfade_samples = min(
                self.overlap_frames * self.samples_per_frame // 2,
                len(self.prev_tail),
                len(chunk),
            )

            if crossfade_samples > 0:
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
                fade_out = 1.0 - fade_in

                chunk[:crossfade_samples] = (
                    chunk[:crossfade_samples] * fade_in
                    + self.prev_tail[-crossfade_samples:] * fade_out
                )

        self.prev_tail = audio[
            end_sample : end_sample + self.overlap_frames * self.samples_per_frame // 2
        ].clone()

        self.emitted_frames += self.chunk_frames

        return chunk.numpy()

    def flush(self) -> np.ndarray | None:
        if len(self.aligned_frames) <= self.emitted_frames:
            return None

        left = max(0, self.emitted_frames - self.overlap_frames)

        window = torch.stack(self.aligned_frames[left:], dim=0).to(
            device=self.model.device,
            dtype=torch.long,
        )

        with torch.no_grad(), torch.inference_mode():
            audio = self.model._decode(window).float().cpu()

        start_sample = (self.emitted_frames - left) * self.samples_per_frame
        tail = audio[start_sample:].clone()

        if self.prev_tail is not None:
            crossfade_samples = min(
                self.overlap_frames * self.samples_per_frame // 2,
                len(self.prev_tail),
                len(tail),
            )

            if crossfade_samples > 0:
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
                fade_out = 1.0 - fade_in

                tail[:crossfade_samples] = (
                    tail[:crossfade_samples] * fade_in
                    + self.prev_tail[-crossfade_samples:] * fade_out
                )

        fade_samples = min(DEFAULT_SAMPLE_RATE // 100, len(tail))

        if fade_samples > 0:
            tail[-fade_samples:] *= torch.linspace(1.0, 0.0, fade_samples)

        self.emitted_frames = len(self.aligned_frames)

        return tail.numpy()


@torch.inference_mode()
def generate_stream_pcm(
    model: Dia,
    text: str,
    *,
    on_pcm_chunk: PcmCallback,
    max_tokens: int = 3072,
    cfg_scale: float = 3.0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    cfg_filter_top_k: int = 45,
    use_torch_compile: bool = True,
    chunk_frames: int = 20,
    overlap_frames: int = 10,
    verbose: bool = False,
) -> None:
    batch_size = 1

    audio_eos_value = model.config.eos_token_id
    audio_pad_value = model.config.pad_token_id
    delay_pattern = model.config.delay_pattern
    max_delay_pattern = max(delay_pattern)
    delay_pattern_Cx = torch.tensor(
        delay_pattern,
        device=model.device,
        dtype=torch.long,
    )

    model.model.eval()

    if use_torch_compile and not hasattr(model, "_compiled"):
        model._prepare_generation = torch.compile(
            model._prepare_generation,
            dynamic=True,
            fullgraph=True,
        )
        model._decoder_step = torch.compile(
            model._decoder_step,
            fullgraph=True,
            mode="max-autotune",
        )
        model._compiled = True

    text_tokens = [model._encode_text(text)]
    text_tensor = model._pad_text_input(text_tokens)

    audio_prompt = [None]

    total_start_time = time.time()

    dec_state, dec_output = model._prepare_generation(
        text_tensor,
        audio_prompt,
        max_tokens=max_tokens,
    )

    prefill_step = dec_output.prefill_steps[0]
    dec_step = min(dec_output.prefill_steps) - 1
    current_idx = torch.tensor([dec_step], device=model.device)

    eos_detected_Bx = torch.zeros(
        (batch_size,),
        dtype=torch.bool,
        device=model.device,
    )
    eos_countdown_Bx = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=model.device,
    )
    finished_step_Bx = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=model.device,
    )

    bos_over = False

    delay_resolver = DelayResolver(delay_pattern)
    dac_streamer = StreamingDacWindow(
        model=model,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
    )

    chunk_index = 0

    if verbose:
        print("generate_stream: starting generation loop", flush=True)
        start_time = time.time()

    while dec_step < max_tokens:
        if (eos_countdown_Bx == 0).all():
            break

        current_step_idx = dec_step + 1

        torch.compiler.cudagraph_mark_step_begin()

        dec_state.prepare_step(dec_step)

        tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(
            2,
            dim=0,
        )

        pred_BxC = model._decoder_step(
            tokens_Bx1xC,
            dec_state,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            current_idx,
        )

        current_idx += 1

        active_mask_Bx = eos_countdown_Bx != 0
        eos_trigger_Bx = torch.zeros_like(active_mask_Bx)

        if active_mask_Bx.any():
            is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (
                pred_BxC[active_mask_Bx, 0] == audio_eos_value
            )
            is_max_len = current_step_idx >= max_tokens - max_delay_pattern
            eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len

        eos_detected_Bx |= eos_trigger_Bx
        start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)

        if start_countdown_mask_Bx.any():
            eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
            finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

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

        if not bos_over:
            bos_over = all(
                dec_step - current_prefill_step > max_delay_pattern
                for current_prefill_step in dec_output.prefill_steps
            )

        dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

        if current_step_idx >= prefill_step:
            delayed_frame = dec_output.generated_tokens[
                0,
                current_step_idx,
                :,
            ]

            delay_resolver.push(delayed_frame)

            while delay_resolver.has_aligned():
                aligned_frame = delay_resolver.pop()

                for chunk in dac_streamer.push(aligned_frame):
                    chunk_index += 1
                    on_pcm_chunk(
                        chunk,
                        DEFAULT_SAMPLE_RATE,
                        chunk_index,
                        False,
                    )

        dec_step += 1

        if verbose and dec_step % 86 == 0:
            duration = time.time() - start_time

            if duration > 0:
                print(
                    f"generate_stream step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x",
                    flush=True,
                )

            start_time = time.time()

    while delay_resolver.has_aligned():
        aligned_frame = delay_resolver.pop()

        for chunk in dac_streamer.push(aligned_frame):
            chunk_index += 1
            on_pcm_chunk(
                chunk,
                DEFAULT_SAMPLE_RATE,
                chunk_index,
                False,
            )

    final_chunk = dac_streamer.flush()

    if final_chunk is not None and len(final_chunk) > 0:
        chunk_index += 1
        on_pcm_chunk(
            final_chunk,
            DEFAULT_SAMPLE_RATE,
            chunk_index,
            True,
        )

    if verbose:
        total_duration = time.time() - total_start_time
        print(
            f"generate_stream: chunks={chunk_index}, total duration={total_duration:.3f}s",
            flush=True,
        )

    del dec_state
