import threading
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


class StandardOutputStreamer:
    """Incrementally emits PCM using Dia.generate's output extraction path."""

    def __init__(
        self,
        model: Dia,
        prefill_step: int,
        max_delay_pattern: int,
        chunk_frames: int,
        overlap_frames: int,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.prefill_step = prefill_step
        self.max_delay_pattern = max_delay_pattern
        self.chunk_samples = chunk_frames * SAMPLE_RATE_RATIO
        self.lookahead_samples = overlap_frames * SAMPLE_RATE_RATIO
        self.verbose = verbose
        self.emitted_samples = 0
        self.first_frame_logged = False

    def emit_available(
        self,
        dec_output,
        available_until_step: int,
        *,
        final_length_frames: int | None = None,
    ) -> list[np.ndarray]:
        available_delayed_frames = available_until_step - self.prefill_step + 1

        if final_length_frames is None:
            valid_frames = available_delayed_frames - self.max_delay_pattern
        else:
            valid_frames = final_length_frames
            available_delayed_frames = valid_frames + self.max_delay_pattern

        if valid_frames <= 0 or available_delayed_frames <= 0:
            return []

        available_samples = valid_frames * SAMPLE_RATE_RATIO
        if (
            final_length_frames is None
            and available_samples - self.emitted_samples
            < self.chunk_samples + self.lookahead_samples
        ):
            return []

        delayed_codes = dec_output.generated_tokens[
            0:1,
            self.prefill_step : self.prefill_step + available_delayed_frames,
            :,
        ]
        lengths = torch.tensor(
            [valid_frames],
            dtype=torch.long,
            device=self.model.device,
        )

        audio = self.model._generate_output(delayed_codes, lengths)[0]

        if audio is None or len(audio) <= self.emitted_samples:
            return []

        if not self.first_frame_logged and self.verbose:
            print(
                "generate_stream: first emitted generated frame "
                f"index={self.prefill_step} valid_frame=0 "
                f"available_delayed_frames={available_delayed_frames}",
                flush=True,
            )
            self.first_frame_logged = True

        chunks: list[np.ndarray] = []
        is_final = final_length_frames is not None

        while self.emitted_samples < len(audio):
            remaining = len(audio) - self.emitted_samples

            if (
                not is_final
                and remaining < self.chunk_samples + self.lookahead_samples
            ):
                break

            take = remaining if is_final else self.chunk_samples
            start = self.emitted_samples
            end = start + take
            chunks.append(np.asarray(audio[start:end], dtype=np.float32))
            self.emitted_samples = end

        return chunks


@torch.inference_mode()
def generate_stream_pcm(
    model: Dia,
    text: str,
    *,
    on_pcm_chunk: PcmCallback,
    audio_prompt: torch.Tensor | str | None = None,
    cancel_event: threading.Event | None = None,
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

    # Resolve audio_prompt into the list form expected by _prepare_generation.
    if isinstance(audio_prompt, str):
        resolved_audio_prompt: list[torch.Tensor | None] = [model.load_audio(audio_prompt)]
    elif isinstance(audio_prompt, torch.Tensor):
        resolved_audio_prompt = [audio_prompt]
    else:
        resolved_audio_prompt = [None]
    audio_prompt_enabled = resolved_audio_prompt[0] is not None
    audio_prompt_frames = (
        int(resolved_audio_prompt[0].shape[0]) if audio_prompt_enabled else 0
    )

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

    total_start_time = time.time()

    dec_state, dec_output = model._prepare_generation(
        text_tensor,
        resolved_audio_prompt,
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

    output_streamer = StandardOutputStreamer(
        model=model,
        prefill_step=prefill_step,
        max_delay_pattern=max_delay_pattern,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        verbose=verbose,
    )

    chunk_index = 0
    # Hold-back buffer: we keep one chunk pending so we can tag the truly last
    # emitted chunk as is_final=True exactly once.
    pending_chunk: np.ndarray | None = None

    if verbose:
        print(
            "generate_stream: audio_prompt "
            f"enabled={audio_prompt_enabled} prompt_frames={audio_prompt_frames} "
            f"prefill_step={prefill_step} initial_dec_step={dec_step} "
            f"max_delay={max_delay_pattern}",
            flush=True,
        )
        print("generate_stream: starting generation loop", flush=True)
        start_time = time.time()

    while dec_step < max_tokens:
        if cancel_event is not None and cancel_event.is_set():
            del dec_state
            return

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

        if current_step_idx >= prefill_step + max_delay_pattern:
            for chunk in output_streamer.emit_available(
                dec_output,
                current_step_idx,
            ):
                if pending_chunk is not None:
                    chunk_index += 1
                    on_pcm_chunk(pending_chunk, DEFAULT_SAMPLE_RATE, chunk_index, False)
                elif verbose:
                    print(
                        f"generate_stream: first PCM chunk samples={len(chunk)}",
                        flush=True,
                    )
                pending_chunk = chunk

        dec_step += 1

        if verbose and dec_step % 86 == 0:
            duration = time.time() - start_time

            if duration > 0:
                print(
                    f"generate_stream step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x",
                    flush=True,
                )

            start_time = time.time()

    final_step = dec_step + 1
    finished_step_Bx[finished_step_Bx == -1] = final_step - max_delay_pattern
    final_length_frames = int(
        torch.clamp(finished_step_Bx[0] - prefill_step, min=0).item()
    )
    final_available_until_step = prefill_step + final_length_frames + max_delay_pattern - 1

    final_chunks = output_streamer.emit_available(
        dec_output,
        final_available_until_step,
        final_length_frames=final_length_frames,
    )

    for chunk in final_chunks:
        if pending_chunk is not None:
            chunk_index += 1
            on_pcm_chunk(pending_chunk, DEFAULT_SAMPLE_RATE, chunk_index, False)
        elif verbose:
            print(
                f"generate_stream: first PCM chunk samples={len(chunk)}",
                flush=True,
            )
        pending_chunk = chunk

    if pending_chunk is not None:
        chunk_index += 1
        on_pcm_chunk(pending_chunk, DEFAULT_SAMPLE_RATE, chunk_index, True)
        pending_chunk = None

    if verbose:
        total_duration = time.time() - total_start_time
        print(
            f"generate_stream: chunks={chunk_index}, total duration={total_duration:.3f}s",
            flush=True,
        )

    del dec_state
