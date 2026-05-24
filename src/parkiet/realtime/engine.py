from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
import threading
from types import TracebackType
from typing import Protocol

import numpy as np
import torch

from parkiet.dia.model import DEFAULT_SAMPLE_RATE, Dia

from .audio_buffer import AudioFrameBuffer, BufferedAudioFrame
from .config import RealtimeTTSConfig
from .exceptions import BackendSynthesisError, SessionClosedError, SessionNotFoundError
from .frame_encoder import pcm16_frame_bytes
from .scheduler import RealtimeSynthesisScheduler
from .segmenter import DutchAwarePhraseSegmenter
from .session import RealtimeTTSSession
from .types import (
    AudioChunk,
    AudioFrame,
    PhraseSynthesisMetrics,
    RealtimePhrase,
    RealtimeSynthesisResult,
    RealtimeTextInput,
    SessionMetrics,
    SessionState,
)


class DiaLike(Protocol):
    def generate(
        self,
        text: str | list[str],
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 45,
        audio_prompt=None,
        audio_prompt_path=None,
        use_cfg_filter=None,
        verbose: bool = False,
        stream_probe_dir: str | None = None,
        stream_probe_every_tokens: int = 86,
        stream_callback=None,
    ) -> np.ndarray | list[np.ndarray]:
        ...


@dataclass(slots=True)
class SessionRuntime:
    session: RealtimeTTSSession
    config: RealtimeTTSConfig
    audio_buffer: AudioFrameBuffer
    scheduler: RealtimeSynthesisScheduler | None
    lock: threading.RLock
    condition: threading.Condition
    stop_event: threading.Event
    worker_thread: threading.Thread | None = None


class DiaRealtimeBackend:
    def __init__(
        self,
        model: DiaLike,
        *,
        use_torch_compile: bool = False,
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.8,
        top_p: float = 0.90,
        cfg_filter_top_k: int = 50,
        collect_timings: bool = False,
        model_load_time_ms: float | None = None,
    ):
        self.model = model
        self.use_torch_compile = use_torch_compile
        self.max_tokens = max_tokens
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.top_p = top_p
        self.cfg_filter_top_k = cfg_filter_top_k
        self.collect_timings = collect_timings
        self.model_load_time_ms = model_load_time_ms
        self.last_timing_breakdown: dict[str, float | int | list[int] | bool | None] = {}

    def __call__(self, phrase: RealtimePhrase, config: RealtimeTTSConfig) -> RealtimeSynthesisResult:
        timings: dict[str, float | int | list[int] | bool | None] = {
            "model_load_ms": self.model_load_time_ms,
            "text_encode_ms": 0.0,
            "prepare_generation_ms": 0.0,
            "decoder_loop_ms": 0.0,
            "decoder_step_calls": 0,
            "generate_output_ms": 0.0,
            "dac_decode_ms": 0.0,
            "cpu_transfer_ms": 0.0,
            "generated_tokens": 0,
            "generated_token_lengths": [],
            "total_ms": 0.0,
            "gpu_peak_memory_bytes": None,
            "use_torch_compile": self.use_torch_compile,
            "max_tokens": self.max_tokens,
            "cfg_scale": self.cfg_scale,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cfg_filter_top_k": self.cfg_filter_top_k,
        }

        with _DiaGenerateProfiler(self.model, timings, enabled=self.collect_timings):
            waveform = self.model.generate(
                phrase.text,
                max_tokens=self.max_tokens,
                cfg_scale=self.cfg_scale,
                temperature=self.temperature,
                top_p=self.top_p,
                cfg_filter_top_k=self.cfg_filter_top_k,
                use_torch_compile=self.use_torch_compile,
                verbose=False,
                stream_callback=None,
            )
        if isinstance(waveform, list):
            waveform = waveform[0]

        waveform_np = np.asarray(waveform, dtype=np.float32).reshape(-1)
        self.last_timing_breakdown = timings
        chunk = AudioChunk(
            session_id=phrase.session_id,
            phrase_index=phrase.index,
            text=phrase.text,
            sample_rate=DEFAULT_SAMPLE_RATE,
            waveform=waveform_np,
            metadata={"timings": timings},
        )
        metrics = PhraseSynthesisMetrics(
            phrase_index=phrase.index,
            phrase_text=phrase.text,
            queued_at=phrase.queued_at,
            synthesis_finished_at=chunk.generated_at,
            audio_duration_ms=1000.0 * float(waveform_np.shape[0]) / float(DEFAULT_SAMPLE_RATE),
        )
        return RealtimeSynthesisResult(phrase=phrase, audio_chunk=chunk, metrics=metrics)

    @classmethod
    def from_local_paths(
        cls,
        *,
        config_path: str | Path = "config.json",
        checkpoint_path: str | Path = "weights/dia-nl-v1.pth",
        compute_dtype: str = "float32",
        device: torch.device | None = None,
        load_dac: bool = True,
        use_torch_compile: bool = False,
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.8,
        top_p: float = 0.90,
        cfg_filter_top_k: int = 50,
        collect_timings: bool = False,
    ) -> "DiaRealtimeBackend":
        load_started = perf_counter()
        model = Dia.from_local(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            compute_dtype=compute_dtype,
            device=device,
            load_dac=load_dac,
        )
        model_load_time_ms = (perf_counter() - load_started) * 1000.0
        return cls(
            model,
            use_torch_compile=use_torch_compile,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            collect_timings=collect_timings,
            model_load_time_ms=model_load_time_ms,
        )


class _DiaGenerateProfiler:
    def __init__(
        self,
        model: DiaLike,
        timings: dict[str, float | int | list[int] | bool | None],
        *,
        enabled: bool,
    ):
        self.model = model
        self.timings = timings
        self.enabled = enabled
        self._orig_encode_text = None
        self._orig_prepare_generation = None
        self._orig_decoder_step = None
        self._orig_generate_output = None
        self._orig_decode = None
        self._orig_tensor_cpu = None
        self._device = getattr(model, "device", None)

    def __enter__(self) -> "_DiaGenerateProfiler":
        if not self.enabled:
            self._reset_peak_memory()
            self.timings["_call_started_at"] = perf_counter()
            return self

        self._reset_peak_memory()
        self.timings["_call_started_at"] = perf_counter()

        self._orig_encode_text = getattr(self.model, "_encode_text", None)
        self._orig_prepare_generation = getattr(self.model, "_prepare_generation", None)
        self._orig_decoder_step = getattr(self.model, "_decoder_step", None)
        self._orig_generate_output = getattr(self.model, "_generate_output", None)
        self._orig_decode = getattr(self.model, "_decode", None)
        self._orig_tensor_cpu = torch.Tensor.cpu

        if self._orig_encode_text is not None:
            def wrapped_encode_text(text):
                return self._time_call("text_encode_ms", self._orig_encode_text, text)
            self.model._encode_text = wrapped_encode_text

        if self._orig_prepare_generation is not None:
            def wrapped_prepare_generation(*args, **kwargs):
                return self._time_call("prepare_generation_ms", self._orig_prepare_generation, *args, **kwargs)
            self.model._prepare_generation = wrapped_prepare_generation

        if self._orig_decoder_step is not None:
            def wrapped_decoder_step(*args, **kwargs):
                self.timings["decoder_step_calls"] = int(self.timings["decoder_step_calls"]) + 1
                return self._time_call("decoder_loop_ms", self._orig_decoder_step, *args, **kwargs)
            self.model._decoder_step = wrapped_decoder_step

        if self._orig_decode is not None:
            def wrapped_decode(*args, **kwargs):
                return self._time_call("dac_decode_ms", self._orig_decode, *args, **kwargs)
            self.model._decode = wrapped_decode

        if self._orig_generate_output is not None:
            def wrapped_generate_output(generated_codes, lengths_Bx):
                token_lengths = [int(value) for value in lengths_Bx.detach().cpu().tolist()]
                self.timings["generated_token_lengths"] = token_lengths
                self.timings["generated_tokens"] = sum(token_lengths)
                return self._time_call("generate_output_ms", self._orig_generate_output, generated_codes, lengths_Bx)
            self.model._generate_output = wrapped_generate_output

        def timed_tensor_cpu(tensor, *args, **kwargs):
            self._synchronize()
            started = perf_counter()
            result = self._orig_tensor_cpu(tensor, *args, **kwargs)
            self._synchronize()
            self.timings["cpu_transfer_ms"] = float(self.timings["cpu_transfer_ms"]) + (
                (perf_counter() - started) * 1000.0
            )
            return result

        torch.Tensor.cpu = timed_tensor_cpu
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.enabled:
            if self._orig_encode_text is not None:
                self.model._encode_text = self._orig_encode_text
            if self._orig_prepare_generation is not None:
                self.model._prepare_generation = self._orig_prepare_generation
            if self._orig_decoder_step is not None:
                self.model._decoder_step = self._orig_decoder_step
            if self._orig_generate_output is not None:
                self.model._generate_output = self._orig_generate_output
            if self._orig_decode is not None:
                self.model._decode = self._orig_decode
            if self._orig_tensor_cpu is not None:
                torch.Tensor.cpu = self._orig_tensor_cpu
        self._synchronize()
        self.timings["total_ms"] = (perf_counter() - float(self.timings.pop("_call_started_at"))) * 1000.0
        if isinstance(self._device, torch.device) and self._device.type == "cuda":
            self.timings["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(self._device))

    def _time_call(self, key: str, fn, *args, **kwargs):
        self._synchronize()
        started = perf_counter()
        result = fn(*args, **kwargs)
        self._synchronize()
        self.timings[key] = float(self.timings[key]) + ((perf_counter() - started) * 1000.0)
        return result

    def _synchronize(self) -> None:
        if isinstance(self._device, torch.device) and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _reset_peak_memory(self) -> None:
        if isinstance(self._device, torch.device) and self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)


class RealtimeTTSEngine:
    def __init__(
        self,
        *,
        config: RealtimeTTSConfig | None = None,
        backend=None,
        segmenter: DutchAwarePhraseSegmenter | None = None,
        scheduler: RealtimeSynthesisScheduler | None = None,
    ):
        self.config = config or RealtimeTTSConfig()
        self.segmenter = segmenter or DutchAwarePhraseSegmenter(self.config)
        self.backend = backend
        self.scheduler = scheduler or (
            RealtimeSynthesisScheduler(backend=self.backend, config=self.config)
            if self.backend is not None
            else None
        )
        self._sessions: dict[str, SessionRuntime] = {}

    def create_session(
        self,
        session_id: str,
        config_override: dict[str, object] | None = None,
    ) -> RealtimeTTSSession:
        if session_id in self._sessions:
            raise ValueError(f"Session already exists: {session_id}")
        lock = threading.RLock()
        session_config = replace(self.config, **(config_override or {}))
        runtime = SessionRuntime(
            session=RealtimeTTSSession(session_id=session_id),
            config=session_config,
            audio_buffer=AudioFrameBuffer(session_config),
            scheduler=(
                RealtimeSynthesisScheduler(backend=self.backend, config=session_config)
                if self.backend is not None
                else None
            ),
            lock=lock,
            condition=threading.Condition(lock),
            stop_event=threading.Event(),
        )
        self._sessions[session_id] = runtime
        return runtime.session

    def accept_text(self, session_id: str, text: str, is_final: bool = True) -> list[RealtimePhrase]:
        return self._accept_input(
            RealtimeTextInput(session_id=session_id, text=text, is_final=is_final, is_delta=False)
        )

    def accept_text_delta(
        self,
        session_id: str,
        text_delta: str,
        is_final: bool = False,
    ) -> list[RealtimePhrase]:
        return self._accept_input(
            RealtimeTextInput(session_id=session_id, text=text_delta, is_final=is_final, is_delta=True)
        )

    def synthesize_pending(self, session_id: str) -> list[RealtimeSynthesisResult]:
        results: list[RealtimeSynthesisResult] = []
        while True:
            result = self.synthesize_next_phrase(session_id)
            if result is None:
                break
            results.append(result)
        return results

    def synthesize_next_phrase(self, session_id: str) -> RealtimeSynthesisResult | None:
        runtime = self._get_runtime(session_id)
        if runtime.scheduler is None:
            raise RuntimeError("RealtimeTTSEngine requires a synthesis backend")

        with runtime.lock:
            session = runtime.session
            if session.state == SessionState.CLOSED:
                raise SessionClosedError(f"Realtime session is closed: {session_id}")
            if session.interrupt_requested:
                return None
            if session.is_synthesizing or not session.phrase_queue:
                return None
            phrase = session.phrase_queue.popleft()
            epoch = session.generation_epoch
            started_at = datetime.utcnow()
            session.mark_synthesis_started(phrase, started_at)

        start_clock = perf_counter()
        try:
            result = runtime.scheduler.synthesize_phrase(phrase)
        except Exception as exc:
            with runtime.lock:
                runtime.session.note_synthesis_finished_without_audio()
                runtime.session.note_buffer_level_ms(runtime.audio_buffer.buffer_level_ms())
            raise BackendSynthesisError(
                f"Realtime backend synthesis failed for phrase {phrase.index} in session {session_id}"
            ) from exc
        generation_ms = (perf_counter() - start_clock) * 1000.0
        result.metrics.synthesis_started_at = started_at
        result.metrics.synthesis_finished_at = result.audio_chunk.generated_at
        result.metrics.generation_ms = generation_ms

        with runtime.lock:
            session = runtime.session
            if (
                session.state == SessionState.CLOSED
                or session.interrupt_requested
                or session.generation_epoch != epoch
            ):
                session.note_synthesis_finished_without_audio()
                session.note_buffer_level_ms(runtime.audio_buffer.buffer_level_ms())
                return None

            frame_count = runtime.audio_buffer.push_chunk(result.audio_chunk)
            result.metrics.frame_count = frame_count
            result.metrics.audio_duration_ms = 1000.0 * float(result.audio_chunk.waveform.shape[0]) / float(
                result.audio_chunk.sample_rate
            )
            session.enqueue_audio(result.audio_chunk, generation_ms=generation_ms, frame_count=frame_count)
            session.note_buffer_level_ms(runtime.audio_buffer.buffer_level_ms())
            runtime.condition.notify_all()
            return result

    def start_session_worker(self, session_id: str) -> None:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            session = runtime.session
            if session.state == SessionState.CLOSED:
                raise SessionClosedError(f"Realtime session is closed: {session_id}")
            if runtime.worker_thread is not None and runtime.worker_thread.is_alive():
                return
            runtime.stop_event.clear()
            thread = threading.Thread(
                target=self._worker_loop,
                args=(session_id,),
                name=f"parkiet-realtime-{session_id}",
                daemon=True,
            )
            runtime.worker_thread = thread
            session.set_worker_running(True)
            thread.start()

    def stop_session_worker(self, session_id: str) -> None:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            runtime.stop_event.set()
            runtime.condition.notify_all()
            thread = runtime.worker_thread
        if thread is not None:
            thread.join()
        with runtime.lock:
            runtime.worker_thread = None
            runtime.session.set_worker_running(False)

    def tick(self, session_id: str) -> RealtimeSynthesisResult | None:
        return self.synthesize_next_phrase(session_id)

    def read_audio_frame(
        self,
        session_id: str,
        *,
        return_silence_when_empty: bool | None = None,
    ) -> AudioFrame | None:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            session = runtime.session
            if session.state == SessionState.CLOSED:
                raise SessionClosedError(f"Realtime session is closed: {session_id}")

            if not runtime.audio_buffer.has_buffered_audio() and self._is_session_complete(session, runtime):
                if session.completion_emitted:
                    return None
                frame = self._build_output_frame(
                    runtime,
                    payload=self._empty_payload(runtime.config.output_sample_format),
                    is_silence=False,
                    is_final=True,
                    source_phrase_index=None,
                )
                session.completion_emitted = True
                session.next_frame_index += 1
                session.note_frame_read(
                    datetime.utcnow(),
                    runtime.audio_buffer.buffer_level_ms(),
                    is_silence=False,
                )
                return frame

            buffered_frame = runtime.audio_buffer.read_frame(
                return_silence_when_empty=return_silence_when_empty,
            )

            if buffered_frame is None:
                return None

            is_final = False
            if buffered_frame.is_silence and session.state != SessionState.INTERRUPTED and session.has_pending_work():
                session.note_underrun()
            if (
                not buffered_frame.is_silence
                and self._is_session_complete_after_buffer_read(session, runtime)
                and not session.completion_emitted
            ):
                is_final = True
                session.completion_emitted = True

            payload = self._encode_output_payload(buffered_frame.samples, runtime.config.output_sample_format)
            frame = self._build_output_frame(
                runtime,
                payload=payload,
                is_silence=buffered_frame.is_silence,
                is_final=is_final,
                source_phrase_index=buffered_frame.source_phrase_index,
            )
            session.next_frame_index += 1
            session.note_frame_read(
                datetime.utcnow(),
                runtime.audio_buffer.buffer_level_ms(),
                is_silence=buffered_frame.is_silence,
            )
            return frame

    def interrupt(self, session_id: str) -> None:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            if runtime.session.state == SessionState.CLOSED:
                raise SessionClosedError(f"Realtime session is closed: {session_id}")
            runtime.session.request_interrupt()
            runtime.session.clear_pending()
            runtime.audio_buffer.flush()
            runtime.condition.notify_all()

    def close_session(self, session_id: str) -> None:
        runtime = self._get_runtime(session_id)
        self.stop_session_worker(session_id)
        with runtime.lock:
            runtime.audio_buffer.flush()
            runtime.session.close()
        del self._sessions[session_id]

    def get_session(self, session_id: str) -> RealtimeTTSSession | None:
        runtime = self._sessions.get(session_id)
        return runtime.session if runtime else None

    def get_session_metrics(self, session_id: str) -> dict[str, object]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            session = runtime.session
            return {
                "accepted_text_chars": session.metrics.accepted_text_chars,
                "generated_phrase_count": session.metrics.generated_phrase_count,
                "generated_audio_ms": session.metrics.generated_audio_ms,
                "frames_read": session.metrics.frames_read,
                "generated_frames_returned": session.metrics.generated_frames_returned,
                "silence_frames_returned": session.metrics.silence_frames_returned,
                "underrun_count": session.metrics.underrun_count,
                "buffer_level_ms": session.metrics.buffer_level_ms,
                "is_worker_running": session.is_worker_running,
                "is_synthesizing": session.is_synthesizing,
                "pending_text_length": len(session.pending_text),
                "queued_phrase_count": len(session.phrase_queue),
            }

    def get_session_state(self, session_id: str) -> dict[str, object]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            session = runtime.session
            return {
                "session_id": session.session_id,
                "is_active": session.state == SessionState.ACTIVE,
                "is_worker_running": session.is_worker_running,
                "is_synthesizing": session.is_synthesizing,
                "is_interrupted": session.state == SessionState.INTERRUPTED,
                "is_ready_for_playback": self._is_ready_for_playback_locked(session, runtime),
                "has_pending_text": bool(session.pending_text),
                "has_queued_phrases": bool(session.phrase_queue),
                "has_buffered_audio": runtime.audio_buffer.has_buffered_audio(),
                "has_inflight_work": session.has_pending_work(),
            }

    def get_phrase_metrics(self, session_id: str) -> list[PhraseSynthesisMetrics]:
        session = self._get_runtime(session_id).session
        return [session.phrase_metrics[index] for index in sorted(session.phrase_metrics)]

    def get_buffer_level_ms(self, session_id: str) -> float:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            return runtime.audio_buffer.buffer_level_ms()

    def is_ready_for_playback(self, session_id: str) -> bool:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            return self._is_ready_for_playback_locked(runtime.session, runtime)

    def _accept_input(self, text_input: RealtimeTextInput) -> list[RealtimePhrase]:
        runtime = self._get_runtime(text_input.session_id)
        with runtime.lock:
            session = runtime.session
            if session.state == SessionState.CLOSED:
                raise SessionClosedError(f"Realtime session is closed: {text_input.session_id}")
            if session.interrupt_requested:
                session.reactivate()

            self._validate_text_input(text_input)
            session.append_input(text_input)
            phrases, remainder = self.segmenter.split(
                session_id=text_input.session_id,
                text=session.pending_text,
                start_index=session.next_phrase_index,
                is_final=text_input.is_final,
            )
            session.enqueue_phrases(phrases, remainder)
            session.note_buffer_level_ms(runtime.audio_buffer.buffer_level_ms())
            runtime.condition.notify_all()
            return phrases

    def _worker_loop(self, session_id: str) -> None:
        runtime = self._get_runtime(session_id)
        try:
            while not runtime.stop_event.is_set():
                with runtime.lock:
                    session = runtime.session
                    should_wait = (
                        session.state != SessionState.CLOSED
                        and not runtime.stop_event.is_set()
                        and (session.interrupt_requested or not session.phrase_queue or session.is_synthesizing)
                    )
                    if should_wait:
                        runtime.condition.wait(timeout=0.05)
                        continue
                try:
                    self.synthesize_next_phrase(session_id)
                except BackendSynthesisError:
                    runtime.stop_event.set()
        finally:
            with runtime.lock:
                runtime.session.set_worker_running(False)
                runtime.worker_thread = None
                runtime.stop_event.clear()

    def _get_runtime(self, session_id: str) -> SessionRuntime:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            raise SessionNotFoundError(f"Unknown realtime TTS session: {session_id}")
        return runtime

    def _validate_text_input(self, text_input: RealtimeTextInput) -> None:
        if not isinstance(text_input.text, str):
            raise TypeError("Realtime text input must be a string")
        if text_input.text == "" and not text_input.is_final:
            raise ValueError("Realtime text input cannot be empty unless is_final=True")

    def _is_ready_for_playback_locked(self, session: RealtimeTTSSession, runtime: SessionRuntime) -> bool:
        buffer_level_ms = runtime.audio_buffer.buffer_level_ms()
        if buffer_level_ms >= runtime.config.start_playback_when_buffer_ms:
            return True
        if session.state == SessionState.INTERRUPTED:
            return False
        return not session.has_pending_work()

    def _is_session_complete(self, session: RealtimeTTSSession, runtime: SessionRuntime) -> bool:
        return (
            session.final_input_received
            and not session.pending_text
            and not session.phrase_queue
            and not session.is_synthesizing
            and not runtime.audio_buffer.has_buffered_audio()
            and session.state == SessionState.ACTIVE
        )

    def _is_session_complete_after_buffer_read(self, session: RealtimeTTSSession, runtime: SessionRuntime) -> bool:
        return self._is_session_complete(session, runtime)

    def _encode_output_payload(self, samples: np.ndarray, sample_format: str) -> bytes | np.ndarray:
        if sample_format == "pcm16":
            return pcm16_frame_bytes(samples)
        if sample_format == "float32":
            return np.asarray(samples, dtype=np.float32).copy()
        raise ValueError(f"Unsupported output sample format: {sample_format}")

    def _empty_payload(self, sample_format: str) -> bytes | np.ndarray:
        if sample_format == "pcm16":
            return b""
        if sample_format == "float32":
            return np.zeros(0, dtype=np.float32)
        raise ValueError(f"Unsupported output sample format: {sample_format}")

    def _build_output_frame(
        self,
        runtime: SessionRuntime,
        *,
        payload: bytes | np.ndarray,
        is_silence: bool,
        is_final: bool,
        source_phrase_index: int | None,
    ) -> AudioFrame:
        session = runtime.session
        return AudioFrame(
            session_id=session.session_id,
            sequence_number=session.next_frame_index,
            payload=payload,
            sample_rate=runtime.config.output_sample_rate,
            frame_duration_ms=runtime.config.frame_duration_ms if not is_final else (
                0 if (isinstance(payload, bytes) and len(payload) == 0)
                or (isinstance(payload, np.ndarray) and payload.size == 0)
                else runtime.config.frame_duration_ms
            ),
            sample_format=runtime.config.output_sample_format,
            channels=1,
            is_silence=is_silence,
            is_final=is_final,
            buffer_level_ms=runtime.audio_buffer.buffer_level_ms(),
            underrun_count=session.metrics.underrun_count,
            source_phrase_index=source_phrase_index,
        )
