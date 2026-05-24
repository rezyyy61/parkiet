from __future__ import annotations

from collections.abc import Callable

from .config import RealtimeTTSConfig
from .types import RealtimePhrase, RealtimeSynthesisResult


BackendSynthesisCallable = Callable[[RealtimePhrase, RealtimeTTSConfig], RealtimeSynthesisResult]


class RealtimeSynthesisScheduler:
    def __init__(self, backend: BackendSynthesisCallable, config: RealtimeTTSConfig):
        self.backend = backend
        self.config = config

    def synthesize_phrase(self, phrase: RealtimePhrase) -> RealtimeSynthesisResult:
        return self.backend(phrase, self.config)
