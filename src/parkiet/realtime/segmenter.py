from __future__ import annotations

import re

from .config import RealtimeTTSConfig
from .types import RealtimePhrase


VOICE_TAG_PATTERN = re.compile(r"^\s*(\[[^\]]+\])\s*")
WHITESPACE_PATTERN = re.compile(r"\s+")
CLAUSE_PATTERN = re.compile(r"[^.!?,;:]+[.!?,;:]?")
HARD_BOUNDARY = frozenset({".", "!", "?", ";", ":"})
SOFT_BOUNDARY = frozenset({","})


class DutchAwarePhraseSegmenter:
    def __init__(self, config: RealtimeTTSConfig):
        self.config = config

    def split(
        self,
        session_id: str,
        text: str,
        start_index: int,
        *,
        is_final: bool = False,
    ) -> tuple[list[RealtimePhrase], str]:
        normalized = self._normalize(text)
        if not normalized:
            return [], ""

        voice_tag, content = self._extract_voice_tag(normalized)
        clauses = self._split_clauses(content)

        ready_phrases: list[str] = []
        remainder = ""
        current = ""

        for clause in clauses:
            candidate = clause if not current else f"{current} {clause}"
            if current and len(candidate) > self.config.max_phrase_chars:
                ready_phrases.append(current)
                current = ""
                long_parts = self._split_long_text(clause)
                for index, part in enumerate(long_parts):
                    is_last_part = index == len(long_parts) - 1
                    if not is_last_part:
                        ready_phrases.append(part)
                    else:
                        current = part
                if current and self._should_flush(current, is_final=False):
                    ready_phrases.append(current)
                    current = ""
                continue

            current = candidate
            if self._should_flush(current, is_final=False):
                ready_phrases.append(current)
                current = ""

        if current:
            if is_final or self._should_flush(current, is_final=True):
                ready_phrases.append(current)
            else:
                remainder = self._with_voice_tag(voice_tag, current)

        phrases = [
            RealtimePhrase(
                session_id=session_id,
                index=start_index + offset,
                text=self._with_voice_tag(voice_tag, phrase_text),
                voice_tag=voice_tag,
                source_text=phrase_text,
                is_final=is_final and remainder == "" and offset == len(ready_phrases) - 1,
            )
            for offset, phrase_text in enumerate(ready_phrases)
        ]
        return phrases, remainder

    def _normalize(self, text: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", text).strip()

    def _extract_voice_tag(self, text: str) -> tuple[str, str]:
        match = VOICE_TAG_PATTERN.match(text)
        if not match:
            return self.config.voice_tag, text
        return match.group(1), text[match.end() :].strip()

    def _split_clauses(self, text: str) -> list[str]:
        return [match.group(0).strip() for match in CLAUSE_PATTERN.finditer(text) if match.group(0).strip()]

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.config.max_phrase_chars:
            return [text]

        parts: list[str] = []
        current_words: list[str] = []
        for word in text.split(" "):
            candidate_words = current_words + [word]
            candidate = " ".join(candidate_words)
            if current_words and len(candidate) > self.config.max_phrase_chars:
                parts.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words

        if current_words:
            parts.append(" ".join(current_words))
        return parts

    def _should_flush(self, text: str, *, is_final: bool) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if len(normalized) >= self.config.max_phrase_chars:
            return True

        boundary = normalized[-1]
        if boundary in HARD_BOUNDARY:
            return is_final or len(normalized) >= self.config.min_phrase_chars
        if boundary in SOFT_BOUNDARY:
            return len(normalized) >= max(self.config.min_phrase_chars, self.config.max_phrase_chars // 2)
        return is_final and len(normalized) > 0

    def _with_voice_tag(self, voice_tag: str, text: str) -> str:
        content = self._normalize(text)
        if not content:
            return voice_tag
        return f"{voice_tag} {content}"
