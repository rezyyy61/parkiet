from __future__ import annotations


class RealtimeTTSError(Exception):
    pass


class SessionNotFoundError(RealtimeTTSError):
    pass


class SessionClosedError(RealtimeTTSError):
    pass


class BackendSynthesisError(RealtimeTTSError):
    pass
