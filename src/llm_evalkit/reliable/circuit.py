from __future__ import annotations

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # failing, rejecting calls
    HALF_OPEN = "half_open"  # probing for recovery


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """Prevent cascading failures by stopping calls to a failing service.

    Usage::

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # context manager form
        with breaker:
            result = call_llm(prompt)

        # or call form
        result = breaker.call(lambda: call_llm(prompt))

    State transitions:
    - CLOSED -> OPEN: after ``failure_threshold`` consecutive failures
    - OPEN -> HALF_OPEN: after ``recovery_timeout`` seconds
    - HALF_OPEN -> CLOSED: on first success
    - HALF_OPEN -> OPEN: on first failure
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.exceptions = exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN:
            assert self._opened_at is not None
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def _on_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def call(self, fn):
        if self.state is CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit is open. Retry after {self.recovery_timeout}s."
            )
        try:
            result = fn()
            self._on_success()
            return result
        except self.exceptions as exc:
            self._on_failure()
            raise exc

    def __enter__(self):
        if self.state is CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit is open. Retry after {self.recovery_timeout}s."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            self._on_failure()
            return False
        if exc_type is None:
            self._on_success()
        return False
