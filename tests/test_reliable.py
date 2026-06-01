import pytest

from llm_evalgate.reliable import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    retry,
    with_fallback,
    with_fallback_chain,
)

# --- retry ---

def test_retry_succeeds_first_try():
    calls = []

    @retry(max_attempts=3, backoff=0.0)
    def fn():
        calls.append(1)
        return "ok"

    assert fn() == "ok"
    assert len(calls) == 1


def test_retry_succeeds_on_second_attempt():
    calls = []

    @retry(max_attempts=3, backoff=0.0)
    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("not yet")
        return "ok"

    assert fn() == "ok"
    assert len(calls) == 2


def test_retry_exhausted_raises():
    @retry(max_attempts=3, backoff=0.0)
    def fn():
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        fn()


def test_retry_only_catches_specified_exceptions():
    @retry(max_attempts=3, backoff=0.0, exceptions=(ValueError,))
    def fn():
        raise TypeError("wrong type")

    with pytest.raises(TypeError):
        fn()


# --- with_fallback ---

def test_fallback_primary_succeeds():
    result = with_fallback(primary=lambda: "primary", fallback=lambda: "fallback")
    assert result == "primary"


def test_fallback_primary_fails_uses_fallback():
    def fail():
        raise RuntimeError("primary down")

    result = with_fallback(primary=fail, fallback=lambda: "fallback")
    assert result == "fallback"


def test_fallback_chain_first_succeeds():
    result = with_fallback_chain([lambda: "a", lambda: "b", lambda: "c"])
    assert result == "a"


def test_fallback_chain_first_two_fail():
    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("down")

    result = with_fallback_chain([fail, fail, lambda: "c"])
    assert result == "c"
    assert len(calls) == 2


def test_fallback_chain_all_fail():
    def fail():
        raise RuntimeError("always")

    with pytest.raises(RuntimeError):
        with_fallback_chain([fail, fail])


def test_fallback_chain_empty_raises():
    with pytest.raises(ValueError):
        with_fallback_chain([])


# --- CircuitBreaker ---

def test_circuit_starts_closed():
    cb = CircuitBreaker()
    assert cb.state is CircuitState.CLOSED


def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass
    assert cb.state is CircuitState.OPEN


def test_circuit_open_raises_circuit_error():
    cb = CircuitBreaker(failure_threshold=1)
    try:
        with cb:
            raise RuntimeError("fail")
    except RuntimeError:
        pass
    with pytest.raises(CircuitOpenError):
        with cb:
            pass


def test_circuit_resets_on_success():
    cb = CircuitBreaker(failure_threshold=3)
    try:
        with cb:
            raise RuntimeError("fail")
    except RuntimeError:
        pass
    assert cb._failure_count == 1
    with cb:
        pass
    assert cb.state is CircuitState.CLOSED
    assert cb._failure_count == 0


def test_circuit_call_form():
    cb = CircuitBreaker()
    result = cb.call(lambda: "ok")
    assert result == "ok"
