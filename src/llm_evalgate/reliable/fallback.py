from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_fallback(
    primary: Callable[[], T],
    fallback: Callable[[], T],
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Call ``primary``; on failure call ``fallback``.

    Usage::

        result = with_fallback(
            primary=lambda: call_opus(prompt),
            fallback=lambda: call_sonnet(prompt),
        )
    """
    try:
        return primary()
    except exceptions:
        return fallback()


def with_fallback_chain(
    callables: list[Callable[[], T]],
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Try each callable in order, returning the first success.

    Raises the last exception if all callables fail.

    Usage::

        result = with_fallback_chain([
            lambda: call_opus(prompt),
            lambda: call_sonnet(prompt),
            lambda: call_haiku(prompt),
        ])
    """
    if not callables:
        raise ValueError("callables list is empty")
    last_exc: Exception | None = None
    for fn in callables:
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]
