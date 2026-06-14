"""Shared utilities for all agents."""

from __future__ import annotations

import logging

import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Errors that are safe to retry: network blips, timeouts, rate limits, 5xx.
# Non-transient errors (auth failures, bad requests) are NOT retried.
_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def create_message_with_retry(
    client: anthropic.AsyncAnthropic,
    **kwargs,
) -> anthropic.types.Message:
    """
    Calls client.messages.create(**kwargs) with automatic retry on transient errors.

    Retries up to 3 times total with exponential back-off (2 s → 4 s → 8 s, capped
    at 30 s).  After all attempts are exhausted the original exception is re-raised.

    Retried:   APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
    Not retried: AuthenticationError, BadRequestError, and any other non-transient error
    """
    return await client.messages.create(**kwargs)
