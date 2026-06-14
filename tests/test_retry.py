"""
Tests for agents/utils.py — create_message_with_retry

Verifies:
  - Transient errors (connection, timeout, rate-limit, 5xx) are retried
  - Non-transient errors (auth, bad request) fail immediately without retry
  - On eventual success after transient failures, the response is returned
  - After exhausting all attempts the original exception is re-raised
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_client() -> MagicMock:
    client = MagicMock(spec=anthropic.AsyncAnthropic)
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


def _fake_response(text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.stop_reason = "end_turn"
    return resp


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestCreateMessageWithRetry:

    @pytest.mark.asyncio
    async def test_returns_response_on_success(self):
        """First-attempt success — no retries needed."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        client.messages.create.return_value = _fake_response("hello")

        result = await create_message_with_retry(client, model="m", max_tokens=10,
                                                  messages=[])
        assert result.content[0].text == "hello"
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_api_connection_error(self):
        """APIConnectionError (network blip) triggers a retry."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        client.messages.create.side_effect = [
            anthropic.APIConnectionError(request=MagicMock()),
            _fake_response("recovered"),
        ]

        with patch("agents.utils.wait_exponential", return_value=MagicMock(sleep=AsyncMock())):
            with patch("tenacity.nap.sleep", new=AsyncMock()):
                result = await create_message_with_retry(
                    client, model="m", max_tokens=10, messages=[]
                )

        assert client.messages.create.call_count == 2
        assert result.content[0].text == "recovered"

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self):
        """APITimeoutError is retried."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        client.messages.create.side_effect = [
            anthropic.APITimeoutError(request=MagicMock()),
            _fake_response("done"),
        ]

        with patch("tenacity.nap.sleep", new=AsyncMock()):
            result = await create_message_with_retry(
                client, model="m", max_tokens=10, messages=[]
            )

        assert client.messages.create.call_count == 2
        assert result.content[0].text == "done"

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_error(self):
        """RateLimitError (429) is retried."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        rate_limit_err = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        client.messages.create.side_effect = [rate_limit_err, _fake_response("ok")]

        with patch("tenacity.nap.sleep", new=AsyncMock()):
            result = await create_message_with_retry(
                client, model="m", max_tokens=10, messages=[]
            )

        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_internal_server_error(self):
        """InternalServerError (5xx) is retried."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        server_err = anthropic.InternalServerError(
            message="server error",
            response=MagicMock(status_code=500, headers={}),
            body={},
        )
        client.messages.create.side_effect = [server_err, _fake_response("ok")]

        with patch("tenacity.nap.sleep", new=AsyncMock()):
            result = await create_message_with_retry(
                client, model="m", max_tokens=10, messages=[]
            )

        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_authentication_error(self):
        """AuthenticationError is non-transient — must not be retried."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        auth_err = anthropic.AuthenticationError(
            message="invalid api key",
            response=MagicMock(status_code=401, headers={}),
            body={},
        )
        client.messages.create.side_effect = auth_err

        with pytest.raises(anthropic.AuthenticationError):
            await create_message_with_retry(client, model="m", max_tokens=10, messages=[])

        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_retry_bad_request_error(self):
        """BadRequestError (400) is non-transient — must not be retried."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        bad_req = anthropic.BadRequestError(
            message="bad request",
            response=MagicMock(status_code=400, headers={}),
            body={},
        )
        client.messages.create.side_effect = bad_req

        with pytest.raises(anthropic.BadRequestError):
            await create_message_with_retry(client, model="m", max_tokens=10, messages=[])

        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_reraises_after_max_attempts_exhausted(self):
        """After 3 failed attempts the last transient error is re-raised."""
        from agents.utils import create_message_with_retry

        client = _mock_client()
        client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )

        with patch("tenacity.nap.sleep", new=AsyncMock()):
            with pytest.raises(anthropic.APIConnectionError):
                await create_message_with_retry(
                    client, model="m", max_tokens=10, messages=[]
                )

        assert client.messages.create.call_count == 3
