"""Gateway recovery is bounded and preserves complete prepared requests."""

import json
from contextlib import asynccontextmanager

import httpx
import pytest

from vector_core.embeddings.client import (
    EmbeddingClient,
    EmbeddingInputRejectedError,
    EmbeddingRequestTooLargeError,
    EmbeddingServiceError,
)


@pytest.mark.parametrize(
    "statuses",
    [
        [502, 200],
        [503, 200],
        [504, 200],
        [502, 503, 200],
        ["connect", 502, 200],
        [504, "timeout", 200],
        ["connect", "timeout", 503],
        [502, 503, 504],
        [502, 502, 502],
        [503, 503, 503],
        [504, 504, 504],
        [400],
        [401],
        [403],
        [404],
        [413],
        [422],
        [500],
        [502, 400],
    ],
)
async def test_gateway_retry_budget_payload_limiter_and_final_error(
    statuses, tmp_path, monkeypatch
):
    requests = []
    responses = []
    events: list[str | float] = []

    @asynccontextmanager
    async def acquire():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    async def sleep(delay):
        assert events[-1] == "exit"
        events.append(delay)

    def respond(request):
        requests.append(request)
        status = statuses[len(requests) - 1]
        if status == "connect":
            raise httpx.ConnectError("connection interrupted", request=request)
        if status == "timeout":
            raise httpx.ReadTimeout("response timed out", request=request)
        response = (
            httpx.Response(
                status,
                json={
                    "data": [
                        {"index": 1, "embedding": [3.0, 4.0]},
                        {"index": 0, "embedding": [1.0, 2.0]},
                    ]
                },
            )
            if status == 200
            else httpx.Response(status, text=f"response {len(requests)}: gateway or input error")
        )
        responses.append(response)
        return response

    monkeypatch.setattr("vector_core.utils.retry.asyncio.sleep", sleep)
    async with EmbeddingClient(
        base_url="http://example.test",
        model="generic",
        dim=2,
        profile="raw",
        document_prefix="passage: ",
        global_concurrency=0,
        cache_path=tmp_path / "cache.db",
    ) as client:
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        monkeypatch.setattr(client._request_limiter, "acquire", acquire)
        texts = [" first\n", "界🙂 second\t"]
        final_status = statuses[-1]
        if final_status == 200:
            client._circuit_failure_count = 1
            assert await client.embed_batch(texts) == [[1.0, 2.0], [3.0, 4.0]]
            assert client._circuit_failure_count == 0
        else:
            error_type = (
                EmbeddingInputRejectedError
                if final_status in {400, 422}
                else EmbeddingRequestTooLargeError
                if final_status == 413
                else EmbeddingServiceError
            )
            with pytest.raises(error_type, match=str(final_status)) as caught:
                await client.embed_batch(texts)
            cause = caught.value.__cause__
            assert isinstance(cause, httpx.HTTPStatusError)
            assert cause.request is requests[-1]
            assert cause.response is responses[-1]
            assert cause.response.status_code == final_status
            assert cause.response.text in str(caught.value)
            assert client._circuit_failure_count == int(final_status >= 500)

    assert len(requests) == len(statuses)
    assert all(request.content == requests[0].content for request in requests)
    assert json.loads(requests[0].content)["input"] == ["passage: " + text for text in texts]
    expected_events: list[str | float] = ["enter", "exit"]
    for delay in (1.0, 2.0)[: len(statuses) - 1]:
        expected_events.extend([delay, "enter", "exit"])
    assert events == expected_events
