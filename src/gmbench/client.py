"""Thin GoodMem REST client covering the surface this harness needs.

The harness talks to the REST API rather than the generated SDK so the request
bodies it sends are visible in one place and can be quoted in a write-up. The
retrieve endpoint answers with a server-sent event stream whose ``data`` frames
are ``RetrieveMemoryEvent`` messages in JSON form.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

import httpx

from .ratelimit import RateLimiter

CHAT_POST_PROCESSOR = "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory"

# The server rate-limits per principal and answers 429 with a Retry-After hint.
# Benchmark runs push thousands of requests through, so backing off correctly is
# load-bearing rather than defensive.
_RETRY_STATUS = frozenset({429, 502, 503, 504})
_MAX_ATTEMPTS = 10
_BASE_BACKOFF_SECONDS = 0.5
_MAX_BACKOFF_SECONDS = 30.0
# The development instance sustains roughly six requests a second before it
# starts answering 429. Pacing under that is what makes a thousands-of-requests
# run finish; retries alone just rediscover the ceiling on every call.
_DEFAULT_REQUESTS_PER_SECOND = 5.0
# Proxied LLM calls are served by the upstream provider rather than by GoodMem's
# own query path, so they get their own, looser budget.
_DEFAULT_LLM_REQUESTS_PER_SECOND = 15.0

_T = TypeVar("_T")


class GoodMemError(RuntimeError):
    """Raised when the server answers with a non-success status."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One reranked chunk plus the metadata of its parent memory."""

    memory_id: str
    chunk_text: str
    chunk_sequence_number: int
    relevance_score: float
    metadata: dict[str, Any]


class GoodMemClient:
    """Synchronous GoodMem REST client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        tls_verify: bool = True,
        timeout_seconds: float = 300.0,
        max_attempts: int = _MAX_ATTEMPTS,
        requests_per_second: float = _DEFAULT_REQUESTS_PER_SECOND,
        llm_requests_per_second: float = _DEFAULT_LLM_REQUESTS_PER_SECOND,
    ) -> None:
        if not api_key:
            raise GoodMemError("GOODMEM_API_KEY is not set")
        self._max_attempts = max(1, max_attempts)
        self._limiter = RateLimiter(requests_per_second)
        self._llm_limiter = RateLimiter(llm_requests_per_second)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            verify=tls_verify,
            timeout=httpx.Timeout(timeout_seconds, connect=30.0),
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GoodMemClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- plumbing ----------------------------------------------------------

    def _with_retries(
        self,
        describe: str,
        attempt_fn: Callable[[], _T],
        limiter: RateLimiter | None = None,
    ) -> _T:
        """Runs ``attempt_fn``, retrying throttled and transient failures.

        Args:
          describe: Short label used in the error message if every attempt fails.
          attempt_fn: Callable that performs one attempt. It should raise
            ``_Throttled`` to request a retry.

        Returns:
          Whatever ``attempt_fn`` returns on its first successful attempt.

        Raises:
          GoodMemError: If every attempt was throttled or transient.
        """
        bucket = limiter or self._limiter
        last: _Throttled | None = None
        for attempt in range(self._max_attempts):
            bucket.acquire()
            try:
                return attempt_fn()
            except _Throttled as throttled:
                last = throttled
                if attempt == self._max_attempts - 1:
                    break
                delay = _backoff_seconds(attempt, throttled.retry_after)
                # Stand the whole pool down, not just this worker: the budget is
                # shared, so letting the others keep firing guarantees the retry
                # arrives into an empty bucket again.
                bucket.pause(delay)
                time.sleep(delay)
        assert last is not None
        raise GoodMemError(
            f"{describe} still failing after {self._max_attempts} attempts: "
            f"{last.status} {last.body[:300]}"
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        def attempt() -> Any:
            response = self._client.request(method, path, **kwargs)
            if response.status_code in _RETRY_STATUS:
                raise _Throttled(
                    response.status_code, response.text[:600], _retry_after(response)
                )
            if response.status_code >= 400:
                raise GoodMemError(
                    f"{method} {path} -> {response.status_code}: {response.text[:600]}"
                )
            if response.status_code == 204 or not response.content:
                return None
            return response.json()

        return self._with_retries(f"{method} {path}", attempt)

    # -- spaces ------------------------------------------------------------

    def create_space(
        self,
        name: str,
        embedder_id: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
        separators: tuple[str, ...],
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Creates a space bound to one embedder with character-based chunking."""
        body = {
            "name": name,
            "labels": labels or {},
            "spaceEmbedders": [{"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": list(separators),
                    "keepStrategy": "KEEP_END",
                    "lengthMeasurement": "CHARACTER_COUNT",
                }
            },
        }
        return self._request("POST", "/v1/spaces", json=body)

    def list_spaces(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/v1/spaces") or {}
        return payload.get("spaces", [])

    def delete_space(self, space_id: str) -> None:
        self._request("DELETE", f"/v1/spaces/{space_id}")

    # -- memories ----------------------------------------------------------

    def batch_create_memories(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Creates memories in one call; each entry needs spaceId and originalContent."""
        payload = self._request("POST", "/v1/memories:batchCreate", json={"requests": requests})
        return (payload or {}).get("results", [])

    def list_memories(
        self, space_id: str, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Lists every memory in a space, following pagination.

        Args:
          space_id: Space to list.
          status: Optional processing-status filter, e.g. ``COMPLETED``.

        Returns:
          Memory records including their metadata and processing status.
        """
        out: list[dict[str, Any]] = []
        params: dict[str, Any] = {"pageSize": 500}
        if status:
            params["statusFilter"] = status
        token: str | None = None
        while True:
            if token:
                params["nextToken"] = token
            payload = self._request("GET", f"/v1/spaces/{space_id}/memories", params=params) or {}
            out.extend(payload.get("memories", []))
            token = payload.get("nextToken") or None
            if not token:
                return out

    def count_memories(self, space_id: str, *, status: str | None = "COMPLETED") -> int:
        """Counts memories in a space, optionally restricted to one processing status."""
        return len(self.list_memories(space_id, status=status))

    def delete_memory(self, memory_id: str) -> None:
        """Deletes one memory."""
        self._request("DELETE", f"/v1/memories/{memory_id}")

    # -- proxied inference -------------------------------------------------

    def chat_completion(
        self,
        llm_id: str,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Runs a chat completion through GoodMem's provider proxy.

        The proxy forwards to the registered upstream endpoint and injects the
        stored credential, so the benchmark can hold the reader and judge
        prompts byte-identical to the paper's without the provider key ever
        reaching this process.

        Args:
          llm_id: UUID of the registered LLM to route through.
          messages: Chat messages in OpenAI format.
          model: Model identifier to request.
          temperature: Sampling temperature, when the model accepts one.
          max_tokens: Completion token cap.

        Returns:
          The assistant message content, or an empty string if absent.

        Raises:
          GoodMemError: If the proxy or the upstream provider rejects the call.
        """
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        def attempt() -> str:
            response = self._client.post(
                f"/v1/proxy/llms/{llm_id}/chat/completions", json=body
            )
            if response.status_code in _RETRY_STATUS:
                raise _Throttled(
                    response.status_code, response.text[:600], _retry_after(response)
                )
            if response.status_code >= 400:
                raise GoodMemError(
                    f"chat_completion({model}) -> {response.status_code}: "
                    f"{response.text[:400]}"
                )
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                raise GoodMemError(f"chat_completion({model}) returned no choices")
            return (choices[0].get("message") or {}).get("content") or ""

        return self._with_retries(
            f"POST /v1/proxy/llms/{llm_id}/chat/completions", attempt, self._llm_limiter
        )

    # -- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        message: str,
        space_id: str,
        *,
        dense_candidates: int,
        final_k: int,
        reranker_id: str,
        space_filter: str | None = None,
        ef_search: int | None = None,
        iterative_scan: str | None = None,
        max_scan_tuples: int | None = None,
        chronological_resort: bool = False,
    ) -> list[RetrievedChunk]:
        """Runs dense retrieval then cross-encoder reranking in one request.

        ``dense_candidates`` sizes the vector-search stage and ``final_k`` caps
        what survives reranking, giving the paper's two-stage selection.

        Args:
          message: The retrieval query, sent verbatim to embedder and reranker.
          space_id: Space to search.
          dense_candidates: Vector-search depth (``requestedSize``).
          final_k: Number of reranked chunks to keep (``max_results``).
          reranker_id: UUID of the registered reranker.
          space_filter: Optional metadata filter expression.
          ef_search: HNSW candidate-list size for this request.
          iterative_scan: HNSW iterative-scan mode.
          max_scan_tuples: Cap on tuples visited during iterative scans.
          chronological_resort: Ask the server to re-sort by memory creation time.

        Returns:
          Reranked chunks in the order the server emitted them.
        """
        space_key: dict[str, Any] = {"spaceId": space_id}
        if space_filter:
            space_key["filter"] = space_filter

        config: dict[str, Any] = {"reranker_id": reranker_id, "max_results": final_k}
        if chronological_resort:
            config["chronological_resort"] = True

        body: dict[str, Any] = {
            "message": message,
            "spaceKeys": [space_key],
            "requestedSize": dense_candidates,
            "fetchMemory": True,
            "fetchMemoryContent": False,
            "postProcessor": {"name": CHAT_POST_PROCESSOR, "config": config},
        }
        hnsw: dict[str, Any] = {}
        if ef_search:
            hnsw["efSearch"] = ef_search
        if iterative_scan:
            hnsw["iterativeScan"] = iterative_scan
        if max_scan_tuples:
            hnsw["maxScanTuples"] = max_scan_tuples
        if hnsw:
            body["hnsw"] = hnsw

        return self._collect_retrieval(body)

    def _collect_retrieval(self, body: dict[str, Any]) -> list[RetrievedChunk]:
        """Streams a retrieve response, retrying when throttled."""
        return self._with_retries(
            "POST /v1/memories:retrieve", lambda: self._collect_retrieval_once(body)
        )

    def _collect_retrieval_once(self, body: dict[str, Any]) -> list[RetrievedChunk]:
        """Streams one retrieve response and assembles the reranked result set."""
        memories: list[dict[str, Any]] = []
        chunks: list[RetrievedChunk] = []
        rerank_set_id: str | None = None
        vector_set_id: str | None = None

        with self._client.stream(
            "POST", "/v1/memories:retrieve", json=body, headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code in _RETRY_STATUS:
                response.read()
                raise _Throttled(
                    response.status_code, response.text[:600], _retry_after(response)
                )
            if response.status_code >= 400:
                response.read()
                raise GoodMemError(
                    f"retrieve -> {response.status_code}: {response.text[:600]}"
                )
            for event in _iter_sse_data(response.iter_lines()):
                if "resultSetBoundary" in event:
                    boundary = event["resultSetBoundary"]
                    if boundary.get("kind") != "BEGIN":
                        continue
                    stage = boundary.get("stageName", "")
                    if stage == "rerank":
                        rerank_set_id = boundary.get("resultSetId")
                    else:
                        vector_set_id = boundary.get("resultSetId")
                elif "memoryDefinition" in event:
                    memories.append(event["memoryDefinition"])
                elif "retrievedItem" in event:
                    item = event["retrievedItem"].get("chunk")
                    if not item:
                        continue
                    chunks.append(_to_chunk(item, memories))
                elif "status" in event:
                    code = event["status"].get("code")
                    if code in {"RERANKING_FAILED", "EMBEDDER_FAILED", "VECTOR_SEARCH_FAILED"}:
                        raise GoodMemError(f"retrieval degraded: {event['status']}")

        if rerank_set_id is None:
            raise GoodMemError("retrieve returned no rerank stage; reranker did not run")
        del vector_set_id
        return [chunk for chunk in chunks if chunk.metadata.get("_result_set") == rerank_set_id]


class _Throttled(Exception):
    """Internal signal that a request should be retried after a pause."""

    def __init__(self, status: int, body: str, retry_after: float | None) -> None:
        super().__init__(f"{status}: {body[:200]}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float | None:
    """Reads the Retry-After header as seconds, when present and numeric."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    """Returns how long to wait, preferring the server's hint over our own guess."""
    if retry_after is not None:
        return min(_MAX_BACKOFF_SECONDS, retry_after + random.uniform(0.0, 0.25))
    exponential = _BASE_BACKOFF_SECONDS * (2**attempt)
    return min(_MAX_BACKOFF_SECONDS, exponential) * random.uniform(0.6, 1.4)


def _to_chunk(item: dict[str, Any], memories: list[dict[str, Any]]) -> RetrievedChunk:
    """Builds a chunk record, resolving its parent memory's metadata."""
    chunk = item.get("chunk", {})
    index = int(item.get("memoryIndex", -1))
    memory = memories[index] if 0 <= index < len(memories) else {}
    metadata = dict(memory.get("metadata") or {})
    metadata["_result_set"] = item.get("resultSetId")
    return RetrievedChunk(
        memory_id=chunk.get("memoryId", memory.get("memoryId", "")),
        chunk_text=chunk.get("chunkText", ""),
        chunk_sequence_number=int(chunk.get("chunkSequenceNumber", 0)),
        relevance_score=float(item.get("relevanceScore", 0.0)),
        metadata=metadata,
    )


def _iter_sse_data(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Yields decoded ``data:`` payloads from a server-sent event stream."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue
