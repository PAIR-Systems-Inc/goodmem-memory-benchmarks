"""The GoodMem requests used by the benchmark.

Model calls use the provider proxy; retrieval uses a server-sent event stream.
The stream contains both results and status messages. A successful HTTP response
therefore needs a second check: did the requested result set arrive in full?

Request pacing is shared across threads. Provider throttles found inside a
stream enter the same retry loop as HTTP throttles.
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

# HTTP 429 responses carry a Retry-After hint. Transient server failures use
# the same retry loop with an exponential backoff.
_RETRY_STATUS = frozenset({429, 502, 503, 504})

# These statuses mean retrieval was incomplete or degraded. The caller needs
# an error here, since grading partial evidence would confound retrieval with
# a service failure.
_FATAL_RETRIEVAL_STATUS = frozenset({
    "EMBEDDER_FAILED",
    "EMBEDDER_UNAVAILABLE",
    "EMBEDDER_TIMEOUT",
    "VECTOR_SEARCH_FAILED",
    "VECTOR_SEARCH_PARTIAL",
    "VECTOR_SEARCH_TIMEOUT",
    "SPACE_INACCESSIBLE",
    "SPACE_NOT_FOUND",
    "SPACE_NO_EMBEDDERS",
    "CHUNK_NOT_FOUND",
    "MEMORY_LOAD_FAILED",
    "MEMORY_CONTENT_UNAVAILABLE",
    "RERANKING_FAILED",
    "RATE_LIMITED",
    "RESOURCE_EXHAUSTED",
    "CONFIGURATION_ERROR",
    "INVALID_ARGUMENT",
    "NOT_FOUND",
    "PERMISSION_DENIED",
    "FAILED_PRECONDITION",
})

# Emitted when no LLM is configured on the post-processor, which is the normal
# case here: this harness runs the reader itself.
_BENIGN_RETRIEVAL_STATUS = frozenset({"FEATURE_DISABLED", "LLM_CAPABILITY_INFERRED"})
_MAX_ATTEMPTS = 10
_BASE_BACKOFF_SECONDS = 0.5
_MAX_BACKOFF_SECONDS = 30.0
# Stay below the development instance's observed limit of about six requests
# per second. Configuration can override these defaults for another instance.
_DEFAULT_REQUESTS_PER_SECOND = 5.0
# Proxied LLM calls are served by the upstream provider rather than by GoodMem's
# own query path, so they get their own, looser budget.
_DEFAULT_LLM_REQUESTS_PER_SECOND = 15.0

_T = TypeVar("_T")


class GoodMemError(RuntimeError):
    """A request failed or returned data the benchmark cannot use."""


@dataclass(frozen=True, slots=True)
class Completion:
    """A model response with its stop reason and token usage.

    The stop reason distinguishes a finished answer from one cut off at the
    token limit, even when both requests succeeded.
    """

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One reranked chunk plus the metadata of its parent memory."""

    memory_id: str
    chunk_text: str
    chunk_sequence_number: int
    relevance_score: float
    metadata: dict[str, Any]
    result_set_id: str = ""


def space_embedder_id(record: dict[str, Any]) -> str:
    """Pick the embedder a space record is bound to.

    A space may carry several embedders with retrieval weights; the one with the
    highest weight is the one a plain query goes through. Returns "" when the
    record names none, so a caller can fall back rather than assert.
    """
    bindings = record.get("spaceEmbedders") or []
    if not bindings:
        return str(record.get("embedderId") or "")
    best = max(bindings, key=lambda b: float(b.get("defaultRetrievalWeight") or 0.0))
    return str(best.get("embedderId") or "")


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
                # Delay new reservations on the shared schedule, then wait
                # before this worker retries.
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
        """List every space, following pagination.

        Callers use this list to find existing spaces before creating new ones.
        """
        out: list[dict[str, Any]] = []
        params: dict[str, Any] = {"pageSize": 500}
        token: str | None = None
        while True:
            if token:
                params["nextToken"] = token
            payload = self._request("GET", "/v1/spaces", params=params) or {}
            out.extend(payload.get("spaces", []))
            token = payload.get("nextToken") or None
            if not token:
                return out

    def space_embedder_id(self, space_id: str) -> str:
        """Return the embedder a space is bound to, or "" if it cannot be read.

        A space carries its own embedder, chosen when it was created. Retrieval
        uses that one, whatever the client's config says, so this is the only
        source that describes what a run actually queried.
        """
        record = self._request("GET", f"/v1/spaces/{space_id}") or {}
        return space_embedder_id(record)

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

    def memory_status(self, memory_id: str) -> str:
        """Read one memory's processing status.

        Polling by id keeps each readiness check independent of the store's size.
        Listing the whole store after every addition would make the BEAM timing
        include an increasingly expensive scan.
        """
        record = self._request("GET", f"/v1/memories/{memory_id}") or {}
        return str(record.get("processingStatus") or "")

    def delete_memory(self, memory_id: str) -> None:
        """Deletes one memory."""
        self._request("DELETE", f"/v1/memories/{memory_id}")

    def describe_model(self, kind: str, model_id: str) -> dict[str, str]:
        """Resolve an embedder, reranker or LLM registration to a model description.

        ``kind`` is the corresponding API collection name: ``embedders``,
        ``rerankers`` or ``llms``. The returned provider and model name identify the
        configuration across instances, where registration UUIDs differ. If the
        lookup fails, return empty labels so reporting can still finish.
        """
        if not model_id:
            return {"provider": "", "model": "", "display_name": ""}
        try:
            record = self._request("GET", f"/v1/{kind}/{model_id}") or {}
        except GoodMemError:
            return {"provider": "", "model": "", "display_name": ""}
        return {
            "provider": str(record.get("providerType") or ""),
            "model": str(record.get("modelIdentifier") or ""),
            "display_name": str(record.get("displayName") or ""),
        }

    # -- proxied inference -------------------------------------------------

    def chat_completion(
        self,
        llm_id: str,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Request a completion through a registered model's provider proxy.

        The proxy adds the stored credential and forwards the OpenAI-format
        messages. Return the text with its stop reason and token counts.

        ``llm_id`` selects the registration; ``model`` selects the upstream model.
        Omit ``temperature`` for models that do not accept it. ``max_tokens`` caps
        the completion length. A rejected request or empty response raises
        GoodMemError.
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
            choice = choices[0]
            text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                raise GoodMemError(
                    f"chat_completion({model}) returned empty content "
                    f"(finish_reason={choice.get('finish_reason')!r})"
                )
            usage = payload.get("usage") or {}
            return Completion(
                text=text,
                finish_reason=str(choice.get("finish_reason") or ""),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            )

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

    def retrieve_dense(
        self,
        message: str,
        space_id: str,
        *,
        candidates: int,
        ef_search: int | None = None,
        iterative_scan: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return dense candidates for a reranker running outside GoodMem.

        The candidates retain their parent memory's metadata so the chosen chunks
        can later be resolved to conversation turns.
        """
        body: dict[str, Any] = {
            "message": message,
            "spaceKeys": [{"spaceId": space_id}],
            "requestedSize": candidates,
            "fetchMemory": True,
            "fetchMemoryContent": False,
        }
        hnsw: dict[str, Any] = {}
        if ef_search:
            hnsw["efSearch"] = ef_search
        if iterative_scan:
            hnsw["iterativeScan"] = iterative_scan
        if hnsw:
            body["hnsw"] = hnsw
        return self._with_retries(
            "POST /v1/memories:retrieve (dense)",
            lambda: self._collect_dense_once(body),
        )

    def _collect_dense_once(self, body: dict[str, Any]) -> list[RetrievedChunk]:
        """Streams a dense-only retrieve, verifying the stream completed."""
        memories: list[dict[str, Any]] = []
        chunks: list[RetrievedChunk] = []
        set_id: str | None = None
        closed = False

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
                    if boundary.get("kind") == "BEGIN":
                        set_id = boundary.get("resultSetId")
                    elif boundary.get("resultSetId") and boundary["resultSetId"] == set_id:
                        closed = True
                elif "memoryDefinition" in event:
                    memories.append(event["memoryDefinition"])
                elif "retrievedItem" in event:
                    item = event["retrievedItem"].get("chunk")
                    if item is None:
                        raise GoodMemError("retrieved_item carried no chunk reference")
                    chunks.append(_to_chunk(item, memories))
                elif "status" in event:
                    _check_status(event["status"])

        if not closed:
            raise GoodMemError("dense retrieve stream ended before its result set closed")
        return chunks

    def _collect_retrieval(self, body: dict[str, Any]) -> list[RetrievedChunk]:
        """Streams a retrieve response, retrying when throttled."""
        return self._with_retries(
            "POST /v1/memories:retrieve", lambda: self._collect_retrieval_once(body)
        )

    def _collect_retrieval_once(self, body: dict[str, Any]) -> list[RetrievedChunk]:
        """Read a retrieval stream and assemble its reranked chunks.

        Track each result set from BEGIN to END and check its advertised item
        count. Raise GoodMemError for an incomplete or malformed stream, or for a
        status reporting degraded retrieval.
        """
        memories: list[dict[str, Any]] = []
        chunks: list[RetrievedChunk] = []
        rerank_set_id: str | None = None
        rerank_expected: int | None = None
        rerank_closed = False

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
                    set_id = boundary.get("resultSetId")
                    if boundary.get("kind") == "BEGIN":
                        if boundary.get("stageName") != "rerank":
                            continue
                        rerank_set_id = set_id
                        expected = boundary.get("expectedItems")
                        # Zero means "unspecified" in the wire contract.
                        rerank_expected = int(expected) if expected else None
                    elif set_id and set_id == rerank_set_id:
                        # The server names the stage only on BEGIN, so the close
                        # is matched by result-set id, which the wire contract
                        # guarantees is shared by a set's boundaries and items.
                        rerank_closed = True
                elif "memoryDefinition" in event:
                    memories.append(event["memoryDefinition"])
                elif "retrievedItem" in event:
                    item = event["retrievedItem"].get("chunk")
                    if item is None:
                        raise GoodMemError(
                            "retrieved_item carried no chunk reference; this harness "
                            "only requests chunk results"
                        )
                    chunks.append(_to_chunk(item, memories))
                elif "status" in event:
                    _check_status(event["status"])

        if rerank_set_id is None:
            raise GoodMemError(
                "retrieve returned no rerank stage; the reranker did not run, so "
                "results would be dense-only and not comparable"
            )
        if not rerank_closed:
            raise GoodMemError(
                "retrieve stream ended before the rerank result set was closed; "
                "the response was truncated"
            )

        reranked = [chunk for chunk in chunks if chunk.result_set_id == rerank_set_id]
        if rerank_expected is not None and len(reranked) != rerank_expected:
            raise GoodMemError(
                f"rerank stage promised {rerank_expected} items but delivered "
                f"{len(reranked)}"
            )
        return reranked


def _is_throttle(message: str) -> bool:
    """Recognize a provider throttle in a stream status message.

    The stream has no structured upstream status code, so this matches only
    known throttle messages. Other reranker failures remain errors.
    """
    low = message.lower()
    return (
        "429" in low
        or "too many requests" in low
        or "rate limit" in low
        or "rate_limit" in low
    )


def _check_status(status: dict[str, Any]) -> None:
    """Fails the request when the server reports a degraded result set.

    Raises:
      GoodMemError: If the status is a known degradation, or is unrecognised.
    """
    code = status.get("code", "")
    if code in _BENIGN_RETRIEVAL_STATUS:
        return
    message = str(status.get("message", ""))
    if code in _FATAL_RETRIEVAL_STATUS:
        # A provider throttle can arrive after HTTP 200, inside the stream.
        # Convert it to the signal understood by the shared retry loop.
        if _is_throttle(message):
            raise _Throttled(429, f"{code}: {message}", None)
        raise GoodMemError(f"retrieval degraded ({code}): {message}")
    # An unknown status may change the meaning of the results. Require an
    # explicit decision about it before accepting the response.
    raise GoodMemError(f"unrecognised retrieval status ({code}): {message}")


def _to_chunk(item: dict[str, Any], memories: list[dict[str, Any]]) -> RetrievedChunk:
    """Builds a chunk record, resolving its parent memory's metadata.

    Raises:
      GoodMemError: If the parent memory was not streamed before the item, which
        would leave the chunk without the metadata every downstream stage keys
        on.
    """
    chunk = item.get("chunk", {})
    index = int(item.get("memoryIndex", -1))
    if not 0 <= index < len(memories):
        raise GoodMemError(
            f"retrieved chunk referenced memory index {index} but only "
            f"{len(memories)} memory definitions were streamed"
        )
    memory = memories[index]
    return RetrievedChunk(
        memory_id=chunk.get("memoryId", memory.get("memoryId", "")),
        chunk_text=chunk.get("chunkText", ""),
        chunk_sequence_number=int(chunk.get("chunkSequenceNumber", 0)),
        relevance_score=float(item.get("relevanceScore", 0.0)),
        metadata=dict(memory.get("metadata") or {}),
        result_set_id=item.get("resultSetId", ""),
    )


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


def _iter_sse_data(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Decode the JSON payload of each SSE data frame.

    Malformed JSON raises GoodMemError. Skipping a bad frame could discard a
    retrieved item while leaving the rest of the response apparently valid.
    """
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError as exc:
            raise GoodMemError(
                f"malformed event frame in retrieve stream: {payload[:200]}"
            ) from exc
