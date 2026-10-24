"""Measure how ingestion time changes as a BEAM conversation grows.

The default run adds one message, waits for it to become searchable, then adds
the next. GoodMem accepts a memory before embedding finishes, so timing ends
only when its status reaches COMPLETED. Each tick records that elapsed time
alongside the accumulated conversation length.

Keeping one message in flight measures the latency of sequential ingestion.
Batch throughput is a separate measurement.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .client import GoodMemClient, GoodMemError
from .config import RunConfig
from .ratelimit import RateLimiter

# The split the paper uses. Conversation "1" is the one with exactly 1,710
# messages; the other 34 in the split run 1,556 to 3,008.
PAPER_CONVERSATION_ID = "1"
PAPER_MESSAGE_COUNT = 1710


@dataclass(frozen=True, slots=True)
class Message:
    """One message in a BEAM conversation."""

    index: int
    role: str
    content: str
    session: int
    time_anchor: str | None

    def render(self) -> str:
        """Renders the message as ingested text."""
        stamp = f"[{self.time_anchor}] " if self.time_anchor else ""
        return f"{stamp}{self.role}: {self.content}"


@dataclass(frozen=True, slots=True)
class Conversation:
    """One ordered BEAM conversation."""

    conversation_id: str
    title: str
    category: str
    messages: tuple[Message, ...]

    @property
    def approximate_tokens(self) -> int:
        """Estimate context length at four characters per token.

        For the reference conversation this gives about 1.06M tokens; an o200k
        tokenizer counts 970,963. Use the estimate to label the time curve.
        """
        return sum(len(message.content) for message in self.messages) // 4


def load(path: Path, conversation_id: str = PAPER_CONVERSATION_ID) -> Conversation:
    """Loads one conversation from the BEAM parquet split.

    Args:
      path: Path to a BEAM split parquet, e.g. ``1M-00000-of-00001.parquet``.
      conversation_id: Which conversation to take.

    Returns:
      The conversation with its messages flattened into ingestion order.

    Raises:
      GoodMemError: If pyarrow is unavailable or the conversation is absent.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GoodMemError("BEAM needs pyarrow; add it to the environment") from exc

    table = pq.read_table(path)
    ids = table.column("conversation_id").to_pylist()
    if conversation_id not in ids:
        raise GoodMemError(
            f"conversation {conversation_id!r} not in {path.name}; present: {ids[:8]}"
        )
    row = ids.index(conversation_id)
    chat = table.column("chat").to_pylist()[row]
    seed = table.column("conversation_seed").to_pylist()[row] or {}

    messages: list[Message] = []
    for session_index, session in enumerate(chat):
        for raw in session:
            messages.append(
                Message(
                    # The dataset's own id is globally monotonic across the
                    # conversation, so it is the ingestion order; fall back to
                    # positional order if it is ever absent.
                    index=int(raw["id"]) if raw.get("id") is not None else len(messages),
                    role=str(raw.get("role") or "unknown"),
                    content=str(raw.get("content") or ""),
                    session=session_index,
                    time_anchor=raw.get("time_anchor"),
                )
            )
    messages.sort(key=lambda message: message.index)
    return Conversation(
        conversation_id=conversation_id,
        title=str(seed.get("title") or ""),
        category=str(seed.get("category") or ""),
        messages=tuple(messages),
    )


@dataclass(frozen=True, slots=True)
class Tick:
    """Cumulative cost after one message has been ingested and is queryable."""

    message_index: int
    cumulative_messages: int
    cumulative_chars: int
    approximate_cumulative_tokens: int
    add_seconds: float
    ready_seconds: float
    cumulative_seconds: float


def _wait_ready(
    client: GoodMemClient, memory_id: str, timeout_seconds: float
) -> float:
    """Wait for a memory to reach COMPLETED and return the elapsed seconds.

    Raise GoodMemError on timeout or embedding failure. A failed message cannot
    contribute a successful ingestion time.
    """
    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        status = client.memory_status(memory_id)
        if status == "COMPLETED":
            return time.monotonic() - started
        if status == "FAILED":
            raise GoodMemError(f"memory {memory_id} failed to embed")
        if time.monotonic() > deadline:
            raise GoodMemError(
                f"memory {memory_id} still {status or 'unknown'} after "
                f"{timeout_seconds:.0f}s"
            )
        time.sleep(0.05)


def measure(
    client: GoodMemClient,
    config: RunConfig,
    conversation: Conversation,
    *,
    space_id: str,
    limit: int | None = None,
    poll_every: int = 1,
) -> Iterator[Tick]:
    """Ingest messages in order, yielding cumulative time at each readiness check.

    The supplied space must be empty. ``limit`` caps the message count for a
    short run. Keep ``poll_every=1`` for the paper's protocol; larger values
    wait only for every Nth memory and the final memory.
    """
    limiter = RateLimiter.per_minute(config.ingest.embed_requests_per_minute)
    messages = conversation.messages[:limit] if limit else conversation.messages
    cumulative_seconds = 0.0
    cumulative_chars = 0

    with tqdm(total=len(messages), desc="beam ingest", unit="msg") as progress:
        for position, message in enumerate(messages, start=1):
            body = {
                "spaceId": space_id,
                "originalContent": message.render(),
                "contentType": "text/plain",
                "metadata": {
                    "conversation_id": conversation.conversation_id,
                    "message_index": message.index,
                    "role": message.role,
                    "session": message.session,
                },
            }
            limiter.acquire()
            add_started = time.monotonic()
            results = client.batch_create_memories([body])
            if len(results) != 1 or "memory" not in results[0]:
                raise GoodMemError(f"message {message.index} was not accepted: {results}")
            memory_id = results[0]["memory"]["memoryId"]
            add_seconds = time.monotonic() - add_started

            ready_seconds = 0.0
            if position % poll_every == 0 or position == len(messages):
                ready_seconds = _wait_ready(
                    client, memory_id, config.ingest.ready_timeout_seconds
                )

            cumulative_chars += len(message.content)
            cumulative_seconds += add_seconds + ready_seconds
            progress.update(1)

            if ready_seconds or position == len(messages):
                yield Tick(
                    message_index=message.index,
                    cumulative_messages=position,
                    cumulative_chars=cumulative_chars,
                    approximate_cumulative_tokens=cumulative_chars // 4,
                    add_seconds=add_seconds,
                    ready_seconds=ready_seconds,
                    cumulative_seconds=cumulative_seconds,
                )


def ticks_path(config: RunConfig) -> Path:
    return config.run_dir / "beam-ingest.jsonl"


def summary_path(config: RunConfig) -> Path:
    return config.run_dir / "beam-summary.json"


def write_summary(config: RunConfig, conversation: Conversation, ticks: list[Tick]) -> Path:
    """Write the ingestion summary and return its path."""
    final = ticks[-1] if ticks else None
    document: dict[str, Any] = {
        "benchmark": "beam-1m",
        "run_tag": config.run_tag,
        "conversation_id": conversation.conversation_id,
        "conversation_title": conversation.title,
        "messages_total": len(conversation.messages),
        "messages_ingested": final.cumulative_messages if final else 0,
        "approximate_tokens": final.approximate_cumulative_tokens if final else 0,
        "cumulative_seconds": final.cumulative_seconds if final else 0.0,
        "cumulative_hours": (final.cumulative_seconds / 3600.0) if final else 0.0,
        "mean_seconds_per_message": (
            final.cumulative_seconds / final.cumulative_messages if final else 0.0
        ),
        "embedder_id": config.models.embedder_id,
        "method": (
            "one message per Add, outer concurrency one, each Add followed by a poll "
            "until the memory reports COMPLETED"
        ),
    }
    path = summary_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")

    with ticks_path(config).open("w") as handle:
        for tick in ticks:
            handle.write(json.dumps(asdict(tick)) + "\n")
    return path
