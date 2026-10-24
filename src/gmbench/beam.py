"""BEAM: sequential ingestion cost over a conversation approaching one million tokens.

This is not an accuracy benchmark. The paper streams one ordered 1,710-message
conversation into each memory system a message at a time, waiting for each Add
to be ready before the next begins, and plots cumulative ingest time against
context length. That isolates what a memory system costs to fill, which is where
systems that run an LLM over every message diverge sharply from ones that do not.

Two things make a number here honest, and both are easy to get wrong:

- **Readiness, not acceptance.** GoodMem's create returns once the memory is
  persisted, with embedding still queued. Timing that would measure the API, not
  the system, so each Add is followed by a poll until the memory reports
  COMPLETED. The paper does the same, calling it "Add-to-ready".
- **One message at a time.** Outer concurrency is one by construction: no
  batching, no overlap. A pipelined ingest would be a different, better number
  measuring something the paper did not measure.
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
        """A rough token count, four characters per token.

        Deliberately rough: the point of this benchmark is elapsed time against
        context length, and the paper's own x-axis is approximate. A real
        tokenizer count for this conversation is about 971k against the ~950k
        this estimate gives, so it is close enough to label an axis and not
        close enough to quote as a measurement.
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
    client: GoodMemClient, space_id: str, expected: int, timeout_seconds: float
) -> float:
    """Blocks until the space reports ``expected`` completed memories.

    Returns:
      Seconds spent waiting.

    Raises:
      GoodMemError: On timeout, or if a memory failed to embed.
    """
    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        statuses = [
            memory.get("processingStatus") for memory in client.list_memories(space_id)
        ]
        if "FAILED" in statuses:
            raise GoodMemError(
                f"a memory failed to embed after {statuses.count('COMPLETED')} completed; "
                "ingestion cost cannot be measured against a partial store"
            )
        if statuses.count("COMPLETED") >= expected:
            return time.monotonic() - started
        if time.monotonic() > deadline:
            raise GoodMemError(
                f"readiness timed out at {statuses.count('COMPLETED')} of {expected}"
            )
        time.sleep(0.25)


def measure(
    client: GoodMemClient,
    config: RunConfig,
    conversation: Conversation,
    *,
    space_id: str,
    limit: int | None = None,
    poll_every: int = 1,
) -> Iterator[Tick]:
    """Streams the conversation one message at a time, yielding cumulative cost.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      conversation: The conversation to stream.
      space_id: An empty space to ingest into.
      limit: Stop after this many messages, for a short run.
      poll_every: Check readiness every N messages instead of every one. One
        matches the paper; a larger value trades fidelity for wall-clock when
        only the shape of the curve is wanted.

    Yields:
      One tick per readiness check.
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
            add_seconds = time.monotonic() - add_started

            ready_seconds = 0.0
            if position % poll_every == 0 or position == len(messages):
                ready_seconds = _wait_ready(
                    client, space_id, position, config.ingest.ready_timeout_seconds
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
    """Writes the ingestion-cost summary and returns where it landed."""
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
