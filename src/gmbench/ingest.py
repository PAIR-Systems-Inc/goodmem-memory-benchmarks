"""Space provisioning and turn-level ingestion.

Each LoCoMo conversation becomes its own space, which keeps retrieval scoped to
the conversation a question is about without relying on a metadata filter, and
matches the paper's per-conversation index. Each conversation turn becomes its
own memory, carrying its dialogue id, session, speaker and session date as
metadata so evidence can be scored and neighbouring turns recovered exactly.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .client import GoodMemClient, GoodMemError
from .config import RunConfig
from .locomo import Conversation, Turn
from .ratelimit import RateLimiter

SPACE_LABEL_KEY = "gmbench-run"


@dataclass(frozen=True, slots=True)
class SpaceBinding:
    """Maps a conversation to the space holding its turns."""

    conversation_id: str
    space_id: str
    space_name: str
    turn_count: int


def manifest_path(config: RunConfig) -> Path:
    """Returns the space-manifest path, which is keyed to the ingest, not the run."""
    return config.ingest_dir / "spaces.json"


def load_manifest(config: RunConfig) -> dict[str, SpaceBinding]:
    """Reads the space manifest written by a previous provision/ingest run."""
    path = manifest_path(config)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {key: SpaceBinding(**value) for key, value in raw.items()}


def save_manifest(config: RunConfig, bindings: dict[str, SpaceBinding]) -> None:
    path = manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in bindings.items()}, indent=2))


def _memory_request(space_id: str, conversation_id: str, turn: Turn) -> dict[str, Any]:
    """Builds one batchCreate entry for a single conversation turn."""
    return {
        "spaceId": space_id,
        "originalContent": turn.render(),
        "contentType": "text/plain",
        "metadata": {
            "conversation_id": conversation_id,
            "dia_id": turn.dia_id,
            "speaker": turn.speaker,
            "session_id": turn.session_id,
            "turn_index": turn.turn_index,
            "session_date": turn.session_date.isoformat(),
        },
    }


def provision(
    client: GoodMemClient, config: RunConfig, conversations: list[Conversation]
) -> dict[str, SpaceBinding]:
    """Creates one space per conversation, reusing any already in the manifest.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      conversations: Conversations to provision spaces for.

    Returns:
      Conversation id to space binding, including pre-existing entries.
    """
    if not config.models.embedder_id:
        raise GoodMemError("models.embedder_id is not configured")

    bindings = load_manifest(config)
    for conversation in conversations:
        if conversation.conversation_id in bindings:
            continue
        name = f"gmbench-locomo-{conversation.conversation_id}-{config.run_tag}"
        space = client.create_space(
            name,
            config.models.embedder_id,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=config.chunking.separators,
            labels={SPACE_LABEL_KEY: config.run_tag},
        )
        bindings[conversation.conversation_id] = SpaceBinding(
            conversation_id=conversation.conversation_id,
            space_id=space["spaceId"],
            space_name=name,
            turn_count=len(conversation.turns),
        )
        save_manifest(config, bindings)
    return bindings


def _ingest_turns(
    client: GoodMemClient,
    config: RunConfig,
    conversation: Conversation,
    binding: SpaceBinding,
    turns: list[Turn],
    progress: tqdm,
    limiter: RateLimiter,
) -> int:
    """Uploads the given turns of one conversation, in paced batches."""
    requests = [
        _memory_request(binding.space_id, conversation.conversation_id, turn)
        for turn in turns
    ]
    size = config.ingest.batch_size
    created = 0
    for start in range(0, len(requests), size):
        batch = requests[start : start + size]
        limiter.acquire(len(batch))
        results = client.batch_create_memories(batch)
        failed = [r for r in results if "memory" not in r]
        if failed:
            raise GoodMemError(
                f"{conversation.conversation_id}: {len(failed)} of {len(batch)} "
                f"memories rejected; first error: {failed[0]}"
            )
        created += len(results)
        progress.update(len(batch))
    return created


def _indexed_dia_ids(client: GoodMemClient, space_id: str) -> tuple[set[str], list[str]]:
    """Returns the dialogue ids that are indexed, and the ids of failed memories.

    Args:
      client: Connected GoodMem client.
      space_id: Space to inspect.

    Returns:
      The set of dialogue ids whose memory reached COMPLETED, and the memory ids
      of every memory that did not and is not still in flight.
    """
    done: set[str] = set()
    failed: list[str] = []
    for memory in client.list_memories(space_id):
        dia_id = (memory.get("metadata") or {}).get("dia_id")
        status = memory.get("processingStatus")
        if status == "COMPLETED":
            if isinstance(dia_id, str):
                done.add(dia_id)
        elif status == "FAILED":
            failed.append(memory["memoryId"])
    return done, failed


def ingest(
    client: GoodMemClient,
    config: RunConfig,
    conversations: list[Conversation],
    bindings: dict[str, SpaceBinding],
    *,
    skip_populated: bool = True,
) -> dict[str, int]:
    """Ingests every turn, then repairs anything that failed, until all are indexed.

    Ingestion is paced to stay under the embedding provider's requests-per-minute
    ceiling, and any memory that fails anyway is deleted and re-created, because
    a partially indexed conversation would silently depress retrieval coverage.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      conversations: Conversations to ingest.
      bindings: Conversation id to space binding from provisioning.
      skip_populated: Unused; kept so earlier call sites stay valid. Missing
        turns are always computed from what is actually indexed.

    Returns:
      Conversation id to the number of turns indexed.

    Raises:
      GoodMemError: If turns are still missing after the configured repair rounds.
    """
    del skip_populated
    limiter = RateLimiter.per_minute(config.ingest.embed_requests_per_minute)

    for round_index in range(config.ingest.repair_rounds + 1):
        work: list[tuple[Conversation, SpaceBinding, list[Turn]]] = []
        for conversation in conversations:
            binding = bindings[conversation.conversation_id]
            done, failed = _indexed_dia_ids(client, binding.space_id)
            for memory_id in failed:
                client.delete_memory(memory_id)
            missing = [turn for turn in conversation.turns if turn.dia_id not in done]
            if missing:
                work.append((conversation, binding, missing))

        if not work:
            break

        pending_turns = sum(len(turns) for _, _, turns in work)
        label = "ingest turns" if round_index == 0 else f"repair round {round_index}"
        if round_index and pending_turns:
            tqdm.write(f"{label}: re-creating {pending_turns} turns that did not index")

        with tqdm(total=pending_turns, desc=label, unit="turn") as progress:
            with ThreadPoolExecutor(max_workers=config.ingest.space_concurrency) as pool:
                futures = [
                    pool.submit(
                        _ingest_turns,
                        client,
                        config,
                        conversation,
                        binding,
                        turns,
                        progress,
                        limiter,
                    )
                    for conversation, binding, turns in work
                ]
                for future in futures:
                    future.result()

        _wait_for_settle(client, config, conversations, bindings)

    counts: dict[str, int] = {}
    still_missing: dict[str, int] = {}
    for conversation in conversations:
        done, _ = _indexed_dia_ids(client, bindings[conversation.conversation_id].space_id)
        counts[conversation.conversation_id] = len(done)
        shortfall = len(conversation.turns) - len(done)
        if shortfall > 0:
            still_missing[conversation.conversation_id] = shortfall

    if still_missing:
        raise GoodMemError(
            f"turns still unindexed after {config.ingest.repair_rounds} repair rounds: "
            f"{still_missing}; lower ingest.embed_requests_per_minute and re-run"
        )
    return counts


def _wait_for_settle(
    client: GoodMemClient,
    config: RunConfig,
    conversations: list[Conversation],
    bindings: dict[str, SpaceBinding],
) -> None:
    """Waits until no memory is left PENDING or PROCESSING in any space.

    A FAILED memory is a settled outcome, so this returns once nothing is still
    in flight and leaves the decision about failures to the repair loop.
    """
    deadline = time.monotonic() + config.ingest.ready_timeout_seconds
    total_expected = sum(len(c.turns) for c in conversations)

    with tqdm(total=total_expected, desc="indexing", unit="turn") as progress:
        seen = 0
        while True:
            in_flight = 0
            settled = 0
            for conversation in conversations:
                space_id = bindings[conversation.conversation_id].space_id
                for memory in client.list_memories(space_id):
                    if memory.get("processingStatus") in {"PENDING", "PROCESSING"}:
                        in_flight += 1
                    else:
                        settled += 1
            if settled > seen:
                progress.update(settled - seen)
                seen = settled
            if in_flight == 0:
                return
            if time.monotonic() > deadline:
                raise GoodMemError(
                    f"indexing timed out with {in_flight} memories still in flight"
                )
            time.sleep(config.ingest.ready_poll_seconds)
