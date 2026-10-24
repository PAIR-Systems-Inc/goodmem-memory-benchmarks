"""Running MemoryAgentBench through GoodMem, and scoring it.

Scoped to Conflict Resolution. The other capability groups ship as a verified
adapter rather than a scored run, because their contexts are enormous -- ReDial
is 5.7M characters and InfBench is a hundred books at up to 3.2M each -- and
ingesting them all would dominate everything else this repository measures. A
partial score presented as a total would be exactly the kind of quiet wrongness
the harness exists to prevent, so the gap is stated rather than papered over.

Conflict Resolution is the cheapest group and the most interesting one for a
memory system: a fact's precedence lives in its serial number, so the task asks
directly whether that survives being stored and retrieved. Single-hop and
multi-hop share a byte-identical context, so the fact pool is ingested once and
queried by both.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import jsonl
from .client import GoodMemClient, GoodMemError
from .config import RunConfig
from .memoryagentbench import Block, Task, Unit
from .prompts import MAB_QUERIES, MAB_READER_SYSTEM, MAB_READER_USER
from .ratelimit import RateLimiter

_SERIAL_PREFIX_RE = re.compile(r"^(\d+)\.\s*")


@dataclass(slots=True)
class Outcome:
    """One question's prediction and whether it matched."""

    qid: str
    source: str
    capability: str
    question: str
    answers: list[str]
    prediction: str
    correct: bool
    retrieved_serials: list[int]
    gold_serial: int | None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Outcome":
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})


def outcomes_path(config: RunConfig, capability: str) -> Path:
    return config.run_dir / f"mab-{capability}-{config.models.reader_model}.jsonl"


def score(prediction: str, answers: list[str]) -> bool:
    """Scores a prediction against the acceptable gold strings.

    MemoryAgentBench grades these by substring containment, and ``answers`` is a
    set of acceptable paraphrases rather than one string, so any of them
    appearing in the prediction counts. Matching is case-insensitive and
    whitespace-normalised; nothing else is normalised, because stripping
    punctuation or stemming would quietly loosen the metric.
    """
    if not prediction.strip():
        return False
    haystack = " ".join(prediction.lower().split())
    for answer in answers:
        needle = " ".join(str(answer).lower().split())
        if needle and needle in haystack:
            return True
    return False


def gold_serial_for(units: list[Unit], answers: list[str]) -> int | None:
    """Finds the highest serial among facts stating a gold answer.

    This is the fact the task wants selected, so recording it lets a wrong
    answer be attributed: a prediction can fail because the authoritative fact
    was never retrieved, or because it was retrieved and the reader picked an
    older one. Those are different problems and the scores alone cannot tell
    them apart.
    """
    best: int | None = None
    for unit in units:
        if unit.serial is None:
            continue
        text = unit.text.lower()
        if any(" ".join(str(a).lower().split()) in text for a in answers):
            best = unit.serial if best is None else max(best, unit.serial)
    return best


def render_context(units: list[Unit]) -> str:
    """Renders retrieved facts as a knowledge pool, in serial order.

    Serial order matters: the reader is instructed that a larger serial is
    newer, so presenting the pool in that order is what makes the rule
    applicable rather than something it has to reconstruct from scrambled lines.
    """
    ordered = sorted(units, key=lambda unit: (unit.serial is None, unit.serial or 0))
    return "[Knowledge Pool]\n" + "\n".join(unit.text for unit in ordered)


def provision_and_ingest(
    client: GoodMemClient, config: RunConfig, block: Block
) -> tuple[str, list[Unit]]:
    """Creates a space for one block and ingests its units.

    Returns:
      The space id and the units that were stored.

    Raises:
      GoodMemError: If any unit is rejected or fails to embed.
    """
    units = block.units()
    if not units:
        raise GoodMemError(f"{block.block_id}: context produced no ingestion units")

    name = f"gmbench-mab-{block.label}-{config.run_tag}"[:255]
    space = client.create_space(
        name,
        config.models.embedder_id,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
        labels={"gmbench-run": config.run_tag},
    )
    space_id = space["spaceId"]

    limiter = RateLimiter.per_minute(config.ingest.embed_requests_per_minute)
    size = config.ingest.batch_size
    requests = [
        {
            "spaceId": space_id,
            "originalContent": unit.text,
            "contentType": "text/plain",
            "metadata": {"block_id": block.block_id, "source": block.source, **unit.metadata},
        }
        for unit in units
    ]
    with tqdm(total=len(requests), desc=f"ingest {block.source}", unit="unit") as bar:
        for start in range(0, len(requests), size):
            batch = requests[start : start + size]
            limiter.acquire(len(batch))
            results = client.batch_create_memories(batch)
            if len(results) != len(batch):
                raise GoodMemError(
                    f"{block.block_id}: sent {len(batch)} units, server returned "
                    f"{len(results)} results"
                )
            rejected = [r for r in results if "memory" not in r]
            if rejected:
                raise GoodMemError(f"{block.block_id}: {len(rejected)} units rejected")
            bar.update(len(batch))

    deadline = time.monotonic() + config.ingest.ready_timeout_seconds
    while True:
        statuses = [m.get("processingStatus") for m in client.list_memories(space_id)]
        if "FAILED" in statuses:
            raise GoodMemError(
                f"{block.block_id}: {statuses.count('FAILED')} units failed to embed"
            )
        if statuses.count("COMPLETED") >= len(units):
            break
        if time.monotonic() > deadline:
            raise GoodMemError(
                f"{block.block_id}: indexing timed out at "
                f"{statuses.count('COMPLETED')} of {len(units)}"
            )
        time.sleep(config.ingest.ready_poll_seconds)

    return space_id, units


def _answer_one(
    client: GoodMemClient,
    config: RunConfig,
    block: Block,
    space_id: str,
    units: list[Unit],
    task: Task,
    llm_id: str,
) -> Outcome:
    """Retrieves, reads and scores one question."""
    by_serial = {unit.serial: unit for unit in units if unit.serial is not None}
    prediction = ""
    error: str | None = None
    retrieved: list[Unit] = []
    try:
        chunks = client.retrieve(
            task.question,
            space_id,
            dense_candidates=config.retrieval.dense_candidates,
            final_k=config.retrieval.final_k,
            reranker_id=config.models.reranker_id,
            ef_search=config.retrieval.ef_search,
            iterative_scan=config.retrieval.iterative_scan,
        )
        for chunk in chunks:
            serial = chunk.metadata.get("serial")
            unit = by_serial.get(int(serial)) if serial is not None else None
            retrieved.append(
                unit
                if unit is not None
                else Unit(
                    unit_id=chunk.memory_id,
                    text=chunk.chunk_text,
                    ordinal=-1,
                    serial=int(serial) if serial is not None else None,
                )
            )
        template = MAB_QUERIES.get(block.source)
        if template is None:
            raise GoodMemError(f"no reader template for source {block.source}")
        completion = client.chat_completion(
            llm_id,
            [
                {
                    "role": "system",
                    "content": MAB_READER_SYSTEM.format(context=render_context(retrieved)),
                },
                {
                    "role": "user",
                    "content": MAB_READER_USER.format(
                        task_query=template.format(question=task.question),
                        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                },
            ],
            model=config.models.reader_model,
            temperature=0.0,
            max_tokens=256,
        )
        prediction = completion.text
    except Exception as exc:  # noqa: BLE001 - recorded, surfaced in the report
        error = f"{type(exc).__name__}: {exc}"

    answers = list(task.answers)
    return Outcome(
        qid=task.qid,
        source=task.source,
        capability=task.capability,
        question=task.question,
        answers=answers,
        prediction=prediction,
        correct=score(prediction, answers) if not error else False,
        retrieved_serials=[u.serial for u in retrieved if u.serial is not None],
        gold_serial=gold_serial_for(units, answers),
        error=error,
    )


def run_capability(
    client: GoodMemClient,
    config: RunConfig,
    blocks: list[Block],
    capability: str,
    *,
    resume: bool = True,
) -> list[Outcome]:
    """Ingests and scores every block in one capability group.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      blocks: All loaded blocks; only those in ``capability`` are used.
      capability: Which group to run.
      resume: Keep outcomes already on disk and only do the remainder.

    Returns:
      One outcome per question in the group.
    """
    selected = [block for block in blocks if block.capability == capability]
    if not selected:
        raise GoodMemError(f"no blocks for capability {capability!r}")
    try:
        llm_id = config.models.llm_id_for(config.models.reader_model)
    except KeyError as exc:
        raise GoodMemError(str(exc)) from exc

    path = outcomes_path(config, capability)
    already = jsonl.completed_keys(path) if resume else set()
    if not resume and path.exists():
        path.unlink()

    with jsonl.ShardWriter(path) as shard:
        for block in selected:
            todo = [task for task in block.tasks if task.qid not in already]
            if not todo:
                continue
            space_id, units = provision_and_ingest(client, config, block)
            with tqdm(total=len(todo), desc=f"score {block.source}", unit="q") as bar:
                with ThreadPoolExecutor(max_workers=config.retrieval.concurrency) as pool:
                    futures = [
                        pool.submit(
                            _answer_one,
                            client,
                            config,
                            block,
                            space_id,
                            units,
                            task,
                            llm_id,
                        )
                        for task in todo
                    ]
                    for future in as_completed(futures):
                        shard.write(future.result().to_json())
                        bar.update(1)

    return [Outcome.from_json(row) for row in jsonl.read_rows(path)]


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    """Aggregates per-source accuracy plus retrieval attribution.

    The attribution is the useful part. A wrong answer here has two quite
    different causes -- the authoritative fact was never retrieved, or it was
    retrieved and an older one won -- and accuracy alone conflates them.
    """
    by_source: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        bucket = by_source.setdefault(
            outcome.source,
            {"questions": 0, "correct": 0, "gold_retrieved": 0, "errors": 0},
        )
        bucket["questions"] += 1
        bucket["correct"] += int(outcome.correct)
        bucket["errors"] += int(bool(outcome.error))
        if outcome.gold_serial is not None and outcome.gold_serial in outcome.retrieved_serials:
            bucket["gold_retrieved"] += 1

    for bucket in by_source.values():
        n = bucket["questions"] or 1
        bucket["accuracy"] = 100.0 * bucket["correct"] / n
        bucket["gold_retrieval_rate"] = 100.0 * bucket["gold_retrieved"] / n

    total = len(outcomes) or 1
    return {
        "questions": len(outcomes),
        "accuracy": 100.0 * sum(o.correct for o in outcomes) / total,
        "errors": sum(1 for o in outcomes if o.error),
        "by_source": by_source,
    }


def write_summary(config: RunConfig, capability: str, document: dict[str, Any]) -> Path:
    """Writes the capability summary and returns where it landed."""
    path = config.run_dir / f"mab-{capability}-summary-{config.models.reader_model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "memoryagentbench",
        "capability": capability,
        "run_tag": config.run_tag,
        "reader_model": config.models.reader_model,
        "retrieval": {
            "dense_candidates": config.retrieval.dense_candidates,
            "final_k": config.retrieval.final_k,
        },
        "scoring": "substring containment against the acceptable gold strings",
        "scope_note": (
            "Only this capability group was scored. The other three ship as a "
            "verified adapter; their contexts run to millions of characters per "
            "block and were out of budget here."
        ),
        **document,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
