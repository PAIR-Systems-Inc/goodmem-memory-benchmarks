"""Run the four MemoryAgentBench capability groups through GoodMem.

Each dataset block supplies a context and the questions to ask of it. We index
that context in its own space, retrieve evidence for each question, then apply
the reader prompt and grader chosen for its source. The adapter in
memoryagentbench.py defines how contexts divide into memories.

Most graders return a binary score. Book summaries receive fractional keypoint
coverage, so aggregation uses Outcome.partial throughout. Source scores are
combined according to capability_score at the end of the run.
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
from .memoryagentbench import GRADERS, Block, Task, Unit
from .prompts import (
    MAB_QUERIES,
    MAB_READER_SYSTEM,
    MAB_READER_USER,
    MAB_SUMMARY_JUDGE_SYSTEM,
    MAB_SUMMARY_JUDGE_USER,
)
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
    # Summarisation is graded by keypoint coverage, which is fractional; every
    # other source is binary and stores 0.0 or 1.0 here. Aggregation uses this
    # rather than the boolean, so coverage is not rounded away.
    partial: float = 0.0
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Outcome":
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})


def outcome_id(block: Block, task: Task) -> str:
    """Qualify a dataset question id with its block id.

    Question ids repeat across blocks: EventQA, for example, uses no0 through
    no99 in each block. Writing and resuming must use the same qualified id.
    """
    return f"{block.block_id}:{task.qid}"


def outcomes_path(config: RunConfig, capability: str) -> Path:
    return config.run_dir / f"mab-{capability}-{config.models.reader_model}.jsonl"


def score(prediction: str, answers: list[str]) -> bool:
    """Check whether a prediction contains any acceptable answer.

    Matching ignores case and repeated whitespace. Word boundaries keep short
    answers distinct: "yes" must not match "yesterday", nor "50" match "150".
    Punctuation is otherwise preserved.
    """
    if not prediction.strip():
        return False
    haystack = " ".join(prediction.lower().split())
    for answer in answers:
        needle = " ".join(str(answer).lower().split())
        if not needle:
            continue
        pattern = re.escape(needle)
        # Only anchor on a side that begins or ends with a word character;
        # a needle like "(2001)" has no boundary to anchor against.
        left = r"\b" if needle[0].isalnum() else ""
        right = r"\b" if needle[-1].isalnum() else ""
        if re.search(f"{left}{pattern}{right}", haystack):
            return True
    return False


_ENTITY_RE = re.compile(r"\b(?:[A-Z0-9][\w'’.-]*)(?:\s+(?:of|the|de|van|von|al)\s+|\s+)?(?:[A-Z0-9][\w'’.-]*)*")
# Words that appear in the templated questions themselves and so carry no
# entity information.
_TEMPLATE_WORDS = frozenset(
    """based provided knowledge pool what which who where when whose how is are was were
    the a an of in on at to for with and or name current sport religion country city
    location associated affiliated founded created born died spouse famous passed away
    come into existence did does do you your my i""".split()
)


def question_entities(question: str) -> list[str]:
    """Extracts the entity mentions a templated question is asking about.

    The conflict-resolution questions are generated from a handful of relation
    templates, so what distinguishes one from another is the entity name. Those
    are the capitalised runs left after the template vocabulary is removed.

    Args:
      question: The question text.

    Returns:
      Candidate entity strings, longest first, lowercased.
    """
    candidates: list[str] = []
    for match in _ENTITY_RE.finditer(question):
        phrase = match.group(0).strip(" ?.,")
        if not phrase:
            continue
        words = [w for w in phrase.split() if w.lower() not in _TEMPLATE_WORDS]
        if not words:
            continue
        rebuilt = " ".join(words)
        # A single short token is as likely to be a stray capitalised word as an
        # entity, so require either several words or a substantial one.
        if len(words) > 1 or len(rebuilt) >= 4:
            candidates.append(rebuilt.lower())
    return sorted(set(candidates), key=len, reverse=True)


def gold_serial_for(
    units: list[Unit], answers: list[str], question: str
) -> int | None:
    """Find the newest fact naming the question's entity and an acceptable answer.

    An answer such as "basketball" may occur in facts about many entities, so
    both must match. Among matching facts, the largest serial takes precedence.

    Return None if no single fact meets both conditions. This is common for
    multi-hop questions, whose answer may belong to an intermediate entity.
    The result supports retrieval attribution; it does not grade the answer.
    """
    entities = question_entities(question)
    if not entities:
        return None
    wanted = [" ".join(str(a).lower().split()) for a in answers if str(a).strip()]
    best: int | None = None
    for unit in units:
        if unit.serial is None:
            continue
        text = " ".join(unit.text.lower().split())
        if not any(answer in text for answer in wanted):
            continue
        if not any(entity in text for entity in entities):
            continue
        best = unit.serial if best is None else max(best, unit.serial)
    return best


def render_context(units: list[Unit]) -> str:
    """Present retrieved facts in serial order.

    The reader prompt gives larger serials precedence when facts conflict.
    """
    ordered = sorted(units, key=lambda unit: (unit.serial is None, unit.serial or 0))
    return "[Knowledge Pool]\n" + "\n".join(unit.text for unit in ordered)


def _indexed_unit_ids(
    client: GoodMemClient, space_id: str
) -> tuple[set[str], list[str]]:
    """Returns the unit ids that reached COMPLETED, and failed memory ids."""
    done: set[str] = set()
    failed: list[str] = []
    for memory in client.list_memories(space_id):
        unit_id = (memory.get("metadata") or {}).get("unit_id")
        status = memory.get("processingStatus")
        if status == "COMPLETED":
            if isinstance(unit_id, str):
                done.add(unit_id)
        elif status == "FAILED":
            failed.append(memory["memoryId"])
    return done, failed


def _settle_units(
    client: GoodMemClient, config: RunConfig, space_id: str, block: Block
) -> None:
    """Waits until nothing in one block's space is PENDING or PROCESSING.

    A FAILED memory is a settled outcome; the repair loop decides what to do
    about it.
    """
    deadline = time.monotonic() + config.ingest.ready_timeout_seconds
    while True:
        statuses = [m.get("processingStatus") for m in client.list_memories(space_id)]
        in_flight = sum(s in {"PENDING", "PROCESSING"} for s in statuses)
        if in_flight == 0:
            return
        if time.monotonic() > deadline:
            raise GoodMemError(
                f"{block.block_id}: indexing timed out with {in_flight} memories in flight"
            )
        time.sleep(config.ingest.ready_poll_seconds)


def space_name(block: Block, run_tag: str) -> str:
    """Name a block's space using its block id and run tag.

    Blocks can share a source label while containing different documents.
    Each needs its own space to keep those documents out of other questions.
    """
    return f"gmbench-mab-{block.block_id}-{run_tag}"[:255]


def provision_and_ingest(
    client: GoodMemClient, config: RunConfig, block: Block
) -> tuple[str, list[Unit]]:
    """Creates a space for one block and ingests its units.

    Returns:
      The space id and the units that were stored.

    Raises:
      GoodMemError: If a unit is rejected or remains unindexed after repairs.
    """
    units = block.units()
    if not units:
        raise GoodMemError(f"{block.block_id}: context produced no ingestion units")

    name = space_name(block, config.run_tag)

    # Reuse a matching space, then let the repair loop fill any gaps.
    existing = next((s for s in client.list_spaces() if s.get("name") == name), None)
    space_id = ""
    if existing is not None:
        space_id = existing["spaceId"]
        stored = client.list_memories(space_id)
        owners = {(m.get("metadata") or {}).get("block_id") for m in stored}
        # Verify ownership in the stored metadata before reusing the space.
        # Its name alone does not establish which documents it contains.
        if owners - {block.block_id}:
            tqdm.write(
                f"{block.block_id}: space holds other blocks {sorted(owners)}; rebuilding it"
            )
            client.delete_space(space_id)
            space_id = ""
        else:
            # Completed memories can be kept even if other units need repair.
            completed = sum(m.get("processingStatus") == "COMPLETED" for m in stored)
            if completed >= len(units):
                tqdm.write(f"{block.source}: reusing {len(units)} facts already stored")
                return space_id, units
            tqdm.write(f"{block.source}: resuming at {completed} of {len(units)} units")

    if not space_id:
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

    # A settled pass may contain failed embeddings. Delete those memories and
    # submit the missing units again, up to the configured repair limit.
    for round_index in range(config.ingest.repair_rounds + 1):
        done, failed = _indexed_unit_ids(client, space_id)
        for memory_id in failed:
            client.delete_memory(memory_id)
        missing = [unit for unit in units if unit.unit_id not in done]
        if not missing:
            break

        label = f"ingest {block.source}" if round_index == 0 else f"repair {block.source} #{round_index}"
        if round_index:
            tqdm.write(f"{label}: re-creating {len(missing)} units that did not index")

        size = config.ingest.batch_size
        requests = [
            {
                "spaceId": space_id,
                "originalContent": unit.text,
                "contentType": "text/plain",
                "metadata": {
                    "block_id": block.block_id,
                    "source": block.source,
                    **unit.metadata,
                },
            }
            for unit in missing
        ]
        with tqdm(total=len(requests), desc=label, unit="unit") as bar:
            for offset in range(0, len(requests), size):
                batch = requests[offset : offset + size]
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

        _settle_units(client, config, space_id, block)

    done, _ = _indexed_unit_ids(client, space_id)
    shortfall = [unit.unit_id for unit in units if unit.unit_id not in done]
    if shortfall:
        raise GoodMemError(
            f"{block.block_id}: {len(shortfall)} of {len(units)} units still unindexed "
            f"after {config.ingest.repair_rounds} repair rounds, e.g. {shortfall[:3]}; "
            f"lower ingest.embed_requests_per_minute and re-run"
        )
    return space_id, units


def _answer_one(
    client: GoodMemClient,
    config: RunConfig,
    block: Block,
    space_id: str,
    units: list[Unit],
    task: Task,
    llm_id: str,
    titles: dict[str, str] | None = None,
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
            # A book summary is asked for at 1,000 to 1,200 words; capping it at
            # a short-answer budget would truncate it and grade the truncation.
            max_tokens=4096 if GRADERS.get(block.source) == "summary" else 256,
        )
        prediction = completion.text
    except Exception as exc:  # noqa: BLE001 - recorded, surfaced in the report
        error = f"{type(exc).__name__}: {exc}"

    answers = list(task.answers)
    grader = GRADERS.get(block.source, "contains")
    partial = 0.0
    if error:
        correct = False
    elif grader == "label":
        correct = score_label(prediction, answers)
    elif grader == "choice":
        correct = score_choice(prediction, answers)
    elif grader == "recall":
        correct = score_recall(prediction, answers, titles or {})
    elif grader == "summary":
        coverage = score_summary(client, config, prediction, list(task.keypoints))
        if coverage is None:
            error = "summary judge unavailable or unparseable"
            correct = False
        else:
            partial = coverage
            # Preserve fractional coverage for aggregation. The binary field
            # records whether at least half the keypoints were covered.
            correct = partial >= 0.5
    else:
        correct = score(prediction, answers)
    if grader != "summary":
        partial = 1.0 if correct else 0.0

    return Outcome(
        qid=outcome_id(block, task),
        source=task.source,
        capability=task.capability,
        question=task.question,
        answers=answers,
        prediction=prediction,
        correct=correct,
        partial=partial,
        retrieved_serials=[u.serial for u in retrieved if u.serial is not None],
        gold_serial=gold_serial_for(units, answers, task.question),
        error=error,
    )


def _retry_errored(path: Path) -> set[str]:
    """Remove errored outcomes from the shard and return the completed ids.

    Rewriting first leaves room for each retried question to append one result
    without duplicating its failed attempt.
    """
    if not path.exists():
        return set()
    kept = [row for row in jsonl.read_rows(path) if not row.get("error")]
    dropped = sum(1 for _ in jsonl.read_rows(path)) - len(kept)
    if dropped:
        tqdm.write(f"resume: retrying {dropped} questions that errored")
        with path.open("w") as handle:
            for row in kept:
                handle.write(json.dumps(row) + "\n")
    return {str(row["qid"]) for row in kept}


def run_capability(
    client: GoodMemClient,
    config: RunConfig,
    blocks: list[Block],
    capability: str,
    *,
    dataset_dir: Path,
    resume: bool = True,
) -> list[Outcome]:
    """Ingests and scores every block in one capability group.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      blocks: All loaded blocks; only those in ``capability`` are used.
      capability: Which group to run.
      dataset_dir: Dataset root, for the entity map ReDial scoring needs.
      resume: Keep outcomes already on disk and only do the remainder.

    Returns:
      One outcome per question in the group.
    """
    selected = [block for block in blocks if block.capability == capability]
    if not selected:
        raise GoodMemError(f"no blocks for capability {capability!r}")

    # Check prompt coverage before spending requests on ingestion.
    missing = sorted({b.source for b in selected if b.source not in MAB_QUERIES})
    if missing:
        raise GoodMemError(f"no reader template for sources: {missing}")
    ungraded = sorted({b.source for b in selected if b.source not in GRADERS})
    if ungraded:
        raise GoodMemError(f"no grader for sources: {ungraded}")
    try:
        llm_id = config.models.llm_id_for(config.models.reader_model)
    except KeyError as exc:
        raise GoodMemError(str(exc)) from exc

    titles: dict[str, str] = {}
    if any(GRADERS.get(b.source) == "recall" for b in selected):
        entity_file = dataset_dir / "entity2id.json"
        if not entity_file.exists():
            raise GoodMemError(
                f"{entity_file} is missing; ReDial gold answers are integer movie "
                "ids and cannot be scored without it"
            )
        titles = movie_titles(json.loads(entity_file.read_text()))
    if any(GRADERS.get(b.source) == "summary" for b in selected) and not config.models.judge_llm_id:
        raise GoodMemError(
            "models.judge_llm_id is not configured; InfBench summaries are graded "
            "by keypoint coverage, which needs a judge"
        )

    path = outcomes_path(config, capability)
    if not resume and path.exists():
        path.unlink()
    already = _retry_errored(path) if resume else set()

    with jsonl.ShardWriter(path) as shard:
        for block in selected:
            todo = [
                task for task in block.tasks if outcome_id(block, task) not in already
            ]
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
                            titles,
                        )
                        for task in todo
                    ]
                    for future in as_completed(futures):
                        shard.write(future.result().to_json())
                        bar.update(1)

    return [Outcome.from_json(row) for row in jsonl.read_rows(path)]


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    """Compute source accuracy and, where possible, attribute retrieval failures.

    For questions with an identifiable gold fact, track whether it reached the
    reader. This distinguishes missing evidence from an incorrect answer made
    with that evidence available.
    """
    by_source: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        bucket = by_source.setdefault(
            outcome.source,
            {
                "questions": 0,
                "correct": 0,
                "gold_retrieved": 0,
                "gold_identifiable": 0,
                "errors": 0,
            },
        )
        bucket["questions"] += 1
        bucket["correct"] += outcome.partial
        bucket["errors"] += int(bool(outcome.error))
        if outcome.gold_serial is not None:
            bucket["gold_identifiable"] += 1
            if outcome.gold_serial in outcome.retrieved_serials:
                bucket["gold_retrieved"] += 1

    for bucket in by_source.values():
        n = bucket["questions"] or 1
        bucket["accuracy"] = 100.0 * bucket["correct"] / n
        identifiable = bucket["gold_identifiable"]
        # Conditioned on the questions where a single fact both names the entity
        # and states the answer. For multi-hop that is almost none, because the
        # answer-bearing fact names the bridging entity instead.
        bucket["gold_retrieval_rate"] = (
            100.0 * bucket["gold_retrieved"] / identifiable if identifiable else None
        )
        bucket["gold_identifiable_rate"] = 100.0 * identifiable / n

    total = len(outcomes) or 1
    return {
        "questions": len(outcomes),
        "accuracy": capability_score(by_source),
        "accuracy_question_weighted": 100.0 * sum(o.partial for o in outcomes) / total,
        "aggregation": (
            "per the paper's appendix D.5: the mean of source scores, except "
            "test-time learning, where the five in-context sources are averaged "
            "first and then combined equally with ReDial"
        ),
        "errors": sum(1 for o in outcomes if o.error),
        "by_source": by_source,
    }


def capability_score(by_source: dict[str, dict[str, Any]]) -> float:
    """Combine source scores according to appendix D.5.

    Each source has equal weight within its capability, except in Test-Time
    Learning. There the five in-context-learning sources form one group with
    the same weight as ReDial:

        TTL = (ReDial + mean(ICL_1..ICL_5)) / 2

    ``by_source`` maps source names to dictionaries containing ``accuracy``.
    An empty mapping returns 0.0.
    """
    if not by_source:
        return 0.0
    icl = [b["accuracy"] for name, b in by_source.items() if name.startswith("icl_")]
    redial = [b["accuracy"] for name, b in by_source.items() if name.startswith("recsys_")]
    if icl and redial:
        return (sum(redial) / len(redial) + sum(icl) / len(icl)) / 2
    scores = [b["accuracy"] for b in by_source.values()]
    return sum(scores) / len(scores)


def write_summary(config: RunConfig, capability: str, document: dict[str, Any]) -> Path:
    """Write the capability summary and return its path."""
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


# -- per-source graders ---------------------------------------------------

_LABEL_RE = re.compile(r"label\s*:?\s*(-?\d+)", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"(-?\d+)\s*$")
_CHOICE_PREFIX_RE = re.compile(r"^\s*([A-Z])[.)]\s*")
_URI_TITLE_RE = re.compile(r"/resource/([^>]+)>?\s*$")
_RECOMMENDATION_RE = re.compile(r"^\s*\d+\s*[.):]\s*(.+?)\s*$", re.MULTILINE)

REDIAL_RECALL_AT = 5


def score_label(prediction: str, answers: list[str]) -> bool:
    """Parse the predicted numeric label and compare it for exact equality.

    The reader is asked for ``label: <int>``. Substring matching would confuse
    labels such as 7 and 71.
    """
    if not prediction.strip():
        return False
    match = _LABEL_RE.search(prediction) or _TRAILING_NUMBER_RE.search(prediction.strip())
    if not match:
        return False
    predicted = match.group(1)
    return any(predicted == str(a).strip() for a in answers)


def score_choice(prediction: str, answers: list[str]) -> bool:
    """Compare multiple-choice answers after removing their option letters.

    DetectiveQA's gold letters can disagree with the options in its prompts,
    so the answer text determines the score.
    """
    if not prediction.strip():
        return False
    stripped: list[str] = []
    for answer in answers:
        text = _CHOICE_PREFIX_RE.sub("", str(answer)).strip()
        if text:
            stripped.append(text)
    return score(prediction, stripped) if stripped else False


def movie_titles(entity2id: dict[str, int]) -> dict[str, str]:
    """Inverts the entity map into movie id to title.

    ReDial gold answers are integer DBpedia ids, which are meaningless against a
    reader that was asked for film names. The id resolves to a resource URI
    whose last segment is the title.
    """
    out: dict[str, str] = {}
    for uri, identifier in entity2id.items():
        match = _URI_TITLE_RE.search(uri)
        if match:
            out[str(identifier)] = match.group(1).replace("_", " ")
    return out


def score_recall(
    prediction: str, answers: list[str], titles: dict[str, str], at: int = REDIAL_RECALL_AT
) -> bool:
    """Scores a recommendation list by whether a gold film is in the top ``at``.

    Args:
      prediction: The reader's numbered recommendation list.
      answers: Gold movie ids as strings.
      titles: Movie id to title, from :func:`movie_titles`.
      at: How many recommendations count, matching the benchmark's Recall@k.

    Returns:
      True when any gold film appears among the first ``at`` recommendations.
    """
    wanted = [titles.get(str(a), "") for a in answers]
    wanted = [w for w in wanted if w]
    if not wanted:
        return False
    items = _RECOMMENDATION_RE.findall(prediction)[:at]
    if not items:
        # An unnumbered reply still gets a chance, on its first lines.
        items = [line for line in prediction.splitlines() if line.strip()][:at]
    haystack = " ".join(" ".join(items).lower().split())
    for title in wanted:
        needle = " ".join(title.lower().split())
        # Drop a DBpedia disambiguator such as "(film)" before matching.
        needle = re.sub(r"\s*\([^)]*\)\s*$", "", needle)
        if needle and needle in haystack:
            return True
    return False


def score_summary(
    client: GoodMemClient, config: RunConfig, summary: str, keypoints: list[str]
) -> float | None:
    """Ask the judge what fraction of the required plot points a summary covers.

    Return a score in [0, 1], or None if the request fails or the response
    cannot be parsed. A valid score of zero means no keypoints were covered;
    None records an evaluation error.
    """
    points = [str(k).strip() for k in keypoints if str(k).strip()]
    if not points or not summary.strip():
        return 0.0
    listed = "\n".join(f"{i + 1}. {point}" for i, point in enumerate(points))
    try:
        text = client.chat_completion(
            config.models.judge_llm_id,
            [
                {"role": "system", "content": MAB_SUMMARY_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": MAB_SUMMARY_JUDGE_USER.format(
                        summary=summary[:60000], keypoints=listed
                    ),
                },
            ],
            model=config.models.judge_model,
        ).text
    except Exception:  # noqa: BLE001 - surfaced as an error, not as zero coverage
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        verdict = json.loads(match.group(0))
        covered = float(verdict.get("covered", 0))
        total = float(verdict.get("total") or len(points))
    except (ValueError, TypeError):
        return None
    if total <= 0:
        return None
    return max(0.0, min(1.0, covered / total))
