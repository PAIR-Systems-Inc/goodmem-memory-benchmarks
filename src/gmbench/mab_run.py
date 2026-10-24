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
    """Returns the identity of one question, for both writing and resuming.

    The dataset's qa_pair_ids are not unique across blocks: all five EventQA
    blocks number their questions no0..no99 while asking 500 different
    questions, so the block has to qualify them.

    This is one function because the two uses have to agree, and they did not.
    Outcomes were written qualified while resume compared the raw dataset id, so
    a resumed run matched nothing, re-ran every question and appended to the
    existing shard -- doubling both the rows and the question total the summary
    reported.
    """
    return f"{block.block_id}:{task.qid}"


def outcomes_path(config: RunConfig, capability: str) -> Path:
    return config.run_dir / f"mab-{capability}-{config.models.reader_model}.jsonl"


def score(prediction: str, answers: list[str]) -> bool:
    """Scores a prediction against the acceptable gold strings.

    MemoryAgentBench grades by containment, and ``answers`` is a set of
    acceptable paraphrases rather than one string, so any of them counts.
    Matching is case-insensitive and whitespace-normalised, and anchored at word
    boundaries -- bare substring containment is wrong on the short answers these
    sources use, where ``yes`` matches "yesterday" and ``50`` matches "150",
    turning a wrong answer into a scored point. Nothing else is normalised;
    stripping punctuation or stemming would loosen the metric silently.
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
    """Finds the newest fact that both names the question's entity and states a gold answer.

    Matching on the answer alone does not work: a gold string like "basketball"
    appears in dozens of facts about unrelated entities, and the highest-serial
    one among those is almost never the fact the question is about. The fact that
    matters is the latest one about *this* entity, which is what the reader is
    supposed to prefer over its earlier, conflicting sibling.

    Returns ``None`` when no single fact satisfies both -- which is the normal
    case for multi-hop, where the answer-bearing fact names the bridging entity
    rather than the one in the question. That is a property of the task, not a
    measurement failure, and the caller reports it as such.

    Args:
      units: Every fact in the pool.
      answers: Acceptable gold answer strings.
      question: The question, used to identify its entity.

    Returns:
      The highest serial among qualifying facts, or None if there are none.
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

    # Idempotent: a re-run should reuse a fact pool that is already stored
    # rather than fail on the name, and should rebuild one that is incomplete
    # rather than query a partial store.
    existing = next((s for s in client.list_spaces() if s.get("name") == name), None)
    if existing is not None:
        space_id = existing["spaceId"]
        statuses = [m.get("processingStatus") for m in client.list_memories(space_id)]
        if statuses.count("COMPLETED") >= len(units) and "FAILED" not in statuses:
            tqdm.write(f"{block.source}: reusing {len(units)} facts already stored")
            return space_id, units
        tqdm.write(
            f"{block.source}: existing space holds {statuses.count('COMPLETED')} of "
            f"{len(units)} facts; rebuilding it"
        )
        client.delete_space(space_id)

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
            # A summary is credited by coverage; the binary flag marks the
            # majority covered, so the two numbers do not disagree in spirit.
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

    # Preflight. A source without a reader template fails every one of its
    # questions identically, which costs a full ingest and a few hundred
    # requests to discover something knowable up front.
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
    already = jsonl.completed_keys(path) if resume else set()
    if not resume and path.exists():
        path.unlink()

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
    """Aggregates per-source accuracy plus retrieval attribution.

    The attribution is the useful part. A wrong answer here has two quite
    different causes -- the authoritative fact was never retrieved, or it was
    retrieved and an older one won -- and accuracy alone conflates them.
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
    """Aggregates source scores the way the paper's appendix D.5 specifies.

    Three capabilities "average their constituent source scores". Test-Time
    Learning does not: its five in-context sources are averaged together first,
    and that average is then weighted equally against ReDial --

        TTL = (ReDial + mean(ICL_1..ICL_5)) / 2

    A flat mean over its six sources instead gives the five in-context sources
    five sixths of the weight rather than a half, which on our run reads 63.83
    where the specified formula reads 47.50. Neither is a mean over questions,
    which would let a source's question count decide its influence.

    Args:
      by_source: Source name to its bucket, each carrying an ``accuracy``.

    Returns:
      The capability score, or 0.0 when there are no sources.
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


# -- per-source graders ---------------------------------------------------

_LABEL_RE = re.compile(r"label\s*:?\s*(-?\d+)", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"(-?\d+)\s*$")
_CHOICE_PREFIX_RE = re.compile(r"^\s*([A-Z])[.)]\s*")
_URI_TITLE_RE = re.compile(r"/resource/([^>]+)>?\s*$")
_RECOMMENDATION_RE = re.compile(r"^\s*\d+\s*[.):]\s*(.+?)\s*$", re.MULTILINE)

REDIAL_RECALL_AT = 5


def score_label(prediction: str, answers: list[str]) -> bool:
    """Scores an in-context-learning answer by exact numeric label.

    Containment is wrong here. The labels are bare integers, so a prediction of
    ``7`` would be scored correct against a gold of ``71`` and vice versa. The
    reader is instructed to emit ``label: <int>``, so the label is parsed and
    compared as a number.
    """
    if not prediction.strip():
        return False
    match = _LABEL_RE.search(prediction) or _TRAILING_NUMBER_RE.search(prediction.strip())
    if not match:
        return False
    predicted = match.group(1)
    return any(predicted == str(a).strip() for a in answers)


def score_choice(prediction: str, answers: list[str]) -> bool:
    """Scores a multiple-choice answer on its text rather than its letter.

    The released DetectiveQA items are not internally consistent: the first
    item's gold is ``"C. The Brandt couple"`` while option C in its own prompt
    reads ``"Suicide"``. The letter therefore cannot be trusted, so both sides
    are stripped to the answer text and compared on that.
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
    """Scores a book summary by how many required keypoints it covers.

    Returns:
      Coverage in [0, 1], or None when the judge could not be reached or its
      reply could not be parsed. None rather than 0.0 on purpose: a judge
      outage and a summary that covers nothing are the same number otherwise,
      and the first is a broken measurement while the second is a result.

    The benchmark ships keypoints as the rubric for this source, so coverage is
    the metric; containment against a 1,000-word reference summary would
    measure nothing.
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
