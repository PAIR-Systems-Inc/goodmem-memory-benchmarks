"""Retrieval, turn expansion, and context assembly.

GoodMem returns the reranked chunks. Because the harness holds the source
conversation locally, each selected chunk is resolved back to its turn by
dialogue id, expanded to the turns either side of it, de-duplicated, and
rendered in chronological order. That is this harness's analogue of the paper's
display-time turn expansion, and it is what gives the reader the dependency
paths that bare top-k windows tend to cut through.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .client import GoodMemClient, GoodMemError, RetrievedChunk
from .config import RunConfig
from .ingest import SpaceBinding
from . import exact_reranker, validate
from .locomo import Conversation, Question, Turn


@dataclass(slots=True)
class RetrievalRecord:
    """Everything one question's retrieval produced, saved for later stages."""

    qid: str
    conversation_id: str
    category: int
    question: str
    answer: str
    gold_evidence: tuple[str, ...]
    gold_unresolvable: tuple[str, ...]
    selected_dia_ids: list[str]
    context_dia_ids: list[str]
    context: str
    scores: list[float] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "conversation_id": self.conversation_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "gold_evidence": list(self.gold_evidence),
            "gold_unresolvable": list(self.gold_unresolvable),
            "selected_dia_ids": self.selected_dia_ids,
            "context_dia_ids": self.context_dia_ids,
            "context": self.context,
            "scores": self.scores,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "RetrievalRecord":
        return cls(
            qid=raw["qid"],
            conversation_id=raw["conversation_id"],
            category=int(raw["category"]),
            question=raw["question"],
            answer=raw["answer"],
            gold_evidence=tuple(raw.get("gold_evidence", ())),
            gold_unresolvable=tuple(raw.get("gold_unresolvable", ())),
            selected_dia_ids=list(raw.get("selected_dia_ids", ())),
            context_dia_ids=list(raw.get("context_dia_ids", ())),
            context=raw.get("context", ""),
            scores=list(raw.get("scores", ())),
        )


def records_path(config: RunConfig) -> Path:
    return config.run_dir / "retrieval.jsonl"


def ranked_dia_ids(chunks: list[RetrievedChunk]) -> list[str]:
    """Pulls the dialogue id off each reranked chunk, preserving rank order.

    Every memory this harness writes carries a ``dia_id``, so one missing means
    the space holds something this run did not put there. Dropping it quietly
    would shrink the evidence for that question with nothing to show why.

    Raises:
      GoodMemError: If a reranked chunk carries no string ``dia_id``.
    """
    out: list[str] = []
    for chunk in chunks:
        dia_id = chunk.metadata.get("dia_id")
        if not isinstance(dia_id, str) or not dia_id:
            raise GoodMemError(
                f"reranked chunk {chunk.memory_id} has no dia_id metadata; the "
                "space may hold memories from another run"
            )
        out.append(dia_id)
    return out


def expand(
    conversation: Conversation, dia_ids: list[str], radius: int
) -> tuple[list[str], list[Turn]]:
    """Resolves selected dialogue ids to turns and expands each by ``radius`` neighbours.

    Args:
      conversation: Source conversation holding every turn.
      dia_ids: Selected dialogue ids in rank order, highest relevance first.
      radius: Number of turns to include either side of each selection.

    Returns:
      The selected dialogue ids in rank order, and the expanded, de-duplicated
      turns in chronological order.

    Raises:
      GoodMemError: If a dialogue id is not present in the conversation.
    """
    selected: list[str] = []
    keep: dict[int, Turn] = {}
    unknown: list[str] = []
    for dia_id in dia_ids:
        turn = conversation.by_dia_id.get(dia_id)
        if turn is None:
            unknown.append(dia_id)
            continue
        selected.append(dia_id)
        for neighbor in conversation.neighbors(turn.turn_index, radius):
            keep[neighbor.turn_index] = neighbor
    if unknown:
        raise GoodMemError(
            f"{conversation.conversation_id}: retrieved dialogue ids absent from the "
            f"dataset: {unknown[:5]}; the index and the dataset disagree"
        )
    ordered = [keep[index] for index in sorted(keep)]
    return selected, ordered


def render_context(turns: list[Turn]) -> str:
    """Renders expanded turns as reader evidence, grouped under session headers."""
    lines: list[str] = []
    previous_session: int | None = None
    previous_index: int | None = None
    for turn in turns:
        if turn.session_id != previous_session:
            if lines:
                lines.append("")
            lines.append(
                f"--- Session {turn.session_id} "
                f"({turn.session_date.strftime('%d %B %Y, %I:%M %p')}) ---"
            )
        elif previous_index is not None and turn.turn_index != previous_index + 1:
            lines.append("[...]")
        lines.append(turn.render())
        previous_session = turn.session_id
        previous_index = turn.turn_index
    return "\n".join(lines)


def _retrieve_one(
    client: GoodMemClient,
    config: RunConfig,
    conversation: Conversation,
    binding: SpaceBinding,
    question: Question,
) -> RetrievalRecord:
    """Retrieves, expands and renders the evidence for one question."""
    chunks = client.retrieve(
        question.question,
        binding.space_id,
        dense_candidates=config.retrieval.dense_candidates,
        final_k=config.retrieval.final_k,
        reranker_id=config.models.reranker_id,
        ef_search=config.retrieval.ef_search,
        iterative_scan=config.retrieval.iterative_scan,
        max_scan_tuples=config.retrieval.max_scan_tuples,
    )
    if not chunks:
        raise GoodMemError(
            f"{question.qid}: retrieval returned no reranked chunks; an empty "
            "context would be scored as a wrong answer rather than a failure"
        )
    selected, turns = expand(
        conversation, ranked_dia_ids(chunks), config.retrieval.neighbor_turns
    )
    gold, unresolvable = conversation.resolvable_evidence(question)
    return RetrievalRecord(
        qid=question.qid,
        conversation_id=question.conversation_id,
        category=question.category,
        question=question.question,
        answer=question.answer,
        gold_evidence=gold,
        gold_unresolvable=unresolvable,
        selected_dia_ids=selected,
        context_dia_ids=[turn.dia_id for turn in turns],
        context=render_context(turns),
        scores=[round(chunk.relevance_score, 6) for chunk in chunks],
    )


def run(
    client: GoodMemClient,
    config: RunConfig,
    conversations: list[Conversation],
    bindings: dict[str, SpaceBinding],
    *,
    limit_per_conversation: int | None = None,
) -> list[RetrievalRecord]:
    """Retrieves evidence for every question and writes it to the run directory.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      conversations: Conversations whose questions to answer.
      bindings: Conversation id to space binding.
      limit_per_conversation: Optional cap on questions per conversation, for
        smoke runs.

    Returns:
      One record per question, in conversation then dataset order.
    """
    if not config.models.reranker_id:
        raise GoodMemError("models.reranker_id is not configured")

    jobs: list[tuple[Conversation, SpaceBinding, Question]] = []
    for conversation in conversations:
        binding = bindings[conversation.conversation_id]
        questions = conversation.questions
        if limit_per_conversation is not None:
            questions = questions[:limit_per_conversation]
        jobs.extend((conversation, binding, question) for question in questions)

    path = records_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Written as each question lands rather than in one go at the end: a full
    # sweep is thousands of rate-limited requests, and losing all of them to a
    # late failure would be the expensive kind of mistake.
    records: list[RetrievalRecord] = []
    with path.open("w") as handle:
        with tqdm(total=len(jobs), desc="retrieve", unit="q") as progress:
            with ThreadPoolExecutor(max_workers=config.retrieval.concurrency) as pool:
                futures = [
                    pool.submit(_retrieve_one, client, config, conversation, binding, question)
                    for conversation, binding, question in jobs
                ]
                for future in futures:
                    record = future.result()
                    records.append(record)
                    handle.write(json.dumps(record.to_json()) + "\n")
                    handle.flush()
                    progress.update(1)
    return records


def load_records(
    config: RunConfig, expected_qids: list[str] | None = None
) -> list[RetrievalRecord]:
    """Reads retrieval records written by a previous retrieve stage.

    Args:
      config: Active run configuration.
      expected_qids: When given, the question ids the file must cover exactly.
        A short file is otherwise indistinguishable from a complete one.

    Returns:
      The records, ordered by question id for determinism across stages.

    Raises:
      GoodMemError: If the file is absent.
      IncompleteStageError: If it does not cover ``expected_qids`` exactly.
    """
    path = records_path(config)
    if not path.exists():
        raise GoodMemError(f"no retrieval records at {path}; run the retrieve stage first")
    with path.open() as handle:
        records = [
            RetrievalRecord.from_json(json.loads(line)) for line in handle if line.strip()
        ]
    if expected_qids is not None:
        validate.require("retrieve", expected_qids, [record.qid for record in records])
    return sorted(records, key=lambda record: record.qid)


def collect_dense(
    client: GoodMemClient,
    config: RunConfig,
    conversations: list[Conversation],
    bindings: dict[str, SpaceBinding],
    *,
    limit_per_conversation: int | None = None,
) -> list[exact_reranker.Candidates]:
    """Fetches the dense candidate set for every question, without reranking.

    This is the first of three phases for running the paper's own reranker,
    which no endpoint here serves correctly. Writing the candidates out lets the
    reranker run in its own environment, so this package needs no GPU.
    """
    jobs: list[tuple[Conversation, SpaceBinding, Question]] = []
    for conversation in conversations:
        binding = bindings[conversation.conversation_id]
        questions = conversation.questions
        if limit_per_conversation is not None:
            questions = questions[:limit_per_conversation]
        jobs.extend((conversation, binding, question) for question in questions)

    rows: list[exact_reranker.Candidates] = []
    with tqdm(total=len(jobs), desc="dense candidates", unit="q") as progress:
        with ThreadPoolExecutor(max_workers=config.retrieval.concurrency) as pool:
            futures = {
                pool.submit(
                    client.retrieve_dense,
                    question.question,
                    binding.space_id,
                    candidates=config.retrieval.dense_candidates,
                    ef_search=config.retrieval.ef_search,
                    iterative_scan=config.retrieval.iterative_scan,
                ): (conversation, question)
                for conversation, binding, question in jobs
            }
            for future in as_completed(futures):
                conversation, question = futures[future]
                chunks = future.result()
                if not chunks:
                    raise GoodMemError(f"{question.qid}: dense retrieval returned nothing")
                rows.append(
                    exact_reranker.Candidates(
                        qid=question.qid,
                        conversation_id=question.conversation_id,
                        query=question.question,
                        dia_ids=tuple(ranked_dia_ids(chunks)),
                        documents=tuple(chunk.chunk_text for chunk in chunks),
                    )
                )
                progress.update(1)

    path = exact_reranker.candidates_path(config.run_dir)
    exact_reranker.write_candidates(path, rows)
    return rows


def assemble_from_scores(
    config: RunConfig,
    conversations: list[Conversation],
    *,
    scores_file: Path | None = None,
) -> list[RetrievalRecord]:
    """Builds reader contexts from an external reranker's scores.

    The third phase. Expansion, de-duplication and rendering are identical to
    the in-GoodMem path; only the ranking came from elsewhere.
    """
    by_conversation = {c.conversation_id: c for c in conversations}
    questions = {
        question.qid: question
        for conversation in conversations
        for question in conversation.questions
    }
    candidates = exact_reranker.read_candidates(
        exact_reranker.candidates_path(config.run_dir)
    )
    scores = exact_reranker.read_scores(
        scores_file or exact_reranker.scores_path(config.run_dir), candidates
    )

    path = records_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[RetrievalRecord] = []
    with path.open("w") as handle:
        for candidate, selected_ids in exact_reranker.iter_ranked(
            candidates, scores, config.retrieval.final_k
        ):
            conversation = by_conversation[candidate.conversation_id]
            question = questions[candidate.qid]
            selected, turns = expand(
                conversation, selected_ids, config.retrieval.neighbor_turns
            )
            gold, unresolvable = conversation.resolvable_evidence(question)
            record = RetrievalRecord(
                qid=question.qid,
                conversation_id=question.conversation_id,
                category=question.category,
                question=question.question,
                answer=question.answer,
                gold_evidence=gold,
                gold_unresolvable=unresolvable,
                selected_dia_ids=selected,
                context_dia_ids=[turn.dia_id for turn in turns],
                context=render_context(turns),
                scores=[round(s, 6) for s in sorted(scores[candidate.qid], reverse=True)][
                    : config.retrieval.final_k
                ],
            )
            records.append(record)
            handle.write(json.dumps(record.to_json()) + "\n")
    return sorted(records, key=lambda record: record.qid)
