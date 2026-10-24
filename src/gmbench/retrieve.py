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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .client import GoodMemClient, GoodMemError, RetrievedChunk
from .config import RunConfig
from .ingest import SpaceBinding
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
            selected_dia_ids=list(raw.get("selected_dia_ids", ())),
            context_dia_ids=list(raw.get("context_dia_ids", ())),
            context=raw.get("context", ""),
            scores=list(raw.get("scores", ())),
        )


def records_path(config: RunConfig) -> Path:
    return config.run_dir / "retrieval.jsonl"


def ranked_dia_ids(chunks: list[RetrievedChunk]) -> list[str]:
    """Pulls the dialogue id off each reranked chunk, preserving rank order."""
    return [
        dia_id for chunk in chunks if isinstance(dia_id := chunk.metadata.get("dia_id"), str)
    ]


def expand(
    conversation: Conversation, dia_ids: list[str], radius: int
) -> tuple[list[str], list[Turn]]:
    """Resolves selected dialogue ids to turns and expands each by ``radius`` neighbours.

    Args:
      conversation: Source conversation holding every turn.
      dia_ids: Selected dialogue ids in rank order, highest relevance first.
      radius: Number of turns to include either side of each selection.

    Returns:
      The dialogue ids that resolved to a known turn, in rank order, and the
      expanded, de-duplicated turns in chronological order.
    """
    selected: list[str] = []
    keep: dict[int, Turn] = {}
    for dia_id in dia_ids:
        turn = conversation.by_dia_id.get(dia_id)
        if turn is None:
            continue
        selected.append(dia_id)
        for neighbor in conversation.neighbors(turn.turn_index, radius):
            keep[neighbor.turn_index] = neighbor
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
    selected, turns = expand(
        conversation, ranked_dia_ids(chunks), config.retrieval.neighbor_turns
    )
    return RetrievalRecord(
        qid=question.qid,
        conversation_id=question.conversation_id,
        category=question.category,
        question=question.question,
        answer=question.answer,
        gold_evidence=question.evidence,
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


def load_records(config: RunConfig) -> list[RetrievalRecord]:
    """Reads retrieval records written by a previous retrieve stage."""
    path = records_path(config)
    if not path.exists():
        raise GoodMemError(f"no retrieval records at {path}; run the retrieve stage first")
    with path.open() as handle:
        return [RetrievalRecord.from_json(json.loads(line)) for line in handle if line.strip()]
