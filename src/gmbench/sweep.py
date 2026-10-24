"""Grid sweep over retrieval depth and expansion radius, scored on coverage.

Two facts make this cheap. Reranked order is stable under truncation, so one
retrieval pass at the largest ``final_k`` also answers every smaller one. And
expansion is a client-side step over the local dataset, so every
``neighbor_turns`` value is free once the selections are known. A whole grid
therefore costs one retrieval pass per distinct ``ef_search``, not one per cell.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from .client import GoodMemClient
from .config import RunConfig
from .coverage import compute
from .ingest import SpaceBinding
from .locomo import Conversation, Question
from .retrieve import RetrievalRecord, expand, ranked_dia_ids, render_context


@dataclass(frozen=True, slots=True)
class SweepCell:
    """Coverage for one grid point."""

    ef_search: int
    final_k: int
    neighbor_turns: int
    questions: int
    full_coverage: float
    evidence_recall: float
    mean_context_turns: float
    mean_context_chars: float


def stratified_sample(
    conversations: list[Conversation], size: int
) -> list[tuple[Conversation, Question]]:
    """Samples questions spread evenly across conversations and categories.

    Args:
      conversations: Conversations to sample from.
      size: Target number of questions; the whole set is used if smaller.

    Returns:
      Conversation and question pairs, deterministic for a given input.
    """
    buckets: dict[tuple[str, int], list[tuple[Conversation, Question]]] = defaultdict(list)
    for conversation in conversations:
        for question in conversation.questions:
            buckets[(conversation.conversation_id, question.category)].append(
                (conversation, question)
            )

    keys = sorted(buckets)
    picked: list[tuple[Conversation, Question]] = []
    depth = 0
    while len(picked) < size:
        added = False
        for key in keys:
            if depth < len(buckets[key]):
                picked.append(buckets[key][depth])
                added = True
                if len(picked) == size:
                    return picked
        if not added:
            break
        depth += 1
    return picked


def _probe(
    client: GoodMemClient,
    config: RunConfig,
    bindings: dict[str, SpaceBinding],
    sample: list[tuple[Conversation, Question]],
    ef_search: int,
    max_final_k: int,
) -> list[tuple[Conversation, Question, list[str]]]:
    """Retrieves once per question at the largest depth in the grid."""
    out: list[tuple[Conversation, Question, list[str]]] = []
    with tqdm(total=len(sample), desc=f"probe ef={ef_search}", unit="q") as progress:
        for conversation, question in sample:
            chunks = client.retrieve(
                question.question,
                bindings[conversation.conversation_id].space_id,
                dense_candidates=config.retrieval.dense_candidates,
                final_k=max_final_k,
                reranker_id=config.models.reranker_id,
                ef_search=ef_search,
                iterative_scan=config.retrieval.iterative_scan,
                max_scan_tuples=config.retrieval.max_scan_tuples,
            )
            out.append((conversation, question, ranked_dia_ids(chunks)))
            progress.update(1)
    return out


def _score(
    probes: list[tuple[Conversation, Question, list[str]]],
    final_k: int,
    neighbor_turns: int,
) -> tuple[int, float, float, float, float]:
    """Scores one grid cell from cached probes, with no further retrieval."""
    records: list[RetrievalRecord] = []
    for conversation, question, ranked in probes:
        selected, turns = expand(conversation, ranked[:final_k], neighbor_turns)
        records.append(
            RetrievalRecord(
                qid=question.qid,
                conversation_id=question.conversation_id,
                category=question.category,
                question=question.question,
                answer=question.answer,
                gold_evidence=question.evidence,
                selected_dia_ids=selected,
                context_dia_ids=[turn.dia_id for turn in turns],
                context=render_context(turns),
            )
        )
    overall = compute(records)[-1]
    return (
        overall.questions,
        overall.full_coverage,
        overall.evidence_recall,
        overall.mean_context_turns,
        overall.mean_context_chars,
    )


def run(
    client: GoodMemClient,
    config: RunConfig,
    conversations: list[Conversation],
    bindings: dict[str, SpaceBinding],
    *,
    sample_size: int,
    final_ks: tuple[int, ...],
    neighbor_turn_values: tuple[int, ...],
    ef_searches: tuple[int, ...],
) -> list[SweepCell]:
    """Sweeps the grid and returns one cell per combination.

    Args:
      client: Connected GoodMem client.
      config: Active run configuration.
      conversations: Conversations to sample from.
      bindings: Conversation id to space binding.
      sample_size: Number of questions to sample.
      final_ks: Rerank depths to evaluate.
      neighbor_turn_values: Expansion radii to evaluate.
      ef_searches: HNSW candidate-list sizes to probe.

    Returns:
      Cells sorted by full coverage, highest first.
    """
    sample = stratified_sample(conversations, sample_size)
    max_final_k = max(final_ks)
    cells: list[SweepCell] = []

    for ef_search in ef_searches:
        probes = _probe(client, config, bindings, sample, ef_search, max_final_k)
        for final_k in sorted(final_ks):
            for neighbor_turns in sorted(neighbor_turn_values):
                questions, full, recall, turns, chars = _score(probes, final_k, neighbor_turns)
                cells.append(
                    SweepCell(
                        ef_search=ef_search,
                        final_k=final_k,
                        neighbor_turns=neighbor_turns,
                        questions=questions,
                        full_coverage=full,
                        evidence_recall=recall,
                        mean_context_turns=turns,
                        mean_context_chars=chars,
                    )
                )

    return sorted(cells, key=lambda cell: cell.full_coverage, reverse=True)
