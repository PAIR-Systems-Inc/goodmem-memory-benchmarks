"""Compare retrieval settings using annotated-evidence coverage.

For each ef_search value, retrieve at the largest depth in the grid. Smaller
depths use prefixes of that ranking, and all neighbor expansions use local
turns. The grid therefore needs one retrieval pass per ef_search value.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from .client import GoodMemClient, GoodMemError
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
    """Sample evenly across conversations and question categories.

    This gives small categories more influence than they have in the full
    dataset: LoCoMo has 96 open-domain questions and 841 single-hop questions.
    Use the sample to compare settings, then measure the chosen setting on all
    questions. Selection is deterministic for a given input and target size.
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


@dataclass(frozen=True, slots=True)
class RerankerVerdict:
    """How well one reranker surfaces annotated evidence."""

    reranker_id: str
    display_name: str
    model: str
    questions: int
    gold_at_k: float
    failed: int

    @property
    def usable(self) -> bool:
        """Whether all calls succeeded and at least some gold evidence was retrieved.

        The caller separately compares coverage with the full-depth reference.
        """
        return self.failed == 0 and self.gold_at_k > 0.0


def verify_rerankers(
    client: GoodMemClient,
    config: RunConfig,
    conversations: list[Conversation],
    bindings: dict[str, SpaceBinding],
    *,
    reranker_ids: list[str],
    sample_size: int = 24,
    final_k: int = 15,
) -> tuple[float, list[RerankerVerdict]]:
    """Compare rerankers with a full-depth reference on the same question sample.

    For each ranking, measure how often annotated evidence appears within
    ``final_k``. The reference requests all candidates through the first
    reranker, then takes its top-k prefix. It is also reranked; an independent
    dense baseline would require retrieve_dense. Return the reference coverage
    and each reranker's coverage and failed request count.
    """
    sample = [
        (conversation, question)
        for conversation, question in stratified_sample(conversations, sample_size * 3)
        if question.evidence
    ][:sample_size]

    def gold_for(conversation: Conversation, question: Question) -> set[str]:
        return set(conversation.resolvable_evidence(question)[0])

    baseline_hits = 0
    cached: list[tuple[Conversation, Question, list[str]]] = []
    with tqdm(total=len(sample), desc="dense baseline", unit="q") as bar:
        for conversation, question in sample:
            chunks = client.retrieve(
                question.question,
                bindings[conversation.conversation_id].space_id,
                dense_candidates=config.retrieval.dense_candidates,
                final_k=config.retrieval.dense_candidates,
                reranker_id=reranker_ids[0],
                ef_search=config.retrieval.ef_search,
                iterative_scan=config.retrieval.iterative_scan,
            )
            ranked = ranked_dia_ids(chunks)
            cached.append((conversation, question, ranked))
            bar.update(1)

    verdicts: list[RerankerVerdict] = []
    for reranker_id in reranker_ids:
        described = client.describe_model("rerankers", reranker_id)
        hits = failed = 0
        with tqdm(total=len(sample), desc=described.get("model", reranker_id)[:22], unit="q") as bar:
            for conversation, question in sample:
                try:
                    chunks = client.retrieve(
                        question.question,
                        bindings[conversation.conversation_id].space_id,
                        dense_candidates=config.retrieval.dense_candidates,
                        final_k=final_k,
                        reranker_id=reranker_id,
                        ef_search=config.retrieval.ef_search,
                        iterative_scan=config.retrieval.iterative_scan,
                    )
                except GoodMemError:
                    failed += 1
                    bar.update(1)
                    continue
                top = set(ranked_dia_ids(chunks))
                hits += bool(gold_for(conversation, question) & top)
                bar.update(1)
        scored = len(sample) - failed
        verdicts.append(
            RerankerVerdict(
                reranker_id=reranker_id,
                display_name=described.get("display_name", ""),
                model=described.get("model", ""),
                questions=scored,
                gold_at_k=100.0 * hits / scored if scored else 0.0,
                failed=failed,
            )
        )

    # The cached full-depth response has already passed through the first
    # reranker. Its prefix is the reference used by this diagnostic.
    for conversation, question, ranked in cached:
        baseline_hits += bool(gold_for(conversation, question) & set(ranked[:final_k]))
    baseline = 100.0 * baseline_hits / len(cached) if cached else 0.0
    return baseline, verdicts
