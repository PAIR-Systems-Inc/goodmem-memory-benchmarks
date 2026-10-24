"""Evidence coverage: a reader-free diagnostic for tuning retrieval.

LoCoMo annotates each question with the dialogue ids that support its answer.
Comparing those against the dialogue ids present in the assembled context gives
a direct measure of evidence access -- the bottleneck the paper identifies as
dominant (its Finding 1) -- without spending a single reader or judge token.
Coverage bounds accuracy from above, so it is the right thing to tune on before
committing to a scored run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .locomo import CATEGORY_NAMES
from .retrieve import RetrievalRecord


@dataclass(frozen=True, slots=True)
class CoverageStats:
    """Coverage over one slice of questions."""

    label: str
    questions: int
    excluded: int
    full_coverage: float
    partial_coverage: float
    evidence_recall: float
    selected_hit_rate: float
    mean_context_turns: float
    mean_context_chars: float


def _summarize(label: str, records: list[RetrievalRecord]) -> CoverageStats:
    # Only questions with at least one reference this conversation actually
    # contains can be scored. The rest are excluded and counted, rather than
    # charged to retrieval as a miss.
    scored = [r for r in records if r.gold_evidence]
    excluded = sum(
        1 for r in records if not r.gold_evidence and r.gold_unresolvable
    )
    full = partial = 0
    recall_sum = 0.0
    selected_hits = 0
    for record in scored:
        gold = {e for e in record.gold_evidence if e}
        context = set(record.context_dia_ids)
        found = gold & context
        recall_sum += len(found) / len(gold)
        if found == gold:
            full += 1
        if found:
            partial += 1
        if gold & set(record.selected_dia_ids):
            selected_hits += 1
    n = len(scored) or 1
    return CoverageStats(
        label=label,
        questions=len(scored),
        excluded=excluded,
        full_coverage=100.0 * full / n,
        partial_coverage=100.0 * partial / n,
        evidence_recall=100.0 * recall_sum / n,
        selected_hit_rate=100.0 * selected_hits / n,
        mean_context_turns=sum(len(r.context_dia_ids) for r in records) / (len(records) or 1),
        mean_context_chars=sum(len(r.context) for r in records) / (len(records) or 1),
    )


def compute(records: list[RetrievalRecord]) -> list[CoverageStats]:
    """Computes coverage overall and per question category.

    Args:
      records: Retrieval records produced by the retrieve stage.

    Returns:
      Per-category stats followed by an overall row. ``questions`` counts the
      questions that could be scored; ``excluded`` counts those whose only
      annotated references are absent from their own conversation, which happens
      a handful of times in the released dataset.
    """
    out: list[CoverageStats] = []
    for category in sorted({r.category for r in records}):
        slice_ = [r for r in records if r.category == category]
        out.append(_summarize(CATEGORY_NAMES.get(category, str(category)), slice_))
    out.append(_summarize("overall", records))
    return out
