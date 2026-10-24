"""Completeness gates between stages.

Every stage here writes a JSONL file that the next stage reads, and each row is
keyed by question id. A short file is the dangerous failure: it looks exactly
like a complete one, so a stage that silently processed 800 of 1,540 questions
still produces a plausible table. The paper is explicit that this is the thing
to guard -- appendix D.4 requires "the expected question count and a unique
identifier linking each question to its context, answer, and judgments" -- so
nothing downstream consumes a file without checking it against the question set
it was supposed to cover.
"""

from __future__ import annotations

from dataclasses import dataclass


class IncompleteStageError(RuntimeError):
    """Raised when a stage's output does not cover the expected question set."""


@dataclass(frozen=True, slots=True)
class Completeness:
    """The difference between what a stage produced and what it owed."""

    stage: str
    expected: int
    produced: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    duplicated: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.duplicated)

    def describe(self) -> str:
        """Returns a message naming the first few ids in each problem class."""
        parts = [f"{self.stage}: expected {self.expected} rows, found {self.produced}"]
        for label, ids in (
            ("missing", self.missing),
            ("unexpected", self.unexpected),
            ("duplicated", self.duplicated),
        ):
            if ids:
                shown = ", ".join(ids[:5])
                more = f" (+{len(ids) - 5} more)" if len(ids) > 5 else ""
                parts.append(f"{len(ids)} {label}: {shown}{more}")
        return "; ".join(parts)


def check(stage: str, expected_qids: list[str], produced_qids: list[str]) -> Completeness:
    """Compares a stage's output ids against the ids it was asked to cover.

    Args:
      stage: Stage name, used in the message.
      expected_qids: The question ids the stage should have covered.
      produced_qids: The question ids actually present in its output, in file
        order, so duplicates are detectable.

    Returns:
      A completeness report; inspect ``ok`` or call ``describe``.
    """
    expected = set(expected_qids)
    seen: set[str] = set()
    duplicated: list[str] = []
    for qid in produced_qids:
        if qid in seen:
            duplicated.append(qid)
        seen.add(qid)
    return Completeness(
        stage=stage,
        expected=len(expected),
        produced=len(produced_qids),
        missing=tuple(sorted(expected - seen)),
        unexpected=tuple(sorted(seen - expected)),
        duplicated=tuple(sorted(set(duplicated))),
    )


def require(stage: str, expected_qids: list[str], produced_qids: list[str]) -> None:
    """Raises unless a stage's output covers exactly the expected question set.

    Raises:
      IncompleteStageError: If any question is missing, duplicated, or unexpected.
    """
    report = check(stage, expected_qids, produced_qids)
    if not report.ok:
        raise IncompleteStageError(report.describe())
