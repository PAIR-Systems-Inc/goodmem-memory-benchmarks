"""Driving the paper's own reranker, which no endpoint here serves correctly.

The paper's table 11 uses ``Qwen/Qwen3-Reranker-4B``. That model is generative:
it emits no similarity score. It is prompted with a yes/no relevance question
and the score is the probability it puts on "yes" at the next position. An
endpoint that serves the weights behind a generic scoring API without that
template returns numbers that are not relevance -- the instance's vLLM
deployment ranks an obviously relevant passage 81st of 128, with every score
squeezed into a 0.26-wide band, and puts gold evidence in the top-15 16.7% of
the time against 83.3% for no reranking at all.

So the reranker is driven directly, in three phases, which also keeps a GPU
dependency out of this package:

1. ``retrieve --dense-only`` asks GoodMem for the dense candidate set and writes
   it here.
2. ``tools/qwen3_reranker.py`` scores those pairs with the checkpoint, applying
   the documented chat template and the paper's appendix G.1 instruction.
3. ``retrieve --from-scores`` assembles reader contexts from the scores.

This changes nothing about what is measured. The paper's pipeline is index,
dense retrieve, rerank, read; GoodMem supplies the index and the dense retrieval
whether the reranker sits inside its post-processor or beside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .client import GoodMemError


@dataclass(frozen=True, slots=True)
class Candidates:
    """One question's dense candidate set, awaiting external reranking."""

    qid: str
    conversation_id: str
    query: str
    dia_ids: tuple[str, ...]
    documents: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "dia_ids": list(self.dia_ids),
            "documents": list(self.documents),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Candidates":
        return cls(
            qid=raw["qid"],
            conversation_id=raw["conversation_id"],
            query=raw["query"],
            dia_ids=tuple(raw["dia_ids"]),
            documents=tuple(raw["documents"]),
        )


def candidates_path(run_dir: Path) -> Path:
    return run_dir / "dense-candidates.jsonl"


def scores_path(run_dir: Path) -> Path:
    return run_dir / "rerank-scores.jsonl"


def provenance_path(run_dir: Path) -> Path:
    return run_dir / "reranker-provenance.json"


# What the scores in this run were produced by. Written when contexts are
# assembled from an external reranker, and read when the summary is built.
EXTERNAL_RERANKER = {
    "provider": "EXTERNAL",
    "model": "Qwen/Qwen3-Reranker-4B",
    "display_name": "Qwen3-Reranker-4B, driven directly (paper table 11)",
    "note": (
        "Scored outside GoodMem by tools/qwen3_reranker.py, applying the "
        "yes/no relevance template the checkpoint requires with the paper's "
        "appendix G.1 instruction. int8 weights. GoodMem supplied the index "
        "and the dense top-128."
    ),
}


def write_provenance(run_dir: Path, scorer: dict[str, Any] | None = None) -> Path:
    """Records which reranker produced this run's ranking.

    Without this the summary describes the reranker named in the config, which
    for an externally reranked run is a model that was never called -- a false
    provenance record in the one file a reader checks.
    """
    path = provenance_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorer or EXTERNAL_RERANKER, indent=2) + "\n")
    return path


def read_provenance(run_dir: Path) -> dict[str, Any] | None:
    """Returns the recorded reranker for this run, or None if ranked in GoodMem."""
    path = provenance_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_candidates(path: Path, rows: list[Candidates]) -> None:
    """Writes the candidate sets for an external reranker to score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json()) + "\n")


def read_candidates(path: Path) -> list[Candidates]:
    """Reads candidate sets written by the dense-only retrieve phase."""
    if not path.exists():
        raise GoodMemError(f"no dense candidates at {path}; run retrieve --dense-only")
    with path.open() as handle:
        return [Candidates.from_json(json.loads(line)) for line in handle if line.strip()]


def read_scores(path: Path, expected: list[Candidates]) -> dict[str, list[float]]:
    """Reads external rerank scores and checks they line up with the candidates.

    Args:
      path: JSONL of ``{qid, scores}``.
      expected: The candidate sets the scores should cover.

    Returns:
      Question id to one score per candidate, in candidate order.

    Raises:
      GoodMemError: If a question is missing, unexpected, or its score count
        does not match its candidate count. A silent mismatch would rerank each
        question against another question's scores, which produces a plausible
        and entirely meaningless result.
    """
    if not path.exists():
        raise GoodMemError(f"no rerank scores at {path}; run tools/qwen3_reranker.py")

    by_qid: dict[str, list[float]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_qid[str(row["qid"])] = [float(s) for s in row["scores"]]

    wanted = {c.qid: len(c.documents) for c in expected}
    missing = sorted(set(wanted) - set(by_qid))
    unexpected = sorted(set(by_qid) - set(wanted))
    if missing or unexpected:
        raise GoodMemError(
            f"rerank scores cover {len(by_qid)} questions, candidates have "
            f"{len(wanted)}; {len(missing)} missing, {len(unexpected)} unexpected"
        )
    bad = [
        qid for qid, count in wanted.items() if len(by_qid[qid]) != count
    ]
    if bad:
        first = bad[0]
        raise GoodMemError(
            f"{len(bad)} questions have the wrong number of scores, e.g. {first}: "
            f"{len(by_qid[first])} scores for {wanted[first]} candidates"
        )
    return by_qid


def rank(candidate: Candidates, scores: list[float], final_k: int) -> list[str]:
    """Returns the top ``final_k`` dialogue ids by score, highest first.

    Ties keep dense order, which is the order GoodMem returned them in, so the
    result is deterministic given the same scores.
    """
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [candidate.dia_ids[i] for i in order[:final_k]]


def iter_ranked(
    candidates: list[Candidates], scores: dict[str, list[float]], final_k: int
) -> Iterator[tuple[Candidates, list[str]]]:
    """Yields each candidate set with its reranked selection."""
    for candidate in candidates:
        yield candidate, rank(candidate, scores[candidate.qid], final_k)
