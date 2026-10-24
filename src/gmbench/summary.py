"""Run summaries: the durable, committable record of a result.

The per-question shards are large and gitignored, so a finished run needs a
small artifact that states what was measured, under what configuration, how
complete it was, and with what uncertainty. Quoting a point estimate against a
published point estimate is itself a way to be quietly wrong, so the interval
travels with the number.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import RunConfig
from .judge import JudgeRecord
from .locomo import CATEGORY_NAMES
from .reader import AnswerRecord
from .retrieve import RetrievalRecord

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261023


@dataclass(frozen=True, slots=True)
class Score:
    """One slice's accuracy with a bootstrap interval."""

    slice: str
    questions: int
    accuracy: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class Integrity:
    """Whether anything in the run was a failure rather than a measurement."""

    questions: int
    reader_errors: int
    reader_truncated: int
    reader_missing_final_marker: int
    judge_infrastructure_failures: int
    judge_unparseable: int
    judgements_expected: int
    judgements_present: int

    @property
    def clean(self) -> bool:
        """True when every scored zero came from the judge, not from plumbing."""
        return (
            self.reader_errors == 0
            and self.reader_truncated == 0
            and self.judge_infrastructure_failures == 0
            and self.judge_unparseable == 0
            and self.judgements_present == self.judgements_expected
        )

    def problems(self) -> list[str]:
        """Returns one line per contamination source, empty when clean."""
        out: list[str] = []
        if self.reader_errors:
            out.append(f"{self.reader_errors} reader calls failed")
        if self.reader_truncated:
            out.append(f"{self.reader_truncated} reader answers hit the token cap")
        if self.reader_missing_final_marker:
            out.append(
                f"{self.reader_missing_final_marker} answers had no FINAL ANSWER marker, "
                "so the whole response was graded"
            )
        if self.judge_infrastructure_failures:
            out.append(
                f"{self.judge_infrastructure_failures} judge calls failed and scored zero"
            )
        if self.judge_unparseable:
            out.append(f"{self.judge_unparseable} judge responses had no usable label")
        if self.judgements_present != self.judgements_expected:
            out.append(
                f"{self.judgements_present} of {self.judgements_expected} judgements present"
            )
        return out


def bootstrap_ci(
    means: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Returns a percentile bootstrap 95% interval over per-question means.

    Args:
      means: Per-question mean scores in [0, 1].
      resamples: Bootstrap resample count.
      seed: Fixed so a rerun of the report reproduces the interval.

    Returns:
      The 2.5th and 97.5th percentiles, as percentages.
    """
    if not means:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(means)
    draws = sorted(100.0 * sum(rng.choices(means, k=n)) / n for _ in range(resamples))
    return (draws[int(0.025 * resamples)], draws[int(0.975 * resamples)])


def scores(judgements: list[JudgeRecord]) -> list[Score]:
    """Computes per-category and overall accuracy with intervals."""
    buckets: dict[str, list[float]] = {}
    for record in judgements:
        name = CATEGORY_NAMES.get(record.category, str(record.category))
        buckets.setdefault(name, []).append(record.mean_score)
        buckets.setdefault("overall", []).append(record.mean_score)

    out: list[Score] = []
    for name in sorted(buckets, key=lambda key: (key == "overall", key)):
        means = buckets[name]
        low, high = bootstrap_ci(means)
        out.append(
            Score(
                slice=name,
                questions=len(means),
                accuracy=100.0 * sum(means) / len(means),
                ci_low=low,
                ci_high=high,
            )
        )
    return out


def integrity(
    answers: list[AnswerRecord], judgements: list[JudgeRecord], repetitions: int
) -> Integrity:
    """Counts every way this run could be a failure rather than a measurement."""
    return Integrity(
        questions=len(answers),
        reader_errors=sum(1 for answer in answers if answer.failed),
        reader_truncated=sum(1 for answer in answers if answer.truncated),
        reader_missing_final_marker=sum(
            1 for answer in answers if "FINAL ANSWER" not in answer.generated.upper()
        ),
        judge_infrastructure_failures=sum(
            record.infrastructure_failures for record in judgements
        ),
        judge_unparseable=sum(record.unparseable for record in judgements),
        judgements_expected=len(judgements) * repetitions,
        judgements_present=sum(len(record.scores) for record in judgements),
    )


def build(
    config: RunConfig,
    records: list[RetrievalRecord],
    answers: list[AnswerRecord],
    judgements: list[JudgeRecord],
) -> dict[str, object]:
    """Assembles the summary document for a finished run."""
    checks = integrity(answers, judgements, config.models.judge_repetitions)
    evidence_chars = [len(record.context) for record in records]
    return {
        "benchmark": "locomo",
        "run_tag": config.run_tag,
        "ingest_tag": config.ingest_tag,
        "reader_model": config.models.reader_model,
        "judge_model": config.models.judge_model,
        "judge_repetitions": config.models.judge_repetitions,
        "embedder_id": config.models.embedder_id,
        "reranker_id": config.models.reranker_id,
        "retrieval": {
            "dense_candidates": config.retrieval.dense_candidates,
            "final_k": config.retrieval.final_k,
            "neighbor_turns": config.retrieval.neighbor_turns,
            "ef_search": config.retrieval.ef_search,
            "iterative_scan": config.retrieval.iterative_scan,
        },
        "chunking": {
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "length_measurement": "CHARACTER_COUNT",
        },
        "evidence": {
            "mean_chars": sum(evidence_chars) / len(evidence_chars) if evidence_chars else 0.0,
            "mean_turns": (
                sum(len(r.context_dia_ids) for r in records) / len(records) if records else 0.0
            ),
        },
        "integrity": {**asdict(checks), "clean": checks.clean, "problems": checks.problems()},
        "scores": [asdict(score) for score in scores(judgements)],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }


def summary_path(config: RunConfig) -> Path:
    return config.run_dir / f"summary-{config.models.reader_model}.json"


def write(config: RunConfig, document: dict[str, object]) -> Path:
    """Writes the summary document and returns where it landed."""
    path = summary_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path
