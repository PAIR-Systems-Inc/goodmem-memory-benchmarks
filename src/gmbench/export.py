"""Builds the committable artifact set for a finished run.

A public reproduction is only checkable if the evidence ships with it, but the
per-question contexts are tens of megabytes and do not belong in a repository.
So the full reader and judge output ship compressed, and the retrieval stage
ships as an index: which turns were selected and reached the reader, plus a
digest of the rendered context. That is enough to recompute coverage
independently and to audit what was retrieved, without redistributing the
conversations themselves -- which are the benchmark's to license, not ours.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig
from .judge import judgements_path
from .reader import answers_path
from .retrieve import load_records, records_path
from .summary import summary_path


@dataclass(frozen=True, slots=True)
class Artifact:
    """One exported file and why it is small enough to commit."""

    path: Path
    bytes_written: int
    note: str


def _gzip_copy(source: Path, target: Path) -> int:
    """Compresses ``source`` to ``target`` and returns the bytes written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as raw, gzip.open(target, "wb", compresslevel=9) as packed:
        shutil.copyfileobj(raw, packed)
    return target.stat().st_size


def _retrieval_index(config: RunConfig, target: Path) -> int:
    """Writes the retrieval index: selections, context ids, and a context digest."""
    records = load_records(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", compresslevel=9) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "qid": record.qid,
                        "conversation_id": record.conversation_id,
                        "category": record.category,
                        "gold_evidence": list(record.gold_evidence),
                        "selected_dia_ids": record.selected_dia_ids,
                        "context_dia_ids": record.context_dia_ids,
                        "scores": record.scores,
                        "context_chars": len(record.context),
                        "context_sha256": hashlib.sha256(
                            record.context.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                + "\n"
            )
    return target.stat().st_size


def run(config: RunConfig) -> list[Artifact]:
    """Exports the committable artifacts for this run.

    Args:
      config: Active run configuration.

    Returns:
      One entry per artifact written.
    """
    out_dir = config.run_dir / "export"
    model = config.models.reader_model
    artifacts: list[Artifact] = []

    summary_source = summary_path(config)
    if summary_source.exists():
        target = out_dir / summary_source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_source, target)
        artifacts.append(
            Artifact(target, target.stat().st_size, "scores, config and integrity counts")
        )

    judge_source = judgements_path(config)
    if judge_source.exists():
        target = out_dir / f"judgements-{model}.jsonl.gz"
        artifacts.append(
            Artifact(
                target, _gzip_copy(judge_source, target), "every judgement and its label"
            )
        )

    answer_source = answers_path(config)
    if answer_source.exists():
        target = out_dir / f"answers-{model}.jsonl.gz"
        artifacts.append(
            Artifact(
                target,
                _gzip_copy(answer_source, target),
                "full reader responses, including the chain of thought",
            )
        )

    if records_path(config).exists():
        target = out_dir / "retrieval-index.jsonl.gz"
        artifacts.append(
            Artifact(
                target,
                _retrieval_index(config, target),
                "selected and delivered turn ids plus a context digest; the "
                "rendered contexts are not redistributed",
            )
        )

    return artifacts


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Whether re-running retrieval produced the same contexts as a prior export."""

    compared: int
    identical: int
    changed: tuple[str, ...]
    absent: tuple[str, ...]

    @property
    def stable(self) -> bool:
        return not (self.changed or self.absent)


def compare_to_export(config: RunConfig, export_index: Path) -> DriftReport:
    """Checks current retrieval contexts against the digests in a prior export.

    Retrieval runs again whenever the pipeline changes, and the reader answers
    on disk were generated from the contexts of the run before it. If the
    contexts are unchanged, those answers remain valid and only the cheap stages
    need redoing; if they drifted, the answers are stale and must be regenerated.
    Guessing wrong either wastes a paid pass or, worse, scores answers against
    evidence they never saw.

    Args:
      config: Active run configuration.
      export_index: Path to a previously exported ``retrieval-index.jsonl.gz``.

    Returns:
      A drift report; inspect ``stable``.
    """
    prior: dict[str, str] = {}
    opener = gzip.open if export_index.suffix == ".gz" else open
    with opener(export_index, "rt") as handle:  # type: ignore[operator]
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # An export index carries the digest; a raw retrieval shard carries
            # the context itself, so accept either and hash on the fly.
            digest = row.get("context_sha256")
            if digest is None:
                context = row.get("context")
                if context is None:
                    continue
                digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
            prior[row["qid"]] = digest

    identical = 0
    changed: list[str] = []
    absent: list[str] = []
    for record in load_records(config):
        digest = hashlib.sha256(record.context.encode("utf-8")).hexdigest()
        expected = prior.get(record.qid)
        if expected is None:
            absent.append(record.qid)
        elif expected == digest:
            identical += 1
        else:
            changed.append(record.qid)

    return DriftReport(
        compared=identical + len(changed),
        identical=identical,
        changed=tuple(sorted(changed)),
        absent=tuple(sorted(absent)),
    )
