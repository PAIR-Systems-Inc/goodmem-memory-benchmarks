"""Prepare a run's evidence for publication.

Reader and judge outputs are compressed in full. Retrieval is exported as turn
ids and a digest of each context, leaving the source conversations in the local
dataset. The ids support coverage checks; the digests let a later run detect
whether its reader would see different text.
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
    """An exported file with its size and a description."""

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
    """Export the available run artifacts and return a record of each file written."""
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

    # MemoryAgentBench stores its predictions and grades together in an
    # outcomes file, with a separate summary for each capability.
    for summary in sorted(config.run_dir.glob("mab-*-summary-*.json")):
        target = out_dir / summary.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, target)
        artifacts.append(
            Artifact(target, target.stat().st_size, "capability scores and per-source detail")
        )
    for outcomes in sorted(config.run_dir.glob("mab-*-gpt-*.jsonl")):
        target = out_dir / f"outcomes-{model}.jsonl.gz"
        artifacts.append(
            Artifact(
                target,
                _gzip_copy(outcomes, target),
                "every question, its retrieved units, prediction and grade",
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
    """Compare current reader contexts with a previous retrieval export.

    Matching digests establish that the context text is unchanged. A changed
    context requires a new answer before it can be scored as part of this run.
    ``export_index`` may be a compressed export or a raw retrieval shard.
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
