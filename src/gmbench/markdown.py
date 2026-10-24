"""Generates the README's results tables from the summary files.

Hand-typing a published number is the same class of mistake as every other one
this harness guards against: transcription is silent and nobody re-checks it. So
the tables in the README are generated from the ``summary-*.json`` a run wrote,
and regenerating them is how the README stays true after a re-run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .report import PAPER_BEST_AGENTIC, PAPER_RAW_RAG

_ROW_ORDER = ("single_hop", "multi_hop", "temporal", "open_domain", "overall")
_PRETTY = {
    "single_hop": "Single-hop",
    "multi_hop": "Multi-hop",
    "temporal": "Temporal",
    "open_domain": "Open-domain",
    "overall": "**Overall**",
}


@dataclass(frozen=True, slots=True)
class Row:
    """One configuration's overall result."""

    label: str
    reader: str
    accuracy: float
    ci_low: float
    ci_high: float
    questions: int
    clean: bool
    embedder: str
    reranker: str
    final_k: int


def load_summaries(run_dirs: dict[str, Path]) -> list[Row]:
    """Reads every summary under the given labelled run directories.

    Args:
      run_dirs: Label to run directory.

    Returns:
      Rows ordered by label then reader.
    """
    rows: list[Row] = []
    for label, directory in run_dirs.items():
        for path in sorted(directory.glob("summary-*.json")):
            doc = json.loads(path.read_text())
            overall = next(
                (s for s in doc["scores"] if s["slice"] == "overall"), None
            )
            if overall is None:
                continue
            rows.append(
                Row(
                    label=label,
                    reader=doc["reader_model"],
                    accuracy=overall["accuracy"],
                    ci_low=overall["ci_low"],
                    ci_high=overall["ci_high"],
                    questions=overall["questions"],
                    clean=bool(doc["integrity"]["clean"]),
                    embedder=(doc.get("embedder") or {}).get("model", ""),
                    reranker=(doc.get("reranker") or {}).get("model", ""),
                    final_k=int(doc.get("retrieval", {}).get("final_k", 0)),
                )
            )
    return sorted(rows, key=lambda row: (row.label, row.reader))


def verdict(accuracy: float, low: float, high: float, reference: float | None) -> str:
    """Describes where a published point estimate sits against our interval."""
    if reference is None:
        return "—"
    if reference < low:
        return f"ahead (+{accuracy - reference:.2f})"
    if reference > high:
        return f"behind ({accuracy - reference:.2f})"
    return f"tie ({accuracy - reference:+.2f})"


def headline_table(rows: list[Row]) -> str:
    """Renders the overall-accuracy comparison as markdown."""
    lines = [
        "| Configuration | Reader | GoodMem | 95% CI | RAW-RAG | vs RAW-RAG "
        "| Best agentic | vs best agentic |",
        "|---|---|---:|---|---:|---|---:|---|",
    ]
    for row in rows:
        paper = PAPER_RAW_RAG.get(row.reader, {}).get("overall")
        agentic = PAPER_BEST_AGENTIC.get(row.reader, {}).get("overall")
        lines.append(
            f"| {row.label} | `{row.reader}` | **{row.accuracy:.2f}** "
            f"| [{row.ci_low:.2f}, {row.ci_high:.2f}] "
            f"| {paper:.2f} | {verdict(row.accuracy, row.ci_low, row.ci_high, paper)} "
            f"| {agentic:.2f} | {verdict(row.accuracy, row.ci_low, row.ci_high, agentic)} |"
            if paper is not None and agentic is not None
            else f"| {row.label} | `{row.reader}` | **{row.accuracy:.2f}** "
            f"| [{row.ci_low:.2f}, {row.ci_high:.2f}] | — | — | — | — |"
        )
    return "\n".join(lines)


def category_table(summary_path: Path) -> str:
    """Renders one run's per-category breakdown as markdown."""
    doc = json.loads(summary_path.read_text())
    reader = doc["reader_model"]
    paper = PAPER_RAW_RAG.get(reader, {})
    agentic = PAPER_BEST_AGENTIC.get(reader, {})
    by_slice = {s["slice"]: s for s in doc["scores"]}

    lines = [
        f"| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for key in _ROW_ORDER:
        score = by_slice.get(key)
        if score is None:
            continue
        reference = paper.get(key)
        best = agentic.get(key)
        lines.append(
            f"| {_PRETTY[key]} | {score['questions']} | {score['accuracy']:.2f} "
            f"| [{score['ci_low']:.2f}, {score['ci_high']:.2f}] "
            f"| {reference:.2f} | {best:.2f} "
            f"| {verdict(score['accuracy'], score['ci_low'], score['ci_high'], reference)} |"
            if reference is not None and best is not None
            else f"| {_PRETTY[key]} | {score['questions']} | {score['accuracy']:.2f} "
            f"| [{score['ci_low']:.2f}, {score['ci_high']:.2f}] | — | — | — |"
        )
    return "\n".join(lines)


def coverage_table(summary_path: Path) -> str:
    """Renders the evidence budget line for one run."""
    doc = json.loads(summary_path.read_text())
    evidence = doc.get("evidence", {})
    return (
        f"Mean evidence supplied to the reader: "
        f"{evidence.get('mean_turns', 0):.0f} turns, "
        f"{evidence.get('mean_chars', 0):,.0f} characters "
        f"(~{evidence.get('mean_chars', 0) / 4:,.0f} tokens)."
    )


def document(rows: list[Row], run_dirs: dict[str, Path]) -> str:
    """Renders the whole results document.

    Every number here is read from a run's ``summary-*.json``. Nothing is typed,
    because a transcription error in a published table is as silent as any other
    failure this harness is built to prevent.
    """
    parts = [
        "# Results",
        "",
        "Generated by `gmbench table`. Do not hand-edit — re-run it instead.",
        "",
        "Scores are question-weighted accuracy over the 1,540 non-adversarial",
        "LoCoMo questions, each the mean of three independent judgements. The",
        "interval is a percentile bootstrap 95% interval over per-question means.",
        "",
        '"Tie" means the published point estimate falls inside our interval: the',
        "data cannot distinguish them. The published figures carry comparable",
        "uncertainty of their own, which is not shown here and would widen every",
        "comparison further.",
        "",
        "## Overall",
        "",
        headline_table(rows),
        "",
        "## Configurations",
        "",
    ]
    seen: set[str] = set()
    for row in rows:
        if row.label in seen:
            continue
        seen.add(row.label)
        parts.append(
            f"- **{row.label}** — embedder `{row.embedder or 'unknown'}`, "
            f"reranker `{row.reranker or 'unknown'}`, "
            f"128 dense candidates to top-{row.final_k}"
        )
    parts.append("")

    for row in rows:
        path = run_dirs[row.label] / f"summary-{row.reader}.json"
        parts += [
            f"## {row.label} — `{row.reader}`",
            "",
            category_table(path),
            "",
            coverage_table(path),
            "",
            f"Integrity: {'clean' if row.clean else 'CONTAMINATED'} "
            f"({row.questions} questions scored).",
            "",
        ]
    return "\n".join(parts)
