"""Build a results document from saved run summaries.

LoCoMo summaries provide scores and bootstrap intervals for each reader.
MemoryAgentBench and BEAM have their own summary formats and section renderers.
The tables use recorded values; interpreting them belongs in the methodology.
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


def document(
    rows: list[Row], run_dirs: dict[str, Path], results_dir: Path | None = None
) -> str:
    """Render LoCoMo tables and any available summaries from the other benchmarks."""
    parts = [
        "# Results",
        "",
        "Generated from run summaries by `gmbench table`.",
        "",
        "LoCoMo accuracy gives each question equal weight. A question's score is",
        "the mean of its recorded judgements; the 95% interval is a percentile",
        "bootstrap over those question means.",
        "",
        '"Tie" means the published point estimate falls inside this run\'s interval.',
        "This comparison does not include uncertainty in the published estimate",
        "or establish statistical equivalence. Configuration names below come from",
        "the saved summaries; see the [methodology](docs/methodology.md) for model",
        "attribution and other limits of the comparison.",
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
            f"Integrity: {'clean' if row.clean else 'needs review'} "
            f"({row.questions} questions scored).",
            "",
        ]

    # Each renderer returns an empty section when its summary is absent.
    if results_dir is not None:
        for extra in (
            reranker_section(results_dir),
            mab_section(results_dir),
            beam_section(results_dir),
        ):
            if extra:
                parts += ["---", ""] + extra
    return "\n".join(parts)


# MemoryAgentBench reference scores: RAW-RAG with gpt-4o-mini, paper table 4.
MAB_PAPER_RAW_RAG: dict[str, float] = {
    "accurate_retrieval": 67.85,
    "test_time_learning": 47.68,
    "long_range_understanding": 31.74,
    "conflict_resolution": 46.00,
}
_MAB_PRETTY = {
    "accurate_retrieval": "Accurate Retrieval",
    "test_time_learning": "Test-Time Learning",
    "long_range_understanding": "Long-Range Understanding",
    "conflict_resolution": "Conflict Resolution",
}
_MAB_ORDER = (
    "accurate_retrieval",
    "test_time_learning",
    "long_range_understanding",
    "conflict_resolution",
)


def _find_summary(results_dir: Path, pattern: str) -> Path | None:
    """Return the first matching export summary in path order, or None."""
    hits = sorted(results_dir.glob(pattern))
    return hits[0] if hits else None


def mab_section(results_dir: Path) -> list[str]:
    """Renders the MemoryAgentBench tables from each capability's summary.

    Returns an empty list when no capability has been scored, so a partial
    checkout does not produce a section full of blanks.
    """
    found: dict[str, dict] = {}
    source_of: dict[str, str] = {}
    for path in sorted(results_dir.glob("*/export/mab-*-summary-*.json")):
        run = path.parent.parent.name
        # Retain archived runs on disk while excluding them from the tables.
        if "broken" in run or run.startswith("_"):
            continue
        data = json.loads(path.read_text())
        capability = data["capability"]
        if capability in found:
            raise ValueError(
                f"two live runs report {capability}: {source_of[capability]} and {run}; "
                "archive the superseded one"
            )
        found[capability] = data
        source_of[capability] = run
    if not found:
        return []

    lines = [
        "## MemoryAgentBench",
        "",
        "Aggregation follows the paper's appendix D.5: three capabilities average",
        "their source scores, while Test-Time Learning averages its five",
        "in-context sources first and then weights that against ReDial equally.",
        "",
        'Here, "ahead" and "behind" describe differences between single-run',
        "scores. No comparison interval is estimated for these results.",
        "",
        "| Capability | N | GoodMem | RAW-RAG | Verdict | Errors |",
        "|---|---:|---:|---:|---|---:|",
    ]
    scores, questions = [], 0
    for key in _MAB_ORDER:
        data = found.get(key)
        if data is None:
            continue
        ours = float(data["accuracy"])
        theirs = MAB_PAPER_RAW_RAG[key]
        delta = ours - theirs
        call = "ahead" if delta > 0 else "behind"
        scores.append(ours)
        questions += int(data["questions"])
        lines.append(
            f"| {_MAB_PRETTY[key]} | {data['questions']} | {ours:.2f} | "
            f"{theirs:.2f} | {call} ({delta:+.2f}) | {data['errors']} |"
        )
    if len(scores) == len(_MAB_ORDER):
        ours = sum(scores) / len(scores)
        theirs = sum(MAB_PAPER_RAW_RAG[k] for k in _MAB_ORDER) / len(_MAB_ORDER)
        delta = ours - theirs
        lines.append(
            f"| **Overall** (mean of four) | {questions} | **{ours:.2f}** | "
            f"{theirs:.2f} | {'ahead' if delta > 0 else 'behind'} ({delta:+.2f}) | — |"
        )
    else:
        missing = [_MAB_PRETTY[k] for k in _MAB_ORDER if k not in found]
        lines.append(f"| **Overall** | — | not computed | — | missing: {', '.join(missing)} | — |")

    lines += ["", "### By source", "", "| Capability | Source | N | Accuracy | Errors |", "|---|---|---:|---:|---:|"]
    for key in _MAB_ORDER:
        data = found.get(key)
        if data is None:
            continue
        for source, stats in data.get("by_source", {}).items():
            lines.append(
                f"| {_MAB_PRETTY[key]} | `{source}` | {stats['questions']} | "
                f"{stats['accuracy']:.2f} | {stats['errors']} |"
            )
    lines.append("")
    return lines


def beam_section(results_dir: Path) -> list[str]:
    """Renders the BEAM ingestion-cost summary, if one was measured."""
    path = _find_summary(results_dir, "*/export/beam-summary.json")
    if path is None:
        return []
    d = json.loads(path.read_text())
    return [
        "## BEAM-1M — ingestion cost",
        "",
        "Messages are added in conversation order. Each must reach COMPLETED",
        "before the next is submitted, so elapsed time includes the wait for",
        "indexing. This measures sequential ingestion latency.",
        "",
        "| | |",
        "|---|---:|",
        f"| Conversation | `{d['conversation_id']}` — {d['conversation_title']} |",
        f"| Messages ingested | {d['messages_ingested']} of {d['messages_total']} |",
        f"| Approximate tokens | {d['approximate_tokens']:,} |",
        f"| Total time | {d['cumulative_hours']:.2f} h ({d['cumulative_seconds']:.0f} s) |",
        f"| Mean per message | {d['mean_seconds_per_message']:.3f} s |",
        "",
    ]


def reranker_section(results_dir: Path) -> list[str]:
    """Renders the reranker-substitution comparison, if it was measured."""
    path = _find_summary(results_dir, "*/export/reranker-comparison.json")
    if path is None:
        return []
    d = json.loads(path.read_text())
    stages = d["stages"]
    label = {
        "dense": "GoodMem dense top-15, no rerank",
        "substitute": "+ hosted `qwen3-rerank`",
        "paper_exact": "+ local `Qwen3-Reranker-4B`",
    }
    lines = [
        "## Reranker evidence coverage",
        "",
        f"This comparison covers {d['questions']:,} of {d['questions_total']:,} questions",
        "with resolvable evidence annotations. Each stage uses the same saved",
        f"candidate sets ({d['dense_candidates_per_question']} candidates per question).",
        "Coverage is measured before neighbor expansion.",
        "",
        "| Stage | All gold evidence in top-15 | Mean gold recall |",
        "|---|---:|---:|",
    ]
    for key in ("dense", "substitute", "paper_exact"):
        s = stages[key]
        lines.append(
            f"| {label[key]} | {s['all_gold_in_top_15_pct']:.1f}% | "
            f"{s['mean_gold_recall_pct']:.1f}% |"
        )
    lines.append("")
    return lines
