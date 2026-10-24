"""Terminal reporting, with the paper's published scores alongside ours."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .coverage import CoverageStats
from .summary import Integrity, Score
from .sweep import SweepCell

# RAW-RAG, LoCoMo, text-embedding-3-small + Cohere Rerank-v3.5 (paper table 10).
# Reference percentages by reader and question category.
PAPER_RAW_RAG: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "single_hop": 94.57,
        "multi_hop": 85.93,
        "temporal": 82.04,
        "open_domain": 73.61,
        "overall": 89.07,
    },
    "gpt-4.1-mini": {
        "single_hop": 96.79,
        "multi_hop": 92.32,
        "temporal": 90.86,
        "open_domain": 80.21,
        "overall": 93.70,
    },
}

# Best agentic system per column in the same table, for context.
PAPER_BEST_AGENTIC: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "single_hop": 93.94,
        "multi_hop": 84.04,
        "temporal": 80.69,
        "open_domain": 73.96,
        "overall": 88.12,
    },
    "gpt-4.1-mini": {
        "single_hop": 95.12,
        "multi_hop": 88.30,
        "temporal": 91.59,
        "open_domain": 76.04,
        "overall": 91.69,
    },
}

_ROW_ORDER = ("single_hop", "multi_hop", "temporal", "open_domain", "overall")


def coverage_table(stats: list[CoverageStats], console: Console | None = None) -> None:
    """Show annotated-evidence coverage before and after context expansion.

    Use this to inspect retrieval without making reader or judge calls.
    """
    console = console or Console()
    table = Table(title="Evidence coverage (annotated-evidence questions only)")
    table.add_column("Slice", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Excl", justify="right")
    table.add_column("Full", justify="right")
    table.add_column("Any", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("In top-k", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Chars", justify="right")

    for stat in stats:
        style = "bold cyan" if stat.label == "overall" else None
        table.add_row(
            stat.label,
            str(stat.questions),
            str(stat.excluded) if stat.excluded else "-",
            f"{stat.full_coverage:.2f}",
            f"{stat.partial_coverage:.2f}",
            f"{stat.evidence_recall:.2f}",
            f"{stat.selected_hit_rate:.2f}",
            f"{stat.mean_context_turns:.1f}",
            f"{stat.mean_context_chars:.0f}",
            style=style,
        )
    console.print(table)
    if any(stat.excluded for stat in stats):
        console.print(
            "[dim]Excl: questions whose annotated evidence points only at turns "
            "absent from their own conversation. They are dropped rather than "
            "scored as retrieval misses.[/dim]"
        )


def integrity_panel(checks: Integrity, console: Console | None = None) -> None:
    """Show evaluation errors and incomplete responses before displaying accuracy.

    These counts help distinguish a low model score from an unreliable run.
    """
    console = console or Console()
    if checks.clean:
        console.print(
            f"[green]Integrity: clean[/green] — {checks.questions} questions, "
            f"{checks.judgements_present} judgements, no reader or judge failures."
        )
        return
    console.print("[bold red]Integrity: needs review[/bold red]")
    for problem in checks.problems():
        console.print(f"  [red]•[/red] {problem}")
    console.print(
        "[yellow]Failed requests score zero. Responses without a final-answer "
        "marker are graded in full. Review these counts before using the scores.[/yellow]"
    )


def accuracy_table(
    scores: list[Score],
    reader_model: str,
    console: Console | None = None,
) -> None:
    """Compare per-category scores with the paper's reference figures.

    Each verdict locates the published point estimate within or outside this
    run's bootstrap interval. It does not account for uncertainty in the
    published score.
    """
    console = console or Console()
    paper = PAPER_RAW_RAG.get(reader_model, {})
    agentic = PAPER_BEST_AGENTIC.get(reader_model, {})
    by_slice = {score.slice: score for score in scores}

    table = Table(title=f"LoCoMo accuracy, reader={reader_model}")
    table.add_column("Category", style="bold")
    table.add_column("N", justify="right")
    table.add_column("GoodMem", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("RAW-RAG", justify="right")
    table.add_column("vs", justify="right")
    table.add_column("Agentic", justify="right")
    table.add_column("Verdict")

    for key in _ROW_ORDER:
        score = by_slice.get(key)
        if score is None:
            continue
        reference = paper.get(key)
        best = agentic.get(key)
        if reference is None:
            verdict, delta = "-", "-"
        elif reference < score.ci_low:
            verdict, delta = "[green]ahead[/green]", f"{score.accuracy - reference:+.2f}"
        elif reference > score.ci_high:
            verdict, delta = "[red]behind[/red]", f"{score.accuracy - reference:+.2f}"
        else:
            verdict = "[dim]tie (reference within CI)[/dim]"
            delta = f"{score.accuracy - reference:+.2f}"
        table.add_row(
            key,
            str(score.questions),
            f"{score.accuracy:.2f}",
            f"[{score.ci_low:.2f}, {score.ci_high:.2f}]",
            f"{reference:.2f}" if reference is not None else "-",
            delta,
            f"{best:.2f}" if best is not None else "-",
            verdict,
            style="bold cyan" if key == "overall" else None,
        )
    console.print(table)
    console.print(
        "[dim]Verdict compares the published point estimate against our interval. "
        "It does not account for uncertainty in the published figure or establish "
        "statistical equivalence.[/dim]"
    )
    if not paper:
        console.print(
            f"[yellow]No published RAW-RAG column for reader '{reader_model}'; "
            "comparison columns are blank.[/yellow]"
        )


def sweep_table(cells: list[SweepCell], console: Console | None = None) -> None:
    """Show sweep configurations in descending order of full evidence coverage.

    Context size is shown alongside coverage because selecting more text can
    improve coverage while increasing the reader's cost.
    """
    console = console or Console()
    table = Table(title="Retrieval sweep (coverage on sampled questions)")
    table.add_column("ef", justify="right")
    table.add_column("final_k", justify="right")
    table.add_column("±turns", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Full", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("~Tokens", justify="right")

    for index, cell in enumerate(cells):
        table.add_row(
            str(cell.ef_search),
            str(cell.final_k),
            str(cell.neighbor_turns),
            str(cell.questions),
            f"{cell.full_coverage:.2f}",
            f"{cell.evidence_recall:.2f}",
            f"{cell.mean_context_turns:.1f}",
            f"{cell.mean_context_chars:.0f}",
            f"{cell.mean_context_chars / 4:.0f}",
            style="bold green" if index == 0 else None,
        )
    console.print(table)
    console.print(
        "[dim]~Tokens is chars/4, a rough gauge against the paper's ~4,800-token "
        "LoCoMo evidence mean.[/dim]"
    )
