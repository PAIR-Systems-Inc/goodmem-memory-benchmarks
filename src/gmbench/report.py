"""Terminal reporting, with the paper's published scores alongside ours."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .coverage import CoverageStats
from .summary import Integrity, Score
from .sweep import SweepCell

# RAW-RAG, LoCoMo, text-embedding-3-small + Cohere Rerank-v3.5 (paper table 10).
# These are the numbers to beat, by reader. Single/multi/temporal/open/overall.
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
    """Prints evidence coverage per category.

    Coverage bounds accuracy from above, so this is the table to tune on before
    spending reader and judge tokens.
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
    """States whether any scored zero was plumbing rather than judgement.

    This prints before the accuracy table on purpose. A number that looks
    reasonable but is partly made of failed requests is the failure mode worth
    catching, and it is invisible in the accuracy table itself.
    """
    console = console or Console()
    if checks.clean:
        console.print(
            f"[green]Integrity: clean[/green] — {checks.questions} questions, "
            f"{checks.judgements_present} judgements, no reader or judge failures."
        )
        return
    console.print("[bold red]Integrity: CONTAMINATED[/bold red]")
    for problem in checks.problems():
        console.print(f"  [red]•[/red] {problem}")
    console.print(
        "[yellow]Scores below include these as zeros. Re-run the affected stage "
        "before quoting them.[/yellow]"
    )


def accuracy_table(
    scores: list[Score],
    reader_model: str,
    console: Console | None = None,
) -> None:
    """Prints measured accuracy, with intervals, against the published rows.

    The interval is the point of this table. Differences of a point or two on
    these slice sizes are inside sampling noise, and the published figures carry
    comparable uncertainty of their own, so a bare delta invites reading a win
    or a regression into nothing.

    Args:
      scores: Per-slice scores with bootstrap intervals.
      reader_model: Reader model name, used to pick the comparison column.
      console: Optional console to print to.
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
            verdict = "[dim]tie (within noise)[/dim]"
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
        "The published figures have their own uncertainty, which is not shown and "
        "would widen every comparison further.[/dim]"
    )
    if not paper:
        console.print(
            f"[yellow]No published RAW-RAG column for reader '{reader_model}'; "
            "comparison columns are blank.[/yellow]"
        )


def sweep_table(cells: list[SweepCell], console: Console | None = None) -> None:
    """Prints sweep results, best full coverage first.

    The context columns matter as much as coverage: the paper's expanded LoCoMo
    evidence averages roughly 4,800 tokens, so a cell that wins on coverage
    while spending far more than that is buying it with budget rather than with
    better retrieval.
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
