"""Terminal reporting, with the paper's published scores alongside ours."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .coverage import CoverageStats
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
            f"{stat.full_coverage:.2f}",
            f"{stat.partial_coverage:.2f}",
            f"{stat.evidence_recall:.2f}",
            f"{stat.selected_hit_rate:.2f}",
            f"{stat.mean_context_turns:.1f}",
            f"{stat.mean_context_chars:.0f}",
            style=style,
        )
    console.print(table)


def accuracy_table(
    measured: dict[str, tuple[int, float]],
    reader_model: str,
    console: Console | None = None,
) -> None:
    """Prints measured accuracy against the paper's RAW-RAG and best-agentic rows.

    Args:
      measured: Category name to (question count, accuracy) from the judge stage.
      reader_model: Reader model name, used to pick the comparison column.
      console: Optional console to print to.
    """
    console = console or Console()
    paper = PAPER_RAW_RAG.get(reader_model, {})
    agentic = PAPER_BEST_AGENTIC.get(reader_model, {})

    table = Table(title=f"LoCoMo accuracy, reader={reader_model}")
    table.add_column("Category", style="bold")
    table.add_column("N", justify="right")
    table.add_column("GoodMem", justify="right")
    table.add_column("Paper RAW-RAG", justify="right")
    table.add_column("vs RAW-RAG", justify="right")
    table.add_column("Best agentic", justify="right")
    table.add_column("vs agentic", justify="right")

    for key in _ROW_ORDER:
        if key not in measured:
            continue
        count, score = measured[key]
        reference = paper.get(key)
        best = agentic.get(key)
        delta = f"{score - reference:+.2f}" if reference is not None else "-"
        delta_style = ""
        if reference is not None:
            delta_style = "green" if score >= reference else "red"
        table.add_row(
            key,
            str(count),
            f"{score:.2f}",
            f"{reference:.2f}" if reference is not None else "-",
            f"[{delta_style}]{delta}[/{delta_style}]" if delta_style else delta,
            f"{best:.2f}" if best is not None else "-",
            f"{score - best:+.2f}" if best is not None else "-",
            style="bold cyan" if key == "overall" else None,
        )
    console.print(table)
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
