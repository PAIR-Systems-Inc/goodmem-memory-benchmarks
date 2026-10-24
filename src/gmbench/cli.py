"""Command-line entry point for the GoodMem memory benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import config as config_module
from . import beam, coverage, export, ingest, judge, mab_run, markdown
from . import memoryagentbench as mab
from . import reader, report, retrieve, summary, sweep
from .client import GoodMemClient
from .locomo import load as load_locomo

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "locomo.toml"


def _replace(cfg: config_module.RunConfig, **changes: object) -> config_module.RunConfig:
    """Returns a copy of the frozen run config with the given fields replaced."""
    fields = {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}
    fields.update(changes)
    return config_module.RunConfig(**fields)  # type: ignore[arg-type]


def _load(args: argparse.Namespace) -> config_module.RunConfig:
    cfg = config_module.load(args.config, run_tag=args.run_tag)
    if getattr(args, "ingest_tag", None):
        cfg = _replace(cfg, ingest_tag=args.ingest_tag)
    if getattr(args, "reader_model", None):
        models = config_module.ModelConfig(
            **{
                **{f: getattr(cfg.models, f) for f in cfg.models.__dataclass_fields__},
                "reader_model": args.reader_model,
            }
        )
        cfg = _replace(cfg, models=models)
    return cfg


def _conversations(cfg: config_module.RunConfig, limit: int | None):
    convs = load_locomo(
        cfg.dataset_path, conversations=cfg.conversations, categories=cfg.categories
    )
    return convs[:limit] if limit else convs


def _client(cfg: config_module.RunConfig) -> GoodMemClient:
    return GoodMemClient(
        cfg.server.base_url,
        cfg.server.api_key,
        tls_verify=cfg.server.tls_verify,
        timeout_seconds=cfg.server.timeout_seconds,
        requests_per_second=cfg.server.requests_per_second,
        llm_requests_per_second=cfg.server.llm_requests_per_second,
    )


def cmd_provision(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    convs = _conversations(cfg, args.conversations_limit)
    with _client(cfg) as client:
        bindings = ingest.provision(client, cfg, convs)
    console.print(f"[green]Provisioned {len(bindings)} spaces[/green] -> {ingest.manifest_path(cfg)}")
    return 0


def cmd_ingest(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    convs = _conversations(cfg, args.conversations_limit)
    with _client(cfg) as client:
        bindings = ingest.provision(client, cfg, convs)
        counts = ingest.ingest(client, cfg, convs, bindings, skip_populated=not args.force)
    total = sum(counts.values())
    console.print(f"[green]Indexed {total} turns across {len(counts)} conversations[/green]")
    return 0


def cmd_retrieve(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    convs = _conversations(cfg, args.conversations_limit)
    bindings = ingest.load_manifest(cfg)
    missing = [c.conversation_id for c in convs if c.conversation_id not in bindings]
    if missing:
        console.print(f"[red]No space for {missing}; run ingest first[/red]")
        return 1
    with _client(cfg) as client:
        records = retrieve.run(
            client, cfg, convs, bindings, limit_per_conversation=args.questions_limit
        )
    console.print(f"[green]Retrieved evidence for {len(records)} questions[/green]")
    report.coverage_table(coverage.compute(records), console)
    return 0


def _int_list(raw: str) -> tuple[int, ...]:
    """Parses a comma-separated integer list from the command line."""
    return tuple(int(part) for part in raw.split(",") if part.strip())


def cmd_sweep(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    convs = _conversations(cfg, args.conversations_limit)
    bindings = ingest.load_manifest(cfg)
    missing = [c.conversation_id for c in convs if c.conversation_id not in bindings]
    if missing:
        console.print(f"[red]No space for {missing}; run ingest first[/red]")
        return 1
    with _client(cfg) as client:
        cells = sweep.run(
            client,
            cfg,
            convs,
            bindings,
            sample_size=args.sample,
            final_ks=_int_list(args.final_k),
            neighbor_turn_values=_int_list(args.neighbor_turns),
            ef_searches=_int_list(args.ef_search),
        )
    report.sweep_table(cells, console)
    best = cells[0]
    console.print(
        f"[green]Best cell:[/green] ef_search={best.ef_search} final_k={best.final_k} "
        f"neighbor_turns={best.neighbor_turns} -> {best.full_coverage:.2f} full coverage"
    )
    return 0


def _expected_qids(cfg: config_module.RunConfig, args: argparse.Namespace) -> list[str]:
    """Returns the question ids this run is supposed to cover."""
    convs = _conversations(cfg, args.conversations_limit)
    limit = args.questions_limit
    return [
        question.qid
        for conversation in convs
        for question in (
            conversation.questions[:limit] if limit else conversation.questions
        )
    ]


def cmd_coverage(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    records = retrieve.load_records(cfg, _expected_qids(cfg, args))
    report.coverage_table(coverage.compute(records), console)
    return 0


def cmd_read(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    expected = _expected_qids(cfg, args)
    records = retrieve.load_records(cfg, expected)
    with _client(cfg) as client:
        answers = reader.run(client, cfg, records, resume=not args.no_resume)
    failures = sum(1 for answer in answers if answer.failed)
    truncated = sum(1 for answer in answers if answer.truncated)
    console.print(
        f"[green]Generated {len(answers)} answers[/green] "
        f"({failures} failed, {truncated} truncated)"
    )
    if failures or truncated:
        console.print(
            "[yellow]Re-run `read` to retry the failures; it resumes from what is "
            "already on disk.[/yellow]"
        )
    return 0


def cmd_judge(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    expected = _expected_qids(cfg, args)
    answers = reader.load_answers(cfg, expected)
    with _client(cfg) as client:
        judgements = judge.run(client, cfg, answers, resume=not args.no_resume)
    return _emit(cfg, args, console, judgements=judgements, answers=answers)


def _emit(
    cfg: config_module.RunConfig,
    args: argparse.Namespace,
    console: Console,
    *,
    judgements: list[judge.JudgeRecord],
    answers: list[reader.AnswerRecord],
) -> int:
    """Writes the run summary and prints integrity before scores.

    Returns:
      0 when the run is clean, 1 when any scored zero came from plumbing. A
      non-zero exit keeps a contaminated run from passing a CI gate quietly.
    """
    expected = _expected_qids(cfg, args)
    records = retrieve.load_records(cfg, expected)
    with _client(cfg) as client:
        models = {
            "embedder": client.describe_model("embedders", cfg.models.embedder_id),
            "reranker": client.describe_model("rerankers", cfg.models.reranker_id),
        }
    document = summary.build(cfg, records, answers, judgements, models)
    path = summary.write(cfg, document)

    checks = summary.integrity(answers, judgements, cfg.models.judge_repetitions)
    report.integrity_panel(checks, console)
    report.accuracy_table(summary.scores(judgements), cfg.models.reader_model, console)
    console.print(f"[dim]Summary written to {path}[/dim]")
    return 0 if checks.clean else 1


def cmd_report(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    expected = _expected_qids(cfg, args)
    records = retrieve.load_records(cfg, expected)
    report.coverage_table(coverage.compute(records), console)
    answers = reader.load_answers(cfg, expected)
    judgements = judge.load_judgements(cfg, expected)
    return _emit(cfg, args, console, judgements=judgements, answers=answers)


def cmd_export(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    artifacts = export.run(cfg)
    if not artifacts:
        console.print("[yellow]Nothing to export; run the stages first.[/yellow]")
        return 1
    for artifact in artifacts:
        size = artifact.bytes_written
        human = f"{size / 1024:.0f}K" if size < 1024**2 else f"{size / 1024**2:.1f}M"
        console.print(f"  [green]{human:>6}[/green]  {artifact.path.name} — {artifact.note}")
    console.print(f"[green]Exported {len(artifacts)} artifacts[/green] to {cfg.run_dir / 'export'}")
    return 0


def cmd_drift(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    index = Path(args.against)
    if not index.exists():
        console.print(f"[red]No export index at {index}[/red]")
        return 1
    report = export.compare_to_export(cfg, index)
    if report.stable:
        console.print(
            f"[green]Contexts stable[/green] — {report.identical} of {report.compared} "
            "identical to the prior export; existing answers remain valid."
        )
        return 0
    console.print(
        f"[yellow]Contexts drifted[/yellow] — {report.identical} of {report.compared} "
        f"identical, {len(report.changed)} changed, {len(report.absent)} not in the export."
    )
    for qid in report.changed[:5]:
        console.print(f"  changed: {qid}")
    console.print(
        "[yellow]Answers on disk were generated from the previous contexts. "
        "Re-run `read --no-resume` before judging.[/yellow]"
    )
    return 1


def cmd_beam(args: argparse.Namespace, console: Console) -> int:
    """Measures sequential ingestion cost over the BEAM 1M conversation."""
    cfg = _load(args)
    source = Path(args.parquet)
    if not source.exists():
        console.print(f"[red]No BEAM parquet at {source}[/red]")
        return 1
    conversation = beam.load(source, args.conversation)
    console.print(
        f"[bold]{conversation.conversation_id}[/bold] — {len(conversation.messages)} messages, "
        f"~{conversation.approximate_tokens:,} tokens — {conversation.title}"
    )
    if len(conversation.messages) != beam.PAPER_MESSAGE_COUNT:
        console.print(
            f"[yellow]Note: the paper streams {beam.PAPER_MESSAGE_COUNT} messages; "
            f"this conversation has {len(conversation.messages)}.[/yellow]"
        )

    with _client(cfg) as client:
        name = f"gmbench-beam-{conversation.conversation_id}-{cfg.run_tag}"
        space = client.create_space(
            name,
            cfg.models.embedder_id,
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
            separators=cfg.chunking.separators,
            labels={ingest.SPACE_LABEL_KEY: cfg.run_tag},
        )
        console.print(f"[dim]ingesting into a fresh space {space['spaceId']}[/dim]")
        ticks = list(
            beam.measure(
                client,
                cfg,
                conversation,
                space_id=space["spaceId"],
                limit=args.limit,
                poll_every=args.poll_every,
            )
        )

    path = beam.write_summary(cfg, conversation, ticks)
    if ticks:
        final = ticks[-1]
        console.print(
            f"[green]Ingested {final.cumulative_messages} messages "
            f"(~{final.approximate_cumulative_tokens:,} tokens) in "
            f"{final.cumulative_seconds / 60:.1f} min "
            f"({final.cumulative_seconds / final.cumulative_messages:.2f} s/message)[/green]"
        )
    console.print(f"[dim]Summary written to {path}[/dim]")
    return 0


def cmd_mab_inspect(args: argparse.Namespace, console: Console) -> int:
    """Reports the MemoryAgentBench selection without ingesting anything."""
    cfg = _load(args)
    directory = Path(args.dataset or (cfg.dataset_path.parent / "memoryagentbench"))
    blocks = mab.load(directory)
    total = mab.question_total(blocks)

    table = __import__("rich.table", fromlist=["Table"]).Table(
        title=f"MemoryAgentBench selection — {total} questions, {len(blocks)} blocks"
    )
    table.add_column("Capability", style="bold")
    table.add_column("Source")
    table.add_column("Label", style="dim")
    table.add_column("Questions", justify="right")
    table.add_column("Unit")
    seen: set[tuple[str, str]] = set()
    for block in sorted(blocks, key=lambda b: (b.capability, b.source)):
        key = (block.capability, block.source)
        if key in seen:
            continue
        seen.add(key)
        count = sum(len(b.tasks) for b in blocks if b.source == block.source)
        table.add_row(
            block.capability, block.source, block.label, str(count), block.unit_kind
        )
    console.print(table)
    if total != 2071:
        console.print(
            f"[yellow]Selection totals {total}, not the published 2,071. The shipped "
            "file holds 3,671 across 22 length variants; check the variant "
            "choices.[/yellow]"
        )
    else:
        console.print("[green]Matches the published 2,071-question selection.[/green]")

    manifest = cfg.run_dir / "memoryagentbench-manifest.json"
    mab.write_manifest(manifest, blocks)
    console.print(f"[dim]Manifest written to {manifest}[/dim]")
    return 0


def cmd_mab(args: argparse.Namespace, console: Console) -> int:
    """Ingests and scores one MemoryAgentBench capability group."""
    cfg = _load(args)
    directory = Path(args.dataset or (cfg.dataset_path.parent / "memoryagentbench"))
    blocks = mab.load(directory)
    total = mab.question_total(blocks)
    if total != 2071:
        console.print(
            f"[yellow]Loaded {total} questions, not the published 2,071; "
            "check the variant selection.[/yellow]"
        )

    with _client(cfg) as client:
        outcomes = mab_run.run_capability(
            client, cfg, blocks, args.capability, resume=not args.no_resume
        )

    document = mab_run.summarize(outcomes)
    path = mab_run.write_summary(cfg, args.capability, document)

    table = Table(title=f"MemoryAgentBench — {args.capability}, reader={cfg.models.reader_model}")
    table.add_column("Source", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Gold fact retrieved", justify="right")
    table.add_column("Errors", justify="right")
    for source, bucket in sorted(document["by_source"].items()):
        table.add_row(
            source,
            str(bucket["questions"]),
            f"{bucket['accuracy']:.2f}",
            f"{bucket['gold_retrieval_rate']:.2f}",
            str(bucket["errors"]) if bucket["errors"] else "-",
        )
    table.add_row(
        "overall",
        str(document["questions"]),
        f"{document['accuracy']:.2f}",
        "-",
        str(document["errors"]) if document["errors"] else "-",
        style="bold cyan",
    )
    console.print(table)
    console.print(
        "[dim]Gold fact retrieved: the share of questions where the highest-serial "
        "fact stating a gold answer reached the reader. A wrong answer with the "
        "gold fact present is a reading failure; without it, a retrieval one.[/dim]"
    )
    console.print(f"[dim]Summary written to {path}[/dim]")
    return 1 if document["errors"] else 0


def cmd_table(args: argparse.Namespace, console: Console) -> int:
    """Prints the README's results tables, generated from the summary files."""
    cfg = _load(args)
    labelled: dict[str, Path] = {}
    for item in args.runs or [f"{cfg.run_tag}={cfg.run_tag}"]:
        label, _, tag = item.partition("=")
        labelled[label] = cfg.results_dir / (tag or label)

    rows = markdown.load_summaries(labelled)
    if not rows:
        console.print("[yellow]No summaries found; run the stages first.[/yellow]")
        return 1

    text = markdown.document(rows, labelled)
    if args.out:
        out = Path(args.out)
        out.write_text(text)
        console.print(f"[green]Wrote {out}[/green] ({len(rows)} scored runs)")
    else:
        print(text)
    unclean = [r for r in rows if not r.clean]
    if unclean:
        console.print(
            "[red]Integrity problems in: "
            + ", ".join(f"{r.label}/{r.reader}" for r in unclean)
            + "[/red]"
        )
        return 1
    return 0


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    for step in (cmd_ingest, cmd_retrieve, cmd_read, cmd_judge):
        code = step(args, console)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmbench",
        description="Run GoodMem against long-conversation memory benchmarks (LoCoMo).",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to the run TOML."
    )
    parser.add_argument("--run-tag", default=None, help="Overrides the configured run tag.")
    parser.add_argument(
        "--ingest-tag",
        default=None,
        help="Reuse the spaces ingested under this tag instead of this run's own.",
    )
    parser.add_argument(
        "--reader-model", default=None, help="Overrides the configured reader model."
    )
    parser.add_argument(
        "--conversations-limit",
        type=int,
        default=None,
        help="Use only the first N conversations (smoke runs).",
    )
    parser.add_argument(
        "--questions-limit",
        type=int,
        default=None,
        help="Use only the first N questions per conversation (smoke runs).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-ingest even if a space looks populated."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard any existing answers or judgements and redo the stage.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("provision", cmd_provision, "Create one space per conversation."),
        ("ingest", cmd_ingest, "Provision, upload turns, and wait for indexing."),
        ("retrieve", cmd_retrieve, "Retrieve and expand evidence for every question."),
        ("coverage", cmd_coverage, "Report evidence coverage from saved retrievals."),
        ("sweep", cmd_sweep, "Grid-search retrieval depth and expansion on coverage."),
        ("read", cmd_read, "Generate answers from saved retrievals."),
        ("judge", cmd_judge, "Grade saved answers and report accuracy."),
        ("report", cmd_report, "Print coverage and accuracy for a run."),
        ("export", cmd_export, "Write the committable artifact set for a run."),
        ("drift", cmd_drift, "Check current contexts against a prior export's digests."),
        ("beam", cmd_beam, "Measure sequential ingestion cost on the BEAM 1M conversation."),
        ("mab-inspect", cmd_mab_inspect, "Report the MemoryAgentBench question selection."),
        ("mab", cmd_mab, "Ingest and score one MemoryAgentBench capability group."),
        ("table", cmd_table, "Print the README results tables from run summaries."),
        ("run", cmd_run, "Ingest, retrieve, read and judge in one go."),
    ):
        child = sub.add_parser(name, help=help_text)
        child.set_defaults(handler=handler)
        if name == "table":
            child.add_argument(
                "--runs",
                nargs="*",
                default=None,
                help="Labelled runs as Label=run-tag, e.g. 'Paper models=paper-models'.",
            )
            child.add_argument(
                "--out", default=None, help="Write the document here instead of stdout."
            )
        if name == "beam":
            child.add_argument(
                "--parquet",
                default="data/beam/1M-00000-of-00001.parquet",
                help="BEAM split parquet to read.",
            )
            child.add_argument(
                "--conversation",
                default=beam.PAPER_CONVERSATION_ID,
                help="Which conversation to stream (default: the paper's).",
            )
            child.add_argument(
                "--limit", type=int, default=None, help="Stop after N messages."
            )
            child.add_argument(
                "--poll-every",
                type=int,
                default=1,
                help="Readiness check cadence in messages; 1 matches the paper.",
            )
        if name == "mab":
            child.add_argument(
                "--capability",
                default="conflict_resolution",
                choices=[
                    "conflict_resolution",
                    "accurate_retrieval",
                    "test_time_learning",
                    "long_range_understanding",
                ],
                help="Which capability group to run (default: conflict_resolution).",
            )
            child.add_argument(
                "--dataset", default=None, help="MemoryAgentBench dataset root."
            )
        if name == "mab-inspect":
            child.add_argument(
                "--dataset", default=None, help="MemoryAgentBench dataset root."
            )
        if name == "drift":
            child.add_argument(
                "--against",
                required=True,
                help="Path to a prior retrieval-index.jsonl.gz to compare against.",
            )
        if name == "sweep":
            child.add_argument(
                "--sample", type=int, default=200, help="Questions to sample (default 200)."
            )
            child.add_argument(
                "--final-k", default="10,15,20,25", help="Rerank depths to try."
            )
            child.add_argument(
                "--neighbor-turns", default="1,2,3,4", help="Expansion radii to try."
            )
            child.add_argument(
                "--ef-search", default="400", help="HNSW efSearch values to probe."
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()
    try:
        return int(args.handler(args, console))
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001 - surfaced to the shell
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
