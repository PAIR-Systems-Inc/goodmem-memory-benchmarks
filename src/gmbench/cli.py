"""Command-line entry point for the GoodMem memory benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import config as config_module
from . import coverage, ingest, judge, reader, report, retrieve, sweep
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


def cmd_coverage(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    records = retrieve.load_records(cfg)
    report.coverage_table(coverage.compute(records), console)
    return 0


def cmd_read(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    records = retrieve.load_records(cfg)
    if args.questions_limit:
        records = records[: args.questions_limit]
    answers = reader.run(cfg, records)
    failures = sum(1 for a in answers if a.error)
    console.print(f"[green]Generated {len(answers)} answers[/green] ({failures} errors)")
    return 0


def cmd_judge(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    answers = reader.load_answers(cfg)
    records = judge.run(cfg, answers)
    report.accuracy_table(judge.accuracy_by_category(records), cfg.models.reader_model, console)
    return 0


def cmd_report(args: argparse.Namespace, console: Console) -> int:
    cfg = _load(args)
    try:
        records = retrieve.load_records(cfg)
        report.coverage_table(coverage.compute(records), console)
    except Exception as exc:  # noqa: BLE001 - reporting is best-effort
        console.print(f"[yellow]No coverage to show: {exc}[/yellow]")
    try:
        judgements = judge.load_judgements(cfg)
        report.accuracy_table(
            judge.accuracy_by_category(judgements), cfg.models.reader_model, console
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]No accuracy to show: {exc}[/yellow]")
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
        ("run", cmd_run, "Ingest, retrieve, read and judge in one go."),
    ):
        child = sub.add_parser(name, help=help_text)
        child.set_defaults(handler=handler)
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
