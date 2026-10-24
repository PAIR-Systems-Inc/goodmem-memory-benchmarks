"""Run configuration loaded from TOML with environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Connection details for the GoodMem instance under test."""

    base_url: str = "https://localhost:8080"
    api_key: str = ""
    tls_verify: bool = False
    timeout_seconds: float = 300.0
    requests_per_second: float = 5.0

    @classmethod
    def from_env(cls, table: dict[str, Any]) -> Self:
        return cls(
            base_url=os.environ.get("GOODMEM_BASE_URL", table.get("base_url", cls.base_url)),
            api_key=os.environ.get("GOODMEM_API_KEY", table.get("api_key", "")),
            tls_verify=_env_bool("GMBENCH_TLS_VERIFY", bool(table.get("tls_verify", False))),
            timeout_seconds=float(table.get("timeout_seconds", cls.timeout_seconds)),
            requests_per_second=float(
                table.get("requests_per_second", cls.requests_per_second)
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """GoodMem-side model registrations, by UUID, plus reader/judge model names."""

    embedder_id: str = ""
    reranker_id: str = ""
    reader_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini-2024-07-18"
    reader_max_tokens: int = 4096
    judge_repetitions: int = 3


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based recursive chunking.

    GoodMem's TOKEN_COUNT length measurement currently falls back to character
    count server-side, so character counts are what actually takes effect. With
    one memory per conversation turn, a window this wide leaves nearly every
    turn as a single chunk and only splits unusually long ones.
    """

    chunk_size: int = 1500
    chunk_overlap: int = 150
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Two-stage retrieval: dense candidates, then cross-encoder rerank."""

    dense_candidates: int = 128
    final_k: int = 15
    neighbor_turns: int = 2
    ef_search: int = 400
    iterative_scan: str = "ITERATIVE_SCAN_RELAXED_ORDER"
    max_scan_tuples: int | None = None
    concurrency: int = 8


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Ingestion batching, pacing, readiness polling and repair.

    One memory per turn means one embedding request per turn, so the embedding
    provider's requests-per-minute ceiling -- not GoodMem -- is what bounds
    ingest. ``embed_requests_per_minute`` paces submission to stay under it;
    ``repair_rounds`` re-drives anything that failed anyway.
    """

    batch_size: int = 64
    space_concurrency: int = 2
    ready_poll_seconds: float = 3.0
    ready_timeout_seconds: float = 3600.0
    embed_requests_per_minute: int = 1500
    repair_rounds: int = 4


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one benchmark run needs."""

    run_tag: str
    dataset_path: Path
    results_dir: Path
    server: ServerConfig
    models: ModelConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    ingest: IngestConfig
    conversations: tuple[str, ...] = ()
    categories: tuple[int, ...] = (1, 2, 3, 4)
    reader_concurrency: int = 50
    openai_api_key: str = field(default="", repr=False)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_tag


def _dataclass_from(cls: type, table: dict[str, Any]) -> Any:
    """Builds a frozen dataclass from a TOML table, ignoring unknown keys."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in table.items():
        if key not in fields:
            continue
        kwargs[key] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def load(path: Path, *, run_tag: str | None = None) -> RunConfig:
    """Loads a run configuration from TOML, applying environment overrides.

    Args:
      path: Path to the TOML configuration file.
      run_tag: Overrides the configured run tag when supplied.

    Returns:
      The assembled run configuration.
    """
    with path.open("rb") as handle:
        doc = tomllib.load(handle)

    root = path.parent.parent
    run = doc.get("run", {})
    dataset = Path(run.get("dataset_path", "data/locomo10.json"))
    results = Path(run.get("results_dir", "results"))

    return RunConfig(
        run_tag=run_tag or run.get("run_tag", "dev"),
        dataset_path=dataset if dataset.is_absolute() else root / dataset,
        results_dir=results if results.is_absolute() else root / results,
        server=ServerConfig.from_env(doc.get("server", {})),
        models=_dataclass_from(ModelConfig, doc.get("models", {})),
        chunking=_dataclass_from(ChunkingConfig, doc.get("chunking", {})),
        retrieval=_dataclass_from(RetrievalConfig, doc.get("retrieval", {})),
        ingest=_dataclass_from(IngestConfig, doc.get("ingest", {})),
        conversations=tuple(run.get("conversations", ())),
        categories=tuple(run.get("categories", (1, 2, 3, 4))),
        reader_concurrency=int(run.get("reader_concurrency", 50)),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
