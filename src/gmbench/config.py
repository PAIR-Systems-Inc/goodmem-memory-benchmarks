"""Load a run's TOML settings and apply environment overrides.

Model registrations belong to a particular GoodMem instance, so they can be
supplied through the environment. Dataset and result paths are resolved from
the repository root, one directory above the config file.
"""

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
    llm_requests_per_second: float = 15.0

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
            llm_requests_per_second=float(
                table.get("llm_requests_per_second", cls.llm_requests_per_second)
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model names and their GoodMem registration UUIDs.

    The environment overrides registration fields in TOML. Reader registrations
    are ``model=uuid`` pairs, so changing the reader also selects its endpoint.
    See .env.example for the supported variables.
    """

    embedder_id: str = ""
    reranker_id: str = ""
    reader_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini-2024-07-18"
    reader_max_tokens: int = 4096
    judge_repetitions: int = 3
    # Reader and judge run through GoodMem's provider proxy, keyed by model name
    # so --reader-model selects both the model and the registration to route it.
    reader_llm_ids: tuple[tuple[str, str], ...] = ()
    judge_llm_id: str = ""

    def llm_id_for(self, model: str) -> str:
        """Returns the registered LLM id that serves ``model``.

        Raises:
          KeyError: If no registration is configured for the model.
        """
        for name, llm_id in self.reader_llm_ids:
            if name == model:
                return llm_id
        raise KeyError(
            f"no reader_llm_ids entry for model {model!r}; add one to the run TOML"
        )


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Recursive chunking measured in characters.

    The default 1,500-character window keeps most LoCoMo turns whole. Longer
    turns may produce several chunks, all carrying the same dialogue id.
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
    """Control how quickly memories are submitted and how long indexing may take.

    ``embed_requests_per_minute`` paces submissions for the embedding provider.
    After each ingestion pass settles, failed memories can be recreated for up to
    ``repair_rounds`` additional passes.
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
    ingest_tag: str
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
        """Where this run's retrievals, answers and judgements are written."""
        return self.results_dir / self.run_tag

    @property
    def ingest_dir(self) -> Path:
        """Directory containing the space manifest.

        Several retrieval runs can reuse one index by sharing an ingest tag. Each
        run still writes its answers under its own run tag.
        """
        return self.results_dir / self.ingest_tag


def _models_from(table: dict[str, Any]) -> ModelConfig:
    """Builds the model config, letting the environment supply instance UUIDs."""
    models: ModelConfig = _dataclass_from(ModelConfig, table)
    overrides: dict[str, Any] = {}
    for env_name, field in (
        ("GMBENCH_EMBEDDER_ID", "embedder_id"),
        ("GMBENCH_RERANKER_ID", "reranker_id"),
        ("GMBENCH_JUDGE_LLM_ID", "judge_llm_id"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            overrides[field] = value

    raw_readers = os.environ.get("GMBENCH_READER_LLM_IDS", "").strip()
    if raw_readers:
        pairs: list[tuple[str, str]] = []
        for item in raw_readers.split(","):
            if not item.strip():
                continue
            model, _, llm_id = item.partition("=")
            if not llm_id:
                raise ValueError(
                    f"GMBENCH_READER_LLM_IDS entry {item!r} is not model=uuid"
                )
            pairs.append((model.strip(), llm_id.strip()))
        overrides["reader_llm_ids"] = tuple(sorted(pairs))

    if not overrides:
        return models
    fields = {name: getattr(models, name) for name in models.__dataclass_fields__}
    fields.update(overrides)
    return ModelConfig(**fields)


def _dataclass_from(cls: type, table: dict[str, Any]) -> Any:
    """Builds a frozen dataclass from a TOML table, ignoring unknown keys."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in table.items():
        if key not in fields:
            continue
        if isinstance(value, dict):
            # A TOML table of model name to LLM id becomes a tuple of pairs so
            # the config stays hashable and frozen.
            kwargs[key] = tuple(sorted(value.items()))
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
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

    resolved_tag = run_tag or run.get("run_tag", "dev")
    return RunConfig(
        run_tag=resolved_tag,
        ingest_tag=run.get("ingest_tag") or resolved_tag,
        dataset_path=dataset if dataset.is_absolute() else root / dataset,
        results_dir=results if results.is_absolute() else root / results,
        server=ServerConfig.from_env(doc.get("server", {})),
        models=_models_from(doc.get("models", {})),
        chunking=_dataclass_from(ChunkingConfig, doc.get("chunking", {})),
        retrieval=_dataclass_from(RetrievalConfig, doc.get("retrieval", {})),
        ingest=_dataclass_from(IngestConfig, doc.get("ingest", {})),
        conversations=tuple(run.get("conversations", ())),
        categories=tuple(run.get("categories", (1, 2, 3, 4))),
        reader_concurrency=int(run.get("reader_concurrency", 50)),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
