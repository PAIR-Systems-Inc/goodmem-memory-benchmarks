"""MemoryAgentBench: one ingested context, many heterogeneous questions.

The shape of this benchmark is "ingest once, query many". Each row is a single
long context plus up to 500 questions about it, and the four capability groups
ask quite different things of the same stored representation: locate a fact,
apply demonstrations seen during ingestion, integrate clues across a whole book,
or respect the precedence of numbered facts. Compression that helps one of those
removes exactly what another needs, which is the point the benchmark makes.

Two things about the released data decide whether a reproduction is comparable:

- **The shipped file is not the published question set.** It carries 22 ``source``
  labels and 3,671 questions, because EventQA ships at three context lengths and
  FactConsolidation at four. The published 2,071 is one length variant per
  source. Counting labels naively over-reports by 77%, so the selection is
  explicit here rather than implied.
- **The ingestion unit differs per source.** A numbered fact, a chat session, a
  labelled demonstration and a line of a novel are not interchangeable units,
  and splitting them wrongly destroys the thing being measured -- most sharply
  for FactConsolidation, where precedence lives in the serial number, and for
  test-time learning, where an example and its label must stay together.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .client import GoodMemError

# One length variant per source, which is what sums to the published 2,071.
# EventQA and FactConsolidation are the only sources with a choice to make; the
# paper reports per-length columns, so these are the defaults a reproduction of
# the headline table needs, and they are overridable.
DEFAULT_VARIANTS: dict[str, str] = {
    "eventqa": "eventqa_131072",
    "factconsolidation_sh": "factconsolidation_sh_32k",
    "factconsolidation_mh": "factconsolidation_mh_32k",
}

CAPABILITIES: dict[str, str] = {
    "eventqa": "accurate_retrieval",
    "longmemeval": "accurate_retrieval",
    "ruler_qa1": "accurate_retrieval",
    "ruler_qa2": "accurate_retrieval",
    "icl_banking77": "test_time_learning",
    "icl_clinic150": "test_time_learning",
    "icl_nlu": "test_time_learning",
    "icl_trec_coarse": "test_time_learning",
    "icl_trec_fine": "test_time_learning",
    "recsys_redial": "test_time_learning",
    "detective_qa": "long_range_understanding",
    "infbench_sum": "long_range_understanding",
    "factconsolidation_sh": "conflict_resolution",
    "factconsolidation_mh": "conflict_resolution",
}

# How each source's context divides into things worth storing separately.
UNIT_DOCUMENT = "document"      # Document N: blocks
UNIT_SESSION = "session"        # windows within a dated chat session
UNIT_DEMONSTRATION = "demo"     # one labelled example
UNIT_FACT = "fact"              # one serially numbered fact
UNIT_PASSAGE = "passage"        # a slice of continuous prose

UNITS: dict[str, str] = {
    "eventqa": UNIT_PASSAGE,
    "longmemeval": UNIT_SESSION,
    "ruler_qa1": UNIT_DOCUMENT,
    "ruler_qa2": UNIT_DOCUMENT,
    "icl_banking77": UNIT_DEMONSTRATION,
    "icl_clinic150": UNIT_DEMONSTRATION,
    "icl_nlu": UNIT_DEMONSTRATION,
    "icl_trec_coarse": UNIT_DEMONSTRATION,
    "icl_trec_fine": UNIT_DEMONSTRATION,
    "recsys_redial": UNIT_DOCUMENT,
    "detective_qa": UNIT_PASSAGE,
    "infbench_sum": UNIT_PASSAGE,
    "factconsolidation_sh": UNIT_FACT,
    "factconsolidation_mh": UNIT_FACT,
}

_LENGTH_SUFFIX_RE = re.compile(r"_(?:\d+|\d+k|full)$", re.IGNORECASE)
_DOCUMENT_SPLIT_RE = re.compile(r"(?m)^(?=Document \d+:)")
_DIALOGUE_SPLIT_RE = re.compile(r"(?m)^(?=Dialogue \d+:)")
_FACT_LINE_RE = re.compile(r"(?m)^(\d+)\.\s+(.*)$")
_LINE_NUMBER_RE = re.compile(r"(?m)^\[(\d+)\]\s?")


def base_source(label: str) -> str:
    """Strips a context-length suffix to get the underlying source name.

    ``eventqa_131072`` and ``factconsolidation_sh_32k`` are length variants of
    ``eventqa`` and ``factconsolidation_sh``. ``longmemeval_s*`` carries a
    literal asterisk in the released data, which also comes off here.
    """
    cleaned = label.rstrip("*")
    if cleaned.startswith("longmemeval"):
        return "longmemeval"
    if cleaned.startswith("icl_"):
        # ICL labels carry a shot count and a balance marker, e.g.
        # icl_banking77_5900shot_balance.
        for name in CAPABILITIES:
            if name.startswith("icl_") and cleaned.startswith(name):
                return name
        return cleaned
    if cleaned.startswith("recsys_redial"):
        return "recsys_redial"
    if cleaned.startswith("detective_qa"):
        return "detective_qa"
    if cleaned.startswith("infbench_sum"):
        return "infbench_sum"
    return _LENGTH_SUFFIX_RE.sub("", cleaned)


@dataclass(frozen=True, slots=True)
class Unit:
    """One thing to store, with where it came from."""

    unit_id: str
    text: str
    ordinal: int
    serial: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {"unit_id": self.unit_id, "ordinal": self.ordinal}
        if self.serial is not None:
            # Precedence for conflict resolution: a larger serial is newer.
            out["serial"] = self.serial
        return out


@dataclass(frozen=True, slots=True)
class Task:
    """One question against an ingested context."""

    qid: str
    source: str
    label: str
    capability: str
    question: str
    answers: tuple[str, ...]
    demo: str | None = None
    keypoints: tuple[str, ...] = ()
    previous_events: str | None = None


@dataclass(frozen=True, slots=True)
class Block:
    """One row: a context to ingest plus the questions asked of it."""

    block_id: str
    label: str
    source: str
    capability: str
    unit_kind: str
    context: str
    tasks: tuple[Task, ...]

    def units(self, *, passage_chars: int = 2000) -> list[Unit]:
        """Splits the context into ingestion units appropriate to its source.

        Args:
          passage_chars: Target size for prose passages, which have no intrinsic
            unit boundary.

        Returns:
          Units in source order.
        """
        return list(_split(self.context, self.unit_kind, passage_chars))


def _split(context: str, kind: str, passage_chars: int) -> Iterator[Unit]:
    """Divides a context according to its source's natural unit."""
    if kind == UNIT_FACT:
        # The serial prefix is precedence, so it is parsed out and kept as
        # metadata as well as left in the text the reader sees.
        for ordinal, match in enumerate(_FACT_LINE_RE.finditer(context)):
            serial = int(match.group(1))
            yield Unit(f"fact-{serial}", match.group(0).strip(), ordinal, serial)
        return

    if kind == UNIT_DOCUMENT:
        splitter = _DOCUMENT_SPLIT_RE if "Document 1:" in context else _DIALOGUE_SPLIT_RE
        parts = [part.strip() for part in splitter.split(context) if part.strip()]
        for ordinal, part in enumerate(parts):
            yield Unit(f"doc-{ordinal}", part, ordinal)
        return

    if kind == UNIT_DEMONSTRATION:
        # Demonstrations are "<utterance>\nlabel: <int>" separated by a blank
        # line. An example must never be split from its label, which is why
        # these are not chunked by size.
        parts = [part.strip() for part in context.split("\n\n") if part.strip()]
        for ordinal, part in enumerate(parts):
            yield Unit(f"demo-{ordinal}", part, ordinal)
        return

    if kind == UNIT_SESSION:
        # A whole session is far too coarse a unit: they average about 14,000
        # characters here, so one embedding has to stand for dozens of unrelated
        # turns and cannot discriminate between them. The paper uses token
        # windows for every MemoryAgentBench source except fact consolidation,
        # and measured against this benchmark the session-level choice cost 40
        # points on this source alone.
        #
        # Each window keeps the session's timestamp line, because these
        # questions are date-sensitive and a window cut from the middle of a
        # session would otherwise lose the only temporal anchor it has.
        ordinal = 0
        for session in _split_sessions(context):
            lines = session.split("\n")
            stamp = lines[0] if lines and lines[0].startswith("Chat Time:") else ""
            body = "\n".join(lines[1:]) if stamp else session
            for window in _pack_prose(body, passage_chars):
                text = f"{stamp}\n{window}" if stamp else window
                yield Unit(f"session-{ordinal}", text, ordinal)
                ordinal += 1
        return

    # Prose: split on paragraph boundaries, packing up to the target size, so a
    # unit is a readable passage rather than an arbitrary cut.
    for ordinal, part in enumerate(_pack_prose(context, passage_chars)):
        yield Unit(f"passage-{ordinal}", part, ordinal)


def _split_sessions(context: str) -> list[str]:
    """Parses the LongMemEval source's stringified session list.

    The context is a Python repr, so it is parsed as a literal rather than as
    JSON. Its shape is *flat* and alternating -- ``['Chat Time: ...', [turns],
    'Chat Time: ...', [turns], ...]`` -- not a list of (stamp, turns) pairs.
    Reading it as pairs iterates the timestamp string character by character and
    yields sessions made of single letters, which is a store that looks
    populated and retrieves nothing.

    Returns:
      One rendered session per timestamp, or the whole context as a single unit
      if it will not parse. Never an empty list: zero units would present an
      empty store as a legitimately empty one.
    """
    import ast

    try:
        parsed = ast.literal_eval(context)
    except (ValueError, SyntaxError, MemoryError):
        return [context]
    if not isinstance(parsed, (list, tuple)):
        return [context]

    def render(stamp: object, turns: object) -> str:
        lines = [str(stamp)] if stamp is not None else []
        for turn in turns if isinstance(turns, (list, tuple)) else ():
            if isinstance(turn, dict):
                lines.append(f"{turn.get('role', '?')}: {turn.get('content', '')}")
            else:
                lines.append(str(turn))
        return "\n".join(lines)

    sessions: list[str] = []
    pending_stamp: object | None = None
    for entry in parsed:
        if isinstance(entry, str):
            # A stamp; the turns for it are the next element.
            if pending_stamp is not None:
                sessions.append(render(pending_stamp, None))
            pending_stamp = entry
        elif isinstance(entry, (list, tuple)):
            if entry and isinstance(entry[0], str) and len(entry) >= 2:
                # Tolerate the paired shape too, in case a future release uses it.
                sessions.append(render(entry[0], entry[1]))
                pending_stamp = None
            else:
                sessions.append(render(pending_stamp, entry))
                pending_stamp = None
    if pending_stamp is not None:
        sessions.append(render(pending_stamp, None))
    return sessions or [context]


def _pack_prose(context: str, target: int) -> list[str]:
    """Packs paragraphs into passages of roughly ``target`` characters."""
    text = _LINE_NUMBER_RE.sub("", context)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    out: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if size and size + len(paragraph) > target:
            out.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        out.append("\n".join(current))
    return out


def load(
    directory: Path, *, variants: dict[str, str] | None = None
) -> list[Block]:
    """Loads the published question selection from the parquet splits.

    Args:
      directory: The dataset root holding ``data/*.parquet``.
      variants: Overrides for which length variant to take per source; defaults
        to :data:`DEFAULT_VARIANTS`.

    Returns:
      Blocks for the selected variants, in source order.

    Raises:
      GoodMemError: If pyarrow is unavailable or no parquet files are found.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GoodMemError("MemoryAgentBench needs pyarrow") from exc

    chosen = {**DEFAULT_VARIANTS, **(variants or {})}
    files = sorted((directory / "data").glob("*.parquet"))
    if not files:
        raise GoodMemError(f"no parquet files under {directory / 'data'}")

    blocks: list[Block] = []
    for file in files:
        table = pq.read_table(file)
        contexts = table.column("context").to_pylist()
        questions = table.column("questions").to_pylist()
        answers = table.column("answers").to_pylist()
        metadata = table.column("metadata").to_pylist()

        for row, (context, qs, ans, meta) in enumerate(
            zip(contexts, questions, answers, metadata)
        ):
            label = str(meta.get("source") or "")
            source = base_source(label)
            # Keep only the chosen length variant for sources that have several.
            wanted = chosen.get(source)
            if wanted and label.rstrip("*") != wanted:
                continue
            capability = CAPABILITIES.get(source)
            if capability is None:
                continue

            ids = meta.get("qa_pair_ids") or []
            keypoints = meta.get("keypoints") or []
            events = meta.get("previous_events") or []
            tasks = tuple(
                Task(
                    qid=str(ids[i]) if i < len(ids) else f"{label}-{row}-{i}",
                    source=source,
                    label=label,
                    capability=capability,
                    question=str(q),
                    answers=tuple(str(a) for a in (ans[i] if i < len(ans) else []) or []),
                    demo=meta.get("demo"),
                    keypoints=tuple(str(k) for k in (keypoints[i:i + 1] or [])),
                    previous_events=events[i] if i < len(events) else None,
                )
                for i, q in enumerate(qs)
            )
            blocks.append(
                Block(
                    block_id=f"{label}-{row}",
                    label=label,
                    source=source,
                    capability=capability,
                    unit_kind=UNITS[source],
                    context=str(context),
                    tasks=tasks,
                )
            )
    return blocks


def question_total(blocks: list[Block]) -> int:
    """Counts questions across blocks."""
    return sum(len(block.tasks) for block in blocks)


def composition(blocks: list[Block]) -> dict[str, int]:
    """Returns question counts per source, for verifying the selection."""
    out: dict[str, int] = {}
    for block in blocks:
        out[block.source] = out.get(block.source, 0) + len(block.tasks)
    return out


def write_manifest(path: Path, blocks: list[Block]) -> None:
    """Records what was selected, so a run states its own question set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "questions": question_total(blocks),
                "blocks": len(blocks),
                "by_source": composition(blocks),
                "by_capability": {
                    capability: sum(
                        len(b.tasks) for b in blocks if b.capability == capability
                    )
                    for capability in sorted({b.capability for b in blocks})
                },
                "labels": sorted({b.label for b in blocks}),
            },
            indent=2,
        )
        + "\n"
    )
