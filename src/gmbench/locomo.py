"""LoCoMo dataset loading.

LoCoMo ships ten multi-session conversations between two named speakers. Each
session carries a date, and each turn a stable dialogue id (``D<session>:<n>``)
that the question set cites as evidence. The 10-conversation release holds 1,986
questions; category 5 is adversarial and the paper excludes it, leaving 1,540.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

_SESSION_RE = re.compile(r"^session_(\d+)$")
_DATE_FORMAT = "%I:%M %p on %d %B, %Y"

CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


@dataclass(frozen=True, slots=True)
class Turn:
    """One speaker utterance inside a session."""

    dia_id: str
    session_id: int
    turn_index: int
    speaker: str
    text: str
    session_date: datetime
    image_caption: str | None = None

    @property
    def timestamp(self) -> str:
        return self.session_date.strftime("%Y-%m-%d %H:%M")

    def render(self) -> str:
        """Renders the turn as reader-visible evidence text."""
        body = self.text
        if self.image_caption:
            body = f"{body} [shared an image: {self.image_caption}]"
        return f"[{self.timestamp}] {self.speaker}: {body}"


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark question about a single conversation."""

    conversation_id: str
    index: int
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]

    @property
    def qid(self) -> str:
        """A stable id that survives across stages and runs.

        The dataset holds a handful of duplicated question strings within a
        conversation, and ``hash`` is salted per process, so position in the
        conversation's question list is what makes this both unique and
        reproducible.
        """
        return f"{self.conversation_id}:q{self.index:04d}:c{self.category}"

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))


@dataclass(frozen=True)
class Conversation:
    """A full multi-session conversation plus its questions.

    This one is deliberately not ``slots=True``: the cached lookup tables below
    need an instance dict, and there are only ten of these objects per run.
    """

    conversation_id: str
    speaker_a: str
    speaker_b: str
    turns: tuple[Turn, ...]
    questions: tuple[Question, ...]

    @cached_property
    def by_dia_id(self) -> dict[str, Turn]:
        return {turn.dia_id: turn for turn in self.turns}

    @cached_property
    def by_turn_index(self) -> dict[int, Turn]:
        return {turn.turn_index: turn for turn in self.turns}

    def neighbors(self, turn_index: int, radius: int) -> list[Turn]:
        """Returns the turn at ``turn_index`` with up to ``radius`` turns either side."""
        lo, hi = turn_index - radius, turn_index + radius
        return [self.by_turn_index[i] for i in range(lo, hi + 1) if i in self.by_turn_index]


def _parse_session_date(raw: str) -> datetime:
    """Parses a LoCoMo session timestamp such as ``1:56 pm on 8 May, 2023``."""
    return datetime.strptime(raw.strip(), _DATE_FORMAT)


def _answer_text(record: dict) -> str:
    """Extracts the reference answer, which is a plain value for every category."""
    for key in ("answer", "adversarial_answer"):
        if key in record and record[key] is not None:
            return str(record[key])
    return ""


def load(path: Path, *, conversations: tuple[str, ...] = (), categories: tuple[int, ...] = ()) -> list[Conversation]:
    """Loads LoCoMo conversations from the release JSON.

    Args:
      path: Path to ``locomo10.json``.
      conversations: When non-empty, keeps only these ``sample_id`` values.
      categories: When non-empty, keeps only questions in these categories.

    Returns:
      Conversations in file order, each with its turns and retained questions.
    """
    records = json.loads(path.read_text())
    wanted_convs = frozenset(conversations)
    wanted_cats = frozenset(categories)
    out: list[Conversation] = []

    for record in records:
        sample_id = record["sample_id"]
        if wanted_convs and sample_id not in wanted_convs:
            continue

        conv = record["conversation"]
        session_ids = sorted(
            int(match.group(1))
            for key in conv
            if (match := _SESSION_RE.match(key)) is not None
        )

        turns: list[Turn] = []
        turn_index = 0
        for session_id in session_ids:
            session_date = _parse_session_date(conv[f"session_{session_id}_date_time"])
            for raw in conv[f"session_{session_id}"]:
                caption = raw.get("blip_caption")
                turns.append(
                    Turn(
                        dia_id=raw["dia_id"],
                        session_id=session_id,
                        turn_index=turn_index,
                        speaker=raw["speaker"],
                        text=raw.get("text", ""),
                        session_date=session_date,
                        image_caption=caption or None,
                    )
                )
                turn_index += 1

        # The index is the question's position in the full, unfiltered question
        # list, so a category filter never renumbers the ids.
        questions = tuple(
            Question(
                conversation_id=sample_id,
                index=position,
                question=qa["question"],
                answer=_answer_text(qa),
                category=int(qa["category"]),
                evidence=tuple(qa.get("evidence") or ()),
            )
            for position, qa in enumerate(record.get("qa", ()))
            if not wanted_cats or int(qa["category"]) in wanted_cats
        )

        out.append(
            Conversation(
                conversation_id=sample_id,
                speaker_a=conv["speaker_a"],
                speaker_b=conv["speaker_b"],
                turns=tuple(turns),
                questions=questions,
            )
        )

    return out
