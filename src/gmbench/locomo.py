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

# The composition of the 10-conversation release, by category. The paper
# evaluates the four non-adversarial categories, which come to 1,540 questions;
# category 5 is adversarial and excluded there. A file that does not match is a
# different release, and a score measured on it would not be comparable.
EXPECTED_CONVERSATIONS = 10
EXPECTED_CATEGORY_COUNTS: dict[int, int] = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
EVALUATED_CATEGORIES: tuple[int, ...] = (1, 2, 3, 4)

# A dialogue reference as the dataset defines it: session and turn.
_DIA_ID_RE = re.compile(r"^D\d+:\d+$")
# Six non-adversarial questions in the 10-conversation release pack several
# references into one string, separated by spaces or semicolons, and two carry
# references that are not recoverable at all ("D", "D:11:26"). Splitting the
# first kind is safe; the second kind is recorded rather than guessed at, since
# inventing a reference would silently change what coverage measures.
_DIA_ID_SPLIT_RE = re.compile(r"[;,\s]+")

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
    evidence_unparsed: tuple[str, ...] = ()

    @property
    def evidence_is_measurable(self) -> bool:
        """True when at least one reference was recoverable.

        A question whose entire evidence list is unparseable cannot be scored
        for coverage. Treating it as a miss would understate retrieval for a
        defect in the dataset rather than in the system under test.
        """
        return bool(self.evidence)

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

    def resolvable_evidence(self, question: Question) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Splits a question's evidence into references this conversation has.

        A handful of references in the released dataset point at turns that are
        not in the conversation (three in the non-adversarial split). Scoring
        those as retrieval misses would charge the system under test for someone
        else's typo, so they are separated out and reported instead.

        Args:
          question: The question whose evidence to resolve.

        Returns:
          The references present in this conversation, and those absent from it.
        """
        known = self.by_dia_id
        present = tuple(ref for ref in question.evidence if ref in known)
        absent = tuple(ref for ref in question.evidence if ref not in known)
        return present, absent

    def neighbors(self, turn_index: int, radius: int) -> list[Turn]:
        """Returns the turn at ``turn_index`` with up to ``radius`` turns either side."""
        lo, hi = turn_index - radius, turn_index + radius
        return [self.by_turn_index[i] for i in range(lo, hi + 1) if i in self.by_turn_index]


class DatasetError(RuntimeError):
    """Raised when the loaded dataset is not the release this harness expects."""


def _parse_session_date(raw: str) -> datetime:
    """Parses a LoCoMo session timestamp such as ``1:56 pm on 8 May, 2023``."""
    return datetime.strptime(raw.strip(), _DATE_FORMAT)


def _normalize_evidence(raw: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Splits evidence references, separating the recoverable from the rest.

    Args:
      raw: The dataset's ``evidence`` value, normally a list of strings.

    Returns:
      Well-formed dialogue references, and the residue that could not be parsed.
    """
    refs: list[str] = []
    residue: list[str] = []
    items = raw if isinstance(raw, (list, tuple)) else ([raw] if raw else [])
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        parts = [part for part in _DIA_ID_SPLIT_RE.split(text) if part]
        for part in parts:
            if _DIA_ID_RE.match(part):
                refs.append(part)
            else:
                residue.append(part)
    # Order-preserving de-duplication; a reference repeated across entries is
    # one piece of required evidence, not two.
    seen: set[str] = set()
    unique = [ref for ref in refs if not (ref in seen or seen.add(ref))]
    return tuple(unique), tuple(residue)


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
        questions: list[Question] = []
        for position, qa in enumerate(record.get("qa", ())):
            category = int(qa["category"])
            if wanted_cats and category not in wanted_cats:
                continue
            refs, residue = _normalize_evidence(qa.get("evidence"))
            questions.append(
                Question(
                    conversation_id=sample_id,
                    index=position,
                    question=qa["question"],
                    answer=_answer_text(qa),
                    category=category,
                    evidence=refs,
                    evidence_unparsed=residue,
                )
            )

        out.append(
            Conversation(
                conversation_id=sample_id,
                speaker_a=conv["speaker_a"],
                speaker_b=conv["speaker_b"],
                turns=tuple(turns),
                questions=tuple(questions),
            )
        )

    _validate(out, filtered_conversations=bool(wanted_convs), filtered_categories=wanted_cats)
    return out


def _validate(
    conversations: list[Conversation],
    *,
    filtered_conversations: bool,
    filtered_categories: frozenset[int],
) -> None:
    """Checks the loaded data against the release the paper evaluates.

    Structural invariants are always checked. The published counts are only
    checked on a full, unfiltered load, since a smoke run deliberately takes a
    subset.

    Raises:
      DatasetError: If the data is internally inconsistent, or is a full load
        that does not match the published composition.
    """
    for conversation in conversations:
        dia_ids = [turn.dia_id for turn in conversation.turns]
        if len(dia_ids) != len(set(dia_ids)):
            raise DatasetError(
                f"{conversation.conversation_id}: duplicate dialogue ids, so evidence "
                "references would be ambiguous"
            )
        for question in conversation.questions:
            if not question.answer.strip():
                raise DatasetError(
                    f"{question.qid}: no reference answer, which would be scored as a "
                    "wrong answer rather than a missing one"
                )

    if filtered_conversations:
        return
    if len(conversations) != EXPECTED_CONVERSATIONS:
        raise DatasetError(
            f"expected {EXPECTED_CONVERSATIONS} conversations, loaded "
            f"{len(conversations)}; this is not the 10-conversation release"
        )

    counts: dict[int, int] = {}
    for conversation in conversations:
        for question in conversation.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
    # An unknown category in the filter means a release this check cannot speak
    # to, so it stays quiet rather than failing on its own ignorance.
    if filtered_categories and not filtered_categories <= set(EXPECTED_CATEGORY_COUNTS):
        return
    wanted = {
        category: total
        for category, total in EXPECTED_CATEGORY_COUNTS.items()
        if not filtered_categories or category in filtered_categories
    }
    if counts != wanted:
        raise DatasetError(
            f"question counts {counts} do not match the published composition "
            f"{wanted}; a score measured here would not be comparable"
        )
