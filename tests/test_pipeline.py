"""Tests for the parts of the harness that decide a score.

The grading rules and the evidence bookkeeping are where a bug would quietly
move a published number, so those are what these cover.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from gmbench.client import RetrievedChunk
from gmbench.coverage import compute
from gmbench.judge import JudgeRecord, accuracy_by_category, parse_label
from gmbench.locomo import Conversation, Question, Turn, load
from gmbench.reader import extract_final_answer
from gmbench.retrieve import RetrievalRecord, expand, ranked_dia_ids, render_context

DATASET = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
needs_dataset = pytest.mark.skipif(not DATASET.exists(), reason="LoCoMo dataset not present")


def _turn(index: int, session: int = 1, text: str = "hello") -> Turn:
    return Turn(
        dia_id=f"D{session}:{index + 1}",
        session_id=session,
        turn_index=index,
        speaker="Amy" if index % 2 == 0 else "Rob",
        text=text,
        session_date=datetime(2023, 5, 8, 13, 56),
    )


def _conversation(turns: list[Turn]) -> Conversation:
    return Conversation(
        conversation_id="conv-test",
        speaker_a="Amy",
        speaker_b="Rob",
        turns=tuple(turns),
        questions=(),
    )


def _chunk(dia_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        memory_id="m-" + dia_id,
        chunk_text="irrelevant",
        chunk_sequence_number=0,
        relevance_score=score,
        metadata={"dia_id": dia_id},
    )


def _ids(*dia_ids: str) -> list[str]:
    return list(dia_ids)


# -- dataset -------------------------------------------------------------


@needs_dataset
def test_dataset_matches_published_counts() -> None:
    """The non-adversarial split must be the 1,540 questions the paper reports."""
    convs = load(DATASET, categories=(1, 2, 3, 4))
    counts: dict[int, int] = {}
    for conv in convs:
        for question in conv.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
    assert len(convs) == 10
    assert counts == {1: 282, 2: 321, 3: 96, 4: 841}
    assert sum(counts.values()) == 1540


@needs_dataset
def test_question_ids_are_unique_and_filter_stable() -> None:
    """Ids must survive a category filter so stages can be run separately."""
    everything = load(DATASET)
    filtered = load(DATASET, categories=(1, 2, 3, 4))
    ids = [q.qid for c in filtered for q in c.questions]
    assert len(ids) == len(set(ids))
    expected = {q.qid for c in everything for q in c.questions if q.category != 5}
    assert set(ids) == expected


def test_turn_render_includes_timestamp_speaker_and_caption() -> None:
    turn = Turn(
        dia_id="D1:5",
        session_id=1,
        turn_index=4,
        speaker="Amy",
        text="Look at this",
        session_date=datetime(2023, 5, 8, 13, 56),
        image_caption="a dog on a wall",
    )
    rendered = turn.render()
    assert rendered.startswith("[2023-05-08 13:56] Amy: Look at this")
    assert "[shared an image: a dog on a wall]" in rendered


# -- expansion and rendering --------------------------------------------


def test_expand_adds_neighbours_dedupes_and_orders() -> None:
    """Overlapping selections must not duplicate turns, and must come back in order."""
    conversation = _conversation([_turn(i) for i in range(10)])
    selected, turns = expand(conversation, _ids("D1:5", "D1:6"), radius=2)
    assert selected == ["D1:5", "D1:6"]
    assert [t.turn_index for t in turns] == [2, 3, 4, 5, 6, 7]


def test_expand_clamps_at_conversation_edges() -> None:
    conversation = _conversation([_turn(i) for i in range(3)])
    _, turns = expand(conversation, _ids("D1:1"), radius=5)
    assert [t.turn_index for t in turns] == [0, 1, 2]


def test_expand_ignores_chunks_without_a_known_turn() -> None:
    conversation = _conversation([_turn(i) for i in range(3)])
    selected, turns = expand(conversation, _ids("D9:99", "D1:2"), radius=0)
    assert selected == ["D1:2"]
    assert [t.dia_id for t in turns] == ["D1:2"]


def test_render_context_marks_session_breaks_and_gaps() -> None:
    turns = [_turn(0, session=1), _turn(5, session=1), _turn(6, session=2)]
    rendered = render_context(turns)
    assert rendered.count("--- Session") == 2
    assert "[...]" in rendered, "a skipped run of turns must be visible to the reader"


# -- coverage ------------------------------------------------------------


def _record(gold: tuple[str, ...], context: list[str], selected: list[str]) -> RetrievalRecord:
    return RetrievalRecord(
        qid="q",
        conversation_id="conv-test",
        category=1,
        question="?",
        answer="a",
        gold_evidence=gold,
        selected_dia_ids=selected,
        context_dia_ids=context,
        context="x" * 10,
    )


def test_coverage_separates_full_partial_and_recall() -> None:
    records = [
        _record(("D1:1", "D1:2"), ["D1:1", "D1:2"], ["D1:1"]),  # full
        _record(("D1:1", "D1:2"), ["D1:1"], ["D1:1"]),  # partial, recall 0.5
        _record(("D1:9",), ["D1:1"], ["D1:1"]),  # miss
    ]
    overall = compute(records)[-1]
    assert overall.label == "overall"
    assert overall.questions == 3
    assert overall.full_coverage == pytest.approx(100 / 3)
    assert overall.partial_coverage == pytest.approx(200 / 3)
    assert overall.evidence_recall == pytest.approx(100 * 1.5 / 3)


def test_coverage_ignores_questions_without_annotated_evidence() -> None:
    records = [_record((), ["D1:1"], ["D1:1"]), _record(("D1:1",), ["D1:1"], ["D1:1"])]
    overall = compute(records)[-1]
    assert overall.questions == 1
    assert overall.full_coverage == pytest.approx(100.0)


# -- reader extraction ---------------------------------------------------


def test_extract_final_answer_takes_text_after_the_marker() -> None:
    response = "## STEP 1\nstuff\n\n## FINAL ANSWER:\n7 May 2023"
    assert extract_final_answer(response) == "7 May 2023"


def test_extract_final_answer_falls_back_to_whole_response() -> None:
    assert extract_final_answer("just an answer") == "just an answer"


# -- judge parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ('{"label": "CORRECT"}', "CORRECT"),
        ('```json\n{"label": "CORRECT"}\n```', "CORRECT"),
        ('```\n{"label": "WRONG"}\n```', "WRONG"),
        ('Reasoning here. {"label": "CORRECT"} done.', "CORRECT"),
        ('  {"label": " correct "}  ', "CORRECT"),
    ],
)
def test_parse_label_accepts_json_envelopes(response: str, expected: str) -> None:
    assert parse_label(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "CORRECT",
        "The answer is CORRECT.",
        "{not json}",
        '{"verdict": "CORRECT"}',
        '["CORRECT"]',
    ],
)
def test_parse_label_rejects_everything_else(response: str) -> None:
    """A bare label with no JSON envelope scores zero, per the paper's rules."""
    assert parse_label(response) is None


def test_parse_label_prefers_the_code_block() -> None:
    response = 'prose {"label": "WRONG"} then\n```json\n{"label": "CORRECT"}\n```'
    assert parse_label(response) == "CORRECT"


# -- aggregation ---------------------------------------------------------


def test_accuracy_averages_repetitions_then_questions() -> None:
    """Accuracy is the question-weighted mean of per-question judgement means."""
    records = [
        JudgeRecord(qid="a", conversation_id="c", category=4, scores=[1, 1, 1]),
        JudgeRecord(qid="b", conversation_id="c", category=4, scores=[1, 0, 0]),
        JudgeRecord(qid="c", conversation_id="c", category=1, scores=[0, 0, 0]),
    ]
    result = accuracy_by_category(records)
    assert result["single_hop"] == (2, pytest.approx(100 * (1 + 1 / 3) / 2))
    assert result["multi_hop"] == (1, pytest.approx(0.0))
    assert result["overall"][0] == 3
    assert result["overall"][1] == pytest.approx(100 * (1 + 1 / 3) / 3)


def test_ranked_dia_ids_preserves_order_and_drops_unlabelled() -> None:
    chunks = [_chunk("D1:3"), RetrievedChunk("m", "t", 0, 0.5, {}), _chunk("D1:1")]
    assert ranked_dia_ids(chunks) == ["D1:3", "D1:1"]


def test_retrieval_record_round_trips_through_json() -> None:
    record = _record(("D1:1",), ["D1:1", "D1:2"], ["D1:1"])
    restored = RetrievalRecord.from_json(json.loads(json.dumps(record.to_json())))
    assert restored.to_json() == record.to_json()


# -- rate limiting -------------------------------------------------------


def test_rate_limiter_spaces_requests_out() -> None:
    """Permits are scheduled, so a burst of callers cannot all go at once."""
    import time

    from gmbench.ratelimit import RateLimiter

    limiter = RateLimiter(per_second=50.0)
    started = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    # Five permits at 50/s is four intervals of 20ms once the first is free.
    assert time.monotonic() - started >= 0.07


def test_rate_limiter_disabled_when_rate_is_not_positive() -> None:
    import time

    from gmbench.ratelimit import RateLimiter

    limiter = RateLimiter(per_second=0.0)
    assert not limiter.enabled
    started = time.monotonic()
    for _ in range(100):
        limiter.acquire()
    assert time.monotonic() - started < 0.05


def test_rate_limiter_pause_holds_subsequent_callers() -> None:
    """A 429 should stand the whole pool down, not just the worker that saw it."""
    import time

    from gmbench.ratelimit import RateLimiter

    limiter = RateLimiter(per_second=1000.0)
    limiter.pause(0.15)
    started = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - started >= 0.12


def test_rate_limiter_per_minute_converts_the_budget() -> None:
    from gmbench.ratelimit import RateLimiter

    limiter = RateLimiter.per_minute(1500)
    assert limiter.enabled
    assert limiter._interval == pytest.approx(60.0 / 1500)
