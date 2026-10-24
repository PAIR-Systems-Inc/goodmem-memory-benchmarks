"""Tests for the parts of the harness that decide a score.

The grading rules and the evidence bookkeeping are where a bug would quietly
move a published number, so those are what these cover.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from gmbench.client import GoodMemError, RetrievedChunk
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


def test_expand_rejects_a_dia_id_the_dataset_does_not_have() -> None:
    """An unresolvable selection means index and dataset disagree, so it must fail."""
    conversation = _conversation([_turn(i) for i in range(3)])
    with pytest.raises(GoodMemError, match="disagree"):
        expand(conversation, _ids("D9:99", "D1:2"), radius=0)


def test_render_context_marks_session_breaks_and_gaps() -> None:
    turns = [_turn(0, session=1), _turn(5, session=1), _turn(6, session=2)]
    rendered = render_context(turns)
    assert rendered.count("--- Session") == 2
    assert "[...]" in rendered, "a skipped run of turns must be visible to the reader"


# -- coverage ------------------------------------------------------------


def _record(
    gold: tuple[str, ...],
    context: list[str],
    selected: list[str],
    *,
    qid: str = "q",
    unresolvable: tuple[str, ...] = (),
) -> RetrievalRecord:
    return RetrievalRecord(
        qid=qid,
        conversation_id="conv-test",
        category=1,
        question="?",
        answer="a",
        gold_evidence=gold,
        gold_unresolvable=unresolvable,
        selected_dia_ids=selected,
        context_dia_ids=context,
        context="x" * 10,
    )


def test_coverage_separates_full_partial_and_recall() -> None:
    records = [
        _record(("D1:1", "D1:2"), ["D1:1", "D1:2"], ["D1:1"], qid="a"),  # full
        _record(("D1:1", "D1:2"), ["D1:1"], ["D1:1"], qid="b"),  # partial, recall 0.5
        _record(("D1:9",), ["D1:1"], ["D1:1"], qid="c"),  # miss
    ]
    overall = compute(records)[-1]
    assert overall.label == "overall"
    assert overall.questions == 3
    assert overall.full_coverage == pytest.approx(100 / 3)
    assert overall.partial_coverage == pytest.approx(200 / 3)
    assert overall.evidence_recall == pytest.approx(100 * 1.5 / 3)


def test_coverage_ignores_questions_without_annotated_evidence() -> None:
    records = [
        _record((), ["D1:1"], ["D1:1"], qid="a"),
        _record(("D1:1",), ["D1:1"], ["D1:1"], qid="b"),
    ]
    overall = compute(records)[-1]
    assert overall.questions == 1
    assert overall.excluded == 0
    assert overall.full_coverage == pytest.approx(100.0)


def test_coverage_excludes_rather_than_penalises_unresolvable_evidence() -> None:
    """A reference absent from its own conversation is the dataset's defect."""
    records = [
        _record(("D1:1",), ["D1:1"], ["D1:1"], qid="a"),
        _record((), ["D1:1"], ["D1:1"], qid="b", unresolvable=("D9:99",)),
    ]
    overall = compute(records)[-1]
    assert overall.questions == 1, "the unresolvable question must not be scored"
    assert overall.excluded == 1, "and must be counted as an exclusion"
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


def test_ranked_dia_ids_preserves_rank_order() -> None:
    assert ranked_dia_ids([_chunk("D1:3"), _chunk("D1:1")]) == ["D1:3", "D1:1"]


def test_ranked_dia_ids_rejects_a_chunk_with_no_dia_id() -> None:
    """Silently dropping one would shrink the evidence with nothing to explain it."""
    chunks = [_chunk("D1:3"), RetrievedChunk("m", "t", 0, 0.5, {})]
    with pytest.raises(GoodMemError, match="no dia_id"):
        ranked_dia_ids(chunks)


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


# -- completeness gates --------------------------------------------------


def test_completeness_accepts_an_exact_cover() -> None:
    from gmbench import validate

    report = validate.check("read", ["a", "b"], ["b", "a"])
    assert report.ok
    validate.require("read", ["a", "b"], ["b", "a"])


def test_completeness_catches_a_short_file() -> None:
    """A short shard is the dangerous case: it looks exactly like a complete one."""
    from gmbench import validate

    report = validate.check("read", ["a", "b", "c"], ["a", "b"])
    assert not report.ok
    assert report.missing == ("c",)
    with pytest.raises(validate.IncompleteStageError, match="1 missing: c"):
        validate.require("read", ["a", "b", "c"], ["a", "b"])


def test_completeness_catches_duplicates_and_strangers() -> None:
    from gmbench import validate

    report = validate.check("judge", ["a", "b"], ["a", "a", "b", "z"])
    assert report.duplicated == ("a",)
    assert report.unexpected == ("z",)
    assert not report.ok


# -- resumable shards ----------------------------------------------------


def test_shard_writer_flushes_each_row(tmp_path) -> None:
    """Rows must be on disk as they land, not at the end of the stage."""
    from gmbench import jsonl

    path = tmp_path / "shard.jsonl"
    with jsonl.ShardWriter(path) as shard:
        shard.write({"qid": "a", "v": 1})
        # Readable from another handle before the writer closes.
        assert [row["qid"] for row in jsonl.read_rows(path)] == ["a"]
        shard.write({"qid": "b", "v": 2})
    assert jsonl.completed_keys(path) == {"a", "b"}


def test_shard_writer_appends_so_a_restart_resumes(tmp_path) -> None:
    from gmbench import jsonl

    path = tmp_path / "shard.jsonl"
    with jsonl.ShardWriter(path) as shard:
        shard.write({"qid": "a"})
    with jsonl.ShardWriter(path) as shard:
        shard.write({"qid": "b"})
    assert jsonl.completed_keys(path) == {"a", "b"}


def test_read_rows_tolerates_a_torn_final_line(tmp_path) -> None:
    """A half-written last row is what an interrupted run leaves behind."""
    from gmbench import jsonl

    path = tmp_path / "shard.jsonl"
    path.write_text('{"qid": "a"}\n{"qid": "b"}\n{"qid": "c", "ge')
    assert [row["qid"] for row in jsonl.read_rows(path)] == ["a", "b"]


def test_read_rows_rejects_corruption_that_is_not_a_torn_tail(tmp_path) -> None:
    from gmbench import jsonl

    path = tmp_path / "shard.jsonl"
    path.write_text('{"qid": "a"}\nnot json at all\n{"qid": "c"}\n')
    with pytest.raises(ValueError, match="corrupt row at line 2"):
        list(jsonl.read_rows(path))


# -- retrieval stream integrity ------------------------------------------


def test_fatal_status_aborts_rather_than_degrading_the_score() -> None:
    """A partial vector search must fail, not quietly return fewer chunks."""
    from gmbench.client import _check_status

    with pytest.raises(GoodMemError, match="VECTOR_SEARCH_PARTIAL"):
        _check_status({"code": "VECTOR_SEARCH_PARTIAL", "message": "partial"})


def test_unknown_status_is_treated_as_fatal() -> None:
    """A warning the server learns to emit should stop a run, not be absorbed."""
    from gmbench.client import _check_status

    with pytest.raises(GoodMemError, match="unrecognised"):
        _check_status({"code": "SOME_FUTURE_WARNING", "message": "hm"})


def test_benign_status_passes() -> None:
    from gmbench.client import _check_status

    _check_status({"code": "FEATURE_DISABLED", "message": "no llm configured"})


def test_chunk_with_an_out_of_range_memory_index_is_rejected() -> None:
    from gmbench.client import _to_chunk

    with pytest.raises(GoodMemError, match="memory index"):
        _to_chunk({"memoryIndex": 3, "chunk": {}}, [{"memoryId": "m"}])


def test_malformed_event_frame_is_rejected() -> None:
    """Skipping a frame loses a retrieved item and lowers coverage unexplained."""
    from gmbench.client import _iter_sse_data

    with pytest.raises(GoodMemError, match="malformed event frame"):
        list(_iter_sse_data(iter(["data: {not json"])))


# -- result-set framing --------------------------------------------------


def _sse(events: list[dict]) -> list[str]:
    import json as _json

    return [f"data: {_json.dumps(event)}" for event in events]


def _fake_stream(monkeypatch, events: list[dict]):
    """Points a client's retrieve at a canned event stream."""
    import contextlib

    from gmbench.client import GoodMemClient

    client = GoodMemClient("https://example.invalid", "key", tls_verify=False)

    class _Response:
        status_code = 200
        text = ""

        def iter_lines(self):
            return iter(_sse(events))

        def read(self):
            return b""

    @contextlib.contextmanager
    def _stream(*args, **kwargs):
        yield _Response()

    monkeypatch.setattr(client._client, "stream", _stream)
    return client


def _begin(set_id: str, expected: int) -> dict:
    return {
        "resultSetBoundary": {
            "resultSetId": set_id,
            "kind": "BEGIN",
            "stageName": "rerank",
            "expectedItems": expected,
        }
    }


def _end(set_id: str) -> dict:
    # The server names the stage only on BEGIN; END carries an empty stageName.
    return {"resultSetBoundary": {"resultSetId": set_id, "kind": "END", "stageName": ""}}


def _memory(dia_id: str) -> dict:
    return {"memoryDefinition": {"memoryId": f"m-{dia_id}", "metadata": {"dia_id": dia_id}}}


def _item(set_id: str, index: int, dia_id: str) -> dict:
    return {
        "retrievedItem": {
            "chunk": {
                "resultSetId": set_id,
                "chunk": {"memoryId": f"m-{dia_id}", "chunkText": dia_id},
                "memoryIndex": index,
                "relevanceScore": 0.9,
            }
        }
    }


def test_retrieve_accepts_a_complete_result_set(monkeypatch) -> None:
    """END carries no stage name, so the close must be matched by result-set id."""
    client = _fake_stream(
        monkeypatch,
        [
            _begin("rs-1", 2),
            _memory("D1:1"),
            _item("rs-1", 0, "D1:1"),
            _memory("D1:2"),
            _item("rs-1", 1, "D1:2"),
            _end("rs-1"),
        ],
    )
    chunks = client.retrieve(
        "q", "space", dense_candidates=8, final_k=2, reranker_id="rr"
    )
    assert [chunk.metadata["dia_id"] for chunk in chunks] == ["D1:1", "D1:2"]


def test_retrieve_rejects_a_truncated_stream(monkeypatch) -> None:
    """A stream cut short would otherwise return a plausible, shorter result set."""
    client = _fake_stream(
        monkeypatch, [_begin("rs-1", 2), _memory("D1:1"), _item("rs-1", 0, "D1:1")]
    )
    with pytest.raises(GoodMemError, match="truncated"):
        client.retrieve("q", "space", dense_candidates=8, final_k=2, reranker_id="rr")


def test_retrieve_rejects_a_short_delivery(monkeypatch) -> None:
    """The server states how many items to expect, so a shortfall is detectable."""
    client = _fake_stream(
        monkeypatch,
        [_begin("rs-1", 3), _memory("D1:1"), _item("rs-1", 0, "D1:1"), _end("rs-1")],
    )
    with pytest.raises(GoodMemError, match="promised 3 items but delivered 1"):
        client.retrieve("q", "space", dense_candidates=8, final_k=3, reranker_id="rr")


def test_retrieve_rejects_a_missing_rerank_stage(monkeypatch) -> None:
    """Without reranking the results are dense-only and not comparable."""
    client = _fake_stream(
        monkeypatch,
        [
            {
                "resultSetBoundary": {
                    "resultSetId": "rs-9",
                    "kind": "BEGIN",
                    "stageName": "vector-search",
                    "expectedItems": 1,
                }
            },
            _memory("D1:1"),
            _item("rs-9", 0, "D1:1"),
            _end("rs-9"),
        ],
    )
    with pytest.raises(GoodMemError, match="no rerank stage"):
        client.retrieve("q", "space", dense_candidates=8, final_k=1, reranker_id="rr")


# -- MemoryAgentBench selection and unit splitting -----------------------

MAB_DIR = Path(__file__).resolve().parents[1] / "data" / "memoryagentbench"
needs_mab = pytest.mark.skipif(
    not (MAB_DIR / "data").exists(), reason="MemoryAgentBench not present"
)


def test_base_source_strips_length_variants() -> None:
    """22 shipped labels collapse to 14 sources; getting this wrong over-reports."""
    from gmbench.memoryagentbench import base_source

    assert base_source("eventqa_131072") == "eventqa"
    assert base_source("eventqa_full") == "eventqa"
    assert base_source("factconsolidation_sh_262k") == "factconsolidation_sh"
    assert base_source("factconsolidation_mh_6k") == "factconsolidation_mh"
    assert base_source("ruler_qa1_197K") == "ruler_qa1"
    assert base_source("longmemeval_s*") == "longmemeval"
    assert base_source("icl_banking77_5900shot_balance") == "icl_banking77"
    assert base_source("recsys_redial_full") == "recsys_redial"
    assert base_source("infbench_sum_eng_shots2") == "infbench_sum"
    assert base_source("detective_qa") == "detective_qa"


@needs_mab
def test_selection_reproduces_the_published_question_count() -> None:
    """One length variant per source is what sums to the paper's 2,071."""
    from gmbench import memoryagentbench as mab

    blocks = mab.load(MAB_DIR)
    assert mab.question_total(blocks) == 2071
    by_capability: dict[str, int] = {}
    for block in blocks:
        by_capability[block.capability] = by_capability.get(block.capability, 0) + len(
            block.tasks
        )
    assert by_capability == {
        "accurate_retrieval": 1000,
        "test_time_learning": 700,
        "long_range_understanding": 171,
        "conflict_resolution": 200,
    }
    assert len(mab.composition(blocks)) == 14, "all 14 sources must be present"


def test_fact_units_keep_their_serial_as_precedence() -> None:
    """Conflict resolution depends on the serial, so it must survive splitting."""
    from gmbench.memoryagentbench import UNIT_FACT, Block

    block = Block(
        block_id="b",
        label="factconsolidation_sh_6k",
        source="factconsolidation_sh",
        capability="conflict_resolution",
        unit_kind=UNIT_FACT,
        context=(
            "Here is a list of facts:\n"
            "0. goaltender is associated with the sport of ice hockey.\n"
            "1. Something unrelated.\n"
            "310. goaltender is associated with the sport of pesapallo.\n"
        ),
        tasks=(),
    )
    units = block.units()
    assert [u.serial for u in units] == [0, 1, 310]
    assert units[0].metadata["serial"] == 0
    # The serial stays in the text too, since the reader needs to see precedence.
    assert units[2].text.startswith("310.")


def test_demonstration_units_never_split_an_example_from_its_label() -> None:
    from gmbench.memoryagentbench import UNIT_DEMONSTRATION, Block

    block = Block(
        block_id="b",
        label="icl_banking77_5900shot_balance",
        source="icl_banking77",
        capability="test_time_learning",
        unit_kind=UNIT_DEMONSTRATION,
        context="utterance one\nlabel: 7\n\nutterance two\nlabel: 12\n",
        tasks=(),
    )
    units = block.units()
    assert len(units) == 2
    assert all("label:" in unit.text for unit in units)


def test_session_splitting_handles_the_flat_alternating_shape() -> None:
    """The released context alternates stamp, turns — not a list of pairs.

    Read as pairs, the timestamp string is iterated character by character and
    the store fills with single letters: populated-looking and useless.
    """
    from gmbench.memoryagentbench import UNIT_SESSION, Block

    context = repr(
        [
            "Chat Time: 2022/11/17 (Thu) 12:04",
            [{"role": "user", "content": "hello there"}],
            "Chat Time: 2022/11/18 (Fri) 09:00",
            [{"role": "assistant", "content": "hi back"}],
        ]
    )
    block = Block(
        block_id="b",
        label="longmemeval_s*",
        source="longmemeval",
        capability="accurate_retrieval",
        unit_kind=UNIT_SESSION,
        context=context,
        tasks=(),
    )
    units = block.units()
    assert len(units) == 2
    assert units[0].text.startswith("Chat Time: 2022/11/17")
    assert "user: hello there" in units[0].text
    assert "assistant: hi back" in units[1].text


def test_session_splitting_falls_back_rather_than_returning_nothing() -> None:
    from gmbench.memoryagentbench import UNIT_SESSION, Block

    block = Block(
        block_id="b",
        label="longmemeval_s*",
        source="longmemeval",
        capability="accurate_retrieval",
        unit_kind=UNIT_SESSION,
        context="this is not a python literal at all",
        tasks=(),
    )
    units = block.units()
    assert len(units) == 1, "an unparseable context must not yield an empty store"


# -- BEAM ----------------------------------------------------------------

BEAM_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "beam" / "1M-00000-of-00001.parquet"
)
needs_beam = pytest.mark.skipif(not BEAM_FILE.exists(), reason="BEAM not present")


@needs_beam
def test_beam_paper_conversation_has_the_published_message_count() -> None:
    from gmbench import beam

    conversation = beam.load(BEAM_FILE)
    assert len(conversation.messages) == beam.PAPER_MESSAGE_COUNT
    # Ingestion order must be the dataset's own monotonic id.
    indices = [message.index for message in conversation.messages]
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert len(set(indices)) == len(indices)
    # Approaching one million tokens.
    assert 800_000 < conversation.approximate_tokens < 1_200_000


@needs_beam
def test_beam_renders_the_session_anchor_only_where_present() -> None:
    from gmbench import beam

    conversation = beam.load(BEAM_FILE)
    anchored = [m for m in conversation.messages if m.time_anchor]
    assert len(anchored) == 10, "one anchor per session"
    assert anchored[0].render().startswith("[March-01-2024] user:")
    plain = next(m for m in conversation.messages if not m.time_anchor)
    assert plain.render().startswith(f"{plain.role}:")


# -- MemoryAgentBench scoring --------------------------------------------


def test_substring_scoring_accepts_any_gold_paraphrase() -> None:
    """RULER ships four paraphrases per question; any of them counts."""
    from gmbench.mab_run import score

    assert score("The answer is pesapallo.", ["pesapallo"])
    assert score("10th and 11th centuries", ["in the 10th and 11th centuries", "10th and 11th centuries"])
    assert score("  PESAPALLO  ", ["pesapallo"]), "matching is case-insensitive"
    assert score("the  10th   and 11th centuries", ["10th and 11th centuries"]), (
        "whitespace is normalised"
    )


def test_scoring_is_anchored_at_word_boundaries() -> None:
    """Bare containment scores wrong answers on these sources' short golds."""
    from gmbench.mab_run import score

    assert not score("yesterday I worked a lot", ["yes"]), "yes inside yesterday"
    assert score("the answer is yes", ["yes"])
    assert not score("about 150 hours a week", ["50"]), "50 inside 150"
    assert score("about 50 hours a week", ["50"])
    assert score("France.", ["France"]), "trailing punctuation must still match"


def test_substring_scoring_rejects_a_miss_and_an_empty_prediction() -> None:
    from gmbench.mab_run import score

    assert not score("ice hockey", ["pesapallo"])
    assert not score("", ["pesapallo"])
    assert not score("   ", ["pesapallo"])
    assert not score("anything", [])


def test_question_entities_strips_the_relation_template() -> None:
    """What distinguishes these questions is the entity, not the relation."""
    from gmbench.mab_run import question_entities

    assert question_entities("Which sport is Salem-Keizer Volcanoes associated with?") == [
        "salem-keizer volcanoes"
    ]
    assert question_entities(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    ) == ["bagratuni dynasty"]


def test_gold_serial_requires_the_entity_not_just_the_answer() -> None:
    """Matching the answer alone picks a fact about something else entirely.

    A gold string like "basketball" occurs in dozens of facts about unrelated
    entities, and the highest-serial one among those is almost never the fact
    the question concerns.
    """
    from gmbench.mab_run import gold_serial_for
    from gmbench.memoryagentbench import Unit

    question = "Which sport is Salem-Keizer Volcanoes associated with?"
    units = [
        Unit("a", "1257. Salem-Keizer Volcanoes is associated with the sport of baseball.", 0, 1257),
        Unit("b", "1722. Salem-Keizer Volcanoes is associated with the sport of basketball.", 1, 1722),
        # Newer, states the gold answer, but about a different entity.
        Unit("c", "2269. Serbian SuperLiga is associated with the sport of basketball.", 2, 2269),
    ]
    assert gold_serial_for(units, ["basketball"], question) == 1722
    assert gold_serial_for(units, ["baseball"], question) == 1257
    assert gold_serial_for(units, ["cricket"], question) is None


def test_gold_serial_is_none_when_the_answer_fact_omits_the_question_entity() -> None:
    """The multi-hop shape: the answer-bearing fact names the bridging entity.

    That is a property of the task rather than a measurement failure, so the
    caller reports it separately instead of counting it as a retrieval miss.
    """
    from gmbench.mab_run import gold_serial_for
    from gmbench.memoryagentbench import Unit

    units = [
        Unit("a", "300. Bagratuni Dynasty is affiliated with the religion of Christianity.", 0, 300),
        Unit("b", "475. Christianity was founded in the city of Taipei.", 1, 475),
    ]
    question = (
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )
    assert gold_serial_for(units, ["Taipei"], question) is None


def test_knowledge_pool_is_rendered_in_serial_order() -> None:
    """The reader is told a larger serial is newer, so order must carry that."""
    from gmbench.mab_run import render_context
    from gmbench.memoryagentbench import Unit

    units = [
        Unit("c", "310. newest.", 0, 310),
        Unit("a", "1. oldest.", 1, 1),
        Unit("b", "223. middle.", 2, 223),
    ]
    rendered = render_context(units)
    assert rendered.startswith("[Knowledge Pool]")
    assert rendered.index("1. oldest.") < rendered.index("223. middle.")
    assert rendered.index("223. middle.") < rendered.index("310. newest.")


def _run_config(tmp_path):
    """A minimal RunConfig pointed at a temporary results directory."""
    from gmbench import config as cfgmod

    return cfgmod.RunConfig(
        run_tag="t",
        ingest_tag="t",
        dataset_path=DATASET,
        results_dir=tmp_path,
        server=cfgmod.ServerConfig(api_key="k"),
        models=cfgmod.ModelConfig(embedder_id="e", reranker_id="r"),
        chunking=cfgmod.ChunkingConfig(),
        retrieval=cfgmod.RetrievalConfig(concurrency=2),
        ingest=cfgmod.IngestConfig(),
    )


def _outcome(
    qid: str,
    *,
    source: str = "factconsolidation_sh",
    capability: str = "conflict_resolution",
    correct: bool = False,
    retrieved: list[int] | None = None,
    gold: int | None = None,
    partial: float | None = None,
):
    """Builds an Outcome, keeping `partial` consistent with `correct`.

    Aggregation reads `partial`, because summarisation is graded by fractional
    keypoint coverage. Every other source stores 1.0 or 0.0 there, so a helper
    keeps the two from drifting apart in tests.
    """
    from gmbench.mab_run import Outcome

    return Outcome(
        qid=qid,
        source=source,
        capability=capability,
        question="?",
        answers=["x"],
        prediction="x",
        correct=correct,
        retrieved_serials=retrieved or [],
        gold_serial=gold,
        partial=float(correct) if partial is None else partial,
    )


def test_summary_separates_a_reading_failure_from_a_retrieval_one() -> None:
    """Accuracy alone cannot tell those apart, so attribution is tracked."""
    from gmbench.mab_run import Outcome, summarize

    outcomes = [
        # Gold fact retrieved and answered: a clean pass.
        _outcome("q1", correct=True, retrieved=[1, 310], gold=310),
        # Gold fact retrieved but the reader chose an older one: a reading failure.
        _outcome("q2", correct=False, retrieved=[2, 400], gold=400),
        # Gold fact never retrieved: a retrieval failure.
        _outcome("q3", correct=False, retrieved=[5, 6], gold=900),
        # No single fact names the entity and states the answer: not attributable.
        _outcome("q4", correct=False, retrieved=[7], gold=None),
    ]
    document = summarize(outcomes)
    bucket = document["by_source"]["factconsolidation_sh"]
    assert bucket["questions"] == 4
    assert bucket["accuracy"] == pytest.approx(25.0)
    assert bucket["gold_identifiable"] == 3, "q4 has no single gold fact"
    assert bucket["gold_retrieved"] == 2
    # Conditioned on the attributable questions, not on all of them.
    assert bucket["gold_retrieval_rate"] == pytest.approx(200 / 3)
    assert bucket["gold_identifiable_rate"] == pytest.approx(75.0)
    assert document["errors"] == 0


def test_summary_reports_no_retrieval_rate_when_nothing_is_attributable() -> None:
    from gmbench.mab_run import Outcome, summarize

    outcomes = [
        _outcome("q1", source="factconsolidation_mh", correct=False, retrieved=[1], gold=None),
    ]
    bucket = summarize(outcomes)["by_source"]["factconsolidation_mh"]
    assert bucket["gold_retrieval_rate"] is None, "a rate over zero questions is not zero"
    assert bucket["gold_identifiable_rate"] == pytest.approx(0.0)


def test_capability_score_is_a_macro_average_over_sources() -> None:
    """The paper averages source scores, not questions (appendix D.5).

    The two coincide only when sources contribute equally many questions. For
    accurate retrieval they do not: EventQA's 500 questions would outweigh
    RULER's 100 fivefold and change the published figure.
    """
    from gmbench.mab_run import Outcome, summarize

    big = [
        _outcome(f"b{i}", source="eventqa", capability="accurate_retrieval", correct=i < 1)
        for i in range(10)
    ]
    small = [
        _outcome("s0", source="ruler_qa1", capability="accurate_retrieval", correct=True)
    ]
    document = summarize(big + small)
    # eventqa 10%, ruler 100% -> macro 55.0; question-weighted would be ~18.2.
    assert document["accuracy"] == pytest.approx(55.0)
    assert document["accuracy_question_weighted"] == pytest.approx(100 * 2 / 11)


def test_session_units_are_windowed_and_keep_their_timestamp() -> None:
    """A whole session is too coarse to embed, and a window needs its date.

    Sessions here average ~14,000 characters, so one embedding standing for the
    whole thing cannot discriminate between dozens of unrelated turns. These
    questions are also date-sensitive, so a window cut from mid-session must
    carry the timestamp forward or it loses its only temporal anchor.
    """
    from gmbench.memoryagentbench import UNIT_SESSION, Block

    long_turn = "user: " + ("a lorem ipsum sentence about travel plans. " * 80)
    context = repr(
        [
            "Chat Time: 2022/11/17 (Thu) 12:04",
            [{"role": "user", "content": long_turn}, {"role": "assistant", "content": long_turn}],
        ]
    )
    block = Block(
        block_id="b",
        label="longmemeval_s*",
        source="longmemeval",
        capability="accurate_retrieval",
        unit_kind=UNIT_SESSION,
        context=context,
        tasks=(),
    )
    units = block.units(passage_chars=1500)
    assert len(units) > 1, "a long session must split into several windows"
    assert all(u.text.startswith("Chat Time: 2022/11/17") for u in units)
    assert max(len(u.text) for u in units) < 6000


@needs_mab
def test_dataset_question_ids_collide_across_blocks() -> None:
    """A guard on the assumption that made the resume key wrong.

    The released qa_pair_ids repeat: EventQA's five blocks each number their
    questions no0..no99 while asking 500 distinct questions. Anything keyed on
    the dataset id alone silently loses four fifths of them.
    """
    from gmbench import memoryagentbench as mab

    blocks = [b for b in mab.load(MAB_DIR) if b.source == "eventqa"]
    ids = [t.qid for b in blocks for t in b.tasks]
    questions = [t.question for b in blocks for t in b.tasks]
    assert len(questions) == 500
    assert len(set(questions)) == 500, "the questions themselves are distinct"
    assert len(set(ids)) == 100, "but their dataset ids are not"
    # Qualifying by block is what restores uniqueness.
    qualified = [f"{b.block_id}:{t.qid}" for b in blocks for t in b.tasks]
    assert len(set(qualified)) == 500


# -- per-source graders --------------------------------------------------


def test_label_grading_is_exact_not_containment() -> None:
    """Bare integer labels: 7 must not score against a gold of 71."""
    from gmbench.mab_run import score_label

    assert score_label("label: 28", ["28"])
    assert score_label("28", ["28"]), "a bare number still parses"
    assert not score_label("label: 7", ["71"]), "7 inside 71"
    assert not score_label("label: 71", ["7"]), "and the reverse"
    assert not score_label("I am not sure", ["28"])


def test_choice_grading_ignores_the_option_letter() -> None:
    """The released DetectiveQA letters do not match their own option text."""
    from gmbench.mab_run import score_choice

    assert score_choice('{"answer": "The Brandt couple"}', ["C. The Brandt couple"])
    assert score_choice('{"answer": "B. The Brandt couple"}', ["C. The Brandt couple"]), (
        "a disagreeing letter must not fail a correct answer"
    )
    assert not score_choice('{"answer": "Suicide"}', ["C. The Brandt couple"])


def test_recall_grading_resolves_movie_ids_to_titles() -> None:
    """ReDial golds are integer DBpedia ids, meaningless against film names."""
    from gmbench.mab_run import movie_titles, score_recall

    titles = movie_titles(
        {
            "<http://dbpedia.org/resource/The_Big_Lebowski>": 7008,
            "<http://dbpedia.org/resource/Fargo_(film)>": 42,
        }
    )
    assert titles["7008"] == "The Big Lebowski"
    top5 = "1. Fargo\n2. The Big Lebowski\n3. Heat\n4. Alien\n5. Up"
    assert score_recall(top5, ["7008"], titles, at=5)
    assert score_recall(top5, ["42"], titles, at=5), "a disambiguator is stripped"
    assert not score_recall(top5, ["7008"], titles, at=1), "Recall@1 is stricter"
    assert not score_recall(top5, ["999"], titles, at=5), "an unknown id cannot match"


def test_recall_grading_needs_the_entity_map() -> None:
    from gmbench.mab_run import score_recall

    assert not score_recall("1. The Big Lebowski", ["7008"], {}, at=5)


def test_every_source_has_a_template_and_a_grader() -> None:
    """A missing entry fails every question of that source identically.

    Discovered at run time it costs a full ingest and hundreds of requests to
    learn something knowable before the run starts, so it is asserted here and
    preflighted in run_capability.
    """
    from gmbench.memoryagentbench import CAPABILITIES, GRADERS
    from gmbench.prompts import MAB_QUERIES

    assert set(GRADERS) == set(CAPABILITIES), "every source needs a grader"
    assert set(MAB_QUERIES) == set(CAPABILITIES), "every source needs a template"


def test_test_time_learning_uses_the_papers_nested_formula() -> None:
    """TTL weights ReDial equally against the mean of five in-context sources.

    A flat mean over its six sources gives the in-context sources five sixths
    of the weight instead of a half. On our measured run that reads 63.83 where
    the specified formula reads 47.50 — against a published 47.68, so the
    difference is the whole comparison.
    """
    from gmbench.mab_run import capability_score

    ttl = {
        "icl_banking77": {"accuracy": 86.0},
        "icl_clinic150": {"accuracy": 86.0},
        "icl_nlu": {"accuracy": 84.0},
        "icl_trec_coarse": {"accuracy": 61.0},
        "icl_trec_fine": {"accuracy": 43.0},
        "recsys_redial": {"accuracy": 23.0},
    }
    # (23 + mean(86,86,84,61,43)) / 2 = (23 + 72) / 2
    assert capability_score(ttl) == pytest.approx(47.5)
    flat = sum(b["accuracy"] for b in ttl.values()) / len(ttl)
    assert flat == pytest.approx(63.833, abs=0.01), "the flat mean it must not be"


def test_other_capabilities_average_their_sources() -> None:
    from gmbench.mab_run import capability_score

    ar = {
        "eventqa": {"accuracy": 48.4},
        "longmemeval": {"accuracy": 11.0},
        "ruler_qa1": {"accuracy": 88.0},
        "ruler_qa2": {"accuracy": 69.0},
    }
    assert capability_score(ar) == pytest.approx(54.1)
    assert capability_score({}) == 0.0


@needs_mab
def test_infbench_question_gets_all_its_keypoints() -> None:
    """InfBench ships one question per block with 25 keypoints for it.

    Taking the i-th element, as DetectiveQA needs, hands InfBench a single
    keypoint out of 25 and grades a book summary against a twenty-fifth of its
    rubric — which scored 2.00 before this was fixed.
    """
    from gmbench import memoryagentbench as mab

    blocks = mab.load(MAB_DIR)
    inf = next(b for b in blocks if b.source == "infbench_sum")
    assert len(inf.tasks) == 1, "one summarization question per book"
    assert len(inf.tasks[0].keypoints) > 10, "and the whole rubric belongs to it"

    detective = next(b for b in blocks if b.source == "detective_qa")
    assert len(detective.tasks) > 1
    assert all(len(t.keypoints) <= 1 for t in detective.tasks), "one each here"


@needs_mab
def test_resume_key_matches_the_written_key() -> None:
    """These two uses must agree, and once did not.

    Outcomes were written with a block-qualified id while resume compared the
    raw dataset id. Nothing ever matched, so a resumed run re-ran every question
    and appended to the existing shard, doubling both the rows and the question
    count the summary reported. Deriving both from one function is the fix.
    """
    from gmbench import memoryagentbench as mab
    from gmbench.mab_run import outcome_id

    block = next(b for b in mab.load(MAB_DIR) if b.source == "factconsolidation_sh")
    task = block.tasks[0]
    written = outcome_id(block, task)
    assert written.startswith(block.block_id)
    assert task.qid in written
    assert outcome_id(block, task) == written

    # And it must separate questions the dataset gives the same id.
    eventqa = [b for b in mab.load(MAB_DIR) if b.source == "eventqa"]
    ids = {outcome_id(b, t) for b in eventqa for t in b.tasks}
    assert len(ids) == 500, "all five blocks' questions stay distinct"


def test_summary_judge_failure_is_not_zero_coverage(monkeypatch) -> None:
    """An outage and a summary covering nothing must not be the same number.

    One is a broken measurement, the other is a result. Returning 0.0 for both
    is how an outage gets published as a score.
    """
    from gmbench import mab_run

    class _Broken:
        def chat_completion(self, *a, **k):
            raise RuntimeError("judge down")

    class _Garbled:
        def chat_completion(self, *a, **k):
            class R:
                text = "I could not comply."
            return R()

    cfg = type(
        "C", (), {"models": type("M", (), {"judge_llm_id": "x", "judge_model": "m"})()}
    )()
    assert mab_run.score_summary(_Broken(), cfg, "a summary", ["point one"]) is None
    assert mab_run.score_summary(_Garbled(), cfg, "a summary", ["point one"]) is None
    # An empty summary genuinely covers nothing, and that is a zero.
    assert mab_run.score_summary(_Broken(), cfg, "   ", ["point one"]) == 0.0


# -- the paper's own reranker, driven externally -------------------------


def test_external_scores_must_line_up_with_the_candidates() -> None:
    """A silent misalignment reranks each question against another's scores.

    That produces a plausible and entirely meaningless result, so the loader
    refuses anything but an exact cover with matching counts.
    """
    from gmbench import exact_reranker as ex
    from gmbench.client import GoodMemError

    cands = [
        ex.Candidates("q1", "c", "?", ("D1:1", "D1:2"), ("a", "b")),
        ex.Candidates("q2", "c", "?", ("D1:3",), ("c",)),
    ]

    def write(tmp: Path, rows: list[dict]) -> Path:
        import json as _json

        path = tmp / "scores.jsonl"
        path.write_text("".join(_json.dumps(r) + "\n" for r in rows))
        return path

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        good = write(tmp, [{"qid": "q1", "scores": [0.1, 0.9]}, {"qid": "q2", "scores": [0.5]}])
        assert ex.read_scores(good, cands)["q1"] == [0.1, 0.9]

        short = write(tmp, [{"qid": "q1", "scores": [0.1, 0.9]}])
        with pytest.raises(GoodMemError, match="missing"):
            ex.read_scores(short, cands)

        miscount = write(
            tmp, [{"qid": "q1", "scores": [0.1]}, {"qid": "q2", "scores": [0.5]}]
        )
        with pytest.raises(GoodMemError, match="wrong number of scores"):
            ex.read_scores(miscount, cands)


def test_external_ranking_is_by_score_and_deterministic_on_ties() -> None:
    from gmbench import exact_reranker as ex

    cand = ex.Candidates(
        "q", "c", "?", ("D1:1", "D1:2", "D1:3", "D1:4"), ("a", "b", "c", "d")
    )
    assert ex.rank(cand, [0.1, 0.9, 0.5, 0.2], 2) == ["D1:2", "D1:3"]
    # Ties keep the dense order GoodMem returned them in.
    assert ex.rank(cand, [0.5, 0.5, 0.5, 0.5], 3) == ["D1:1", "D1:2", "D1:3"]


def test_collect_dense_is_callable_end_to_end(monkeypatch, tmp_path) -> None:
    """A smoke test over the dense path, which a missing import once broke.

    The failure was a NameError that only appeared at runtime because nothing
    exercised this function.
    """
    from gmbench import retrieve as rt
    from gmbench.client import RetrievedChunk
    from gmbench.ingest import SpaceBinding

    turns = [_turn(i) for i in range(5)]
    question = Question(
        conversation_id="conv-test",
        index=0,
        question="who?",
        answer="a",
        category=1,
        evidence=("D1:1",),
    )
    conversation = Conversation(
        conversation_id="conv-test",
        speaker_a="Amy",
        speaker_b="Rob",
        turns=tuple(turns),
        questions=(question,),
    )

    class _Client:
        def retrieve_dense(self, message, space_id, **kwargs):
            return [
                RetrievedChunk(f"m{t.turn_index}", t.render(), 0, 0.5, {"dia_id": t.dia_id})
                for t in turns
            ]

    cfg = _run_config(tmp_path)
    rows = rt.collect_dense(
        _Client(), cfg, [conversation], {"conv-test": SpaceBinding("conv-test", "s", "n", 5)}
    )
    assert len(rows) == 1
    assert rows[0].qid == question.qid
    assert len(rows[0].documents) == 5
    assert (cfg.run_dir / "dense-candidates.jsonl").exists()


# -- one store per block -------------------------------------------------


@needs_mab
def test_every_block_gets_its_own_space_name() -> None:
    """Two blocks must never share a space name.

    A source ships many rows, each a self-contained document store, and several
    rows share one label. Naming the space after the label collided them, so a
    row was answered against whichever row had last been ingested -- and the
    reuse check passed, because a space holding another block's documents is
    full, COMPLETED and indistinguishable from a correct one. It cost 607 of
    2,071 questions, concentrated entirely in the multi-row sources, and it is
    invisible in the output: the reader simply says the context does not
    mention the answer.
    """
    from gmbench import memoryagentbench as mab
    from gmbench.mab_run import space_name

    blocks = mab.load(MAB_DIR)
    names = [space_name(block, "tag") for block in blocks]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"blocks sharing a space name: {sorted(duplicates)[:5]}"
    assert len(names) == len(blocks)


@needs_mab
def test_multi_row_sources_exist_so_the_collision_is_reachable() -> None:
    """Guards the test above against becoming vacuous.

    If every source were single-row the collision could not occur and the
    assertion would pass for the wrong reason.
    """
    from collections import Counter

    from gmbench import memoryagentbench as mab

    per_label = Counter(block.label for block in mab.load(MAB_DIR))
    multi = {label: n for label, n in per_label.items() if n > 1}
    assert multi, "expected sources shipping several rows under one label"
    assert max(multi.values()) >= 10


# -- paginated listings --------------------------------------------------


def test_list_spaces_follows_pagination(monkeypatch) -> None:
    """Every page must be collected, not just the first.

    The caller uses this listing to decide whether a space already exists. A
    space on page two reads as absent, so the create that follows fails with
    409 and the run dies. It stayed hidden while a run made 14 spaces and
    surfaced the moment one made 130.
    """
    from gmbench.client import GoodMemClient

    client = GoodMemClient("https://example.invalid", "key", tls_verify=False)
    pages = [
        {"spaces": [{"name": f"s{i}"} for i in range(50)], "nextToken": "t1"},
        {"spaces": [{"name": f"s{i}"} for i in range(50, 90)], "nextToken": ""},
    ]
    seen: list[dict] = []

    def _request(method, path, params=None, **kwargs):
        assert method == "GET" and path == "/v1/spaces"
        seen.append(dict(params or {}))
        return pages[len(seen) - 1]

    monkeypatch.setattr(client, "_request", _request)
    names = [s["name"] for s in client.list_spaces()]

    assert len(names) == 90, "second page was dropped"
    assert names[-1] == "s89"
    assert "nextToken" not in seen[0]
    assert seen[1]["nextToken"] == "t1", "token from page one was not sent"


def test_throttled_rerank_is_retried_not_recorded_as_a_dead_question() -> None:
    """A 429 inside the stream must be retryable.

    The provider throttle arrives as a RERANKING_FAILED status frame, not an
    HTTP 429, so the transport retry never sees it. Treated as fatal it scores
    the question zero: 248 of 500 EventQA questions died this way and halved
    the source's published score.
    """
    import pytest as _pytest

    from gmbench.client import _check_status, _Throttled

    with _pytest.raises(_Throttled):
        _check_status(
            {
                "code": "RERANKING_FAILED",
                "message": "Reranking failed: Reranking request failed with status 429",
            }
        )


def test_a_rerank_failure_that_is_not_a_throttle_stays_fatal() -> None:
    """Only throttles retry; anything else must still stop the run."""
    import pytest as _pytest

    from gmbench.client import GoodMemError, _check_status

    with _pytest.raises(GoodMemError, match="RERANKING_FAILED"):
        _check_status(
            {"code": "RERANKING_FAILED", "message": "model not found: bad-reranker-id"}
        )


def test_resume_retries_errored_outcomes(tmp_path) -> None:
    """An errored outcome must not count as completed.

    An error scores zero, so treating it as done resumes the run into a
    permanently depressed number -- and it is the questions that failed which
    most need another attempt.
    """
    import json as _json

    from gmbench.mab_run import _retry_errored

    path = tmp_path / "outcomes.jsonl"
    with path.open("w") as handle:
        for qid, err in (("a", None), ("b", "GoodMemError: 429"), ("c", None)):
            handle.write(_json.dumps({"qid": qid, "error": err, "correct": True}) + "\n")

    keep = _retry_errored(path)

    assert keep == {"a", "c"}, "errored question was treated as done"
    remaining = [_json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r["qid"] for r in remaining] == ["a", "c"], "errored row was left in the shard"


def test_mab_overall_verdict_follows_the_sign(tmp_path) -> None:
    """The overall verdict must be computed, not hardcoded.

    It read "behind" regardless, which was right only while the number was.
    """
    import json as _json

    from gmbench.markdown import MAB_PAPER_RAW_RAG, mab_section

    for capability, theirs in MAB_PAPER_RAW_RAG.items():
        out = tmp_path / f"run-{capability}" / "export"
        out.mkdir(parents=True)
        (out / f"mab-{capability}-summary-gpt-4o-mini.json").write_text(
            _json.dumps(
                {
                    "capability": capability,
                    "accuracy": theirs + 5.0,
                    "questions": 100,
                    "errors": 0,
                    "by_source": {},
                }
            )
        )

    overall = [ln for ln in mab_section(tmp_path) if "**Overall**" in ln]
    assert len(overall) == 1
    assert "ahead (+5.00)" in overall[0], overall[0]


def test_an_externally_reranked_run_is_not_described_by_its_config(tmp_path) -> None:
    """The summary must name the reranker that actually ranked the results.

    A --from-scores run is ranked outside GoodMem, but the summary described
    whatever reranker was left in the config. That produced a published record
    reading "VOYAGE rerank-2.5" for a run ranked by Qwen3-Reranker-4B -- a false
    provenance claim in the one file a reader checks.
    """
    from gmbench.exact_reranker import EXTERNAL_RERANKER, read_provenance, write_provenance

    assert read_provenance(tmp_path) is None, "absent provenance must not be invented"

    write_provenance(tmp_path)
    recorded = read_provenance(tmp_path)

    assert recorded is not None
    assert recorded["model"] == EXTERNAL_RERANKER["model"]
    assert "Qwen3-Reranker-4B" in recorded["display_name"]
    assert "int8" in recorded["note"], "the disclosed deviation must travel with it"
