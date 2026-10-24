"""Reader stage: turn retrieved evidence into an answer.

Calls go through GoodMem's provider proxy, which forwards to the registered
upstream model and injects the stored credential. That keeps the provider key
out of this process while still sending the paper's prompt verbatim -- the proxy
passes the body through, so nothing here is reshaped by GoodMem's own
post-processing. The reader is held at the paper's model on purpose: GoodMem is
the retrieval system under test, and a stronger reader would raise the score
without saying anything about retrieval.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import jsonl, validate
from .client import GoodMemClient, GoodMemError
from .config import RunConfig
from .prompts import LOCOMO_READER
from .retrieve import RetrievalRecord

_FINAL_ANSWER_RE = re.compile(r"##\s*FINAL ANSWER:\s*(.*)", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class AnswerRecord:
    """A reader answer alongside the question it belongs to."""

    qid: str
    conversation_id: str
    category: int
    question: str
    answer: str
    generated: str
    final_answer: str
    finish_reason: str = ""
    error: str | None = None

    @property
    def failed(self) -> bool:
        """True when this answer is an infrastructure failure, not a model answer."""
        return bool(self.error) or not self.generated.strip()

    @property
    def truncated(self) -> bool:
        """True when generation stopped at the token cap.

        A truncated chain of thought usually loses the FINAL ANSWER marker, so
        the grader scores the reasoning instead of the answer.
        """
        return self.finish_reason == "length"

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "conversation_id": self.conversation_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "generated": self.generated,
            "final_answer": self.final_answer,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "AnswerRecord":
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in raw.items() if key in fields})


def answers_path(config: RunConfig) -> Path:
    return config.run_dir / f"answers-{config.models.reader_model}.jsonl"


def extract_final_answer(response: str) -> str:
    """Returns the text after ``FINAL ANSWER:``, or the whole response if absent.

    This is the paper's extraction rule; keeping it identical matters because
    the judge grades the extracted span, not the full chain of thought.
    """
    match = _FINAL_ANSWER_RE.search(response)
    return (match.group(1) if match else response).strip()


def _answer_one(
    client: GoodMemClient, config: RunConfig, llm_id: str, record: RetrievalRecord
) -> AnswerRecord:
    """Runs the reader for one question.

    The paper's reader prompt is a single user message with no system message,
    and evaluation reads the text after ``FINAL ANSWER:``.
    """
    prompt = LOCOMO_READER.format(context=record.context, question=record.question)
    generated = ""
    finish_reason = ""
    error: str | None = None
    try:
        completion = client.chat_completion(
            llm_id,
            [{"role": "user", "content": prompt}],
            model=config.models.reader_model,
            temperature=0.0,
            max_tokens=config.models.reader_max_tokens,
        )
        generated = completion.text
        finish_reason = completion.finish_reason
    except Exception as exc:  # noqa: BLE001 - recorded, surfaced by the report
        error = f"{type(exc).__name__}: {exc}"

    return AnswerRecord(
        qid=record.qid,
        conversation_id=record.conversation_id,
        category=record.category,
        question=record.question,
        answer=record.answer,
        generated=generated,
        final_answer=extract_final_answer(generated),
        finish_reason=finish_reason,
        error=error,
    )


def run(
    client: GoodMemClient,
    config: RunConfig,
    records: list[RetrievalRecord],
    *,
    resume: bool = True,
) -> list[AnswerRecord]:
    """Generates an answer per retrieval record, appending each as it lands.

    Args:
      client: Connected GoodMem client, used as the inference proxy.
      config: Active run configuration.
      records: Retrieval records to answer.
      resume: Keep answers already on disk and only generate the remainder.

    Returns:
      One answer record per input record, ordered by question id.

    Raises:
      GoodMemError: If no LLM registration is configured for the reader model.
      IncompleteStageError: If the finished shard does not cover every record.
    """
    try:
        llm_id = config.models.llm_id_for(config.models.reader_model)
    except KeyError as exc:
        raise GoodMemError(str(exc)) from exc

    path = answers_path(config)
    already = jsonl.completed_keys(path) if resume else set()
    if not resume and path.exists():
        path.unlink()
    todo = [record for record in records if record.qid not in already]
    if already:
        tqdm.write(f"read: resuming, {len(already)} answers already on disk")

    with jsonl.ShardWriter(path) as shard:
        with tqdm(
            total=len(todo), desc=f"read:{config.models.reader_model}", unit="q"
        ) as bar:
            with ThreadPoolExecutor(max_workers=config.reader_concurrency) as pool:
                futures = [
                    pool.submit(_answer_one, client, config, llm_id, record)
                    for record in todo
                ]
                for future in as_completed(futures):
                    shard.write(future.result().to_json())
                    bar.update(1)

    return load_answers(config, [record.qid for record in records])


def load_answers(
    config: RunConfig, expected_qids: list[str] | None = None
) -> list[AnswerRecord]:
    """Reads answers written by a previous reader stage.

    Args:
      config: Active run configuration.
      expected_qids: When given, the question ids the shard must cover exactly.

    Returns:
      Answers ordered by question id.

    Raises:
      GoodMemError: If the shard is absent.
      IncompleteStageError: If it does not cover ``expected_qids`` exactly.
    """
    path = answers_path(config)
    if not path.exists():
        raise GoodMemError(f"no answers at {path}; run the read stage first")
    answers = [AnswerRecord.from_json(row) for row in jsonl.read_rows(path)]
    if expected_qids is not None:
        validate.require("read", expected_qids, [answer.qid for answer in answers])
    return sorted(answers, key=lambda answer: answer.qid)
