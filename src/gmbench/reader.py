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

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "conversation_id": self.conversation_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "generated": self.generated,
            "final_answer": self.final_answer,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "AnswerRecord":
        return cls(**raw)


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
    error: str | None = None
    try:
        generated = client.chat_completion(
            llm_id,
            [{"role": "user", "content": prompt}],
            model=config.models.reader_model,
            temperature=0.0,
            max_tokens=config.models.reader_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, then scored as a failure
        error = f"{type(exc).__name__}: {exc}"

    return AnswerRecord(
        qid=record.qid,
        conversation_id=record.conversation_id,
        category=record.category,
        question=record.question,
        answer=record.answer,
        generated=generated,
        final_answer=extract_final_answer(generated),
        error=error,
    )


def run(
    client: GoodMemClient, config: RunConfig, records: list[RetrievalRecord]
) -> list[AnswerRecord]:
    """Generates an answer per retrieval record and writes them to the run directory.

    Args:
      client: Connected GoodMem client, used as the inference proxy.
      config: Active run configuration.
      records: Retrieval records to answer.

    Returns:
      One answer record per input record, in input order.

    Raises:
      GoodMemError: If no LLM registration is configured for the reader model.
    """
    try:
        llm_id = config.models.llm_id_for(config.models.reader_model)
    except KeyError as exc:
        raise GoodMemError(str(exc)) from exc

    answers: list[AnswerRecord | None] = [None] * len(records)
    path = answers_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(records), desc=f"read:{config.models.reader_model}", unit="q") as bar:
        with ThreadPoolExecutor(max_workers=config.reader_concurrency) as pool:
            futures = {
                pool.submit(_answer_one, client, config, llm_id, record): index
                for index, record in enumerate(records)
            }
            for future in as_completed(futures):
                answers[futures[future]] = future.result()
                bar.update(1)

    done = [answer for answer in answers if answer is not None]
    with path.open("w") as handle:
        for answer in done:
            handle.write(json.dumps(answer.to_json()) + "\n")
    return done


def load_answers(config: RunConfig) -> list[AnswerRecord]:
    """Reads answers written by a previous reader stage."""
    path = answers_path(config)
    if not path.exists():
        raise RuntimeError(f"no answers at {path}; run the read stage first")
    with path.open() as handle:
        return [AnswerRecord.from_json(json.loads(line)) for line in handle if line.strip()]
