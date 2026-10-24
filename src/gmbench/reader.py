"""Reader stage: turn retrieved evidence into an answer.

The reader runs against OpenAI directly rather than through GoodMem so the
prompt is byte-identical to the paper's. GoodMem is the retrieval system under
test here; routing generation through it as well would mix the two.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

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
    client: OpenAI, config: RunConfig, record: RetrievalRecord
) -> AnswerRecord:
    """Runs the reader for one question."""
    prompt = LOCOMO_READER.format(context=record.context, question=record.question)
    generated = ""
    error: str | None = None
    try:
        response = client.chat.completions.create(
            model=config.models.reader_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=config.models.reader_max_tokens,
        )
        generated = response.choices[0].message.content or ""
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


def run(config: RunConfig, records: list[RetrievalRecord]) -> list[AnswerRecord]:
    """Generates an answer per retrieval record and writes them to the run directory.

    Args:
      config: Active run configuration.
      records: Retrieval records to answer.

    Returns:
      One answer record per input record, in input order.
    """
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; the reader stage needs it")
    client = OpenAI(api_key=config.openai_api_key)

    answers: list[AnswerRecord] = [None] * len(records)  # type: ignore[list-item]
    with tqdm(total=len(records), desc=f"read:{config.models.reader_model}", unit="q") as bar:
        with ThreadPoolExecutor(max_workers=config.reader_concurrency) as pool:
            futures = {
                pool.submit(_answer_one, client, config, record): index
                for index, record in enumerate(records)
            }
            for future, index in futures.items():
                answers[index] = future.result()
                bar.update(1)

    path = answers_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for answer in answers:
            handle.write(json.dumps(answer.to_json()) + "\n")
    return answers


def load_answers(config: RunConfig) -> list[AnswerRecord]:
    """Reads answers written by a previous reader stage."""
    path = answers_path(config)
    if not path.exists():
        raise RuntimeError(f"no answers at {path}; run the read stage first")
    with path.open() as handle:
        return [AnswerRecord.from_json(json.loads(line)) for line in handle if line.strip()]
