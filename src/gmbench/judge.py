"""Judge stage: grade answers against the reference, following the paper's rules.

The parsing and scoring rules below are the ones specified in appendix G.3 of
the paper. They are strict on purpose: a bare ``CORRECT`` with no JSON envelope
scores zero, and every failure mode stays in the denominator. Reproducing them
exactly is what keeps a score comparable to the published one, since a laxer
parser would quietly inflate it.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from .config import RunConfig
from .locomo import CATEGORY_NAMES
from .prompts import JUDGE_SYSTEM, JUDGE_USER
from .reader import AnswerRecord

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FLAT_OBJECT_RE = re.compile(r"\{[^{}]*\"label\"\s*:\s*\"[^\"]*\"[^{}]*\}", re.DOTALL)


@dataclass(slots=True)
class JudgeRecord:
    """One answer's judgements and its mean score."""

    qid: str
    conversation_id: str
    category: int
    scores: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "conversation_id": self.conversation_id,
            "category": self.category,
            "scores": self.scores,
            "labels": self.labels,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "JudgeRecord":
        return cls(**raw)


def judgements_path(config: RunConfig) -> Path:
    return config.run_dir / f"judgements-{config.models.reader_model}.jsonl"


def parse_label(response: str) -> str | None:
    """Extracts the judge's label using the paper's selection order.

    Args:
      response: Raw judge response text.

    Returns:
      The upper-cased label, or None when no usable JSON label is present.
    """
    if not response or not response.strip():
        return None

    candidates: list[str] = []
    block = _CODE_BLOCK_RE.search(response)
    if block:
        candidates.append(block.group(1))
    else:
        flat = _FLAT_OBJECT_RE.search(response)
        if flat:
            candidates.append(flat.group(0))
        else:
            candidates.append(response.strip())

    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        label = decoded.get("label")
        if isinstance(label, str):
            return label.strip().upper()
    return None


def _judge_once(client: OpenAI, config: RunConfig, answer: AnswerRecord) -> tuple[int, str]:
    """Requests one judgement; any failure scores zero and stays in the denominator."""
    prompt = JUDGE_USER.format(
        question=answer.question,
        golden_answer=answer.answer,
        generated_answer=answer.final_answer,
    )
    try:
        response = client.chat.completions.create(
            model=config.models.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - scored zero, per the paper
        return 0, f"ERROR:{type(exc).__name__}"

    label = parse_label(text)
    if label is None:
        return 0, "UNPARSEABLE"
    return (1 if label == "CORRECT" else 0), label


def _judge_answer(client: OpenAI, config: RunConfig, answer: AnswerRecord) -> JudgeRecord:
    """Collects the configured number of independent judgements for one answer."""
    record = JudgeRecord(
        qid=answer.qid, conversation_id=answer.conversation_id, category=answer.category
    )
    for _ in range(config.models.judge_repetitions):
        score, label = _judge_once(client, config, answer)
        record.scores.append(score)
        record.labels.append(label)
    return record


def run(config: RunConfig, answers: list[AnswerRecord]) -> list[JudgeRecord]:
    """Grades every answer and writes the judgements to the run directory.

    Args:
      config: Active run configuration.
      answers: Answers to grade.

    Returns:
      One judge record per answer, in input order.
    """
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; the judge stage needs it")
    client = OpenAI(api_key=config.openai_api_key)

    records: list[JudgeRecord] = [None] * len(answers)  # type: ignore[list-item]
    total = len(answers) * config.models.judge_repetitions
    with tqdm(total=total, desc=f"judge:{config.models.judge_model}", unit="j") as bar:
        with ThreadPoolExecutor(max_workers=config.reader_concurrency) as pool:
            futures = {
                pool.submit(_judge_answer, client, config, answer): index
                for index, answer in enumerate(answers)
            }
            for future, index in futures.items():
                records[index] = future.result()
                bar.update(config.models.judge_repetitions)

    path = judgements_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json()) + "\n")
    return records


def accuracy_by_category(records: list[JudgeRecord]) -> dict[str, tuple[int, float]]:
    """Computes question-weighted accuracy per category and overall.

    Args:
      records: Judge records to aggregate.

    Returns:
      Category name to (question count, accuracy as a percentage), including an
      ``overall`` entry weighted by question, matching the paper's convention.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for record in records:
        buckets[record.category].append(record.mean_score)

    out: dict[str, tuple[int, float]] = {}
    for category, means in sorted(buckets.items()):
        name = CATEGORY_NAMES.get(category, str(category))
        out[name] = (len(means), 100.0 * sum(means) / len(means))

    all_means = [mean for means in buckets.values() for mean in means]
    if all_means:
        out["overall"] = (len(all_means), 100.0 * sum(all_means) / len(all_means))
    return out


def load_judgements(config: RunConfig) -> list[JudgeRecord]:
    """Reads judgements written by a previous judge stage."""
    path = judgements_path(config)
    if not path.exists():
        raise RuntimeError(f"no judgements at {path}; run the judge stage first")
    with path.open() as handle:
        return [JudgeRecord.from_json(json.loads(line)) for line in handle if line.strip()]
