"""Grade LoCoMo answers using the paper's appendix G.3 rules.

Each answer receives three judgements by default. The parser looks for a JSON
label; a bare CORRECT does not count. Failed calls and unusable labels contribute
zero to the mean, and their counts are retained for the integrity report.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import jsonl, validate
from .client import GoodMemClient, GoodMemError
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

    @property
    def infrastructure_failures(self) -> int:
        """Count judgements that failed to reach the model.

        These contribute zero to accuracy but remain separately visible in the
        integrity report.
        """
        return sum(1 for label in self.labels if label.startswith("ERROR:"))

    @property
    def unparseable(self) -> int:
        """Judgements whose response carried no usable JSON label."""
        return sum(1 for label in self.labels if label == "UNPARSEABLE")

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
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in raw.items() if key in fields})


def judgements_path(config: RunConfig) -> Path:
    return config.run_dir / f"judgements-{config.models.reader_model}.jsonl"


def parse_label(response: str) -> str | None:
    """Read a JSON label using the paper's order of precedence.

    Use the first fenced block if present, otherwise the first flat JSON object,
    otherwise the entire response. Do not try a lower-priority form when the
    selected one fails to parse. Return the uppercased label, or None.
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


def _judge_once(
    client: GoodMemClient, config: RunConfig, answer: AnswerRecord
) -> tuple[int, str]:
    """Requests one judgement; any failure scores zero and stays in the denominator."""
    prompt = JUDGE_USER.format(
        question=answer.question,
        golden_answer=answer.answer,
        generated_answer=answer.final_answer,
    )
    try:
        text = client.chat_completion(
            config.models.judge_llm_id,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=config.models.judge_model,
        ).text
    except Exception as exc:  # noqa: BLE001 - scored zero, per the paper
        return 0, f"ERROR:{type(exc).__name__}"

    label = parse_label(text)
    if label is None:
        return 0, "UNPARSEABLE"
    return (1 if label == "CORRECT" else 0), label


def _judge_answer(
    client: GoodMemClient, config: RunConfig, answer: AnswerRecord
) -> JudgeRecord:
    """Collects the configured number of independent judgements for one answer."""
    record = JudgeRecord(
        qid=answer.qid, conversation_id=answer.conversation_id, category=answer.category
    )
    for _ in range(config.models.judge_repetitions):
        score, label = _judge_once(client, config, answer)
        record.scores.append(score)
        record.labels.append(label)
    return record


def run(
    client: GoodMemClient,
    config: RunConfig,
    answers: list[AnswerRecord],
    *,
    resume: bool = True,
) -> list[JudgeRecord]:
    """Grades every answer, appending each judgement set as it lands.

    Args:
      client: Connected GoodMem client, used as the inference proxy.
      config: Active run configuration.
      answers: Answers to grade.
      resume: Keep judgements already on disk and only grade the remainder.

    Returns:
      One judge record per answer, ordered by question id.

    Raises:
      GoodMemError: If no judge LLM registration is configured.
      IncompleteStageError: If the finished shard does not cover every answer.
    """
    if not config.models.judge_llm_id:
        raise GoodMemError("models.judge_llm_id is not configured")

    path = judgements_path(config)
    already = jsonl.completed_keys(path) if resume else set()
    if not resume and path.exists():
        path.unlink()
    todo = [answer for answer in answers if answer.qid not in already]
    if already:
        tqdm.write(f"judge: resuming, {len(already)} answers already graded")

    reps = config.models.judge_repetitions
    with jsonl.ShardWriter(path) as shard:
        with tqdm(
            total=len(todo) * reps, desc=f"judge:{config.models.judge_model}", unit="j"
        ) as bar:
            with ThreadPoolExecutor(max_workers=config.reader_concurrency) as pool:
                futures = [
                    pool.submit(_judge_answer, client, config, answer) for answer in todo
                ]
                for future in as_completed(futures):
                    shard.write(future.result().to_json())
                    bar.update(reps)

    return load_judgements(config, [answer.qid for answer in answers])


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


def load_judgements(
    config: RunConfig, expected_qids: list[str] | None = None
) -> list[JudgeRecord]:
    """Reads judgements written by a previous judge stage.

    Args:
      config: Active run configuration.
      expected_qids: When given, the question ids the shard must cover exactly.

    Returns:
      Judgements ordered by question id.

    Raises:
      GoodMemError: If the shard is absent.
      IncompleteStageError: If it does not cover ``expected_qids`` exactly.
    """
    path = judgements_path(config)
    if not path.exists():
        raise GoodMemError(f"no judgements at {path}; run the judge stage first")
    records = [JudgeRecord.from_json(row) for row in jsonl.read_rows(path)]
    if expected_qids is not None:
        validate.require("judge", expected_qids, [record.qid for record in records])
    return sorted(records, key=lambda record: record.qid)
