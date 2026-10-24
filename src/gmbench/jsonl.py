"""Save completed work one JSONL row at a time.

The writer flushes each row under a lock so concurrent model calls can share a
file. On restart, a stage reads the saved question ids and submits the remaining
work. The reader tolerates an incomplete final row from an interrupted write,
but reports malformed JSON anywhere earlier in the file.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator


class ShardWriter:
    """Appends rows to a JSONL file, flushing each one.

    Safe to share across worker threads; one lock serialises the writes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a")

    def write(self, row: dict[str, Any]) -> None:
        """Appends one row and pushes it to the operating system."""
        line = json.dumps(row) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yields the decoded rows of a JSONL file, skipping a torn final line.

    Only the last line may be incomplete, and only when a process died
    mid-write. Any earlier bad line means the file is corrupt rather than
    truncated, so it is raised.

    Raises:
      ValueError: If a line other than the last fails to decode.
    """
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                # A partial tail is the expected shape of an interrupted run.
                return
            raise ValueError(f"{path}: corrupt row at line {index + 1}") from exc


def completed_keys(path: Path, key: str = "qid") -> set[str]:
    """Returns the key values already present in a shard, for resuming."""
    return {
        str(row[key])
        for row in read_rows(path)
        if isinstance(row, dict) and row.get(key) is not None
    }
