"""Scores query-document pairs with Qwen3-Reranker-4B, the paper's own checkpoint.

Qwen3-Reranker is generative: it does not emit a similarity score. It is prompted
with a yes/no relevance question and the score is the probability it assigns to
"yes" at the next position. An endpoint that serves the weights behind a generic
scoring API without that template returns numbers that are not relevance, which
is what the instance's vLLM deployment does -- it ranks an obviously relevant
passage 81st of 128 with every score squeezed into a 0.26-wide band.

The template below is Qwen's documented one, and the instruction is the paper's
appendix G.1 wording. Weights load in int8 because the GPU has under 6 GiB free
against roughly 8 GiB of bf16 weights; that is a disclosed deviation, and the
alternative was a different model entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3-Reranker-4B"
# The paper's appendix G.1 relevance instruction, verbatim.
INSTRUCTION = (
    "Given a question and a passage, determine if the passage contains "
    "information relevant to answering the question."
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be '
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    """Loads the checkpoint once and scores batches of pairs."""

    def __init__(self, max_length: int = 1024, batch_size: int = 8) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            dtype=torch.float16,
            device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        ).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.max_length = max_length
        self.batch_size = batch_size

    def _pair(self, query: str, document: str) -> str:
        return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"

    @torch.no_grad()
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Returns P(yes) for each document, which is the relevance score."""
        out: list[float] = []
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        for start in range(0, len(documents), self.batch_size):
            chunk = documents[start : start + self.batch_size]
            encoded = self.tokenizer(
                [self._pair(query, d) for d in chunk],
                truncation=True,
                max_length=budget,
                add_special_tokens=False,
            )["input_ids"]
            padded = self.tokenizer.pad(
                {"input_ids": [self.prefix_ids + ids + self.suffix_ids for ids in encoded]},
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            logits = self.model(**padded).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            out.extend(torch.softmax(pair.float(), dim=1)[:, 1].tolist())
        return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score gmbench dense candidates with Qwen3-Reranker-4B, the "
            "reranker the paper uses, applying its documented template."
        )
    )
    parser.add_argument(
        "--candidates", required=True, help="dense-candidates.jsonl from gmbench."
    )
    parser.add_argument("--out", required=True, help="rerank-scores.jsonl to write.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Pairs per forward pass; lower it if the GPU runs out of memory.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help=(
            "Token budget per pair. LoCoMo turns come to about 70 tokens with "
            "the template, so the default is generous and keeps the batch small "
            "enough to fit beside int8 weights."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep scores already written and only do the remainder.",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.candidates).read_text().splitlines()
        if line.strip()
    ]
    out_path = Path(args.out)
    done: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(str(json.loads(line)["qid"]))
        print(f"resuming, {len(done)} already scored", file=sys.stderr, flush=True)

    todo = [r for r in rows if str(r["qid"]) not in done]
    reranker = Reranker(max_length=args.max_length, batch_size=args.batch_size)
    with out_path.open("a" if args.resume else "w") as handle:
        for index, row in enumerate(todo):
            scores = reranker.score(row["query"], row["documents"])
            handle.write(json.dumps({"qid": row["qid"], "scores": scores}) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"{index + 1}/{len(todo)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
