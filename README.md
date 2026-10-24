# goodmem-memory-benchmarks

Runs GoodMem as the retrieval layer on long-conversation memory benchmarks and
scores it against the published numbers. LoCoMo first.

## What this is measuring, and against what

The reference is **RAW-RAG**, from [*Is Complex Agentic Memory Always Needed?
Naive RAG Is Stronger Than We Thought*](docs/2610.30755v1.pdf)
(arXiv:2610.30755v1). That paper's claim is that naive dense retrieval plus
reranking over raw conversation history beats twelve agentic memory systems on
six of seven benchmark–model settings. GoodMem is architecturally the same
shape, so the question here is not whether the paper reproduces — it is whether
GoodMem, with a stronger embedder and reranker, clears the bar the paper set.

Targets on LoCoMo (paper table 10, question-weighted, adversarial excluded):

| Reader | RAW-RAG overall | Best agentic system | Best agentic system name |
|---|---:|---:|---|
| `gpt-4o-mini` | 89.07 | 88.12 | MemMachine |
| `gpt-4.1-mini` | 93.70 | 91.69 | MemMachine |

RAW-RAG got those with `text-embedding-3-small` and Cohere Rerank-v3.5. This
harness runs **Voyage 4 Large** and **voyage-rerank-2.5**, so the retrieval side
starts from stronger models than the paper used.

Per-category targets live in `src/gmbench/report.py` and are printed next to
every measured score, so a run tells you directly whether it cleared the bar.

## Design

One space per conversation, one memory per conversation turn.

| RAW-RAG | Here |
|---|---|
| 256-token windows, 32-token overlap | One memory per turn; character-based recursive chunking at 1500/150, which leaves nearly every turn whole |
| Flat per-conversation index | One GoodMem space per conversation |
| Top-128 dense candidates | `requestedSize: 128` |
| Cross-encoder rerank to top-15 | `ChatPostProcessor` with `reranker_id` and `max_results: 15` |
| Display-time turn expansion | Each selected turn expanded by ±`neighbor_turns` from the local dataset, de-duplicated, chronological |
| Reader prompt (appendix G.4) | `src/gmbench/prompts.py`, verbatim |
| Judge: 3 reps, temp 0, strict JSON label (appendix G.3) | `src/gmbench/judge.py`, including the parse rules and zero-scoring |

Deliberate departures from the paper, and why:

- **Character chunking, not token.** GoodMem's `TOKEN_COUNT` length measurement
  currently falls back to character count server-side
  (`SplitterFactory.mapLengthMeasurement`). With one memory per turn it barely
  matters: a 1500-character window leaves almost every turn as a single chunk.
  The paper's own ablation finds chunk sizes from 128 to 1024 tokens do not
  significantly move its total.
- **Turns, not windows.** A turn is the natural unit here, it carries an exact
  timestamp and dialogue id as metadata, and it makes expansion exact rather
  than offset arithmetic.
- **Expansion client-side.** GoodMem returns which turns were selected; the
  harness rebuilds the reader context from the source JSON. Nothing is
  reconstructed from chunk offsets, so the evidence is exactly the source text.
- **Reader and judge call OpenAI directly**, not through GoodMem. GoodMem is the
  retrieval system under test; routing generation through it as well would mix
  the two and make the comparison unreadable.
- **HNSW, not exact search.** pgvector HNSW with `efSearch` tuned per request.
  This is a recall knob, exposed in the config.

## Evidence coverage: tune here before spending tokens

LoCoMo annotates every question with the dialogue ids that support its answer.
The `coverage` stage compares those against what is actually in the assembled
context, which measures evidence access — the bottleneck the paper identifies
as dominant — with no reader or judge tokens at all. Coverage bounds accuracy
from above, so it is the right thing to sweep `final_k`, `neighbor_turns` and
`ef_search` on before committing to a scored run.

Columns: **Full** = every gold turn present, **Any** = at least one, **Recall** =
mean fraction of gold turns present, **In top-k** = gold turn among the reranked
selections before expansion.

## Running it

Needs the `uv` toolchain, a reachable GoodMem instance, and `OPENAI_API_KEY` for
the reader and judge stages. Retrieval and coverage need no OpenAI key.

```bash
uv venv --python 3.13 && uv pip install -e .
cp .env.example .env        # fill in GOODMEM_API_KEY and OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)

# Place the LoCoMo 10-conversation release here:
#   data/locomo10.json      (https://github.com/snap-research/locomo)

gmbench ingest              # one space per conversation, one memory per turn
gmbench retrieve            # retrieve, expand, and print coverage
gmbench read                # generate answers (needs OPENAI_API_KEY)
gmbench judge               # grade and print accuracy against the paper
gmbench report              # reprint both tables for a finished run
gmbench run                 # all four in sequence
```

Useful flags: `--run-tag` to keep runs separate, `--reader-model` to switch
reader without editing config, `--conversations-limit` / `--questions-limit` for
smoke runs, `--config` to point at a different TOML.

A run writes everything to `results/<run-tag>/`: `spaces.json` (the space
manifest, reused on re-run), `retrieval.jsonl` (the assembled context per
question), `answers-<model>.jsonl` and `judgements-<model>.jsonl`. Stages are
resumable because each reads the previous stage's file.

## Configuration

`config/locomo.toml`. The knobs that move the score:

- `retrieval.dense_candidates` / `retrieval.final_k` — the two-stage depths.
- `retrieval.neighbor_turns` — turns kept either side of each selection. The
  paper's expanded LoCoMo evidence averages about 4,800 tokens; if coverage is
  short and the context is well under that, this is the first dial to turn.
- `retrieval.ef_search`, `retrieval.iterative_scan` — HNSW recall.
- `models.embedder_id` / `models.reranker_id` — GoodMem registrations, by UUID.

Model ids in the committed config point at the registrations on the development
instance. `goodmem embedder list` and `goodmem reranker list` show what is
available.

## Status

Scaffolding and the LoCoMo path are in place and exercised against a live
instance. Ingest and retrieval are verified end to end; the reader and judge
stages are written to the paper's prompts and parse rules but have not yet been
run, pending an `OPENAI_API_KEY`.
