# GoodMem memory benchmarks

Reproducing the experiments in [*Is Complex Agentic Memory Always Needed? Naive
RAG Is Stronger Than We Thought*](https://arxiv.org/abs/2610.30755) with
[GoodMem](https://github.com/PAIR-Systems-Inc/goodmem) as the context layer.

The paper asks how far retrieval over raw conversation history can go. This
repository uses GoodMem to store that history and supply the passages used to
answer each question. The reader and judge prompts stay fixed so we can study
the context GoodMem provides.

## Results

On LoCoMo, all twelve runs place the paper's RAW-RAG score within the GoodMem
run's 95% bootstrap interval. The results are consistent with the paper's finding
that straightforward retrieval can perform well on these tasks.

The first row uses table 10's retrieval stack: `text-embedding-3-small` with
Cohere Rerank-v3.5, 128 candidates narrowed to 15. It is the one configuration
here that matches a published row model for model.

| LoCoMo configuration | `gpt-4o-mini` | `gpt-4.1-mini` |
|---|---:|---:|
| Table 10 stack: Cohere `rerank-v3.5` | 88.68 | 93.48 |
| Qwen3-Reranker-4B, run locally | 88.66 | 93.42* |
| Hosted Qwen reranker | 88.72 | 93.23 |
| Cohere `rerank-v5.0-pro` | 88.87 | 93.46 |
| Voyage `rerank-2.5` | 87.84 | 94.29 |
| Voyage `rerank-3` | 88.44 | 94.35 |
| RAW-RAG, paper table 10 | 89.07 | 93.70 |

The first four rows share one index and one candidate depth, so the reranker is
the only difference between them. They span 0.21 points with `gpt-4o-mini` and
0.25 with `gpt-4.1-mini`, across four rerankers from three vendors and two
Cohere generations. The two Voyage rows select 30 chunks rather than 15, so they
differ in context budget as well as reranker. See the
[comparison details](docs/methodology.md#retrieval-configurations).

\* Two answers hit the token limit before producing a final-answer marker. Both
received full credit under the paper's extraction rule. Scoring both as wrong
would lower the result from 93.42 to 93.29. The other eleven runs passed the
harness's integrity checks.

MemoryAgentBench covers a broader range of tasks. With `gpt-4o-mini`, GoodMem
scored **52.07**, compared with RAW-RAG's **48.32**, using the paper's aggregation
formula. It scored higher on all four capability groups, with no errors across
the 2,071 questions. These are single-run comparisons.

This run uses the stack the paper's MemoryAgentBench conditions use
(appendix C.3): `text-embedding-3-small`, 128 candidates narrowed to 15, Cohere
Rerank-v3.5. An earlier run on a Voyage index scored 51.44 and is kept under
`results/mab-*-superseded-voyage-stack/`. Unlike LoCoMo, where the reranker
barely moves the score, the retrieval stack matters here: the two runs differ by
up to 7.8 points on an individual source, in both directions.

BEAM measures ingestion time. GoodMem stored its 1,710-message conversation in
**1.43 hours**, averaging **3.004 seconds per message** under sequential
insertion. The measured cost per message changed little as the conversation
grew. This timing includes the wait for each memory to become searchable.

[RESULTS.md](RESULTS.md) contains the full tables and LoCoMo intervals. Read the
[methodology](docs/methodology.md) for the scoring rules and limits of the
comparisons. Standalone LongMemEval-S and PersonaMem-v2 have no adapters here.

## Run a benchmark

You need Python 3.13, [uv](https://github.com/astral-sh/uv), and a reachable
GoodMem instance. Register an embedder and reranker on that instance, along with
the reader models and the `gpt-4o-mini-2024-07-18` judge. GoodMem's provider proxy
uses the credentials stored with those registrations.

From the repository root:

```bash
uv venv --python 3.13
uv pip install -e .
. .venv/bin/activate
cp .env.example .env
```

Fill in `.env` with your instance URL, GoodMem API key and model registration
UUIDs, then load it:

```bash
set -a
. ./.env
set +a
```

The registration variables are `GMBENCH_EMBEDDER_ID`, `GMBENCH_RERANKER_ID`,
`GMBENCH_JUDGE_LLM_ID` and `GMBENCH_READER_LLM_IDS`. The last takes pairs such as
`gpt-4o-mini=<uuid>,gpt-4.1-mini=<uuid>`. Use the UUIDs from your own instance.

Download LoCoMo's ten-conversation release to `data/locomo10.json`; the
[dataset guide](data/README.md) describes the expected files. The default
[configuration](config/locomo.toml) selects 30 of 128 dense candidates and adds
two neighboring turns on each side. The published Voyage runs use this setup.

Run each stage in order:

```bash
gmbench ingest
gmbench retrieve
gmbench read
gmbench judge
gmbench export
```

`ingest` waits for the turns to be indexed. `retrieve` saves the reader's context
and reports evidence coverage. `read` generates an answer for each question;
`judge` scores those answers and writes a summary. Finally, `export` prepares
the files that can be committed alongside the results.

Use `--run-tag` to give an experiment its own output directory. To try another
retrieval setting on an existing index, give it a new run tag and reuse the
original `--ingest-tag`. Put these options before the command:

```bash
gmbench --run-tag my-run --reader-model gpt-4.1-mini read
```

The [running guide](docs/running.md) gives complete commands for the top-15
configuration, the local Qwen scorer, and the other benchmarks. It also covers
resuming work and generating result tables.

## How the code fits together

A LoCoMo run starts in [`locomo.py`](src/gmbench/locomo.py), which loads dated
turns and the questions about them. [`ingest.py`](src/gmbench/ingest.py) gives each
conversation its own GoodMem space and each turn its own memory. The dialogue
id travels with the memory as metadata.

[`retrieve.py`](src/gmbench/retrieve.py) uses those ids to recover selected turns
from the local dataset. It adds neighboring turns, merges overlaps, and puts the
excerpts in conversation order. Saving this context lets both readers work from
the same evidence. [`coverage.py`](src/gmbench/coverage.py) checks whether the
annotated supporting turns made it into that context.

The remaining stages answer and grade the questions. The prompt templates live
in [`prompts.py`](src/gmbench/prompts.py); their wording is part of the benchmark
protocol. [`summary.py`](src/gmbench/summary.py) computes scores and intervals,
while [`markdown.py`](src/gmbench/markdown.py) turns saved summaries into the
results document.

MemoryAgentBench has its own [dataset adapter](src/gmbench/memoryagentbench.py)
and [runner](src/gmbench/mab_run.py) because its sources need different storage
units and graders. [`beam.py`](src/gmbench/beam.py) handles the sequential
ingestion measurement.

## Inspect the evidence

Each run writes to `results/<run-tag>/`. Full retrieval contexts stay local;
compressed exports are committed under `export/`.

| Export | Contents |
|---|---|
| `summary-<model>.json` | LoCoMo scores, intervals and run configuration, with integrity counts |
| `answers-<model>.jsonl.gz` | Reader responses and extracted answers |
| `judgements-<model>.jsonl.gz` | Individual judge responses and scores |
| `retrieval-index.jsonl.gz` | Selected and delivered turn ids, plus a digest of each context |
| `mab-<capability>-summary-<model>.json` | MemoryAgentBench source and capability scores |
| `outcomes-<model>.jsonl.gz` | MemoryAgentBench predictions and grades |
| `beam-summary.json`, `beam-ingest.jsonl.gz` | BEAM totals and per-message timings |

The retrieval index supports coverage checks without including full source
conversations. The `drift` command compares its digests with a new retrieval
pass to identify contexts that have changed.

For local checks:

```bash
uv pip install pytest ruff
pytest -q
```

Dataset-dependent tests skip when their input files are absent. The remaining
tests use local fixtures and simulated service responses.

## License

The harness is [MIT licensed](LICENSE). Download benchmark data from its authors
and follow the terms supplied with each dataset. This repository does not
include the source corpora.
