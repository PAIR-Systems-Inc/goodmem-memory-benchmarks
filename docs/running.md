# Running the experiments

Start with the environment and dataset setup in the [README](../README.md).
Run the commands below from the repository root with the virtual environment
active and `.env` loaded. Model UUIDs always refer to registrations on your own
GoodMem instance.

## LoCoMo with a hosted reranker

`config/locomo-paper.toml` uses 128 dense candidates, 15 reranked selections, and
two neighboring turns on each side. To run the hosted Qwen configuration, set
`GMBENCH_EMBEDDER_ID` to your `text-embedding-3-small` registration and
`GMBENCH_RERANKER_ID` to your DashScope `qwen3-rerank` registration.

Choose fresh tags so your output stays separate from the published runs:

```bash
export GMBENCH_EMBEDDER_ID='your-text-embedding-3-small-uuid'
export GMBENCH_RERANKER_ID='your-hosted-qwen-reranker-uuid'

cfg=config/locomo-paper.toml
run=locomo-hosted

gmbench --config "$cfg" --run-tag "$run" --ingest-tag "$run" ingest
gmbench --config "$cfg" --run-tag "$run" --ingest-tag "$run" \
        verify-reranker
gmbench --config "$cfg" --run-tag "$run" --ingest-tag "$run" retrieve

for model in gpt-4o-mini gpt-4.1-mini; do
  gmbench --config "$cfg" --run-tag "$run" --ingest-tag "$run" \
          --reader-model "$model" read
  gmbench --config "$cfg" --run-tag "$run" --ingest-tag "$run" \
          --reader-model "$model" judge
  gmbench --config "$cfg" --run-tag "$run" --ingest-tag "$run" \
          --reader-model "$model" export
done
```

Check each command's output before proceeding. `verify-reranker` uses a
24-question sample and exits nonzero when an endpoint's top-k coverage falls
below its reference. The reference is a full-depth request through the first
reranker, truncated to k, so it is also reranked. Use `retrieve --dense-only`
to collect an independent dense result set. Inspect the endpoint and its model
template if the check fails.

Retrieval is shared by both readers. The reader model changes the answer and
judgement filenames, so the two sets can live under the same run tag.

For a small trial, add `--conversations-limit 1 --questions-limit 5` before the
command. Keep those limits consistent across retrieval, reading and grading,
and give the trial its own tags. `ingest` stores the whole selected conversation.

## Reuse an index

A run tag names the output directory. An ingest tag locates the space manifest.
To change only the reranker, point the new run at the original ingest and keep
the embedder registration consistent with that index:

```bash
export GMBENCH_EMBEDDER_ID='your-text-embedding-3-small-uuid'
export GMBENCH_RERANKER_ID='your-cohere-rerank-v3.5-uuid'

gmbench --config config/locomo-paper.toml --ingest-tag locomo-hosted \
        verify-reranker --rerankers "$GMBENCH_RERANKER_ID"
gmbench --config config/locomo-paper.toml --run-tag locomo-cohere \
        --ingest-tag locomo-hosted retrieve

for model in gpt-4o-mini gpt-4.1-mini; do
  gmbench --config config/locomo-paper.toml --run-tag locomo-cohere \
          --ingest-tag locomo-hosted --reader-model "$model" read
  gmbench --config config/locomo-paper.toml --run-tag locomo-cohere \
          --ingest-tag locomo-hosted --reader-model "$model" judge
  gmbench --config config/locomo-paper.toml --run-tag locomo-cohere \
          --ingest-tag locomo-hosted --reader-model "$model" export
done
```

Register Cohere Rerank-v3.5 with its provider credential before running this
example. These commands use the supported environment variables for both model
registrations. Reporting reads the embedder from configuration, so it must agree
with the reused index.

## Run Qwen3-Reranker-4B locally

The local scorer uses Qwen's yes/no relevance template and int8 weights. It
requires a CUDA GPU and a separate environment with the dependencies in
[`tools/requirements-reranker.txt`](../tools/requirements-reranker.txt).

```bash
uv venv --python 3.13 .reranker-venv
uv pip install --python .reranker-venv/bin/python \
       -r tools/requirements-reranker.txt
```

The following uses the OpenAI index created as `locomo-hosted` above. Set the
embedder UUID to match it. GoodMem writes the dense candidates, the local model
scores them, and the harness assembles the context from those scores:

```bash
export GMBENCH_EMBEDDER_ID='your-text-embedding-3-small-uuid'

gmbench --config config/locomo-paper.toml --run-tag locomo-local-qwen \
        --ingest-tag locomo-hosted retrieve --dense-only

.reranker-venv/bin/python tools/qwen3_reranker.py \
    --candidates results/locomo-local-qwen/dense-candidates.jsonl \
    --out results/locomo-local-qwen/rerank-scores.jsonl --resume

gmbench --config config/locomo-paper.toml --run-tag locomo-local-qwen \
        --ingest-tag locomo-hosted retrieve --from-scores
```

Then run `read`, `judge` and `export` for each reader with the same config and
tags. Assembly checks that every question has one score per candidate. It also
writes `reranker-provenance.json`, which reporting uses in place of the hosted
reranker registration. This records the int8 precision used by the supplied
scorer.

## Inspect retrieval before scoring

`coverage` reads saved contexts. `sweep` compares selection depth and neighbor
expansion using annotated evidence, without calling the reader:

```bash
gmbench --config config/locomo-paper.toml --run-tag locomo-hosted \
        --ingest-tag locomo-hosted coverage
gmbench --config config/locomo-paper.toml --run-tag locomo-sweep \
        --ingest-tag locomo-hosted sweep \
        --final-k 15,30 --neighbor-turns 1,2,3 --ef-search 400
```

The sweep samples evenly across conversations and categories. Its percentages
therefore differ in weighting from a full LoCoMo run. Compare context size as
well as coverage, then evaluate the chosen settings on all questions.

## Resume or repeat a stage

LoCoMo reader and judge stages append completed rows and resume by question id.
This saves completed requests after an interruption. They also retain rows
recording failed requests; rerunning the command alone does not retry those
rows. To repeat a stage, put `--no-resume` before its command:

```bash
gmbench --config config/locomo-paper.toml --run-tag locomo-hosted \
        --ingest-tag locomo-hosted --no-resume read
gmbench --config config/locomo-paper.toml --run-tag locomo-hosted \
        --ingest-tag locomo-hosted --no-resume judge
```

Repeating `read` replaces that reader's answers, so repeat `judge` afterward.
MemoryAgentBench resumes differently: it drops errored outcomes and retries
those questions while keeping successful results.

If you repeat retrieval, compare its contexts with the prior export before
reusing answers:

```bash
gmbench --config config/locomo-paper.toml --run-tag locomo-hosted \
        --ingest-tag locomo-hosted drift \
        --against results/locomo-hosted/export/retrieval-index.jsonl.gz
```

Run this before replacing the export you intend to compare against. Stable
contexts establish that the evidence text is unchanged; reusing answers also
assumes the reader settings and prompts have stayed the same.

## MemoryAgentBench

Install the optional parquet reader and download the files described in the
[dataset guide](../data/README.md):

```bash
uv pip install pyarrow
gmbench --run-tag mab-selection mab-inspect --dataset data/memoryagentbench
```

The selection should contain 2,071 questions. Use a separate run tag for each
capability: the exported outcomes filename includes the reader model but not
the capability.

```bash
for capability in accurate_retrieval test_time_learning long_range_understanding conflict_resolution; do
  gmbench --config config/locomo-paper.toml --run-tag "mab-$capability" \
          --reader-model gpt-4o-mini mab \
          --dataset data/memoryagentbench --capability "$capability"
  gmbench --config config/locomo-paper.toml --run-tag "mab-$capability" export
done
```

Choose the embedder and reranker registrations before starting. Some sources
contain entire books, so ingestion can take substantially longer than LoCoMo.
The command reports per-source scores and the capability score calculated from
them.

## BEAM

BEAM also needs `pyarrow`. Each invocation creates a new space, so choose an
unused run tag:

```bash
gmbench --run-tag beam-sequential beam \
        --parquet data/beam/1M-00000-of-00001.parquet \
        --conversation 1 --poll-every 1
```

Use `--limit 20` for a short trial with a different tag. Keep `--poll-every 1`
for the sequential protocol: each message must become searchable before the
next is submitted.

BEAM writes `beam-summary.json` and `beam-ingest.jsonl` in the run directory.
The `export` command does not package these files. To prepare them for the
results tables:

```bash
mkdir -p results/beam-sequential/export
cp results/beam-sequential/beam-summary.json results/beam-sequential/export/
gzip -c results/beam-sequential/beam-ingest.jsonl \
    > results/beam-sequential/export/beam-ingest.jsonl.gz
```

## Generate result tables

`table` reads summaries; it does not rerun inference. Run labels are supplied as
`Label=directory`, relative to the configured results directory. To generate a
table from a local run:

```bash
gmbench table --runs 'Hosted Qwen=locomo-hosted' --out local-results.md
```

To read a committed export directly, include `/export` in the directory:

```bash
gmbench table --runs \
        'Hosted Qwen=paper-qwen3/export' \
        'Voyage 2.5=voyage4l-k30/export' \
        'Voyage 3=voyage4l-rerank3/export' \
        --out exported-results.md
```

The generator also includes available MemoryAgentBench and BEAM export
summaries. It rejects duplicate active MemoryAgentBench summaries for the same
capability. Superseded runs are excluded when their directory name contains
`broken` or starts with `_`.

RESULTS.md includes a local Qwen `gpt-4.1-mini` row whose supporting export is
not currently committed. A fresh checkout can regenerate the exported rows;
it cannot reconstruct that additional row from the committed summaries alone.
