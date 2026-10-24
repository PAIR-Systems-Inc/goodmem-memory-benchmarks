# goodmem-memory-benchmarks

Measuring [GoodMem](https://github.com/PAIR-Systems-Inc/goodmem) against the
retrieval baseline from **[*Is Complex Agentic Memory Always Needed? Naive RAG Is
Stronger Than We Thought*](https://arxiv.org/abs/2610.30755)** (arXiv:2610.30755v1), on the
paper's own benchmarks, with the paper's own reader, prompts and grading rules.

**The finding: parity.** On LoCoMo, GoodMem lands statistically indistinguishable
from RAW-RAG — the paper's hand-tuned naive-RAG baseline — and ahead of the
strongest agentic memory system in its comparison. That is the whole claim. It is
deliberately not "GoodMem wins": the differences against RAW-RAG are smaller than
this measurement's own noise, and a repository that overstated them would be the
first thing a reader checked.

---

## Why this comparison

The paper's argument is that dense retrieval plus cross-encoder reranking over
*raw* conversation history matches or beats twelve agentic memory systems —
systems that extract facts, summarise, build graphs, and mutate stored state.
Across four benchmarks RAW-RAG ranked first in six of seven benchmark–model
settings.

GoodMem is a production memory system of that same architectural shape: raw
content in, chunked, embedded, retrieved, reranked. So the question is narrow and
checkable. Swap GoodMem's retrieval in for theirs, hold *everything* else fixed,
and see where the number lands.

Everything downstream of retrieval is pinned to the paper:

| | |
|---|---|
| Readers | `gpt-4o-mini`, `gpt-4.1-mini` |
| Reader prompt | Appendix G.4, verbatim — seven-step chain of thought, `FINAL ANSWER:` extraction |
| Judge | `gpt-4o-mini-2024-07-18`, temperature 0, **three** judgements per answer |
| Grading | Appendix G.3, verbatim, including the strict JSON-label rule where a bare `CORRECT` with no JSON envelope scores **zero** |
| Question set | 1,540 LoCoMo questions, adversarial category 5 excluded |

Those prompts are in [`src/gmbench/prompts.py`](src/gmbench/prompts.py), copied
rather than paraphrased. A stronger reader would raise the score without saying
anything about memory, so the reader is not a variable here.

---

## Results

Full tables for every benchmark are in **[RESULTS.md](RESULTS.md)**, generated
by `gmbench table` from the run summaries rather than typed by hand. The per-question evidence behind every number is in
`results/<run>/export/`.

### LoCoMo

| Configuration | Reader | GoodMem | 95% CI | RAW-RAG | Best agentic |
|---|---|---:|---|---:|---:|
| Paper models | `gpt-4o-mini` | 88.72 | [87.19, 90.24] | 89.07 — tie | 88.12 — tie |
| Paper models | `gpt-4.1-mini` | 93.23 | [91.99, 94.42] | 93.70 — tie | 91.69 — **ahead** |
| Voyage | `gpt-4o-mini` | 87.84 | [86.19, 89.37] | 89.07 — tie | 88.12 — tie |
| Voyage | `gpt-4.1-mini` | **94.29** | [93.14, 95.37] | 93.70 — tie | 91.69 — **ahead** |

Statistically tied with RAW-RAG in all four, and ahead of MemMachine — the
strongest agentic system in the paper's comparison — on both `gpt-4.1-mini` rows.
The paper-models configuration reaches that on **~2,700 evidence tokens against
the paper's ~4,800**, a little over half the context budget.

> **One caveat.** Neither reranker the paper used was usable as served here, so
> the scored rows pair the paper's embedder with its reranker *family* rather than
> its exact checkpoint. That substitution is measured, not assumed: the paper's own
> `Qwen3-Reranker-4B`, driven correctly over the identical index, retrieves 2.4
> points more gold evidence — not enough to move a reader across a tie. See
> [On the reranker](#on-the-reranker). Cohere Rerank-v3.5 remains unmeasured.

### How to read a verdict

Every score carries a bootstrap 95% interval over per-question means. "Tie" means
the published point estimate falls **inside** our interval — the data cannot
distinguish them. On these slice sizes a one- or two-point difference is noise,
and the published figures carry comparable uncertainty of their own, which is not
shown and would widen each comparison further.

### The noise floor is about 0.7 points

Two scored passes of the *identical* configuration, on an unchanged index, gave
88.51 and 87.84 overall. That 0.67-point spread is not sampling error; it is the
hosted reranker. Isolating it: dense top-128 came back bit-identical in order,
set and scores across runs, while the reranker returned identical scores on two
calls and different ones on a third. Query embeddings also jitter slightly for
identical input, though not enough to shift the dense stage.

So retrieval here is **not bit-reproducible**, and the honest consequence is that
any claimed difference below roughly a point — in either direction — is an
artifact. Re-running this repository will not give you these numbers to two
decimal places. It should give you the same verdicts.

---

## How it works

One GoodMem space per conversation, one memory per conversation turn.

| RAW-RAG | Here |
|---|---|
| 256-token windows, 32-token overlap | One memory per turn; character-based recursive chunking at 1500/150, which leaves nearly every turn whole |
| Flat per-conversation index | One GoodMem space per conversation |
| Top-128 dense candidates | `requestedSize: 128` |
| Cross-encoder rerank to top-15 | `ChatPostProcessor` with `reranker_id` and `max_results` |
| Display-time turn expansion | Each selected turn expanded by ±N turns client-side, de-duplicated, chronological |

Retrieval returns *which turns* were selected; the reader context is rebuilt from
the local dataset by dialogue id. So the evidence is exactly the source text, not
something reassembled from chunk offsets.

### Deliberate departures, and why

- **Character chunking, not token.** GoodMem's `TOKEN_COUNT` length measurement
  currently falls back to character count server-side. With one memory per turn
  it barely matters — a 1500-character window leaves almost every turn as a
  single chunk — and the paper's own ablation finds chunk sizes from 128 to 1024
  tokens do not significantly move its total.
- **Turns, not windows.** A turn carries an exact timestamp and dialogue id as
  metadata, which is what makes expansion exact and coverage measurable.
- **Reader and judge through GoodMem's provider proxy.**
  `/v1/proxy/llms/{id}` forwards the request body unchanged and injects the
  stored credential, so the prompt stays byte-identical to the paper's while no
  provider key ever reaches this process.
- **HNSW, not exact search.** pgvector HNSW with `efSearch` tuned per request. A
  sweep of 400 against 800 moved coverage by 0.00 points, so recall is not
  binding at this depth.

### On the reranker

The paper reports two LoCoMo settings, which land within 0.06 points of each
other for the `gpt-4o-mini` reader:

| Setting | Embedder | Reranker | RAW-RAG overall |
|---|---|---|---|
| Table 10 | `text-embedding-3-small` | Cohere Rerank-v3.5 | 89.07 / 93.70 |
| Table 11 | Qwen3-Embedding-4B | Qwen3-Reranker-4B | 89.13 / 94.55 |

Neither was usable as served. Cohere needs a credential this instance does not
have. Qwen3-Reranker-4B is present over vLLM and *inverted* — see below. So the
scored runs use DashScope's hosted `qwen3-rerank`: the paper's reranker family,
correctly served.

**That substitution is not material, and it is measured rather than asserted.**
The paper's own checkpoint was driven directly — applying the yes/no relevance
template it requires, with the paper's appendix G.1 instruction — over the
identical index and the identical dense candidate sets:

| Stage | All gold evidence in top-15 | Mean gold recall |
|---|---:|---:|
| GoodMem dense top-15, no rerank | 69.4% | 76.2% |
| + hosted `qwen3-rerank` (the scored runs) | 79.7% | 86.5% |
| + `Qwen3-Reranker-4B`, the paper's own | **82.1%** | **88.6%** |

*n* = 1,322 questions — those of the 1,540 carrying resolvable evidence
annotations, scored before the pass was halted as having answered its question.

The paper's own reranker retrieves 2.4 points more gold evidence than the
substitute. Both feed a reader whose scored result is already statistically tied
with RAW-RAG, and a 2.4-point coverage difference at the head of a 15-turn
context does not move a reader across that gap. The raw scores are published in
`results/paper-exact/`; reproduce them with `tools/qwen3_reranker.py`, then
`retrieve --from-scores` to assemble contexts and score a full pass.

One disclosed deviation in that measurement: the checkpoint was loaded with int8
weights, because the GPU had 5.84 GiB free against the ~8 GiB bf16 needs.

### A reranker can be worse than no reranker

Qwen3-Reranker is *generative*. It emits no similarity score: it is prompted with
a yes/no relevance question, and the score is the probability it puts on "yes".
An endpoint serving those weights behind a generic scoring API without that
template returns numbers that are not relevance at all.

Not hypothetical. This instance's vLLM deployment put the correct turn in the
top-15 **16.7%** of the time, against **75.0%** for switching reranking off
entirely. Asked *"When did Caroline go to the LGBTQ support group?"* it ranked
the turn reading *"I went to a LGBTQ support group yesterday"* **81st of 128**,
every score compressed into a 0.249–0.510 band — the signature of a generative
reranker scored without its relevance head. The same pipeline scored 55.71
overall with it and 87.84 with a working one: a 32-point swing from one
misconfigured stage. Driven correctly, the same weights rank that turn **first,
at 0.9997**.

| Reranker | Gold evidence in top-15 |
|---|---:|
| `qwen3-rerank`, hosted (DashScope) | **91.7%** |
| `rerank-2.5` / `rerank-3` / `rerank-3-lite` (Voyage) | 83.3% |
| `gte-rerank-v2` (DashScope) | 83.3% |
| *no reranking at all* | 75.0% |
| `Qwen3-Reranker-4B`, self-hosted over vLLM | 16.7% |

That is a 24-question diagnostic sample, which is why the absolute figures differ
from the 1,322-question table above; the ordering is what matters.

The check ships as a command. It costs no reader tokens and exits non-zero when a
reranker scores below the dense baseline, because at that point the stage is not
weak but inverted, and no scored pass should be run on it:

```bash
gmbench --config config/locomo-paper.toml --ingest-tag paper-qwen3 \
        verify-reranker --rerankers <uuid> <uuid> ...
```

The broken run is published too, under
`results/paper-models-broken-reranker/export/` — same pipeline, same reader, same
judge, differing only in the reranker. A broken reranker is silent: it returns
plausible scores in the right shape, and the only symptom is a worse number
somewhere downstream. A 30-point drop could have been the chunking, the embedder,
the prompt, the judge or the index. Coverage located the one wrong stage in a
single command, before a reader token was spent.

### The Voyage configuration

`voyage-4-large` + `rerank-2.5` is reported as a separate row, not the headline,
because a model the paper did not use beating one it did says nothing about the
memory system. Voyage also shipped `rerank-3` on 2026-09-30, which claims to beat
Cohere Rerank-v4.0 Pro; this repository does not use it, for the same reason.

---

## Evidence coverage

LoCoMo annotates which turns support each answer, so retrieval can be scored with
**zero reader tokens**. The `coverage` stage compares the annotated turns against
what actually reached the reader. Coverage bounds accuracy from above, which makes
it the right thing to sweep configurations on before paying for a scored run.

Three defects in the released annotations are handled explicitly rather than
absorbed into the number:

- Six questions pack several references into one string, such as
  `"D22:1 D22:2 D9:10 D9:11"`. These are split. Before fixing this, our own
  coverage figure was **pessimistic**.
- Three questions reference turns absent from their own conversation. These are
  excluded from the denominator and counted — not scored as retrieval misses,
  because they are the dataset's defect rather than the system's.
- Two carry references that are not recoverable at all (`"D"`, `"D:11:26"`). No
  guess is made. `"D30:05"` is probably `"D30:5"`, but inventing a reference
  changes what coverage measures.

---

## Trusting the output

A benchmark that fails loudly is fine. One that fails quietly and reports a
plausible number is not, so the harness is built to make the second impossible.

- **Completeness gates.** Every stage writes one row per question keyed by
  question id, and no stage consumes the previous one's file without checking it
  covers exactly the expected question set. A short shard is otherwise
  indistinguishable from a complete one.
- **Integrity accounting.** Failed reader calls, answers truncated at the token
  cap, errored judge requests and unparseable judge labels are each counted and
  printed **before** the scores; `report` exits non-zero when any are present.
  The paper scores these as zeros and keeps them in the denominator, which this
  matches so the number stays comparable — counting them separately is what stops
  an outage from passing as a low score. Every run published here is clean.
- **Stream verification.** The server frames each result set as BEGIN, items, END
  and states how many items to expect. All three are checked, so a truncated
  response cannot return a shorter, plausible result set.
- **Degradation is fatal.** A partial or timed-out vector search aborts the
  request rather than returning fewer chunks. So does a status code the harness
  does not recognise.
- **Resumable stages.** Reader and judge flush each row as it lands and resume by
  question id, so an interrupted pass costs the remainder rather than all of it.
- **Drift detection.** `drift` compares current contexts against a prior export's
  digests, so a pipeline change cannot be silently scored against answers
  generated from older evidence.
- **Generated tables.** The published numbers come from `gmbench table` reading
  the run summaries. Transcription is as silent a failure as any other.

---

## Benchmarks

| Benchmark | Questions | Status |
|---|---:|---|
| **LoCoMo** | 1,540 | Fully scored, two configurations, both readers |
| **MemoryAgentBench** | 2,071 | **Withdrawn and being re-measured** — a harness bug invalidated 607 questions |
| **BEAM-1M** | — | Fully measured: 1,710 messages, 1.43 h, flat per-message cost |
| LongMemEval-S | 500 | Data acquired and verified, adapter not written |
| PersonaMem-v2 | 5,000 | Data acquired and verified, adapter not written |

### MemoryAgentBench

All four capability groups, all 2,071 published questions, reader `gpt-4o-mini`,
128 dense candidates to top-15.

| Capability | GoodMem | RAW-RAG | Stability across identical runs |
|---|---:|---:|---|
| Accurate Retrieval | 54.10 | 67.85 | 54.08 – 58.48 over four runs |
| Test-Time Learning | 46.35 | 47.68 | 46.35 – 47.50 over two runs |
| Long-Range Understanding | 18.31 | 31.74 | 18.31 – 30.58 over two runs |
| Conflict Resolution | 51.50 | 46.00 | 51.50 in every run |
| **Overall** (mean of four) | **42.56** | **48.32** | — |

> ### These numbers are withdrawn
>
> **A harness bug meant 607 of the 2,071 questions were answered against the
> wrong documents.** The table above is left in place because it was published,
> not because it is correct. Clean figures will replace it.
>
> Each MemoryAgentBench row is a self-contained document store, and several rows
> share one source label. The space holding a row's documents was named after
> the *label* rather than the row, so rows of the same source collided on one
> space — and the guard meant to catch a half-built store could not see it,
> because a space full of **another row's** documents is populated, COMPLETED
> and indistinguishable from a correct one. A row whose unit count did not
> exceed its predecessor's was therefore answered against the previous row's
> documents.
>
> The damage lands entirely on the multi-row sources:
>
> | Source | Rows | Questions answered against the wrong store | Published score |
> |---|---:|---:|---:|
> | `infbench_sum` | 100 | **97 of 100** | 0.00 |
> | `longmemeval` | 5 | **180 of 300** | 11.00 |
> | `eventqa` | 5 | **300 of 500** | 48.40 |
> | `detective_qa` | 10 | **30 of 71** | 36.62 |
> | `ruler_qa1`/`qa2`, all `icl_*`, `recsys_redial`, `factconsolidation_*` | 1 each | **0** | 69–94 |
>
> That split is the capability split. Accurate Retrieval and Long-Range
> Understanding — the two groups reported as behind — are the two containing
> multi-row sources. Test-Time Learning (level) and Conflict Resolution (ahead)
> are built entirely from single-row sources and have no affected questions. It
> also explains the run-to-run variance reported in the third column, which was
> described as reranker noise and was nothing of the kind: which rows were
> poisoned depended on the order their unit counts happened to fall in.
>
> Fixed by keying the space on the row id, and by refusing a space that holds
> any other row's documents. Two regression tests cover it. Accurate Retrieval
> and Long-Range Understanding are being re-run; Test-Time Learning and Conflict
> Resolution are unaffected and stand.

Aggregation follows the paper's appendix D.5. Three capabilities average their
source scores; Test-Time Learning does not — its five in-context sources are
averaged first and then weighted equally against ReDial. Using a flat mean over
TTL's six sources instead reads 63.83 where the specified formula reads 46.35,
so the rule is worth getting right.

#### By source

| Capability | Source | N | Accuracy | Range over repeat runs |
|---|---|---:|---:|---|
| Accurate Retrieval | `ruler_qa1` | 100 | 88.00 | 88.00 – 88.00 |
| Accurate Retrieval | `ruler_qa2` | 100 | 69.00 | 68.00 – 69.00 |
| Accurate Retrieval | `eventqa` | 500 | 48.40 | 48.00 – 59.60 |
| Accurate Retrieval | `longmemeval` | 300 | 11.00 | 11.00 – 19.33 |
| Test-Time Learning | `icl_banking77` | 100 | 87.00 | 86.00 – 87.00 |
| Test-Time Learning | `icl_clinic150` | 100 | 86.00 | 86.00 – 86.00 |
| Test-Time Learning | `icl_nlu` | 100 | 86.00 | 84.00 – 86.00 |
| Test-Time Learning | `icl_trec_coarse` | 100 | 61.00 | 61.00 – 61.00 |
| Test-Time Learning | `icl_trec_fine` | 100 | 41.00 | 41.00 – 43.00 |
| Test-Time Learning | `recsys_redial` | 200 | 20.50 | 20.50 – 23.00 |
| Long-Range Understanding | `detective_qa` | 71 | 36.62 | 36.62 – 59.15 |
| Long-Range Understanding | `infbench_sum` | 100 | 0.00 | 0.00 – 2.00 |
| Conflict Resolution | `factconsolidation_sh` | 100 | 94.00 | 93.00 – 94.00 |
| Conflict Resolution | `factconsolidation_mh` | 100 | 9.00 | 9.00 – 10.00 |

#### Where GoodMem is strong

The five in-context-learning sources (86–87 on Banking77, CLINC150 and NLU) and
RULER (88.00) are the cleanest results in this repository, and both are stable to
within a point across runs. Single-hop fact consolidation reaches
94.00, and for the 92% of those
questions where one fact both names the entity and states the answer, GoodMem put
that exact fact in front of the reader **100% of the
time** — from 2,310 facts of which hundreds are near-identical in form. The
remaining points there are the reader preferring an older conflicting fact, not a
retrieval miss.

#### Where it is weak, and why

**Multi-hop fact consolidation (9.00) is unreachable by
construction.** Only 1% of those questions have
a single fact naming the question's entity *and* stating the answer, because the
answer-bearing fact names the bridging entity instead. Asked *"Where did the
religion associated with the Bagratuni Dynasty come into existence?"*, the gold
fact is `475. Christianity was founded in the city of Taipei.` — zero words
shared with the question. One query cannot reach it, and the paper's baseline is
deliberately single-shot: it *"isolates evidence access from query-time search
control."* Adding a hop would score better and stop being comparable.

**InfBench summarization (0.00) is not a top-k retrieval task.** The reader sees
15 passages, roughly 30,000 characters of a 3.2-million-character book — under
1% — and writes a plausible but generic summary that matches none of the 25
specific plot points the benchmark grades against. The judge is working: it
receives the full rubric and returns 0 of 25 covered. The paper's LRU of 31.74
implies its own InfBench score is also near zero, given DetectiveQA around 60.

**LongMemEval within MemoryAgentBench (11.00) is a genuine weakness worth
chasing.** It is a different setting from the standalone benchmark, where RAW-RAG
reaches 87.53: there each example has an isolated store and a question date to
filter on, while here 60 questions share one pool of 111 sessions including other
questions' distractors. Even allowing for that, 11.00 is low, and it is the
single clearest thing to improve in this repository.

#### Half of these sources are not measurable in one run

| Stable to ≤1 point | Swings 8 points or more |
|---|---|
| `ruler_qa1`, `ruler_qa2`, all five `icl_*`, `factconsolidation_sh`/`_mh` | `eventqa` (11.6), `longmemeval` (8.3), `detective_qa` (22.5) |

The split tracks the metric, not the system. The stable sources have short
factual golds — an entity, a numeric label — graded by exact or boundary match.
The unstable ones ask the reader to generate freely and then grade loosely:
EventQA scores a 45-character narrative continuation by containment, and
DetectiveQA is a near-chance multiple choice over a novel whose decisive clue is
rarely among 15 retrieved passages. The published figures are single runs too, so
comparisons at this granularity should be read as indicative rather than
decisive.

### BEAM-1M

Ingestion cost, not accuracy. One ordered 1,710-message conversation approaching
a million tokens, streamed a message at a time. The measured quantity is
**Add-to-ready**, not Add: GoodMem's create returns once the memory is persisted
with embedding still queued, so timing the call would measure the API rather than
the system. Outer concurrency is one by construction — no batching, no overlap.

**Result: 1,710 messages, ~1.06M tokens, 1.43 hours, 3.00 s/message.**

| System | Cumulative ingest | vs GoodMem |
|---|---:|---|
| RAW-RAG (local in-process index) | 0.33 h | 4.3× faster |
| **GoodMem** | **1.43 h** | — |
| MemOS | 3.25 h | 2.3× slower |
| Mem0 | 3.9 h | 2.7× slower |
| EMem | 8.5 h | 5.9× slower |
| MemMachine | 18.2 h | 12.7× slower |
| Memora | 27.0 h | 18.9× slower |

Comparison figures are the paper's, measured on its hardware against its own
configurations — not a controlled head-to-head. Treat the ordering as meaningful
and the ratios as approximate.

**The shape matters more than the endpoint.** This benchmark exists to catch
ingest cost that grows as history accumulates, which is what happens when a
system re-reads or re-consolidates what it already stored. Across the full run,
per-message cost is flat:

| Segment of the conversation | Add | Add-to-ready | Total |
|---|---:|---:|---:|
| First 20% | 0.206 s | 2.795 s | 3.001 s |
| Middle 20% | 0.206 s | 2.799 s | 3.005 s |
| Last 20% | 0.205 s | 2.799 s | 3.005 s |

Ingesting the 1,710th message into a million-token store costs the same as the
first. The cost is per-job overhead, not a function of store size.

Two honest qualifications. The `Add` column is floored by this harness's own
request pacing, so it is an upper bound on GoodMem's accept latency rather than a
measurement of it. And the 2.8 s to readiness is background-job scheduling under
*strictly sequential single-message adds*, which is the paper's protocol and not
how anyone would bulk-load: the same harness ingests LoCoMo at **51 turns per
second** when batching is allowed, about 150× faster per message. The sequential
number measures per-job latency; the batched one measures throughput. Both are
real, and they answer different questions.

---

## Running it

Needs the [`uv`](https://github.com/astral-sh/uv) toolchain and a reachable
GoodMem instance with an embedder, a reranker and the reader/judge models
registered. **No provider API key is needed in this process** — inference runs
through GoodMem's proxy against the stored credentials.

```bash
uv venv --python 3.13 && uv pip install -e .
cp .env.example .env     # fill in instance URL, API key, and registration UUIDs
set -a && . ./.env && set +a
```

Place the LoCoMo 10-conversation release at `data/locomo10.json` (see
[`data/README.md`](data/README.md) — datasets are not redistributed here).

```bash
gmbench ingest                    # one space per conversation, one memory per turn
gmbench retrieve                  # retrieve, expand, print coverage
gmbench read                      # generate answers
gmbench judge                     # grade, print integrity then scores
gmbench export                    # write the committable evidence set
gmbench table                     # regenerate the README tables
```

Other stages: `sweep` grid-searches retrieval depth and expansion on coverage
alone; `coverage` reports it from saved retrievals; `drift` checks contexts
against a prior export; `beam` measures ingestion cost; `mab-inspect` reports the
MemoryAgentBench selection.

Useful flags: `--run-tag` to keep runs separate, `--ingest-tag` to point a new
retrieval run at an existing ingest, `--reader-model` to switch reader,
`--no-resume` to redo a stage, `--conversations-limit` / `--questions-limit` for
smoke runs.

### What a run writes

`results/<run-tag>/` holds the per-question shards — `retrieval.jsonl`,
`answers-<model>.jsonl`, `judgements-<model>.jsonl` — which are large and
gitignored. `results/<run-tag>/export/` holds the committable evidence, which is
what is published here:

| File | What it is |
|---|---|
| `summary-<model>.json` | Scores with intervals, the full configuration, and integrity counts |
| `judgements-<model>.jsonl.gz` | Every individual judgement and its label |
| `answers-<model>.jsonl.gz` | Full reader responses, including the chain of thought |
| `retrieval-index.jsonl.gz` | Selected and delivered turn ids per question, plus a context digest |

The retrieval *index* ships instead of the contexts themselves: it makes coverage
independently recomputable and retrieval auditable, without redistributing
conversations that are the benchmark's to license.

---

## Licence and provenance

This harness is MIT licensed — see [LICENSE](LICENSE). It redistributes no
benchmark data. Each dataset is licensed by its authors:

| Dataset | Licence |
|---|---|
| LoCoMo | see [snap-research/locomo](https://github.com/snap-research/locomo) |
| MemoryAgentBench | MIT, over upstream corpora with their own terms |
| BEAM | CC BY-SA 4.0 |
| LongMemEval | MIT |
| MEME | CC BY 4.0 |
| PersonaMem-v2 | CC BY 4.0 |

If you re-run this and get something different, that is interesting and worth an
issue — particularly if a verdict changes rather than a decimal.
