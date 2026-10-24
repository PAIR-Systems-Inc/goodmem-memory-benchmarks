# What the benchmarks measure

The comparison replaces RAW-RAG's retrieval implementation with GoodMem while
keeping the reader and grading protocol close to the paper. Retrieval models
and storage boundaries also affect the result, so they need to be considered
alongside the final score. References to the paper below mean
[*Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We
Thought*](https://arxiv.org/abs/2610.30755v1), version 1.

## LoCoMo

LoCoMo contains ten conversations divided into dated sessions. This harness
evaluates the 1,540 questions in categories 1–4, excluding the adversarial
category. Each conversation has its own GoodMem space, so a question can retrieve
only from the conversation it concerns.

One turn becomes one memory. Recursive chunking uses a 1,500-character window
with 150 characters of overlap, which leaves most turns whole. The paper uses
256-token windows with 32-token overlap. Those are different storage boundaries,
even when they ultimately lead the reader to the same evidence.

GoodMem retrieves 128 dense candidates. A reranker selects the best 15 or 30,
depending on the configuration. The harness resolves their dialogue ids against
the local dataset and includes two turns on each side. Overlapping selections
are merged and presented chronologically, with session dates and markers for
gaps. The reader therefore sees source text rather than reconstructed chunk
fragments.

### Retrieval configurations

| Configuration | Embedder | Reranker | Selected chunks |
|---|---|---|---:|
| Hosted Qwen | `text-embedding-3-small` | DashScope `qwen3-rerank` | 15 |
| Local Qwen checkpoint | See attribution note below | `Qwen/Qwen3-Reranker-4B`, int8 | 15 |
| Voyage 2.5 | `voyage-4-large` | `rerank-2.5` | 30 |
| Voyage 3 | `voyage-4-large` | `rerank-3` | 30 |

Paper table 10 uses `text-embedding-3-small` with Cohere Rerank-v3.5; table 11
uses Qwen3-Embedding-4B with Qwen3-Reranker-4B. The hosted Qwen configuration uses
table 10's embedder with a substitute reranker. The local scorer uses table 11's
reranker checkpoint at int8 precision. Neither configuration reproduces a
complete row from the paper. The published results here do not include a Cohere
Rerank-v3.5 run.

Model attribution for `paper-exact` is incomplete. Its summary records
`voyage-4-large`, while its reranker-comparison record names
`text-embedding-3-small`. The committed `gpt-4o-mini` summary also labels the
reranker `rerank-2.5`, whereas the local provenance file names Qwen3-Reranker-4B.
These records do not establish an exact model-matched comparison. The
`gpt-4.1-mini` score appears in RESULTS.md, but its answer and judgement exports
are not currently committed. Treat these rows as supplementary measurements.

GoodMem uses pgvector HNSW search with `ef_search=400`. This is approximate
nearest-neighbor search, another difference from the paper's flat index.

The hosted Qwen run supplies about 2,700 evidence tokens per question; the
Voyage runs supply about 4,800. These counts use characters divided by four as
an estimate. The different selection depths make context budget part of the
comparison.

### Reading and grading

The readers are `gpt-4o-mini` and `gpt-4.1-mini`. Both use the LoCoMo prompt from
appendix G.4 and a 4,096-token completion limit. Reader calls go through
GoodMem's provider proxy, which adds the stored credential.

The harness extracts text after `## FINAL ANSWER:`. If the marker is absent,
it passes the full response to the judge, following the paper's extraction
rule. This matters for truncated responses: the reasoning may contain a correct
answer even though the reader never reaches its final-answer section.

The judge is `gpt-4o-mini-2024-07-18` at temperature zero. Each answer receives
three judgements using appendix G.3's prompt and label parser. The parser first
looks for a fenced JSON block, then a flat JSON object, then tries the entire
response. A bare `CORRECT` without a JSON label scores zero. Failed judge calls
also score zero and remain in the denominator, with a separate error count.

The three judgements are averaged for each question. Category and overall
accuracy are then means of those question scores. This gives every question
equal weight in the overall LoCoMo score.

### Uncertainty and incomplete answers

The 95% intervals come from 10,000 bootstrap resamples of question means, using
a fixed seed. They describe variation across questions in the observed run.
They do not capture all variation from rerunning hosted services, nor do they
account for questions sharing a conversation.

In the tables, “tie” means the paper's point estimate falls within this run's
interval. “Ahead” and “behind” mean it falls below or above that interval.
These labels are descriptive; they are not a statistical equivalence test or a
comparison of two independently estimated intervals. The paper's own sampling
uncertainty is not included.

Two passes of the same configuration scored 88.51 and 87.84, a difference of
0.67 points. That illustrates run-to-run variation, but two observations cannot
establish a general noise threshold or identify all its causes. Small score
differences should be interpreted with that limitation in mind.

The local Qwen run with `gpt-4.1-mini` has two truncated answers out of 1,540.
Both lack a final-answer marker and were judged correct in all three repetitions.
The score is 93.42; treating both as wrong gives 93.29. The integrity report flags
this run. The other seven LoCoMo rows have 4,620 judgements each and no recorded
reader or judge failures.

### Evidence coverage

LoCoMo annotates the turns supporting an answer. The coverage report compares
those ids with both the reranker's selections and the expanded reader context.
This gives a useful way to inspect retrieval before generating answers.

| Measure | Meaning |
|---|---|
| Full | Every resolvable annotated turn is in the expanded context |
| Any | At least one annotated turn is in the expanded context |
| Recall | Mean fraction of annotated turns found per question |
| In top-k | At least one annotated turn was selected before expansion |

Some annotations contain several ids in one string; the loader splits them.
Malformed ids are retained as unparsed data. References to absent turns are
reported separately, and questions without any resolvable evidence are omitted
from coverage scoring. These questions still count toward answer accuracy.

Coverage is not a strict upper bound on accuracy. The reader may find the answer
in an unannotated turn, and retrieving every annotated turn does not guarantee a
correct answer. A sweep is best used to compare coverage and context size, then
followed by a scored run of the chosen configuration.

## MemoryAgentBench

The selected release contributes 2,071 questions across four capability groups.
Some sources ship at several context lengths. The adapter selects
`eventqa_131072` and the `32k` variants of single-hop and multi-hop fact
consolidation, alongside the other sources.

Each dataset block gets its own space. Storage units follow the source:
numbered facts retain their precedence, labelled examples stay with their labels,
and chat windows keep their session date. Prose is packed into passages. The
published runs use `gpt-4o-mini` with 128 dense candidates reranked to 15.

Scoring also follows the source. Short answers use case-insensitive containment
with word boundaries. Numeric labels require exact equality. DetectiveQA uses
answer text because option letters can disagree with the reference text.
ReDial resolves movie ids to titles and measures recall in the recommendation
list. InfBench summaries receive fractional credit for coverage of their
supplied plot keypoints.

Each source contributes equally to its capability score, with one exception:
Test-Time Learning first averages its five in-context-learning sources, then
gives that mean the same weight as ReDial. Overall accuracy is the mean of the
four capability scores. This is the aggregation in appendix D.5; averaging all
2,071 questions together would answer a different question.

| Capability | Questions | GoodMem | RAW-RAG |
|---|---:|---:|---:|
| Accurate Retrieval | 1,000 | 71.48 | 67.85 |
| Test-Time Learning | 700 | 46.35 | 47.68 |
| Long-Range Understanding | 171 | 36.43 | 31.74 |
| Conflict Resolution | 200 | 51.50 | 46.00 |
| Overall | 2,071 | 51.44 | 48.32 |

These runs have no recorded errors. They are single measurements, so the table
alone does not establish the uncertainty of the differences.

Performance varies substantially by source. EventQA scores 93.60 and single-hop
fact consolidation 94.00. Multi-hop fact consolidation scores 9.00: a question
may name one entity while the fact containing its answer names an intermediate
entity. A single retrieval query often misses that connection.

InfBench scores 2.44. Its reader receives 15 passages from a book that can run
to millions of characters, then must cover 25 plot points in a summary. This
provides limited access to material spread across the book. LongMemEval within
MemoryAgentBench scores 36.33; it uses shared context blocks and should not be
compared directly with the standalone LongMemEval-S setting.

## BEAM ingestion time

BEAM streams conversation `1`, containing 1,710 messages. The default protocol
submits one memory and waits for `COMPLETED` before sending the next. The total
therefore includes request time and the wait for embedding and indexing.

| Conversation segment | Create request | Wait for readiness | Total per message |
|---|---:|---:|---:|
| First 20% | 0.206 s | 2.795 s | 3.001 s |
| Middle 20% | 0.206 s | 2.799 s | 3.005 s |
| Last 20% | 0.205 s | 2.799 s | 3.005 s |

The run took 1.43 hours. Per-message latency was nearly constant across these
segments, suggesting that accumulated history did not add a substantial cost in
this experiment. The create-request measurement includes the harness's request
pacing; the wait includes background scheduling and polling. These timings do
not isolate the server's processing time or measure bulk-ingestion throughput.

The paper reports 0.33 hours for RAW-RAG and longer times for several agentic
systems. Those measurements used different hardware and configurations, so the
ratios are not a controlled comparison with this GoodMem run.

## Checks around the measurements

LoCoMo stages check saved question ids against the expected set before using a
previous stage's output. Retrieval verifies stream boundaries and item counts.
A degraded search is an error; recognized provider throttles are retried with a
pause shared by the worker pool. Ingestion repairs failed memories and stops if
the expected turns remain unindexed.

The report separates failed requests from incorrect answers. It also records
truncation and missing final-answer markers. A nonzero report status means a
failure check needs attention, even when an accuracy value was produced.

Saved contexts can be compared with an earlier export using `drift`. If the
context changes, the corresponding reader answers need to be generated again.
Run summaries and exported per-question records provide the evidence behind the
published tables.
