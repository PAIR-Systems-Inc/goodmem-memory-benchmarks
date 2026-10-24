# Results

Generated from run summaries by `gmbench table`.

LoCoMo accuracy gives each question equal weight. A question's score is
the mean of its recorded judgements; the 95% interval is a percentile
bootstrap over those question means.

"Tie" means the published point estimate falls inside this run's interval.
This comparison does not include uncertainty in the published estimate
or establish statistical equivalence. Configuration names below come from
the saved summaries; see the [methodology](docs/methodology.md) for model
attribution and other limits of the comparison.

## Overall

| Configuration | Reader | GoodMem | 95% CI | RAW-RAG | vs RAW-RAG | Best agentic | vs best agentic |
|---|---|---:|---|---:|---|---:|---|
| Hosted substitute | `gpt-4.1-mini` | **93.23** | [91.99, 94.42] | 93.70 | tie (-0.47) | 91.69 | ahead (+1.54) |
| Hosted substitute | `gpt-4o-mini` | **88.72** | [87.19, 90.24] | 89.07 | tie (-0.35) | 88.12 | tie (+0.60) |
| Paper reranker | `gpt-4.1-mini` | **93.42** | [92.19, 94.59] | 93.70 | tie (-0.28) | 91.69 | ahead (+1.73) |
| Paper reranker | `gpt-4o-mini` | **88.66** | [87.10, 90.17] | 89.07 | tie (-0.41) | 88.12 | tie (+0.54) |
| Voyage rerank-2.5 | `gpt-4.1-mini` | **94.29** | [93.14, 95.37] | 93.70 | tie (+0.59) | 91.69 | ahead (+2.60) |
| Voyage rerank-2.5 | `gpt-4o-mini` | **87.84** | [86.19, 89.37] | 89.07 | tie (-1.23) | 88.12 | tie (-0.28) |
| Voyage rerank-3 | `gpt-4.1-mini` | **94.35** | [93.20, 95.45] | 93.70 | tie (+0.65) | 91.69 | ahead (+2.66) |
| Voyage rerank-3 | `gpt-4o-mini` | **88.44** | [86.82, 89.98] | 89.07 | tie (-0.63) | 88.12 | tie (+0.32) |

## Configurations

- **Hosted substitute** — embedder `text-embedding-3-small`, reranker `qwen3-rerank`, 128 dense candidates to top-15
- **Paper reranker** — embedder `voyage-4-large`, reranker `Qwen/Qwen3-Reranker-4B`, 128 dense candidates to top-15
- **Voyage rerank-2.5** — embedder `voyage-4-large`, reranker `rerank-2.5`, 128 dense candidates to top-30
- **Voyage rerank-3** — embedder `voyage-4-large`, reranker `rerank-3`, 128 dense candidates to top-30

## Hosted substitute — `gpt-4.1-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 96.43 | [95.16, 97.58] | 96.79 | 95.12 | tie (-0.36) |
| Multi-hop | 282 | 92.32 | [89.13, 95.15] | 92.32 | 88.30 | tie (-0.00) |
| Temporal | 321 | 90.34 | [87.02, 93.35] | 90.86 | 91.59 | tie (-0.52) |
| Open-domain | 96 | 77.43 | [69.10, 85.42] | 80.21 | 76.04 | tie (-2.78) |
| **Overall** | 1540 | 93.23 | [91.99, 94.42] | 93.70 | 91.69 | tie (-0.47) |

Mean evidence supplied to the reader: 62 turns, 10,775 characters (~2,694 tokens).

Integrity: clean (1540 questions scored).

## Hosted substitute — `gpt-4o-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 94.09 | [92.47, 95.60] | 94.57 | 93.94 | tie (-0.48) |
| Multi-hop | 282 | 87.47 | [83.57, 91.13] | 85.93 | 84.04 | tie (+1.54) |
| Temporal | 321 | 80.69 | [76.22, 84.84] | 82.04 | 80.69 | tie (-1.35) |
| Open-domain | 96 | 72.22 | [63.19, 80.90] | 73.61 | 73.96 | tie (-1.39) |
| **Overall** | 1540 | 88.72 | [87.19, 90.24] | 89.07 | 88.12 | tie (-0.35) |

Mean evidence supplied to the reader: 62 turns, 10,775 characters (~2,694 tokens).

Integrity: clean (1540 questions scored).

## Paper reranker — `gpt-4.1-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 96.16 | [94.81, 97.34] | 96.79 | 95.12 | tie (-0.63) |
| Multi-hop | 282 | 92.43 | [89.36, 95.27] | 92.32 | 88.30 | tie (+0.11) |
| Temporal | 321 | 91.90 | [88.68, 94.70] | 90.86 | 91.59 | tie (+1.04) |
| Open-domain | 96 | 77.43 | [69.10, 85.07] | 80.21 | 76.04 | tie (-2.78) |
| **Overall** | 1540 | 93.42 | [92.19, 94.59] | 93.70 | 91.69 | tie (-0.28) |

Mean evidence supplied to the reader: 60 turns, 10,874 characters (~2,718 tokens).

Integrity: needs review (1540 questions scored).

## Paper reranker — `gpt-4o-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 94.49 | [92.91, 95.96] | 94.57 | 93.94 | tie (-0.08) |
| Multi-hop | 282 | 86.76 | [82.74, 90.43] | 85.93 | 84.04 | tie (+0.83) |
| Temporal | 321 | 79.02 | [74.56, 83.28] | 82.04 | 80.69 | tie (-3.02) |
| Open-domain | 96 | 75.35 | [67.01, 83.68] | 73.61 | 73.96 | tie (+1.74) |
| **Overall** | 1540 | 88.66 | [87.10, 90.17] | 89.07 | 88.12 | tie (-0.41) |

Mean evidence supplied to the reader: 60 turns, 10,874 characters (~2,718 tokens).

Integrity: clean (1540 questions scored).

## Voyage rerank-2.5 — `gpt-4.1-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 96.79 | [95.52, 97.90] | 96.79 | 95.12 | tie (-0.00) |
| Multi-hop | 282 | 93.97 | [91.02, 96.57] | 92.32 | 88.30 | tie (+1.65) |
| Temporal | 321 | 92.11 | [89.10, 94.81] | 90.86 | 91.59 | tie (+1.25) |
| Open-domain | 96 | 80.56 | [72.57, 88.19] | 80.21 | 76.04 | tie (+0.35) |
| **Overall** | 1540 | 94.29 | [93.14, 95.37] | 93.70 | 91.69 | tie (+0.59) |

Mean evidence supplied to the reader: 107 turns, 19,386 characters (~4,847 tokens).

Integrity: clean (1540 questions scored).

## Voyage rerank-2.5 — `gpt-4o-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 94.53 | [92.98, 96.00] | 94.57 | 93.94 | tie (-0.04) |
| Multi-hop | 282 | 87.00 | [82.98, 90.66] | 85.93 | 84.04 | tie (+1.07) |
| Temporal | 321 | 76.12 | [71.44, 80.58] | 82.04 | 80.69 | behind (-5.92) |
| Open-domain | 96 | 70.83 | [61.81, 79.51] | 73.61 | 73.96 | tie (-2.78) |
| **Overall** | 1540 | 87.84 | [86.19, 89.37] | 89.07 | 88.12 | tie (-1.23) |

Mean evidence supplied to the reader: 107 turns, 19,386 characters (~4,847 tokens).

Integrity: clean (1540 questions scored).

## Voyage rerank-3 — `gpt-4.1-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 96.67 | [95.40, 97.82] | 96.79 | 95.12 | tie (-0.12) |
| Multi-hop | 282 | 94.21 | [91.37, 96.69] | 92.32 | 88.30 | tie (+1.89) |
| Temporal | 321 | 92.32 | [89.30, 95.02] | 90.86 | 91.59 | tie (+1.46) |
| Open-domain | 96 | 81.25 | [72.92, 88.54] | 80.21 | 76.04 | tie (+1.04) |
| **Overall** | 1540 | 94.35 | [93.20, 95.45] | 93.70 | 91.69 | tie (+0.65) |

Mean evidence supplied to the reader: 107 turns, 19,284 characters (~4,821 tokens).

Integrity: clean (1540 questions scored).

## Voyage rerank-3 — `gpt-4o-mini`

| Category | N | GoodMem | 95% CI | RAW-RAG | Best agentic | Verdict |
|---|---:|---:|---|---:|---:|---|
| Single-hop | 841 | 94.69 | [93.14, 96.16] | 94.57 | 93.94 | tie (+0.12) |
| Multi-hop | 282 | 88.06 | [84.28, 91.73] | 85.93 | 84.04 | tie (+2.13) |
| Temporal | 321 | 77.36 | [72.79, 81.72] | 82.04 | 80.69 | behind (-4.68) |
| Open-domain | 96 | 71.88 | [62.85, 80.56] | 73.61 | 73.96 | tie (-1.73) |
| **Overall** | 1540 | 88.44 | [86.82, 89.98] | 89.07 | 88.12 | tie (-0.63) |

Mean evidence supplied to the reader: 107 turns, 19,284 characters (~4,821 tokens).

Integrity: clean (1540 questions scored).

---

## Reranker evidence coverage

This comparison covers 1,322 of 1,540 questions
with resolvable evidence annotations. Each stage uses the same saved
candidate sets (128 candidates per question).
Coverage is measured before neighbor expansion.

| Stage | All gold evidence in top-15 | Mean gold recall |
|---|---:|---:|
| GoodMem dense top-15, no rerank | 69.4% | 76.2% |
| + hosted `qwen3-rerank` | 79.7% | 86.5% |
| + local `Qwen3-Reranker-4B` | 82.1% | 88.6% |

---

## MemoryAgentBench

Aggregation follows the paper's appendix D.5: three capabilities average
their source scores, while Test-Time Learning averages its five
in-context sources first and then weights that against ReDial equally.

Here, "ahead" and "behind" describe differences between single-run
scores. No comparison interval is estimated for these results.

| Capability | N | GoodMem | RAW-RAG | Verdict | Errors |
|---|---:|---:|---:|---|---:|
| Accurate Retrieval | 1000 | 71.48 | 67.85 | ahead (+3.63) | 0 |
| Test-Time Learning | 700 | 46.35 | 47.68 | behind (-1.33) | 0 |
| Long-Range Understanding | 171 | 36.43 | 31.74 | ahead (+4.69) | 0 |
| Conflict Resolution | 200 | 51.50 | 46.00 | ahead (+5.50) | 0 |
| **Overall** (mean of four) | 2071 | **51.44** | 48.32 | ahead (+3.12) | — |

### By source

| Capability | Source | N | Accuracy | Errors |
|---|---|---:|---:|---:|
| Accurate Retrieval | `ruler_qa1` | 100 | 87.00 | 0 |
| Accurate Retrieval | `ruler_qa2` | 100 | 69.00 | 0 |
| Accurate Retrieval | `eventqa` | 500 | 93.60 | 0 |
| Accurate Retrieval | `longmemeval` | 300 | 36.33 | 0 |
| Test-Time Learning | `recsys_redial` | 200 | 20.50 | 0 |
| Test-Time Learning | `icl_banking77` | 100 | 87.00 | 0 |
| Test-Time Learning | `icl_clinic150` | 100 | 86.00 | 0 |
| Test-Time Learning | `icl_nlu` | 100 | 86.00 | 0 |
| Test-Time Learning | `icl_trec_coarse` | 100 | 61.00 | 0 |
| Test-Time Learning | `icl_trec_fine` | 100 | 41.00 | 0 |
| Long-Range Understanding | `infbench_sum` | 100 | 2.44 | 0 |
| Long-Range Understanding | `detective_qa` | 71 | 70.42 | 0 |
| Conflict Resolution | `factconsolidation_mh` | 100 | 9.00 | 0 |
| Conflict Resolution | `factconsolidation_sh` | 100 | 94.00 | 0 |

---

## BEAM-1M — ingestion cost

Messages are added in conversation order. Each must reach COMPLETED
before the next is submitted, so elapsed time includes the wait for
indexing. This measures sequential ingestion latency.

| | |
|---|---:|
| Conversation | `1` — Building a Multi-Language AI Chatbot with Contextual Memory |
| Messages ingested | 1710 of 1710 |
| Approximate tokens | 1,061,523 |
| Total time | 1.43 h (5136 s) |
| Mean per message | 3.004 s |
