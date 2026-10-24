# Benchmark data

Download the datasets separately and keep their files under this directory.
Source corpora are not included in the repository. Each dataset's authors
provide its license and any terms inherited from upstream sources.

## LoCoMo

Get the ten-conversation release from
[snap-research/locomo](https://github.com/snap-research/locomo) and place
`locomo10.json` here:

```text
data/locomo10.json
```

The loader checks the release size and question categories. Excluding the
adversarial category leaves 1,540 questions across ten conversations:

| Category | Questions |
|---|---:|
| Single-hop | 841 |
| Multi-hop | 282 |
| Temporal | 321 |
| Open-domain | 96 |

Use `run.dataset_path` in the TOML configuration to select another location.

## MemoryAgentBench

Follow the download instructions in
[HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench).
Keep the parquet splits in a `data/` subdirectory of the dataset root:

```text
data/memoryagentbench/
  data/
    Accurate_Retrieval-*.parquet
    Test_Time_Learning-*.parquet
    Long_Range_Understanding-*.parquet
    Conflict_Resolution-*.parquet
  entity2id.json
```

ReDial scoring needs `entity2id.json` to turn reference movie ids into titles.
The adapter selects one context length per source. Check the expected
2,071-question selection before ingestion:

```bash
uv pip install pyarrow
gmbench --run-tag mab-selection mab-inspect --dataset data/memoryagentbench
```

`--dataset` accepts another root with the same layout. The selected source
variants are defined in
[`memoryagentbench.py`](../src/gmbench/memoryagentbench.py).

## BEAM

Place the BEAM 1M split at:

```text
data/beam/1M-00000-of-00001.parquet
```

The ingestion experiment uses conversation `1`, which contains 1,710 messages.
The `beam` command accepts another file through `--parquet` and another
conversation through `--conversation`. It also requires `pyarrow`.

The [running guide](../docs/running.md) explains the sequential timing protocol
and gives commands for each benchmark. Standalone LongMemEval-S and
PersonaMem-v2 are not supported by the current adapters.
