# Datasets

Benchmark data is not redistributed here. Each dataset is licensed by its own
authors, and this repository only measures against it.

## LoCoMo

Place the 10-conversation release at `data/locomo10.json`. It is available from
the authors at <https://github.com/snap-research/locomo>.

The harness checks what it loads: the non-adversarial split must come to 1,540
questions across 10 conversations, in the category counts the paper reports
(841 single-hop, 282 multi-hop, 321 temporal, 96 open-domain). A file that does
not match is a different release, and the numbers here would not be comparable.
