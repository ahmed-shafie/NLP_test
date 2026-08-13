# Vector-DB v0.8 corpus — what it changed and how it was measured

The input is a 40,955-row CSV (`banking_nlu_vector_db_v08_final.csv`): 1,872
hand-written rows across 29 labels plus the full 39,083-row ArBanking77 corpus
(MSA, Palestinian, and the AraFinNLP Gulf / Moroccan / Tunisian test sets).

## Why it cannot be indexed as it stands

| Problem | Evidence | Handling |
|---|---|---|
| Two label spaces in one column | 77 Arabic customer-service topics ("وصول البطاقة") sit beside executable intents | topics map to `fallback`, the fine label moves to `topic` |
| Test rows inside the index | 369/370 of the public ArBanking77 test rows are in the file | rows tagged `split=test_*` are held out and never indexed |
| Contradictory single tokens | `حوّل` is labelled both `transfer_money` and `currency_conversion`; 51 such texts | dropped |
| Fragments as evidence | "ابعتلو" (transfer) sits 0.86 from "الغِ الامر المستمر", which then routed into a transfer | only `category=sentence` rows of 3+ words are indexed |
| Mislabelled greetings | "أهلاً، أريد تحويل مبلغ" carries the `greet` label | small-talk rows containing an action verb are dropped |

## Held-out slice

7,667 Gulf / Moroccan / Tunisian customer-service questions — dialects with no
representation in the index. None of them is a command, so the only correct
outcome is "no money flow".

## Measured

Classifier-level, index built from the built-in examples plus the CSV:

| Index | gold intent accuracy | OOS → fallback | money flow opened on OOS |
|---|---|---|---|
| built-in examples only (112) | 0.812 | 32.3% | 22.7% |
| + authored CSV rows (1,901) | 0.894 | 66.6% | 19.7% |
| + capped at 30/topic (3,873) | 0.891 | 93.4% | 4.1% |
| + capped at 100/topic (9,601) | 0.875 | 98.0% | 1.4% |
| **+ full corpus (33,315)** | 0.851 | **99.3%** | **0.5%** |

End-to-end through `route_fresh_turn` (the decision the customer actually gets)
on the 303-row gold set:

| Configuration | intent accuracy | money flow opened on the held-out slice |
|---|---|---|
| before | 0.993 | 872/7,667 = 11.4% |
| full corpus, no engine change | 0.977 | — |
| **full corpus + underspecified-request cue** | **1.000** | **141/7,667 = 1.8%** |

The dip in the middle row is one failure mode, not many: with 85% of the index
labelled out of scope, an underspecified request ("أرغب في تحويل مبلغ") retrieves
five customer-service neighbours and is refused. The fix is in the engine, not
in the data — an action verb with no question word and no question mark is a
request to act, so the flow opens and asks for the missing slots.

## Abuse detection under a 31k index

The semantic safety net used to score abuse by its share of the retrieved
neighbours. That share is a property of the index, not of the turn: once 31k
out-of-scope rows are indexed, a genuine insult retrieves four abusive
neighbours and one complaint and slips under the bar. Abuse is now scored by the
similarity of its *nearest* abusive example. Bar chosen on the held-out slice:

| bar | real complaints flagged as abuse |
|---|---|
| 0.70 | 1/7,667 |
| **0.75** | **1/7,667** |
| 0.80 | 0/7,667, but misses genuine insults |

## Reproducing

```bash
export VECTOR_DB_CSV=path/to/banking_nlu_vector_db_v08_final.csv
python -m scripts.build_example_corpus              # rebuild app/nlu/data
python -m research.vector_db_v08.eval_index         # classifier, all variants
python -m research.vector_db_v08.eval_holdout       # end-to-end, held-out slice
python -m research.vector_db_v08.eval_abuse_bar     # abuse false positives
python -m scripts.eval_nlu                          # gold set
```

## Licensing

The 39,083 ArBanking77 rows come from SinaLab (Birzeit University). The dataset
card carries no licence, and the accompanying model's MIT licence does not cover
the data. Treat this corpus as research/evaluation material until SinaLab
confirms terms for commercial use.
