# PoC: Arabic beneficiary-name detection with Hugging Face Transformers

Question asked: *can a Hugging Face model replace the 20k-row name gazetteer that keeps
matching ordinary speech (`الي` registered as a name), and does it survive real Saudi
utterances?*

Answer: **yes, and the decisive ingredient was not the model — it was training it on real
out-of-domain negatives.** The first version scored 100 % on our own data and invented a
beneficiary on 45 % of real Saudi utterances.

Nothing here is wired into the running app: this directory is offline research, and
`app/` is untouched. `results.txt` is the raw output of the run described below.

## Reproduce

```bash
# 1. MASSIVE ar-SA (CC BY 4.0, Amazon Science) -> research/ner_recipient_ar/data/
curl -sSL https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz \
  | tar xz -C /tmp 1.0/data/ar-SA.jsonl 1.0/LICENSE
mkdir -p research/ner_recipient_ar/data
cp /tmp/1.0/data/ar-SA.jsonl research/ner_recipient_ar/data/
cp /tmp/1.0/LICENSE research/ner_recipient_ar/data/MASSIVE-LICENSE

# 2. generated banking utterances + real MASSIVE negatives (train partition only)
python research/ner_recipient_ar/gen_data.py

# 3. fine-tune on CPU (~8 min on 8 cores)
python research/ner_recipient_ar/train_ner.py

# 4. compare every system on every slice
python research/ner_recipient_ar/run_compare.py
```

Data, generated files and the 415 MB checkpoint are git-ignored.

---

## 1. Systems compared

| # | System | What it is |
|---|---|---|
| S1 | **current: regex cue + gazetteer** | `entities.extract_recipient` + `canonicalize_recipient` — `إلى/الى/الي/ل` cue patterns, then spelling correction against `app/data/names.csv` (20 087 rows) |
| S2 | `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner` | off-the-shelf, Apache-2.0, no training |
| S3 | `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner` | off-the-shelf, Apache-2.0, no training |
| S4 | **fine-tuned** `camelbert-mix` + PER head | trained here, 8.1 min on 8 CPU cores |
| S5 | **S4 + beneficiary-directory tie-break** | S4, then the `لـ` reading is disambiguated against the customer's own beneficiary list |

Licences verified from the source, not from memory: CAMeLBERT checkpoints **Apache-2.0**,
MASSIVE **CC BY 4.0** (Amazon Science), `transformers` **Apache-2.0**. No gated model, no
non-commercial clause.

## 2. Evaluation slices

| Slice | Rows | Who wrote it | What it measures |
|---|---|---|---|
| `gold(37)` | 37 | us, before this PoC (`app/eval/nlu_gold.jsonl`) | the Arabic transfer phrasings we already claim to support |
| `hard(20)` | 20 | me, for this PoC (`hard_ar.jsonl`) | name-first / no-cue / dialect / own-account, **8 of them negatives** |
| `massive-neg(400)` | 400 | **nobody here** — MASSIVE `ar-SA`, *test* partition, no `person` annotation | how often a system **invents** a beneficiary on real Saudi speech |
| `massive-per(200)` | 200 | **nobody here** — MASSIVE `ar-SA`, *test* partition, gold `person` slot | can it find a person name in out-of-domain phrasing |

Metric: exact match of the recipient string after `app.nlu.normalize.normalize` (so
`أحمد == احمد`, the same folding the beneficiary lookup uses). Negatives must yield `None`.

MASSIVE is split by **its own partition**: only `train` rows were used for fine-tuning,
only `test` rows for evaluation — no leakage.

## 3. Results

| System | gold(37) | hard(20) | massive-neg(400) invented | massive-per(200) | latency |
|---|---|---|---|---|---|
| S1 current regex + gazetteer | **1.000** | 0.500 | **71 / 400 = 17.8 %** | 0.010 | 1.2 ms |
| S2 camelbert-mix-ner (off the shelf) | 0.432 | 0.700 | 8 / 400 = 2.0 % | 0.605 | 7.9 ms |
| S3 camelbert-msa-ner (off the shelf) | 0.649 | 0.850 | 15 / 400 = 3.8 % | 0.650 | 8.1 ms |
| S4 fine-tuned | 0.946 | 1.000 | 8 / 400 = 2.0 % | 0.750 | 8.5 ms |
| **S5 fine-tuned + directory tie-break** | **1.000** | **1.000** | **8 / 400 = 2.0 %** | **0.760** | 8.5 ms |

`massive-per` understates every system: MASSIVE annotates the attached preposition as part of
the name (gold `لمريم`), which is exactly what we strip. Scoring `مريم` against gold `لمريم` as
correct, S5 reaches **176 / 200 = 0.880**.

`gold(37) = 1.000` for the current system is **not** a compliment: those 37 rows were written
against these regexes. The columns that carry information are `hard` and the two MASSIVE ones.

## 4. The finding that matters

The first fine-tune used only our generated banking sentences plus 30 hand-written negatives:

| version | gold | hard | invented on massive-neg |
|---|---|---|---|
| trained on our data only | 0.919 | 0.950 | **181 / 400 = 45.3 %** |
| + 3 700 real MASSIVE `train` rows as negatives | 0.946 | 1.000 | **8 / 400 = 2.0 %** |

(the accuracy of the first row was measured before the `لل` decoding fix, which is worth a
couple of points there and nothing at all on the invention rate)

Same architecture, same banking accuracy, **22× fewer invented beneficiaries.** A model
trained only on sentences we imagined learns "any unfamiliar token after a preposition is a
name" — literally the gazetteer's bug, relearned. The fix was data, not modelling.

## 5. Error analysis

**The 8 remaining "false positives" are mostly real names that MASSIVE simply did not tag as
`person`** (singers are tagged `artist_name`):

```
خل ارتاح لمحمد عبده جاهزة عشان تشغلها   → pred 'محمد عبده'  (a person; tagged artist_name)
ذكرني بموعدي الثلاثاء مع منى            → pred 'منى'        (a person; gold has no slot)
ملحم بركات                              → pred 'م بركات'    (a person, span cut short)
أرسل email لجدتي يقول بنزورك يوم السبت   → pred 'دتي'        ← a genuine error
```
So the true invention rate is **below 2 %**. Two of the eight are real mistakes, and both are a
span that starts mid-word — which the beneficiary lookup would reject anyway.

The current system's failures on the same slice are not names at all — they are the regex
swallowing whole clauses:

```
كم نحتاج لحين نوصل المطار       → recipient 'حين نوصل المطار'
هل لدي أي طلبات صداقة جديدة     → recipient 'دى أي طلبات صداقة جديدة'
```
and on our own hard slice it produced `recipient='حساب'` for **`حول 500 إلى حسابي`** — an
own-account transfer read as a transfer to a person called "حساب".

## 5b. Second blind check: real banking-app review language

MASSIVE is voice-assistant phrasing. `scrape_reviews.py` pulls banking-app reviews from
**Apple's public customer-reviews feed**, discovering Finance apps per storefront (Saudi and
Egypt so far; the script walks 13 Arabic storefronts) — real customers, in dialect, **writing
about transfers, beneficiaries, balances and bills**. That is the closest public text to our
actual users. 2 249 reviews → 1 078 Arabic → 289 banking-relevant → **411 sentences**.

None of them is a command to an assistant, so there is no recipient to find: every one of the
411 must yield `None`.

| System | invents a beneficiary on 411 real banking sentences |
|---|---|
| current regex + gazetteer | **131 / 411 = 31.9 %** |
| fine-tuned NER + directory | **12 / 411 = 2.9 %** |

Split by storefront (Egypt added on a second pass; Apple rate-limits the feed hard, so the
harvest is incremental and the script resumes):

| storefront | regex + gazetteer | NER + directory |
|---|---|---|
| Saudi (347) | 32.9 % | 2.9 % |
| Egypt (64) | 26.6 % | 3.1 % |

Dialect did not move either system beyond its confidence interval — but 64 Egyptian sentences
is ±11 %, so this is a hint, not a result. Note also that a large share of Egyptian reviews are
written in English or franco-Arabic (`b3ml 7wela`), which neither system addresses at all.

```
تطبيق سيئ مو جاي يحول معلق له يومين            → regex 'ه يمين'
مستحيل تطلع المستفيدين الي عندك و معلوماتهم     → regex 'عندك و معلوماتهم مع انه أول يمديك'
لو الغى المصرف رسوم التحويل لبنوك محلية        → regex 'بنوك محله أيكون أفضل بنك'
```
The ten NER hits are fragments (`ف`, `شر`, `برق`) that the beneficiary lookup rejects. The
gazetteer's are whole clauses that look like a name to the resolver.

The feed is Apple's own published endpoint (no page scraping, no login); the script keeps only
rating and review text and drops the nickname, and the harvested text is git-ignored.

## 5c. ArBanking77: the assistant's out-of-scope behaviour, measured

`eval_arbanking77.py` runs the **whole pipeline** (not just the extractor) over
[ArBanking77](https://huggingface.co/datasets/SinaLab/ArBanking77) (SinaLab / Birzeit
University, ArabicNLP 2023) — BANKING77 arabized into MSA **and Palestinian dialect**.

Its 77 intents are bank *customer-service* topics ("card swallowed", "why verify identity",
"exchange rate"). We support five actions and **none of them is in that list**, so the correct
answer for all 370 rows is: answer or fall back, but do not start a money flow.

| | MSA (179) | Palestinian (191) |
|---|---|---|
| correctly fell back | **58.1 %** | **30.9 %** |
| wrongly answered as balance inquiry | 22.9 % | 22.0 % |
| **started a money flow** | **16 / 179 = 8.9 %** | **31 / 191 = 16.2 %** |
| invented a recipient | 1 / 179 | 4 / 191 |

```
هل تقبل الشيكات؟                          → add_beneficiary
ما هو الدفع المعلق؟                        → pay_bill
شو هي أحسن طريقة لصرف العملات؟             → transfer_money, recipient='صرف العملات؟'
```

Two things this says that the NER work does not:

1. **Intent routing is the weaker half.** The extractor was the thing we measured; the
   classifier is what opens a money flow on "do you accept cheques?".
2. **Dialect halves the fallback rate** (58 % → 31 %) on the *same* 77 intents in the *same*
   translation pipeline — the cleanest dialect-gap number we have.

Caveats: only the ~1 450-row public **sample** is on the Hub (the full ~31 k set is by request
from SinaLab), the corpus is translated from English so it is not natively-typed Arabic, and
the dataset card carries **no licence** — the model repo is MIT, but the data licence must be
confirmed with SinaLab before any non-research use. Run it with
`NLU_LLM_ENABLED=false` (the LLM exception handler otherwise dominates the runtime).

## 6. Cost

* model on disk **415 MB** (fp32); ~450 MB RSS when loaded
* latency on 8 CPU cores, short utterances: **mean 7.9 ms, p50 7.8, p95 8.9, max 13.9 ms**
  → inside the 800 ms budget with three orders of magnitude to spare, unlike an LLM
* training: **8.1 min** on CPU, no GPU, 9 700 rows, 3 epochs
* dynamic int8 quantisation would roughly quarter the size; not needed yet

## 7. Honest caveats

1. **`hard(20)` is mine.** I wrote it *and* then tuned the `لـ` decoding while looking at it,
   so 20/20 is optimistic. The MASSIVE columns are the trustworthy ones.
2. **Still no customer data.** MASSIVE is voice-assistant phrasing, not banking. It proves the
   model does not hallucinate on real Saudi speech; it cannot prove the banking error rate.
3. **Confidence interval**: 2.0 % on 400 rows is ±1.4 % at 95 %. Don't quote "2 %" as exact.
4. **Arabic only.** The English path (`_EN_RECIPIENT_RE`) still needs a capital letter, so
   `send 500 to ahmed hassan` fails — untouched by this PoC.
5. The generated grammar is my imagination too. Its role is to teach the *shape* of banking
   requests; the negatives are what keep it honest.

## 8. Recommendation

Integrate as the **primary** Arabic recipient extractor, keeping everything downstream intact:

```
NER (PER span)  →  proclitic readings ("لمحمد" → محمد | لمحمد)
                →  beneficiary directory match (existing resolve_contact)
                →  disambiguation prompt when >1 match
                →  deterministic confirmation ("(yes/no)")   ← unchanged, Tier A
```
* the gazetteer stops being an *extractor* and remains only a spelling helper — the `الي`
  class of bugs disappears by construction;
* keep the regex as the fallback when the model file is absent, exactly like Stanza today;
* no LLM anywhere on this path, no change to money-critical replies;
* ship the model file as an artefact (or an internal HF repo) — 415 MB does not belong in git;
* add `massive-neg` as a permanent eval slice with a hard gate: **invented recipients ≤ 3 %**.

Not recommended: adopting Rasa. It replaces the dialogue framework we already have and would
not have changed a single number in this table.
