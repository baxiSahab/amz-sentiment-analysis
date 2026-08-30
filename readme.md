# Amazon Review Polarity — fastText vs TF-IDF + Logistic Regression

Sentiment classification on the Amazon Review Polarity dataset (3.6M train / 400k test),
comparing a TF-IDF + logistic regression baseline against a fastText classifier at
matched data sizes, splits and seeds.

---

## Running this

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`fasttext==0.9.3` builds from source and needs a C++ toolchain
(`build-essential` on Debian/Ubuntu, Xcode command line tools on macOS,
MSVC Build Tools on Windows). Install it first if the pip step fails.

### 2. Data

Download the Amazon Review Polarity CSVs and place both in one folder:

```
<your-data-folder>/
    train.csv     3,600,000 rows
    test.csv        400,000 rows
```

Both are headerless with three columns in this order: `label, title, text`,
where `label` is 1 (negative) or 2 (positive).

### 3. Point the notebooks at that folder

Each notebook sets `DATA_DIR` in its first code cell, directly under a banner
reading `REPRODUCIBILITY -- REPLACE THIS WITH YOUR OWN DATA PATH`. Edit that one
line in each of the three notebooks. Nothing else needs changing — `artifacts/`
and `results_csv/` are created relative to the notebook.

### 4. Run in this order

| # | notebook | platform | what it does |
|---|---|---|---|
| 1 | `exploration_and_experiments.ipynb` | any | Data preview, class balance, training-set summary |
| 2 | `baseline_full.ipynb` | Windows **or** WSL/Linux | TF-IDF + LogReg at 700k / 1.5M / 3.6M; `C` and `max_features` sweeps; error analysis |
| 3 | `fasttext_model.ipynb` | **WSL/Linux only** | fastText at the same three sizes; sweeps; error analysis |

`baseline_full.ipynb` detects whether it is running under Windows or WSL and resolves
the same folder either way. `fasttext_model.ipynb` uses a single Linux-style path and
the `fasttext` package, so run it from WSL or Linux.

Run All is idempotent in both model notebooks: every expensive step checks for its own
cached output first, so a re-run goes straight to scoring. Cached artifacts carry a
sidecar `.json` stamp of the config that produced them — change a setting and that step
rebuilds itself.

### 5. Where output goes

| path | contents |
|---|---|
| `artifacts/` | Cached matrices, vectorisers and models. **~5 GB** at full scale. |
| `results_csv/` | All result tables (git-ignored; the numbers below are the record). |
| `~/ft/` | fastText corpora and `.bin` models — **outside the project folder**, on the Linux disk. Reading them through `/mnt/c` is roughly an order of magnitude slower, which is why they live here. |

### 6. Runtime and hardware

At 3.6M rows on the reference machine (WSL2):

| | baseline | fastText |
|---|---|---|
| training | 987.6s (754.1 vectorize + 233.5 fit) | 93.7s |
| peak memory | 3.2 GB design matrix in RAM | streams from disk |

Budget roughly **7.5 GB of free RAM** for the baseline at 3.6M and ~5 GB of disk for
`artifacts/`. To do a quick end-to-end check first, set `SIZES = [200_000, 500_000]`
in the config cell of either model notebook.

**Note on the timing comparison:** fastText runs with `thread=8`; the baseline uses
`LogisticRegression(solver="liblinear")`, which is single-threaded. The wall-clock
figures therefore compare 8 threads against 1 and are not a like-for-like measure of
algorithmic cost. The memory figures are unaffected.

### 7. Optional experiments

`fasttext_model.ipynb` has `RUN_EXPERIMENTS = False` in its config cell. Set it to
`True` to run the noise-floor repeat (step 11) and the `minCount` grid (step 12).

### 8. Reproducibility

`SEED = 42`, `VAL_SIZE = 0.2` and identical sampling are used across both model
notebooks, so the two learning curves are directly comparable. Pinned dependency
versions are in `requirements.txt`.

---

## Results

All accuracies below are on the **same fixed 400,000-row held-out test set**;
validation is a 20% split of each training sample.

## Configurations

|          | setting                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------- |
| Baseline | `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=50_000)` + `LogisticRegression(C=2.5)` |
| fastText | `lr=0.07, epoch=5, dim=10, wordNgrams=3, bucket=10_000_000, minCount=9`                             |

## Headline — test set (400k rows)

| sample size | train rows | baseline | fastText           | delta               |
| ----------- | ---------- | -------- | ------------------ | ------------------- |
| 700,000     | 560,000    | 0.932865 | 0.931220           | −0.165 pp          |
| 1,500,000   | 1,200,000  | 0.936268 | 0.937075           | +0.081 pp           |
| 3,600,000   | 2,880,000  | 0.939375 | **0.941965** | **+0.283 pp** |

Validation accuracy at 3.6M: baseline 0.939650, fastText 0.943404.
Validation−test gap ≤ 0.144 pp for both models at every size.

## Cost

|                       | baseline @3.6M                     | fastText @3.6M                     |
| --------------------- | ---------------------------------- | ---------------------------------- |
| vectorize             | 754.1s                             | —                                 |
| fit                   | 233.5s                             | 93.7s                              |
| **total train** | **987.6s**                   | **93.7s**                    |
| peak train memory     | 3,186.9 MB design matrix           | streams from disk                  |
| inference, 400k test  | 31.9s                              | 11.3s                              |
| vocabulary            | 50,000 (cap, binding at all sizes) | 270,790 words + 10M n-gram buckets |

Baseline training cost by size: 154.9s @700k (620.3 MB), 319.8s @1.5M (1,328.5 MB),
987.6s @3.6M (3,186.9 MB).

## Learning curve — baseline, validation accuracy

| train rows | 20k    | 40k    | 80k    | 160k   | 400k   | 560k     | 1.2M     | 2.88M    |
| ---------- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | -------- |
| val acc    | 0.9056 | 0.9063 | 0.9170 | 0.9234 | 0.9291 | 0.932671 | 0.936733 | 0.939650 |

The 20k–400k points come from an earlier sweep at **`C=3.0`** (not 2.5), each validated on
its own 20% split rather than the 400k test set. They are carried into
`baseline_full.ipynb` §3 as the `PRIOR` dict so the combined curve is continuous.
The 560k / 1.2M / 2.88M points are from the full runs in this repo.
Still rising at 2.88M.

## Sweeps

**Baseline — `C` at 3.6M** (validation)

| C       | 1.5      | 2.0      | **2.5**      | 3.0      | 3.5      |
| ------- | -------- | -------- | ------------------ | -------- | -------- |
| val acc | 0.939364 | 0.939601 | **0.939650** | 0.939617 | 0.939619 |

**Baseline — `max_features` 50k vs 100k** (validation; not run at 3.6M)

| size      | 50k      | 100k     | delta     | vectorize       | fit           | matrix                |
| --------- | -------- | -------- | --------- | --------------- | ------------- | --------------------- |
| 700,000   | 0.932671 | 0.933721 | +0.105 pp | 125.0 → 196.8s | 29.9 → 38.0s | 620.3 → 663.7 MB     |
| 1,500,000 | 0.936733 | 0.938670 | +0.194 pp | 232.9 → 255.3s | 86.9 → 93.8s | 1,328.5 → 1,421.1 MB |

**fastText — `minCount` at 700k** (`wordNgrams=3, lr=0.07`, validation)

| minCount | 1         | 3        | 6        | **9**      |
| -------- | --------- | -------- | -------- | ---------------- |
| vocab    | 1,116,466 | 225,252  | 128,291  | **95,590** |
| val acc  | 0.932157  | 0.932143 | 0.931871 | 0.932050         |

At 3.6M, `minCount=9` cut vocabulary 3,629,672 → 270,790 (13×) with validation
accuracy unchanged (0.943417 → 0.943404).

**fastText — `wordNgrams` at 3.6M** (validation)

| wordNgrams | 2        | 3        |
| ---------- | -------- | -------- |
| val acc    | 0.940713 | 0.943417 |
| fit        | 54.5s    | 93.7s    |

**fastText — `dim` at 700k** (`wordNgrams=2, lr=0.10`, validation)

| dim     | 10       | 20       | 50       |
| ------- | -------- | -------- | -------- |
| val acc | 0.928929 | 0.929621 | 0.929893 |
| fit     | 18.0s    | 14.9s    | 34.4s    |
