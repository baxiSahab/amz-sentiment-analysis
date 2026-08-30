# Results — fastText vs TF-IDF + Logistic Regression

Amazon Polarity sentiment classification. All accuracies on the **same fixed
400,000-row held-out test set**; validation is a 20% split of each training sample.

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
| val acc    | 0.9056 | 0.9062 | 0.9170 | 0.9233 | 0.9291 | 0.932671 | 0.936733 | 0.939650 |

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
