# Head-to-head benchmark

FeatureForge against the public automatic feature tools, on public tables, with every tool's output
scored by the same judge. Code and raw results: [bench/h2h](../bench/h2h).

## Protocol

- **One judge for every arm.** A pinned LightGBM (learning rate 0.05, 63 leaves, feature_fraction 0.5,
  bagging 0.8, lambda_l2 5, early stopping on a random 15% of training rows, then three seeds refit on all
  training rows). There is no tuning or stacking, so a gain is a feature gain. Each tool's output is the raw
  columns plus what it generated.
- **Held-out rows.** Each table is split 80/20. The tools see the held-out rows only without labels, as a
  contest's test file. Time-ordered tables are also scored with the latest 20% held out (below).
- **Tools.** Featuretools 1.31 (Deep Feature Synthesis: per-category aggregates and arithmetic
  transforms, its own selection, at most 500 columns), OpenFE 0.0.12, AutoFeat 2.1.1 (not started past
  5,000 rows or 30 columns, where it timed out or crashed in every run; scored as raw), and CAAFE-style LLM
  features: Claude wrote 3-10 feature functions per table from the column names, types, eight sample rows and
  the description, before any score was seen. As in CAAFE, a proposal is kept only if it lowers the
  4-fold CV loss on the training rows.
- **Lift** is the percent reduction of the held-out loss against the raw columns (log loss, or RMSE; RMSLE
  targets are log1p-transformed). A crash or timeout scores as raw.
- **Leak checks.** With shuffled training labels the new version kept nothing on 12 of 12 tables and the
  held-out loss landed at chance. With the held-out labels replaced by a permuted copy, its output files were
  byte-identical (md5) on Balance Scale, vehicles, Jungle Chess and Phoneme: the builders never read them.

## What changed

The comparison exposed two gaps, and three changes close them:

1. **Dense blocks.** The search screens candidates one at a time, so it missed sets of features that help
   only together (Jungle Chess: every single piece-position difference screened below its parents). Now
   every pairwise difference, sum, ratio and product of up to 20 leading numeric columns (capped at 20
   million cells a block) is tried as one block, plus every low-cardinality category's frequency and each
   leading numeric against its group mean. A block is kept when it lowers the CV loss; the gate judges the
   blocks like any other round. Blocks are label-free and off for time-ordered data.
2. **A cross-validated gate for small tables (200-3,000 rows).** A 20% gate on a few hundred rows cannot tell
   a gain from noise. The whole search runs on four folds and is scored on the fifth, five times, so every
   row is a gate row; the search's own rounds are left out there (their picks overfit the search's CV on
   German credit and Pima diabetes). From 3,000 to 10,000 rows, when the usual gate turns the search down,
   the blocks alone get this cross-validated gate.
3. **One LightGBM thread** for tables under 200,000 cells: same output, about 10x faster on a busy machine.

## Results

**63 public tables** (20 AutoGluon/PMLB, 29 OpenML copies of Kaggle contests and Playground tables, 14
tables from the OpenFE and CAAFE papers), seed 0, with diabetes130 and the time-ordered tables corrected
(below):

| Tool | Mean lift | Median lift | Worst table |
|---|---|---|---|
| **FeatureForge (new)** | **+6.9%** | **+1.6%** | −1.8% |
| FeatureForge (before) | +4.3% | +0.1% | −1.8% |
| Featuretools | +5.0% | +0.6% | −3.6% |
| OpenFE | −166% (Tic-Tac-Toe −10,398%) | 0.0% | |
| AutoFeat | +1.4% | 0.0% | −23.8% |

Before the corrections, average ranks among the five tools and raw were 2.38 (new), 2.94 (before), 3.15
(Featuretools), 4.03 (AutoFeat), 4.21 (OpenFE).

**Tables used while designing vs not.** The blocks and gates were designed while looking at 20 of the 63
tables. On those 20 the new version averages +12.2% (before +5.5%, Featuretools +10.9%); on the other 43,
+5.2% (before +4.4%, Featuretools +2.4%), average rank 1.98 against 3.07 for Featuretools.

**20 fresh OpenML tables**, fixed before any run and never used before (OpenML-CC18 and regression
benchmark tables), 3 seeds, 19 finished:

| Tool | Mean lift | Median lift | Avg rank | Worst | Median build |
|---|---|---|---|---|---|
| **FeatureForge (new)** | **+17.0%** | +0.5% | **1.92** | −0.4% | 124 s |
| FeatureForge (before) | +14.8% | +1.8% | 2.32 | −0.2% | 56 s |
| Featuretools | +6.5% | +2.7% | 2.63 | −41.5% | 4 s |

The new version gained over the old one on Banknote (+66% → +90%), Wilt (+11% → +21%), Brazilian houses
(+12% → +19%) and CPU small (0 → +2.7%), and lost on Ozone (+1.9% → 0), WDBC (+1.4% → +0.3%) and red wine
(+1.8% → −0.4%). Featuretools is ahead by 3-16 points on WDBC, credit approval, kr-vs-kp and blood
transfusion, where FeatureForge keeps nothing.

**Small tables, five splits each** (12 tables under 2,000 rows): the new version +11.6% mean lift, before
+5.5%, Featuretools +11.1%; it is never more than 1.4% below raw. Five splits separate vehicles and steel
plates (new version ahead), Pima diabetes (ahead of Featuretools) and Balance Scale (Featuretools ahead by
1.6 points); the other gaps are inside the run-to-run spread.

**Fresh splits of the biggest gain.** Jungle Chess on three new splits: +36.6%, +36.0%, +35.6% (Featuretools
+27.6%, +27.3%, +26.5%).

**Contests.** The new code switches off on time-ordered data and its gates above 10,000 rows, and four
contests were rebuilt from Kaggle's files, old against new: Home Credit (full train file, five child tables;
the held-out 20% plus Kaggle's test as unlabeled rows) and an IEEE-CIS window (latest 40% of transactions,
last fifth held out) came out byte-identical, AUC 0.766 → 0.803 and 0.932 → 0.957 over raw; full Santander
and full Porto scored the same.

**LLM features** (CAAFE-style, 57 tables): +8.8% mean lift, worst −10.5%. They win where knowing the rules
matters (Jungle Chess +98%, Tic-Tac-Toe +99%, Balance Scale +99.96%, churn +32%) and lose where a guessed
meaning is wrong (spambase −10.5%). Several proposals used outside knowledge (the Friedman formula, Kaggle
forum decodings), flagged in [bench/h2h/llm/README.md](../bench/h2h/llm/README.md).

## Corrections and limits

- **diabetes130.** Patients repeat across rows, and both FeatureForge versions counted each patient's visits
  over the whole file. That count includes the later visits the readmission label is about: it was the whole
  +24% gain (still +24.6% with whole patients held out). With the patient and visit ids dropped the gain is
  +0.4%, which the table above uses. A count over a repeating id is the usual contest trick, but on
  data like this it reads the future.
- **Time-ordered tables** (bike sharing, Ames, House Prices, KC house, kick, SF crime, wind, NYC taxi):
  re-split with the latest 20% held out, the new version (identical to the old there) gains +0.2% on
  average and Featuretools +0.9%. The table above uses these numbers.
- **Runtime.** FeatureForge is the slowest of the three on small tables (median 124 s on the fresh set
  against 4 s for Featuretools); the cross-validated gate runs the search six times. Timings are on a shared
  4-core box running several jobs at once, so all of them are inflated.
- **Typical table.** The median gain is small: the mean comes from large wins on tables whose signal is in
  interactions (geometry, games, physics measurements).
- **Tried and dropped:** the cross-validated gate up to 10,000 rows with the search's rounds (CPU act
  +7.7% → +4.2%, churn +8.1% → +7.4%, four times slower); dropping a block early on one CV repeat (lost on
  eucalyptus and PC1).
