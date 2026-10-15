# Results

All results below use the same protocol:

- **Features only.** Every arm (raw columns, FeatureForge, hand-built features) is scored by one fixed
  bagged LightGBM judge: learning rate 0.03, 63 leaves, min_child_samples 50, feature_fraction 0.6,
  bagging 0.8, lambda_l2 1, early stopping 200. There is no tuning, stacking or ensembling, so a gain
  is a feature gain.
- **Held-out rows are built like the real test.** They are later in time when the test is later, after
  the same gap when the test starts after a gap, and whole groups when the test holds new entities.
  FeatureForge sees the held-out rows only as unlabeled rows, as it would see a contest's test file.
- **Leak control.** With shuffled training labels the held-out score lands at chance.
- **Runtime** is the feature build on a 4-core, 15 GB container, at the default 900 s search budget
  unless noted.

## Kaggle late submissions

FeatureForge was trained on the full train file, with Kaggle's own test file passed as the unlabeled
rows. Scores are the official leaderboard, public / private.

| Contest | Metric | Raw | FeatureForge | Build + judge (s) | Winners' private |
|---|---|---|---|---|---|
| IEEE-CIS Fraud | AUC ↑ | 0.9357 / 0.9093 | 0.9566 / **0.9293** | 1,881 + 1,066 | ~0.946 |
| Home Credit Default Risk | AUC ↑ | 0.7503 / 0.7470 | 0.8009 / **0.7960** | 2,892 + 1,754 | ~0.806 |
| Santander Transaction | AUC ↑ | 0.8979 / 0.8954 | 0.9205 / **0.9176** | 1,234 + 694 | ~0.926 |
| Rossmann Store Sales | RMSPE ↓ | 0.2081 / 0.2393 | 0.1048 / **0.1177** | 38 + 144 | ~0.100 |
| M5 Forecasting Accuracy | WRMSSE ↓ | private 1.862 | private **0.852** | 491 + 1,136 | ~0.52 |
| Favorita Grocery Sales | NWRMSLE ↓ | 0.756 / 0.757 | 0.532 / **0.5626** | 655 + 1,615 | ~0.51 |
| West Nile Virus | AUC ↑ | 0.7258 / 0.6938 | 0.7389 / **0.7233** | 45 + 7 | n/a |
| Porto Seguro | Gini ↑ | 0.279 / 0.284 | 0.279 / 0.284 (nothing kept) | 506 + 128 | ~0.297 |

Each row shows the best FeatureForge entry. Two of them came from earlier versions of the code. West
Nile's 0.7233 is one; the final version scores 0.7162. Santander's 0.9176 is the other; the final
version scores 0.9161, and reruns of the same code range 0.9172-0.9185, which is within noise. Runtimes
are from the recorded entry builds.

On IEEE-CIS, the winners' public hand-built features scored 0.9273 private through the same judge, so
FeatureForge's own features beat them. Two of West Nile's 21 features are label-free row counts, and
they partly reflect a duplicate-row artifact in that contest's files.

## Contests sealed until the final version

These were held back from all development and run once, at seed 0, under the normal memory limit.

| Contest | Metric | Raw | FeatureForge | Build | Peak memory |
|---|---|---|---|---|---|
| Child Mind Institute, Problematic Internet Use | QWK ↑ | 0.4348 | **0.4470** | 82 s | 0.37 GB |
| Student Game Play | macro F1 ↑ | 0.6518 | **0.6828** | 343 s | 3.18 GB |
| OTTO sessions | AUC ↑ | 0.7774 | **0.7881** | 1,143 s | 2.16 GB |
| ISIC 2024, metadata only | pAUC above 80% TPR ↑ | 0.0430 | 0.0336 | 678 s | 2.27 GB |
| Jane Street 2024 | weighted R² ↑ | 0.00721 | 0.00651 | 572 s | 5.56 GB |

## Held-out checks on contests used during development

| Contest | Metric | Raw | FeatureForge | Reference | Build |
|---|---|---|---|---|---|
| IEEE-CIS, latest 20% | AUC ↑ | 0.924 | **0.9541** | winners' hand 0.9423 | 27 min |
| Loan Default (Imperial) | AUC ↑ | 0.735 | **0.9986** | | 30 min |
| Mercari, two holdouts | RMSLE ↓ | 0.571 / 0.575 | **0.470 / 0.473** | | 10 min |
| DSB 2019, as-of event logs | QWK ↑ | 0.378 | **0.559** | | 5 min |
| M5 / Favorita, two windows | RMSE ↓ | 0.782 / 0.754; 0.909 / 0.870 | **0.751 / 0.712; 0.679 / 0.669** | | 1-10 min |
| Amex Default, history (blind) | metric ↑ | 0.784 / 0.773 | **0.794 / 0.783** | | 5-7 min |
| Riiid, outcome history (blind) | AUC ↑ | 0.612 / 0.618 | **0.766 / 0.768** | winners ~0.82 | 27 min |
| Ventilator Pressure (blind) | MAE ↓ | 3.623 | **0.597** | winners' hand 0.556 | 40 min |
| Amazon Employee Access, two seeds (blind) | AUC ↑ | 0.8684 / 0.8641 | **0.9116 / 0.8961** | winners' crosses 0.9147 / 0.8995 | 1 min |
| Telstra, two seeds (blind) | logloss ↓ | 0.670 | **0.514 / 0.522** | hand 0.497 / 0.498 | |
| BNP Paribas, two seeds (blind) | logloss ↓ | 0.4632 / 0.4656 | **0.4502 / 0.4436** | public hand 0.4560 / 0.4582 | |
| Two Sigma rentals, two seeds (blind) | logloss ↓ | 0.577 / 0.588 | **0.545 / 0.545** | winners' hand 0.543 / 0.540 | 8-11 min |
| Allstate Claims | MAE ↓ | 1151 | **1143** | | 8 min |
| PetFinder, two holdouts | QWK ↑ | 0.392 / 0.344 | **0.413 / 0.359** | | 2 min |

## Unseen contests

| Contest | Metric | Raw | FeatureForge | Hand-built, same judge |
|---|---|---|---|---|
| Open Polymer Prediction 2025 | wMAE ↓ | 0.1345 | **0.0508** | 0.0523 (RDKit + Morgan) |
| PLAsTiCC 2018 | logloss ↓ | 2.065 | **0.940** | 1.369 |
| Instacart, 131,209 users | AUC ↑ | 0.6915 | **0.8323** | |
| H&M | metric ↑ | 0.791 | **0.870** | |
| KKBox churn | logloss ↓ | 0.280 | **0.204** | 0.203 |
| TalkingData | AUC ↑ | 0.9606 | **0.9694** | |
| Avazu CTR, last day | AUC ↑ | 0.7443 | **0.7551** | 0.7570 |
| Microsoft Malware | AUC ↑ | 0.6760 | **0.6870** | 0.6827 |
| Expedia hotels, 3% | MAP@5 ↑ | 0.2290 | **0.2490** | 0.2334 |
| Held-back wide 2015 contest (1,930 columns) | AUC ↑ | 0.7737 | **0.7953** | 0.7732 |
| Springleaf (1,935 columns) | AUC ↑ | 0.7641 | **0.7962** | |
| Elo | RMSE ↓ | 3.861 | **3.705** | 3.688 |
| BNP Paribas | logloss ↓ | 0.46052 | **0.45192** | |
| Google Analytics Revenue | RMSE ↓ | 0.3643 | **0.3180** | 0.3165 |
| Linking Writing Processes, 3 seeds | RMSE ↓ | 0.711 | **0.653** | |
| MCTS Variants | RMSE ↓ | 0.4939 | **0.4348** | |

FeatureForge also gained on GA, JPX, Avito, Bosch, Allstate and Homesite. It tied raw on Mercedes,
Bioresponse, DRW, ICR, CIBMTR, Sberbank, Hull and ASHRAE, where its gate kept the raw columns or nothing
moved beyond noise. On suites of public tables it averaged +2.2% over 22 OpenML tables and +3.8% over 20
contest tables.

**Time series** (lower is better): Stormy last-28 0.457 → **0.365**, Stormy events 0.599 → **0.532**,
PFS 1.003 → **0.917**, Rohlik 20.47 → **17.20**, Store Sales 0.481 → **0.413**. On GEFCom the
forecasting check rejected the block, so the result equals raw.

## Known limits

- **ISIC 2024 and Jane Street scored below raw**, from one sealed run each.
  - On ISIC, a high-cardinality recode of `patient_id` passed the gate on log loss but cut the
    contest's partial AUC. The gate judges log loss, not the contest metric.
  - On Jane Street, the panel family's 72 columns are not gated.
- **Very tight memory.** At 70% of a table's own peak memory, some runs still get killed (Home Credit at
  6.0 GB, Expedia with 100 classes, MCTS) or lose their lift (Store Sales, Avazu). At the normal limit
  no run crashed.
- **Very wide tables.** Springleaf's 1,935 columns spend about 625 s in setup, which is not charged to
  the budget. The full build took 3,001 s on a 900 s budget, at a 13.4 GB peak.
- **Library versions.** pandas 3 builds are 7-28% slower than pandas 2.3.3, with identical output.
  Different LightGBM or scikit-learn versions can pick different features.

## Parity with the original

This package was extracted from `b-ionut-r/TabularAML-refactored` at `89fc353`, the head of PR #2. The
original `scripts/contest_features.py` and this package's `featureforge` command were run on the same
inputs, one after the other in one container, with pandas 3.0.5 and the library versions in
`requirements-lock.txt`. The resulting feature files were then compared.

| Inputs | Blocks exercised | Train / test features | Result |
|---|---|---|---|
| Synthetic binary, 8,000 rows | text models, dates, high-cardinality recode, search + gate | 8,000 × 28 / 3,000 × 27 | byte-identical |
| Synthetic regression + 2 child tables | aggregates, latest-state history, child-row models, search | 5,000 × 54 / 2,000 × 53 | byte-identical |
| Synthetic multiclass | JSON fields, list columns, multiclass search | 6,000 × 19 / 2,000 × 18 | byte-identical |
| Synthetic store × item daily sales | forecasting block + held-out check | 16,272 × 48 / 1,008 × 47 | byte-identical |
| West Nile Virus (Kaggle files) | search + gate (21 kept, gate +7.21%) | 10,506 × 53 / 116,293 × 52 | byte-identical |
| Rossmann (Kaggle files, 1.02M rows) | forecasting block (63 columns, check z +140) | 1,017,209 × 81 / 41,088 × 80 | byte-identical |

The ported test suite passes: 77 tests, 1 skipped. The four TabularAML solver tests were left out with the solver.
