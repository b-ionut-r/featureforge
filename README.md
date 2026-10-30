# FeatureForge

Automated, leak-free feature engineering for tabular data, built to win paid tabular contests.

Give it a training table, the test table and any related tables. It writes feature files that any
model or AutoML can train on. It reads the data's structure to decide which feature blocks to build
(child tables, event logs, forecasting horizons, free text, lists, panels), then runs a
residual-guided search over thousands of candidate features. Nothing is kept unless it beats the raw
columns on held-out rows built like the real test file.

```bash
pip install -e .
featureforge --train train.csv --test test.csv --target y --out-dir features/
```

## Results

All results below use one fixed bagged LightGBM judge for every arm, with no tuning or stacking, so
only the features change. Leak checks were run on each contest: with shuffled training labels the
score lands at chance.

**Kaggle late submissions** (official private leaderboard; full train file in, Kaggle's own test file as the unlabeled rows):

| Contest (prize) | Metric | Raw columns | FeatureForge | Winners' private |
|---|---|---|---|---|
| IEEE-CIS Fraud ($20k) | AUC ↑ | 0.9093 | **0.9293** (winners' hand features, same judge: 0.9273) | ~0.946 |
| Home Credit Default Risk ($70k) | AUC ↑ | 0.7470 | **0.7960** | ~0.806 |
| Santander Transaction ($65k) | AUC ↑ | 0.8954 | **0.9176** | ~0.926 |
| Rossmann Store Sales ($35k) | RMSPE ↓ | 0.2393 | **0.1177** | ~0.100 |
| M5 Forecasting Accuracy ($50k) | WRMSSE ↓ | 1.862 | **0.852** | ~0.52 |
| Favorita Grocery Sales ($30k) | NWRMSLE ↓ | 0.757 | **0.5626** | ~0.51 |
| West Nile Virus ($40k) | AUC ↑ | 0.6938 | **0.7233** | n/a |
| Porto Seguro ($25k) | Gini ↑ | 0.284 | 0.284 (the gate kept nothing) | ~0.297 |

**Contests held back from all development and run once** (higher is better):
CMI Problematic Internet Use 0.4348 → **0.4470**, Student Game Play 0.6518 → **0.6828**,
OTTO sessions 0.7774 → **0.7881**, ISIC 2024 0.0430 → 0.0336, Jane Street 2024 0.0072 → 0.0065.

**Other unseen contests, raw → FeatureForge (winners' or public hand features through the same judge):**
Open Polymer wMAE 0.1345 → **0.0508** (0.0523), PLAsTiCC logloss 2.065 → **0.940**,
Instacart AUC 0.6915 → **0.8323**, Riiid AUC 0.612 → **0.723**, Microsoft Malware AUC 0.6760 → **0.6870** (0.6827),
Springleaf AUC 0.7641 → **0.7962**, Avazu AUC 0.7443 → 0.7551 (0.7570), TalkingData AUC 0.9606 → **0.9694**,
Elo RMSE 3.861 → 3.705 (3.688). Across suites of public tables: 22 OpenML tables +2.2%, 20 contest tables +3.8%.

**Against other automatic feature tools** (same judge, [docs/benchmark.md](docs/benchmark.md)): on 20 public
OpenML tables never used in development, mean held-out loss −17.0% against −6.5% for Featuretools, best
average rank; on 63 tables including the OpenFE and CAAFE papers' tables, −6.9% against −5.0% for
Featuretools and −4.3% before the dense blocks and cross-validated gate. With LLM-written domain proposals checked on top
of the search, 20 more unseen tables average −14.4% (search alone −7.3%, the LLM's features alone −11.4%); with one Claude API call per table, as the product makes it, 19 further unseen tables average −14.1% (search alone −11.9%).

West Nile's and Santander's best entries came from earlier versions of the code. The final version
scores 0.7162 and 0.9161 there; Santander's gap is within run-to-run noise. Runtimes, memory peaks and
the full tables are in [docs/results.md](docs/results.md).

## Install

Python 3.10+.

```bash
pip install -e .                       # the package and the `featureforge` command
pip install -r requirements-lock.txt   # the exact library versions every benchmark ran on
```

FeatureForge is deterministic on a given set of library versions: the same data gives byte-identical
features. Different LightGBM or scikit-learn versions can pick different features, so pin
`requirements-lock.txt` when you need results to match. `rdkit` is optional (`pip install -e .[chem]`);
with it installed, molecule strings (SMILES) get chemistry descriptors.

## Use

### Command line

```bash
featureforge \
    --train application_train.csv --test application_test.csv \
    --target TARGET --id SK_ID_CURR \
    --table bureau=bureau.csv --table prev=previous_application.csv \
    --table inst=installments_payments.csv \
    --out-dir features/
```

The output is `features/train_features.parquet`, which includes the target, and
`features/test_features.parquet`. Each `--table` takes the form `name=path[:key[:time]]`. The key
defaults to the main table's id column, or to the one column the two tables share. The time column
defaults to the first column named like a date or a day/month count. `featureforge --help` lists
every option. The defaults are the configuration every result above was built with.

### Python

The whole pipeline:

```python
from featureforge import build_features

train_path, test_path = build_features(
    "train.parquet", "test.parquet", target="y", id="row_id",
    tables={"history": ("history.parquet", "customer_id", "month")},
    budget=900, out_dir="features",
)
```

Domain knowledge can go in as a proposals file: Python functions written by a person or an LLM from the
column names and description, each kept only if it beats the search's output on cross-validation
(`featureforge --proposals proposals.py` on the command line):

```python
# proposals.py
def win_rate(df):          # takes the raw training then test rows, returns one value per row
    return df["wins"] / (df["wins"] + df["losses"]).replace(0, float("nan"))

PROPOSALS = [("win_rate", win_rate)]
```

```python
build_features("train.parquet", "test.parquet", target="y", proposals="proposals.py")
```

Or let Claude write the file: with `ANTHROPIC_API_KEY` set (and `pip install -e .[llm]`), `build_features`
asks Claude for proposals from the column names, types, a few rows and an optional `description=`, with no
target values. The file is saved as `features/proposals_llm.py`, reused on reruns, and judged by the same
check. It is skipped when most column names are placeholders (f_1, V12) and no description is given. `llm="off"`
(`--llm off`) disables it; without a key nothing changes.

The search alone, on one table:

```python
from featureforge import FeatureForge

forge = FeatureForge(time_budget=300).fit(X_train, y_train, X_unlabeled=X_test)
X_train_fe = forge.transform_train(X_train)   # target-dependent features out of fold
X_test_fe = forge.transform(X_test)
```

Always pass the test rows as `X_unlabeled`. Label-free statistics read them, and the gate is built to
resemble them: a time-ordered split when the test lies in the future, and whole groups held out when
the test holds new entities. Use `transform_train` on the training rows, never `transform`, because
only `transform_train` keeps target encodings out of fold.

Many rows per entity (plays per player, visits per patient) and a question about new entities: name
the entity column, `FeatureForge(group_col="player_id")` or `build_features(..., forge_kw={"group_col":
"player_id"})`. The gate, every CV fold, the screening split and out-of-fold encodings then hold out
whole entities, and no time column is looked for. Keep that column in the table: a column passed as
`id=` is dropped before the search, and a named `group_col` that is missing raises an error.

## How it works

```
train, test, child tables
  │
  ├─ related tables ── per-key aggregates · as-of aggregates of event logs · latest-state history
  │                    price paths · outcome history · same-code matches · lookups
  │                    out-of-fold child-row models (kept when a quick CV says they help)
  ├─ JSON fields · list columns · per-unit amounts · intraday panels
  ├─ forecasting block ── kept only if it beats raw on a held-out window shaped like the test;
  │                       when kept it replaces the search
  └─ FeatureForge search on main + top related columns
        rounds of: base model → candidates → residual screen → joint ranking → prefix ladder by CV
        gate: raw vs raw + new features on rows held out before the search, paired z ≥ 1
```

- **Every block switches on from the data's structure.** There are no per-contest flags to set.
  [docs/families.md](docs/families.md) lists what fires and when.
- **Leak rules.** Target-dependent features are out of fold on training rows. Columns built from the
  labels (target encodings, child-row model scores, outcome histories, label-echo columns found in the
  data) only feed same-row features, never cross-row ones such as counts, group statistics, neighbours
  or profiles. Forecasting features use only what is known one horizon before each row.
- **An honest gate.** Rows held out before any search compare the raw columns with raw plus the new
  features under the same model. A set is kept only when its paired per-row gain clears one standard
  error. When the test starts after a gap, each family must also pass on its own.
- **A work clock instead of a wall clock.** Budget decisions are charged in seconds of a 4-core
  reference machine, computed from row and column counts. A busy or fast machine therefore picks the
  same features.
- **Memory guards.** Every heavy stage is priced against the memory free when it starts. When it would
  not fit, the stage first switches to a cheaper mode with the same output, then reads a row sample,
  and only then is it skipped, with a log line. A run never gets killed partway.

[docs/architecture.md](docs/architecture.md) covers the search, the gate, the budget and the memory
model in detail.

## Repository layout

```
src/featureforge/
  forge.py            FeatureForge: the gated, residual-guided search
  pipeline.py         build_features: related tables → blocks → forecasting check → search
  proposals.py        outside domain features, kept only if they pass a check on top of the search
  llm.py              optional: Claude writes the proposals file from the schema (Anthropic API)
  cli.py              the `featureforge` command
  io.py               reading tables the same way on pandas 2 and 3
  screens.py          column families, slots, geo pairs, synthetic rows, gate statistics
  pairscan.py         wide scan of numeric pairs for differences and ratios
  _budget.py          work clock and memory model
  _frame.py           column codes, caches and the shared Context
  _lgb.py             objectives and losses
  specs/              candidate features (each a Spec)
    base.py           the Spec protocol
    numeric.py        arithmetic, row statistics, digits, geo, rotations, linear residuals
    encoding.py       counts, out-of-fold target encodings, family counts, slot bags
    neighbors.py      kNN target means and class distances, ridge GAM, PCA/PLS
    groups.py         group statistics, entity lags and gaps, in-group sequences
    dates.py          calendar parts, event recency
    text.py           writing statistics, TF-IDF SVD, sparse linear models, key crosses
    sequences.py      structured strings: SMILES, protein/DNA, part numbers
    profile.py        hidden-client profiles
  blocks/             structure-triggered blocks outside the search
    relational.py     child tables: aggregates, as-of, matches, lookups, child-row models
    history.py        latest-state and outcome history
    forecast.py       forecasting lags, windows, seasonality and events
    forecast_check.py the forecasting block's held-out check
    lists.py          JSON fields, list columns, per-unit amounts
    panel.py          intraday panels
    returns.py        price paths
  utils/              text views, memory accounting and key-range streaming, task inference
tests/                pytest suite (89 tests)
examples/             runnable quickstart
docs/                 architecture, families, results
```

## Tests

```bash
pip install -e .[dev]
pytest            # about 2 minutes on 4 cores
```

## Provenance

FeatureForge was developed inside TabularAML and benchmarked there. This repository is that work,
extracted from `b-ionut-r/TabularAML-refactored` at commit `89fc353` (PR #2), which holds every
adopted improvement. The extraction was checked by running the original and this package on the same
inputs. Both produced identical feature files on synthetic tables covering every block, and on West
Nile, Rossmann and Home Credit (see [docs/results.md](docs/results.md#parity-with-the-original)).
