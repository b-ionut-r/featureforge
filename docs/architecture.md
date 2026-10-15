# Architecture

FeatureForge has two layers:

1. **The pipeline** (`featureforge.pipeline.build_features`, the `featureforge` command). It turns a
   contest's files into feature files: related tables first, then the blocks that switch on from the
   data's structure, then a forecasting check, then the search.
2. **The search** (`featureforge.FeatureForge`). It runs on one table and keeps a feature set only when
   that set beats the raw columns on held-out rows.

## Pipeline

`build_features(train, test, target, tables=..., ...)` runs these stages in order. Each stage runs
only when the data has the structure it needs, and each is priced against free memory before it starts
(see [Memory guards](#memory-guards)).

1. **Read.** `io.read` reads the tables so that pandas 2 and pandas 3 give the same frames. Text becomes
   categories, mixed `0`/`"0"` cells become one level, list cells are joined, and float64 becomes
   float32. Multiclass labels are coded `0..K-1` for the search and written back as given.
2. **Related tables** (`blocks/relational.py`, `blocks/history.py`, `blocks/returns.py`).
   - A table with one row per key is joined as columns.
   - A keyed child table is aggregated per key (`RelatedTables`). Numerics get mean, min, max, sum,
     std and the mean of the last three rows, and categoricals get level shares. This covers
     grandchildren too.
   - When the main table repeats the key and both tables carry a comparable time column, the child is
     aggregated **as of each main row** (`asof_features`): only strictly earlier child rows count.
   - Latest state (last, last minus mean, last minus previous), price paths (realised volatility) and
     outcome history of event logs that carry the target.
   - Same-code matches (a customer's earlier purchases of this offer's brand) and lookup tables.
   - **Child-row models.** Each child row is labelled with its parent's target, LightGBM learns which
     child rows signal the outcome, and out-of-fold scores are aggregated per parent. They are kept only
     when a quick 3-fold CV on a sample of parents shows they add to the aggregates. On Home Credit
     they help; on Elo they cost an hour and hurt, so the check drops them.
   - A table too big to aggregate whole (Optiver's 160M-row book) is read from parquet one key range at
     a time. The output is the same as aggregating it whole.
3. **JSON, lists and per-unit amounts** (`blocks/lists.py`). JSON fields are flattened. List columns get
   item counts and common-item flags. Skewed amounts get ratios to the small counts they grow with.
4. **Intraday panels** (`blocks/panel.py`). Each entity is compared with all entities at the same
   moment, and with its own last steps of the day.
5. **Forecasting** (`blocks/forecast.py`, `blocks/forecast_check.py`). This switches on when there is a
   date column, an entity key whose rows repeat over time, and test rows after the training period.
   The block gives horizon-aware lags, rolling windows, seasonal and event features. It is checked on
   the last labelled periods, used as a stand-in for the test: the block is refitted on the earlier
   rows, and a quick model with it must beat one without it by a paired z of at least 1.0. When it
   passes, it **replaces** the search. On top of forecasting columns, the search's picks lost on held-out
   data (Favorita 0.669 → 0.689, Recruit 0.519 → 0.525).
6. **Search.** FeatureForge runs on the main columns plus the 200 related columns that a quick LightGBM
   ranks highest. Every related column is still written out. Columns built from labels (child-model
   scores, outcome histories) are passed as `label_cols` and only feed same-row features.
7. **Write.** The test features are written first, then the training features. Related columns are
   spilled to disk during the search and added back a column at a time.

## Search

`FeatureForge.fit(X, y, X_unlabeled)`:

1. **Setup.** The task is inferred. Columns are screened: integer codes, free text, dates,
   structured strings, column families (`px1..px784`), numbered slots, latitude/longitude pairs,
   label-echo columns, fields absent from the test, and synthetic test rows. Three things are detected
   here:
   - a **time column**: a numeric column whose test values lie beyond the training range;
   - a **group column**: a column whose test values are mostly new entities;
   - **hidden entities**: a timestamp minus a "days since X" column that is constant per client.
2. **Gate rows** are held out before any search, built like the test. They are the latest rows when the
   test is later in time, sit after a matching gap when the test starts after a gap, and are whole
   groups when the test holds new entities.
3. **Rounds**, until the budget runs out:
   1. *Base model.* K-fold LightGBM on the current set gives out-of-fold margins and split-gain
      importances.
   2. *Candidates.* Thousands of `Spec`s are built from the most important columns (see `specs/`).
      These are arithmetic, counts, out-of-fold target encodings of keys and pairs, group statistics,
      kNN target means, ridge GAMs, PCA/PLS, text models, key crosses, dates, digits, geo, entity gaps,
      in-group sequences and client profiles, plus a pair scan of near-duplicate numerics. Round 2 and
      later also compose the features selected so far. Cheap families are built first, and no family
      may take more than a third of the build.
   3. *Residual screen.* Each candidate alone trains a tiny LightGBM whose `init_score` is the base
      margin. Its validation gain therefore measures only information the model lacks. The screen runs
      cross-fitted, on several threads.
   4. *Selection.* Survivors are ranked by split gain in a joint model. The best prefix (3, 6, 12, …)
      is chosen by K-fold CV loss, with target encodings re-fitted inside each fold so the CV is not
      optimistic.
4. **Gate.** Raw columns versus raw plus the kept features, under the same model, on the gate rows. The
   set is kept when the winsorised paired per-row gain has z ≥ 1.0, raised to 1.645 under 1,000 gate
   rows. When the test starts after a gap, a failing set is retried with one family dropped at a time.
5. **Final fit.** Every kept feature is refitted on all labelled rows. `transform_train` returns
   out-of-fold values for target-dependent features. `transform` uses the full fit.

## Leak rules

- A target-dependent `Spec` (`target_dep = True`) returns out-of-fold values on training rows. A row's
  own label never enters its own feature.
- Columns derived from labels (target encodings, anything built on them, child-model scores, outcome
  histories, and label-echo columns such as running accuracies found in the data) feed only same-row
  features. A cross-row feature built on them would hand a row its own label through other rows.
- As-of aggregation uses strictly earlier rows. Forecasting features use only what is known one horizon
  before each row. Whole-history summaries of an as-of table use only rows before every main row.
- The check that was run on every contest: with shuffled training labels, the held-out score lands at
  chance (for example IEEE-CIS 0.4636, Riiid 0.514, Home Credit 0.4964).

## Budget: a work clock

`time_budget` is measured in seconds of a 4-core reference machine, **charged from row, column and
candidate counts, never from the wall clock** (`_budget.py`). An earlier timed version picked 12
features or none on the same data from run to run. The first round always gets enough CV fits for its
screen and prefix ladder, and later rounds start only while ten more fits fit in the budget. The base
fit, the gate and the final fits always run. On a 4-core machine, the wall time is roughly the budget
plus setup and the final fits.

## Memory guards

Every heavy stage estimates its working set and compares it with the memory free when it starts. That
is the machine's free memory, or the container's cgroup limit when lower (`utils/memory.py`). The
priced stages are related-table aggregation, history, price paths, child-row models, the
related-column ranking (including LightGBM's bin sample), each search round and every candidate, the
joint model, the gate, each final fit and the forecasting block. When a stage does not fit:

1. it switches to a cheaper mode with the same output (key ranges, one table at a time, a block of rows
   at a time);
2. failing that, it reads a row sample;
3. failing that, it is skipped, and a log line says so.

A run that fits never takes these paths, and its output is unchanged. The gate keeps any result no
worse than the raw columns.

## Determinism

The same data, on the same library versions, gives byte-identical features. Several choices make this
hold:

- the budget is counted in work, not time;
- BLAS runs on one thread inside the search;
- liblinear fits take turns behind a lock;
- LightGBM runs with `deterministic=True` and column-wise histograms;
- candidate caches are keyed and capped so that thread timing cannot change a pick.

Different LightGBM or scikit-learn versions can split differently, which is why
`requirements-lock.txt` exists.
