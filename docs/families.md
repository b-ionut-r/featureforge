# What switches on, and when

There are no per-contest flags. Each family below fires from the data's structure, and the search
families are kept only when they pass the gate.

| Data looks like | What is built | Where | Example |
|---|---|---|---|
| Any table | Arithmetic, counts, out-of-fold target encodings of keys and key pairs, group statistics, kNN target means, ridge GAM, PCA/PLS, key crosses, digits and rounding residues; a pair scan of strong or near-duplicate numeric pairs | `specs/`, `pairscan.py` | Loan Default's `f528 - f527`: AUC 0.735 → 0.998 |
| Hidden client id (ID-like columns plus a "days since opening" anchor) | Entity anchors, composite keys, a client-profile block of per-client means, spreads and distinct counts | `forge.py`, `specs/profile.py` | IEEE-CIS 0.9093 → 0.9293 |
| A family of similar columns (`var_0..var_199`) | Repeat counts and masks per column; synthetic-looking test rows left out of label-free counts | `specs/encoding.py` (`FamilyCount`), `screens.py` | Santander 0.8954 → 0.9176 |
| Numbered slots (`component_id_1..8`) | Bag counts and target maps learned across slots | `specs/encoding.py` (`SlotBag`, `SlotTE`) | Caterpillar |
| Free text | Writing statistics, TF-IDF SVD topics, out-of-fold sparse linear models on words and character n-grams | `specs/text.py` | Mercari RMSLE 0.571 → 0.470 |
| Structured strings (SMILES, protein/DNA, part numbers) | Character n-gram models and SVD; RDKit descriptors and fingerprints when installed | `specs/sequences.py` | Open Polymer wMAE 0.1345 → 0.0508 |
| Dates, or a time column with a low-cardinality event field | Calendar parts; time since or until an entity's last or next event | `specs/dates.py` | Rossmann promotions, holidays |
| Repeating many-level keys with a time column (users, devices) | Time since the previous row, until the next, rows before it; counts per hour or day | `specs/groups.py` (`EntityLag`) | TalkingData 0.9606 → 0.9694 |
| Ordered groups whose whole sequence is in the test (breaths, sessions) | In-group lags, leads, running sums, areas | `specs/groups.py` (`GroupSeq`) | Ventilator MAE 3.62 → 0.60 |
| Latitude/longitude pairs | Haversine distance, bearing, rotated coordinates | `specs/numeric.py` | West Nile |
| Integer codes (300+ thinly spread levels) | Used as keys for counts, encodings and crosses while staying numeric; "prefix + number" strings offered as numbers | `forge.py`, `specs/numeric.py` (`StrNum`) | Amazon access 0.868 → 0.912 |
| Child tables keyed by the main id | Aggregates, latest-state history, price paths, out-of-fold child-row models (kept by a CV check) | `blocks/relational.py`, `blocks/history.py`, `blocks/returns.py` | Home Credit 0.7470 → 0.7960 |
| Event logs with a repeating main key and a time column | As-of aggregates from strictly earlier events; outcome history when the log carries the target | `blocks/relational.py`, `blocks/history.py` | DSB 2019 QWK 0.378 → 0.559; Riiid 0.612 → 0.723 |
| Long-format child tables (a category plus a few columns) | One share column per frequent level, with numerics summed per level | `blocks/relational.py` | Telstra logloss 0.670 → 0.514 |
| Lookup tables (no shared column) | Attributes joined on the matching column or slots | `blocks/relational.py` | Caterpillar's components |
| JSON cells, list cells | Flattened fields; item counts and common-item flags; per-unit amounts | `blocks/lists.py` | Google Analytics, Two Sigma rentals |
| Intraday panel (entity × moment × day) | Rank within the moment, imbalances, own last steps of the day | `blocks/panel.py` | Optiver |
| Date + repeating series key + test after the training period | Horizon-aware lags, rolling windows, seasonal and event features; used alone when the held-out check keeps it | `blocks/forecast.py` | M5 1.862 → 0.852; Favorita 0.757 → 0.563 |
| Test holds new entities (patients, users) | The gate, search folds and encodings split whole groups | `forge.py` | — |
| Test starts after a gap | The gate starts after the same gap; each family must pass on its own | `forge.py` | IEEE-CIS |

## Tried and not adopted

These were neutral or worse on outer holdouts, so they are not in the package:

- row statistics as a default family;
- genetic and tree-guided interaction search, and OpenFE;
- autoencoder and MLP features, and naive-Bayes family maps;
- composite-key aggregations;
- a wider-prefix gate and a period-level gate;
- past-only, fold-averaged and lagged target encodings;
- the search on top of forecasting columns;
- a 300k search-row cap, and a cheaper search model;
- hour-of-day parts from compact stamps, and counting test rows in label-free counts.
