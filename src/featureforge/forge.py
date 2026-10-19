"""FeatureForge: fast, wide, residual-guided feature search with an honest gate.

Search loop (per round):

1. **Base model.** K-fold LightGBM on the current feature set gives out-of-fold
   raw margins and split-gain importances.
2. **Candidates.** Thousands of features from the most important columns:
   pairwise arithmetic (add/sub/mul/div), frequency encodings, out-of-fold
   target encodings of single keys and key pairs, and group statistics
   (mean/std/min/max/deviation/z-score of a numeric within a categorical or
   low-cardinality key). Round 2+ also combine the features selected so far
   with raw columns, so useful second-order features (e.g. ``a*b - c``) can
   emerge.
3. **Residual screening.** Each candidate alone trains a tiny LightGBM whose
   ``init_score`` is the base margin, so its validation gain measures only
   the information the current model lacks. Successive halving over row
   subsets keeps this to milliseconds per candidate; candidates run in
   parallel threads.
4. **Selection.** Survivors are ranked by split gain in a joint model, then the
   best prefix (3, 6, 12, ... features) is chosen by K-fold CV loss.

Finally a **gate**: rows held out before any search compare raw vs. raw+new
features with the same model. If the new features do not lower held-out
loss, nothing is added. This keeps the search honest: a CV gain that only
exists on the folds it was selected on is rejected.

Typical use::

    forge = FeatureForge(time_budget=300).fit(X_train, y_train)
    X_train_fe = forge.transform_train(X_train)   # OOF target encodings
    X_test_fe = forge.transform(X_test)
"""
from __future__ import annotations

import os
import threading
import time
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from featureforge._budget import (_BASE_RSS, _CLASS_CELL_S, _CV_CELL_S, _KEY_ROW_S, _PAIR_CELL_S, _REF_THREADS,
                                  _ROW_PASS_S, _SCAN_SHARE, _SETUP_COL_S, _SPEC_BYTES_PER_ROW, _SPEC_WORK,
                                  _SPEC_WORK_DEFAULT, _TREE_CELL_S, _WU_ROW_SHARE, _spec_bytes, _tree_col_s)
from featureforge._frame import (ARITH_OPS, Context, GROUP_STATS, ROW_TIME, X_nunique, _CODE_CACHE, _COL_CACHE,
                                 _GROUP_CACHE, _as_str, _as_str_frame_col, _columns, _concat_cats, _is_cat,
                                 _key_values, _permute_columns, _renumbered, _unique_names, _warm)
from featureforge._lgb import _fold_steps, _lgb_objective, _loss, _row_loss, _stepped_class_loss, _stepped_loss
from featureforge.screens import (_absent_in_test, _block_z, _later_half, _numbered_codes, _winsor, column_families,
                                  geo_points, slot_families, synthetic_rows)
from featureforge.specs.base import Spec, _release
from featureforge.specs.dates import DatePart, EventRecency, _DATE_PARTS, _is_date_like, _to_days
from featureforge.specs.encoding import (BinnedPairTE, Count, FamilyCount, KeyBinTE, MixedCount, MixedTE, SlotBag,
                                         SlotTE, TargetEnc)
from featureforge.specs.groups import EntityLag, GroupSeq, GroupStat
from featureforge.specs.neighbors import KNNClassDist, KNNTarget, LinearOOF, Projection, _one_blas_thread
from featureforge.specs.numeric import Arith, Digits, GeoPair, LinResid, Rotate, RowStat, StrNum
from featureforge.specs.text import CrossLinearOOF, TextLinearOOF, TextSVD, TextStats, _is_text, _text_nnz
from featureforge.utils.memory import _trim_heap, available_bytes
from featureforge.utils.text import as_text


# Features computed from a row's own values only (no other rows enter them).
_SAME_ROW = (Arith, RowStat, Digits, Rotate, DatePart, TextStats, GeoPair)


class FeatureForge:
    """Residual-guided automated feature engineering.

    Parameters
    ----------
    task : "regression" | "binary" | "multiclass" | None
    log_target : bool
        Search on ``log1p(y)`` (use for RMSLE-scored regression).
    gate_families : bool | "auto"
        When the full set fails the gate, drop feature families one at a time and keep the
        best remaining set that clears the gate by one more standard error. "auto" (default):
        only when the test starts after a gap from the training rows, where the gate also
        starts the same time after the search rows; there a family that leans on the latest
        labelled rows (neighbours' labels, cross-linear key models on IEEE-CIS) fails while the
        client encodings and profiles still carry over.
    label_cols : sequence of str
        Columns of ``X`` computed from the labels before ``fit`` (out-of-fold model scores of
        child rows, outcome histories of an event log): held like label-echo columns, in
        same-row features only.
    label_echo : bool
        Find columns that carry earlier rows' labels (running means or counts of past
        answers in an event log: within an entity, or a pair of entities such as student and
        question, the next row's change tracks this row's label; or lagged answers: a row a
        few rows later holds this row's label) and keep them, and features built on them, out of every cross-row feature
        (group statistics, counts, encodings, neighbours, anchors). Such statistics would
        hand each row its own label through the entity's later rows.
    family_time_share : float
        Largest share of the candidate-building work one family may take (1 disables).
    time_budget : float
        Search budget in seconds of work on the reference container (the work clock: costs
        come from row, column and candidate counts, never from timing, so a run picks the same
        features on any machine). The first round always gets sixteen CV fits, eight of them
        kept for screening and the prefix ladder; later rounds start only while ten fit in the
        budget. Cheap candidate families are built first. The base fit, the gate and the final
        fits always run.
    n_rounds : int
        Search rounds; round r>1 composes previously selected features.
    max_new_features : int
        Cap on features added in total.
    nested_cv : bool | "auto"
        Recompute target encodings inside each CV fold when ranking feature sets,
        so search CV is not optimistic about sparse-key encodings. "auto": on with
        time-blocked search, where encodings fitted on random folds carry the
        validation block's labels (IEEE-CIS latest window 0.948 -> 0.953).
    entities : bool
        Look for hidden entity ids: a timestamp-like column minus a "days since
        X" column is constant per entity (account opening date, first visit);
        such anchors are added as columns and combined with ID-like columns
        into composite keys for counts, target maps and group statistics.
    entity_nums : int
        Numerics aggregated per entity.
    event_gaps : "auto" | "past" | "off"
        With a time column and a repeating many-level key: time since each user's previous
        row, until their next and rows before it, users being the key with the row's other
        codes (TalkingData's ip + app + device + os). "past" leaves out the next row.
    group_col : str | "auto" | None
        Column whose test values are new groups (users, patients). "auto": a
        repeating integer / categorical column with under 20% of the unlabeled
        rows' values seen in training. The gate, search folds, screening split
        and out-of-fold encodings then split whole groups, and target maps over
        that column are not generated.
    time_cv : bool
        With a time column, search CV uses contiguous time blocks as folds and
        screening is measured on the latest rows, so features that only
        interpolate within a period are not selected.
    events : bool
        With a time column: time since / until each entity's last / next row
        with a given level of a low-cardinality column (days since the last
        promotion, until the next holiday). Date columns (datetime or ISO date
        strings) are always expanded into calendar fields as candidates.
    text : bool
        Free-text columns (strings of mostly distinct values, several words each:
        titles, descriptions) get label-free writing statistics and TF-IDF SVD
        topics, and an out-of-fold sparse linear model on their words and character
        n-grams (alone, and all text plus one-hot keys) as candidates.
    pair_scan : bool
        Score differences and ratios of every pair among the 60 strongest numerics and of
        every closely related pair (rank correlation >= 0.9) on a 20k-row sample (5% of the budget), and offer
        the best dozen as candidates (Loan Default's f528 - f527: AUC 0.735 -> 0.998).
    profiles : bool
        With a hidden client id (ID-like columns plus an anchor), one candidate block per
        client key: per-client mean / std of up to 120 numerics and of every re-based
        "days since" column, distinct counts of categoricals, and row counts, label-free
        over all known rows (``profile.ClientProfile``).
    time_col : str | "auto" | None
        Column that orders rows in time. When set, the gate holds out the most
        recent rows instead of a random sample, so features that only work
        within a period (neighbours of the same day, encodings of entities that
        stop appearing) are judged as they will be on a later test set. "auto"
        picks a numeric column whose unlabeled values (``X_unlabeled``) lie
        beyond the training range, as a competition's test period does.
    n_interactions : int
        Number of feature pairs (and half as many triples) mined from the base
        model's tree paths and expanded into explicit interaction features.
    top_numeric, top_keys : int
        How many of the most important numeric / key columns seed candidates.
    gate_frac : float
        Fraction of rows held out (before search) to confirm the final gain.
        0 disables the gate.
    gate_z : float
        Required paired z-score of the (winsorised) per-row gate improvement
        (1.0 ~ 84% one-sided confidence; raised to 1.645 when the gate has fewer
        than 1000 rows). Protects small tables from noise-driven adds.
    """

    def __init__(self, task: Optional[str] = None, log_target: bool = False,
                 time_budget: float = 300, n_rounds: int = 3, max_new_features: int = 60,
                 top_numeric: int = 24, top_keys: int = 10, max_key_cardinality: int = 200,
                 arith_ops: Sequence[str] = ARITH_OPS, group_stats: Sequence[str] = GROUP_STATS,
                 gate_frac: float = 0.2, gate_z: float = 1.0, min_rel_gain: float = 0.001, cv: int = 3,
                 hc_threshold: int = 32, n_interactions: int = 40, novelty_slack: float = 1.0,
                 nested_cv="auto", entities: bool = True,
                 time_col: Optional[str] = "auto", entity_nums: int = 6,
                 events: bool = True, time_cv: bool = True,
                 group_col: Optional[str] = "auto", text: bool = True, drop_synthetic: bool = True, gate_families="auto",
                 family_time_share: float = 1 / 3, label_echo: bool = True, label_cols: Sequence[str] = (),
                 pair_scan: bool = True, profiles: bool = True, event_gaps: str = "auto", random_state: int = 0, n_jobs: int = -1, verbose: bool = True):
        self.task = task
        self.log_target = log_target
        self.time_budget = time_budget
        self.n_rounds = n_rounds
        self.block_numeric = 20
        self.cv_gate = True
        self.cv_gate_rows = 3000
        self.max_new_features = max_new_features
        self.top_numeric = top_numeric
        self.top_keys = top_keys
        self.max_key_cardinality = max_key_cardinality
        self.arith_ops = tuple(arith_ops)
        self.group_stats = tuple(group_stats)
        self.gate_frac = gate_frac
        self.gate_z = gate_z
        self.min_rel_gain = min_rel_gain
        self.cv = cv
        self.hc_threshold = hc_threshold
        self.n_interactions = n_interactions
        self.novelty_slack = novelty_slack
        self.nested_cv = nested_cv
        self.entities = entities
        self.time_col = time_col
        self.entity_nums = entity_nums
        self.event_gaps = event_gaps
        self.events = events
        self.time_cv = time_cv
        self.group_col = group_col
        self.text = text
        self.drop_synthetic = drop_synthetic
        self.family_time_share = family_time_share
        self.label_echo = label_echo
        self.label_cols = label_cols
        self.gate_families = gate_families
        self.pair_scan = pair_scan
        self.profiles = profiles
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------- utilities
    def _log(self, msg):
        if self.verbose:
            print(f"[FeatureForge {time.time() - self._t0:6.1f}s] {msg}", flush=True)

    def _time_left(self) -> float:
        """Budget left on the work clock (reference-container seconds; see ``_CV_CELL_S``)."""
        deadline = getattr(self, "_deadline", None) or self.time_budget
        return deadline - getattr(self, "_work", 0.0)

    def _charge(self, seconds: float):
        self._work = getattr(self, "_work", 0.0) + seconds

    @staticmethod
    def _cv_cost(n_rows: int, n_cols: int, n_fits: int) -> float:
        return _CV_CELL_S / _TREE_CELL_S * _tree_col_s(n_rows) * max(n_cols, 1) * n_fits / 5

    def _threads(self) -> int:
        import os
        if getattr(self, "_cells", 1e9) <= 200_000:
            # A small table's fits take milliseconds; several threads only wait on each other, and on a busy
            # machine they waited seconds (Balance Scale, 500 rows: 57 s on four threads, 6 s on one).
            return 1
        return os.cpu_count() if self.n_jobs in (-1, None) else max(1, self.n_jobs)

    def _folds(self, n_rows, y, k, seed):
        g = getattr(self, "_groups", {}).get(n_rows)
        if g is not None:
            # Whole groups (users) per fold: test rows belong to unseen groups.
            levels, codes = np.unique(g, return_inverse=True)
            fold_of = np.random.default_rng(seed).permutation(len(levels)) % k
            f = fold_of[codes]
            return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]
        if self.task_ == "regression":
            return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n_rows)))
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n_rows), y))

    def _te_folds(self, X: pd.DataFrame, y, seed):
        """Folds for out-of-fold target statistics: five folds, split as ``_folds`` splits them."""
        return self._folds(len(X), y, 5, seed)

    def _model_frame(self, X: pd.DataFrame, recode: Optional[bool] = None) -> pd.DataFrame:
        """Model view of a frame: categoricals as ``category`` dtype, or, when the
        high-cardinality recode is active, those columns as frequency-rank numbers."""
        recode = self.recode_ if recode is None else recode
        X = X.copy(deep=False)  # columns are replaced, never written into
        for c in self.cat_cols_:
            if c not in X.columns:
                continue
            if recode and c in self.rank_maps_:
                X[c] = self._rank(X[c], c)
            else:
                X[c] = pd.Categorical(_as_str(X[c]), categories=self.cat_levels_[c])
        return X

    def _fit_rank_maps(self, *frames: pd.DataFrame):
        """Frequency rank of each level of the high-cardinality categoricals (1 = most common),
        counted over the rows of all ``frames`` (counted per frame, so categoricals are never
        stacked into per-row strings)."""
        self.rank_maps_ = {}
        for c in self.hc_cols_:
            vc = None
            for F in frames:
                col = F[c] if isinstance(F[c].dtype, pd.CategoricalDtype) else F[c].astype(object)
                v = col.value_counts(dropna=False)
                v.index = np.where(pd.isna(v.index.to_numpy(dtype=object)), "__NA__", v.index.astype(object).astype(str))
                v = v.groupby(level=0).sum()
                vc = v if vc is None else vc.add(v, fill_value=0)
            vc = vc[vc > 0]
            # Ranked by frequency, ties by value; levels seen at most twice share one rank per count:
            # ordering them by value spread the many once-seen device IPs over a meaningless range
            # a tree then splits on.
            t = pd.DataFrame({"v": vc.index.to_numpy(dtype=object), "n": vc.to_numpy()}).sort_values(
                ["n", "v"], ascending=[False, True], kind="mergesort")
            r = pd.Series(np.arange(1, len(t) + 1, dtype=float), index=t["v"].to_numpy())
            n_ = t["n"].to_numpy()
            rare = n_ <= 2
            if rare.any():
                r[rare] = r[rare].groupby(n_[rare]).transform("min").to_numpy()
            _warm(r.index)
            self.rank_maps_[c] = r

    def _rank(self, s: pd.Series, c: str) -> np.ndarray:
        return self.rank_maps_[c].reindex(_as_str(s).to_numpy()).to_numpy(dtype=float)

    def _lgb_params(self, lr=0.1, threads=None, **kw):
        p = dict(_lgb_objective(self.task_, self.n_classes_), learning_rate=lr, num_leaves=31,
                 min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=1.0, verbosity=-1, seed=self.random_state, data_random_seed=self.random_state,
                 num_threads=threads or self._threads(), max_cat_to_onehot=8,
                 # LightGBM otherwise picks row- or column-wise histograms by timing both, and
                 # the two sum in a different order: on Airline the same commit's round-one
                 # ranking split on that and kept 41 features or none on alternate runs.
                 deterministic=True, force_col_wise=True)
        # LightGBM bins from a sample of up to 200k rows held as doubles, about 13 bytes per
        # sampled cell: 7-11 GB on Bosch's 4,264 columns. Wide frames bin from fewer rows;
        # frames under ~1,000 columns keep the default.
        width = getattr(self, "model_width_", 0)
        if width and 2.5e9 / (13 * width) < 200_000:
            p["bin_construct_sample_cnt"] = max(20_000, int(2.5e9 / (13 * width)))
        p.update(kw)
        return p

    @staticmethod
    def _sample_cap(p, n_rows, width):
        """Dataset parameters for a matrix of ``width`` columns. The rule above sizes the bin
        sample from the widest frame expected (raw columns plus one per new feature), but a
        family block adds a column per member: two blocks over a 1,879-column family made the
        joint and gate matrices ~5,700 columns, and binning them from 92k rows took ~8 GB on top
        of everything else. The sample is cut to ~4 GB of sampled cells for the actual width;
        where that is no fewer rows than LightGBM would sample anyway, nothing changes."""
        cap = max(20_000, int(4e9 / (13 * max(width, 1))))
        if cap < min(p.get("bin_construct_sample_cnt", 200_000), n_rows):
            p = dict(p, bin_construct_sample_cnt=cap)
        return p

    def _train_params(self, D, lr=0.1, **kw):
        """Training parameters for a prebuilt dataset: its bin sample size as it was built."""
        p = self._lgb_params(lr, **kw)
        if "bin_construct_sample_cnt" in (D.params or {}):
            p["bin_construct_sample_cnt"] = D.params["bin_construct_sample_cnt"]
        return p

    def _fit_eval(self, Xtr, ytr, Xva, yva, lr=0.1, rounds=2000, es=50):
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        b = lgb.train(self._lgb_params(lr), dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        return b

    @staticmethod
    def _lgb_matrix(X: pd.DataFrame):
        """Float32 matrix of a model frame, categoricals as their codes (missing -> NaN, as
        LightGBM's pandas path encodes them), and the categorical column positions."""
        M = np.empty((len(X), X.shape[1]), dtype=np.float32)
        cats = []
        for j, c in enumerate(X.columns):
            s = X[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                cats.append(j)
                k = s.cat.codes.to_numpy()
                M[:, j] = np.where(k < 0, np.nan, k)
            else:
                M[:, j] = s.to_numpy(dtype=np.float32, na_value=np.nan)
        return M, cats

    def _lgb_data(self, X: pd.DataFrame, y, lr=0.1):
        """One binned LightGBM dataset for a whole frame; folds train on its row subsets.
        Binning once per frame instead of once per fold, from a float32 matrix instead of
        float64 copies of each fold, is most of the search's memory and a fifth of its time.
        The raw matrix is not kept: kept, every row subset took its own copy of its rows."""
        import lightgbm as lgb
        M, cats = self._lgb_matrix(X)
        return lgb.Dataset(M, np.asarray(y), categorical_feature=cats,
                           params=self._sample_cap(self._lgb_params(lr), *M.shape)).construct()

    def _lgb_fit_bytes(self, n_rows, width):
        """A LightGBM fit's memory beyond its matrix: the sample it bins from (about 13 bytes a sampled cell, see
        _lgb_params), the binned rows and the row subsets' copies of them (a byte a value each), the histogram
        pool (255 bins of gradient, hessian and count for each column and each of 31 leaves) and, a class each,
        float32 gradients and hessians, float64 scores and the validation rows' predictions."""
        p = self._sample_cap(self._lgb_params(), n_rows, width)
        sample = min(p.get("bin_construct_sample_cnt", 200_000), n_rows)
        return 13 * sample * width + 2 * n_rows * width + 24 * 255 * 31 * width + 24 * n_rows * max(self.n_classes_, 1)

    def _fit_eval_rows(self, D, tr, va, lr=0.1, rounds=2000, es=50):
        """``_fit_eval`` on row subsets of a dataset built by ``_lgb_data``."""
        import lightgbm as lgb
        dtr, dva = D.subset(np.sort(tr)), D.subset(np.sort(va))
        return lgb.train(self._train_params(D, lr), dtr, rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(es, verbose=False)])

    @staticmethod
    def _mine_paths(booster, names, pairs, triples, max_trees=300):
        """Accumulate split gain of parent->child (and grandparent chains) feature pairs.

        Features that a tree splits on in sequence along one path interact in
        the model; their gain is a cheap, model-guided interaction ranking.
        """
        dump = booster.dump_model(num_iteration=booster.best_iteration or None)
        for tree in dump["tree_info"][:max_trees]:
            stack = [(tree["tree_structure"], None, None)]
            while stack:
                node, par, gpar = stack.pop()
                if "split_feature" not in node:
                    continue
                f = names[node["split_feature"]]
                g = float(node.get("split_gain", 0.0))
                if par is not None and par != f:
                    key = tuple(sorted((par, f)))
                    pairs[key] = pairs.get(key, 0.0) + g
                    if gpar is not None and len({gpar, par, f}) == 3:
                        k3 = tuple(sorted((gpar, par, f)))
                        triples[k3] = triples.get(k3, 0.0) + g
                for side in ("left_child", "right_child"):
                    stack.append((node[side], f, par))

    def _nested_values(self, spec, W, y, key, tr, va):
        """Encodings of one CV fold computed from its training rows only.

        Out-of-fold target statistics over the whole search frame still let a
        training row's value depend on the validation rows' labels; for sparse
        keys that makes CV favour target encodings that do not generalise.
        Here the training rows get an inner out-of-fold encoding and the
        validation rows an encoding fitted on the training rows alone.
        """
        ck = (spec.name,) + key
        if ck not in self._nested_cache:
            import copy
            s = copy.copy(spec)
            inner = self._te_folds(W.iloc[tr], y[tr], self.random_state + 7)
            vt = s.fit_transform_oof(W.iloc[tr].reset_index(drop=True), y[tr], self.ctx_, inner)
            vv = s.transform(W.iloc[va].reset_index(drop=True), self.ctx_)
            self._nested_cache[ck] = (np.asarray(vt, dtype=np.float32), np.asarray(vv, dtype=np.float32))
        return self._nested_cache[ck]

    def _cv(self, X: Optional[pd.DataFrame], y, repeats, lr=0.1, mine=False, nested=None, mat=None, give_up=None):
        """Repeated K-fold: OOF raw margins (averaged over repeats), mean loss, summed gain importance.

        ``nested`` maps target-encoding specs to the frame they were built on;
        their columns are then recomputed per fold from the fold's training rows.
        ``mat``: the frame as ``(float32 matrix, categorical positions, column names)``,
        used instead of ``X`` (then None); an optional fourth item picks the matrix's
        columns, so no full copy of the selected columns is kept.
        ``give_up`` (work-clock seconds): when the fit's cost (``_cv_cost``) is larger, return None
        without fitting, keeping the cost in ``_cv_projected``; likewise, after any fold, when the
        folds fitted so far, per fold, price the whole fit above it.
        """
        import lightgbm as lgb
        M, cats, cols, *pick = mat if mat is not None else (*self._lgb_matrix(X), list(X.columns))
        pick = pick[0] if pick else None

        def rows(r):
            return M[r] if pick is None else M[np.ix_(r, pick)]
        n = len(M)
        n_fits = sum(len(f) for f in repeats)
        width = M.shape[1] if pick is None else len(pick)
        prior = self._cv_cost(n, width, n_fits)
        if give_up and prior > give_up:
            self._cv_projected = prior
            return None
        # Each fold is charged from the iterations it trained (early stopping is deterministic):
        # a fit of thousands of trees costs that, not the row-by-column prior.
        self._last_cv_cost = 0.0
        if mine:
            self.pair_gain_, self.triple_gain_ = {}, {}
        oof_mean = np.zeros((n, self.n_classes_)) if self.task_ == "multiclass" else np.zeros(n)
        imp = pd.Series(0.0, index=cols)
        losses = []
        pos = {c: j for j, c in enumerate(cols)}
        specs = [sp for sp in (nested or {}).get("specs", []) if sp.out_names()[0] in pos]
        cat_cols = {cols[i] for i in cats}
        derived = [sp for sp in (nested or {}).get("derived", []) if sp.out_names()[0] in pos
                   and all(p in pos and p not in cat_cols for p in sp.parents)]
        D = None
        if not specs and not derived:
            # One binned dataset; folds train on its row subsets. The raw matrix is not kept:
            # validation rows are read from ``M`` again. A run of a column-major matrix's columns is
            # passed as it is (LightGBM reads it in place; same bins, same model), not copied.
            run = (pick is not None and M.flags.f_contiguous and len(pick) > 0
                   and list(pick) == list(range(pick[0], pick[0] + len(pick))))
            Msel = M if pick is None else M[:, pick[0]:pick[0] + len(pick)] if run else M[:, pick]
            D = lgb.Dataset(Msel, np.asarray(y), categorical_feature=cats,
                            params=self._sample_cap(self._lgb_params(lr), *Msel.shape)).construct()
            Msel = None
        for r_i, folds in enumerate(repeats):
            oof = np.zeros_like(oof_mean)
            for f_i, (tr, va) in enumerate(folds):
                if D is not None:
                    b = self._fit_eval_rows(D, tr, va, lr=lr)
                    Mva = rows(va)
                else:
                    Mtr, Mva = rows(tr), rows(va)
                    for sp in specs:
                        vt, vv = self._nested_values(sp, nested["W"], y, (r_i, f_i), tr, va)
                        for j, col in enumerate(sp.out_names()):
                            Mtr[:, pos[col]] = vt if vt.ndim == 1 else vt[:, j]
                            Mva[:, pos[col]] = vv if vv.ndim == 1 else vv[:, j]
                    for sp in derived:  # in build order: a feature's parents are already this fold's
                        for Mx in (Mtr, Mva):
                            v = np.asarray(sp.transform(pd.DataFrame({p: Mx[:, pos[p]] for p in sp.parents}),
                                                        self.ctx_), dtype=np.float32)
                            for j, col in enumerate(sp.out_names()):
                                Mx[:, pos[col]] = v if v.ndim == 1 else v[:, j]
                    dtr = lgb.Dataset(Mtr, y[tr], categorical_feature=cats, free_raw_data=False)
                    dva = lgb.Dataset(Mva, y[va], reference=dtr)
                    b = lgb.train(self._lgb_params(lr), dtr, 2000, valid_sets=[dva],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])
                    del Mtr, dtr, dva
                fold_cost = _tree_col_s(len(tr)) * b.current_iteration() * width * (self.n_classes_ if self.task_ == "multiclass" else 1)
                self._last_cv_cost += fold_cost
                self._charge(fold_cost)
                done = r_i * len(folds) + f_i + 1
                if give_up and done < n_fits and self._last_cv_cost / done * n_fits > give_up:
                    # Priced from the folds fitted so far, the fit would be discarded for one on fewer
                    # rows (Expedia 3%: 803 s of wall time on 510k rows, all of it thrown away).
                    self._cv_projected = self._last_cv_cost / done * n_fits
                    D = b = None
                    _trim_heap()
                    return None
                oof[va] = b.predict(Mva, num_iteration=b.best_iteration, raw_score=True)
                imp += pd.Series(b.feature_importance("gain"), index=cols)
                if mine:
                    self._mine_paths(b, cols, self.pair_gain_, self.triple_gain_)
            losses.append(_loss(self.task_, y, oof))
            oof_mean += oof / len(repeats)
        # The binned dataset, the selected columns' copy and LightGBM's per-thread buffers are freed
        # here, but glibc keeps them in its arenas: across a prefix ladder of fits on a wide table the
        # process grew by gigabytes it no longer used (1,913 columns: 10.9 -> 13.6 GB).
        D = b = None
        _trim_heap()
        return oof_mean, float(np.mean(losses)), imp

    # ------------------------------------------------------- candidate pool
    def _generate(self, W: pd.DataFrame, imp: pd.Series, selected: List[Spec], round_idx: int) -> List[Spec]:
        num_cols = [c for c in W.columns if c not in self.cat_cols_]
        num_rank = [c for c in imp.sort_values(ascending=False).index if c in num_cols]
        keys_rank = [c for c in imp.sort_values(ascending=False).index if c in self.key_cols_]
        top_num = num_rank[:self.top_numeric]
        top_keys = keys_rank[:self.top_keys]
        existing = set(W.columns) | {sp.name for sp in selected}
        cands: List[Spec] = []

        time_keys = {self.time_col_, self.group_col_, *self.date_cols_, *getattr(self, "time_like_", [])} - {None}

        time_cols = {self.time_col_, *self.date_cols_} - {None}
        # Label echoes, target features and anything built on either: same-row features only. A
        # group statistic of out-of-fold target encodings reads other rows' encodings, which were
        # fitted on this row's label (Avazu: group means of products of two encodings over
        # device_ip took search CV from 0.379 to 0.283 while the gate's held-out loss rose from
        # 0.378 to 0.448).
        echo = set(getattr(self, "echo_cols_", ()))
        for sp in selected:
            if sp.target_dep or echo & set(sp.parents):
                echo |= set(sp.out_names()) | {sp.name}

        def add(spec):
            if ROW_TIME in spec.parents:  # file position: it orders the splits, it is not a feature
                return
            if spec.target_dep and time_keys & set(spec.parents) and not getattr(spec, "time_safe", False):
                return
            # Label-echo columns (and features built on them) only in same-row features.
            if echo and echo & set(spec.parents) and not isinstance(spec, _SAME_ROW):
                return
            # A group's statistic of the date itself (deviation of a row's date from its group's
            # mean date) only tracks where the group sits in time; it drifts by construction.
            if isinstance(spec, GroupStat) and spec.parents[-1] in time_cols:
                return
            if spec.name not in existing:
                existing.add(spec.name)
                cands.append(spec)

        sel_names = [s.name for s in selected if s.n_out == 1 and s.name in num_cols]
        label_free = {s.name for s in selected if not s.target_dep}
        seq = getattr(self, "seq_order_", None)
        if round_idx == 0 and seq is not None:
            g = self.group_col_
            add(GroupSeq(g, seq, None, "step"))
            add(GroupSeq(g, seq, None, "dt"))
            varying = [c for c in num_rank if c not in (g, seq) and c in self.base_cols_
                       and W.groupby(g, sort=False)[c].nunique().mean() > 1.5]
            for c in varying[:4]:
                for kind in ("lag1", "lag2", "lag3", "lead1", "lead2", "diff1", "diff2", "cumsum", "area", "cummax"):
                    add(GroupSeq(g, seq, c, kind))
        if round_idx == 0:
            raw_num = [c for c in self.base_cols_ if c not in self.cat_cols_]
            for stem, cols in column_families(raw_num).items():
                binary = all(set(pd.unique(W[c].dropna())) <= {0, 1} for c in cols)
                stats = ("argmax", "sum") if binary else ("sum", "mean", "std", "max", "min", "argmax",
                                                          "slope", "delta", "npos")
                for st in stats:
                    add(RowStat(cols, st, stem))
            for stem, cols in column_families(raw_num, min_size=8).items():
                cols = [c for c in cols if W[c].nunique() > 20]
                if len(cols) >= 8:
                    add(FamilyCount(cols, "count", stem))
                    add(FamilyCount(cols, "mask", stem))
            # Multi-valued fields stored in numbered slots (components, specs, items).
            for stem, cols, pair in slot_families(W[self.base_cols_], self.cat_cols_):
                add(SlotBag(cols, pair, stem))
                if self.n_classes_ <= 2:
                    add(SlotTE(cols, pair, stem))
            if len(raw_num) >= 3:
                add(RowStat(raw_num, "nonzero", "all"))
                if W[raw_num].isna().any().any():
                    add(RowStat(raw_num, "nan", "all"))
            dense_num = [c for c in top_num if W[c].nunique() > 10]
            if len(W) <= 300_000:
                sizes = sorted({d for d in (2, 4, 8, 16) if d <= len(dense_num)} | {min(len(dense_num), 32)})
                for d in sizes:
                    if d >= 2:
                        ks = (5, 20, 100) if self.n_classes_ <= 2 else (10, 50)
                        add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"top{d}"))
                        if d >= 4:
                            # Importance-weighted metric: distance follows what the model uses.
                            w = np.sqrt(imp.reindex(dense_num[:d]).clip(lower=0).to_numpy() + 1e-12)
                            add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"w{d}", w / w.mean()))
                            add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"rw{d}", w / w.mean(), rank=True))
                            if 2 <= self.n_classes_ <= 10 and len(W) <= 200_000:
                                add(KNNClassDist(dense_num[:d], (1, 2, 4), self.n_classes_, f"w{d}", w / w.mean()))
                for d in (8, 24):
                    if len(dense_num) >= max(4, d // 2) and (d == 8 or len(dense_num) > 8):
                        add(Projection(dense_num[:d], "pca", 4, self.n_classes_, f"top{d}"))
                        add(Projection(dense_num[:d], "pls", 3, self.n_classes_, f"top{d}"))
                if len(dense_num) >= 4:
                    for c in dense_num[:8]:
                        add(LinResid(dense_num[:16], c))
                        add(LinResid(dense_num[:16], c, log=True))
                if len(dense_num) >= 3 and len(W) <= 200_000 and self.n_classes_ <= 2:
                    add(LinearOOF(dense_num[:16], self.n_classes_, "top"))
            # Same-scale sums of 3-4 columns (total area, total spend, ...).
            mag = {c: np.log10(np.nanmedian(np.abs(W[c].to_numpy(dtype=float))) + 1e-9) for c in dense_num[:10]}
            for m in (3, 4):
                for combo in combinations(dense_num[:10], m):
                    if max(mag[c] for c in combo) - min(mag[c] for c in combo) <= 1.0:
                        add(RowStat(list(combo), "sum", "+".join(combo)))
            for a, b in combinations(dense_num[:10], 2):
                add(BinnedPairTE(a, b, self.n_classes_))
            # Geography: distances/bearings between points, rotations, kNN on the map.
            pts = [p for p in geo_points(raw_num) if p[0] in W.columns]
            for p, q in combinations(pts, 2):
                for kind in ("hav", "manh", "bearing"):
                    add(GeoPair(p, q, kind))
            for la, lo in pts:
                for deg in (15, 30, 60, 75, 105, 120, 150, 165):
                    add(Rotate(la, lo, deg))
                if len(W) <= 300_000:
                    ks = (5, 20, 100) if self.n_classes_ <= 2 else (10, 50)
                    add(KNNTarget([la, lo], ks, self.n_classes_, f"geo_{la}"))
            # Structured strings: character n-grams, and molecule descriptors when they parse as SMILES.
            if self.seq_cols_:
                from featureforge.specs.sequences import seq_specs
                for sp in seq_specs(self.seq_cols_, self.smiles_cols_, self.n_classes_):
                    add(sp)
            # Free text: how it is written, its topics, and a sparse linear model on its words.
            for c in self.text_cols_:
                add(TextStats(c))
                add(TextSVD(c))
                add(TextLinearOOF([c], self.n_classes_))
            if self.text_cols_:
                tk = [k for k in keys_rank if k in self.base_cols_ and W[k].nunique() > 2][:12]
                add(TextLinearOOF(self.text_cols_, self.n_classes_, keys=tk, label="text+keys"))
            # Shrunken linear model over one-hot keys and all key pairs.
            lin_keys = [k for k in keys_rank if W[k].nunique() > 2]
            if len(lin_keys) >= 2 and len(W) <= 500_000:
                add(CrossLinearOOF(lin_keys[:8], self.n_classes_, pairs=True))
                add(CrossLinearOOF(lin_keys[:20], self.n_classes_, pairs=False))
            # Key x numeric-band target maps.
            for k in top_keys[:8]:
                for c in dense_num[:8]:
                    if c != k:
                        add(KeyBinTE(k, c, self.n_classes_))
            # Numerics as categories: target encoding of exact values.
            for c in top_num[:12]:
                nu = W[c].nunique()
                if c not in self.key_cols_ and 10 < nu <= len(W) // 4:
                    add(TargetEnc([c], self.n_classes_))
            # How the value is written.
            for c in top_num[:12]:
                x = W[c].to_numpy(dtype=float)
                fin = x[np.isfinite(x)]
                if len(fin) == 0 or W[c].nunique() <= 20:
                    continue
                if np.any(fin != np.round(fin)):
                    add(Digits(c, "frac"))
                    add(Digits(c, "frac100"))
                elif np.nanmax(np.abs(fin)) >= 100:
                    add(Digits(c, "mod10"))
                    add(Digits(c, "mod100"))
            for c in self.numbered_:
                add(StrNum(c))
            for a, b in combinations(top_num, 2):
                for op in self.arith_ops:
                    add(Arith(op, a, b))
            for k in top_keys:
                add(Count([k]))
                if X_nunique(W, k) > 2:
                    add(TargetEnc([k], self.n_classes_))
            for k1, k2 in combinations(top_keys, 2):
                add(Count([k1, k2]))
                add(TargetEnc([k1, k2], self.n_classes_))
            for c in top_num:
                if c not in self.key_cols_:
                    add(Count([c]))
            for k in top_keys:
                for c in top_num:
                    if c == k:
                        continue
                    for st in self.group_stats:
                        add(GroupStat(k, c, st))
            for c in self.date_cols_:
                for part in _DATE_PARTS:
                    add(DatePart(c, part))
            for op, a, b, _ in getattr(self, "wide_pairs_", []):
                add(Arith(op, a, b))
            self._event_candidates(W, imp, add)
            self._entity_candidates(W, imp, top_num, add)
            self._event_gap_candidates(W, imp, add)
            self._interaction_candidates(W, add)
        else:
            # Compose selected features with the strongest raw columns.
            partners = top_num[: max(8, self.top_numeric // 2)]
            for s in sel_names:
                for c in partners:
                    if c == s:
                        continue
                    for op in self.arith_ops:
                        add(Arith(op, s, c))
                # Group statistics of an out-of-fold target feature would leak: other
                # rows' encodings were fitted on this row's label.
                if s in label_free:
                    for k in top_keys[:6]:
                        for st in ("mean", "dev", "z"):
                            add(GroupStat(k, s, st))
            for k1, k2, k3 in list(combinations(top_keys[:6], 3)):
                add(Count([k1, k2, k3]))
                add(TargetEnc([k1, k2, k3], self.n_classes_))
        return cands

    def _event_gap_candidates(self, W, imp, add):
        """Where each row sits among its user's events: time since the user's previous row, until
        their next one, and how many came before. A click log names no user; the repeating key with
        the most levels (a device's ip) together with the row's other codes (app, device, os,
        channel) does, the way TalkingData's winners keyed time-to-next-click. Offered whenever a
        time column orders the rows and such a key repeats; the gate decides. ``event_gaps="past"``
        leaves out the next row (it reads later rows of the same file).

        Only with the test rows passed to ``fit`` (``X_unlabeled``): the gaps are read from the history
        fit holds, so test rows that arrive later in batches (an API feeding Jane Street's or Optiver's
        test) would find their own neighbours missing: no next row at all, and a previous row taken
        from training rather than their batch, unlike every training row."""
        t = getattr(self, "time_col_", None)
        if (self.event_gaps == "off" or t is None or t == ROW_TIME or not self.id_cols_
                or not getattr(self, "has_test_", False)):
            return
        echo = set(getattr(self, "echo_cols_", ()))
        k = self.id_cols_[0]
        if k == t or k in echo:
            return

        def code_like(c):
            if c in self.cat_cols_:
                return c not in self.text_cols_
            x = W[c].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            return len(x) > 0 and bool(np.all(x == np.round(x)))
        codes = [c for c in sorted(self.raw_cols_, key=lambda c: -imp.get(c, 0.0))
                 if c not in (k, t) and c not in echo and c not in getattr(self, "date_cols_", [])
                 and W[c].nunique() > 1 and code_like(c)][:4]
        users = [(k,)] + [(k, *p) for r in (1, 2, 3) for p in combinations(codes, r)]
        if len(codes) > 3:
            users.append((k, *codes))
        kinds = ("prev", "nth") if self.event_gaps == "past" else ("prev", "next", "nth")
        for u in users:
            for kind in kinds:
                sp = EntityLag(list(u) if len(u) > 1 else u[0], t, kind)
                # Planned as a family of their own: as entity lags (about ten times an arithmetic
                # candidate's work, planned first) they shared one family share and were starved.
                sp.family_ = "EventGap"
                add(sp)

    def _entity_candidates(self, W, imp, top_num, add):
        """Counts, target maps and group statistics over ID-like columns and over
        composite entities (ID x anchor, ID pair x anchor)."""
        ids = sorted(self.id_cols_, key=lambda c: -imp.get(c, 0.0))[:4]
        if not ids:
            return
        nums = [c for c in top_num if c not in ids and c not in self.cat_cols_
                and not c.startswith("anchor__")][:self.entity_nums]
        ents = [(k,) for k in ids]
        for _, _, _, a in self.anchors_:
            ents += [(k, a) for k in ids[:3]]
            ents += [(k1, k2, a) for k1, k2 in combinations(ids[:3], 2)]
        if self.profiles and self.anchors_:
            from featureforge.specs.profile import profile_candidates
            rank = list(imp.sort_values(ascending=False).index)
            rebase, seen = [], set()
            for _, t_, s_, d_ in getattr(self, "anchor_found_", []):
                if d_ not in seen and t_ in W.columns and d_ in W.columns:
                    seen.add(d_)
                    rebase.append((t_, s_, d_))
            exclude = ({getattr(self, "time_col_", None)} | set(getattr(self, "echo_cols_", ()))) - {None}
            for sp in profile_candidates(W, ids, self.anchors_, rebase[:12],
                                         [c for c in rank if c in W.columns and c not in self.cat_cols_],
                                         [c for c in rank if c in self.cat_cols_ and c not in self.text_cols_],
                                         exclude=exclude, y=getattr(self, "_yW", None),
                                         time_col=getattr(self, "time_col_", None)):
                add(sp)
        t = getattr(self, "time_col_", None)
        if t is not None:
            # Rows of the entity (or entity pair) at the same time value: clicks by one device in
            # the same hour or day (Avazu, TalkingData).
            for e in [(k,) for k in ids] + list(combinations(ids[:3], 2)):
                if t not in e:
                    add(Count(list(e) + [t]))
        for e in ents:
            if len(e) > 1:
                add(Count(list(e)))
            add(TargetEnc(list(e), self.n_classes_))
            for c in nums:
                if c in e:
                    continue
                for st in ("mean", "dev", "std", "nunique"):
                    add(GroupStat(e if len(e) > 1 else e[0], c, st))


    def _event_candidates(self, W, imp, add, max_levels=4):
        """With a time column: time since / until each entity's last / next row with
        a given level of a low-cardinality column (promotions, holidays, closures)."""
        t = getattr(self, "time_col_", None)
        if t is None or not self.events:
            return
        ids = sorted(self.id_cols_, key=lambda c: -imp.get(c, 0.0))[:2]
        keysets = [[k] for k in ids] or [[]]
        WU = W if self.WU_ is None else self.WU_
        if not getattr(self, "has_test_", False):
            # Times to an entity's last / next flagged row are read from the history fit holds: test rows
            # sent later in batches would miss their own batch's rows (see _event_gap_candidates).
            return
        for f in self.base_cols_:
            if f == t or f in ids or f in self.date_cols_:
                continue
            vc = WU[f].value_counts(normalize=True)
            if not 2 <= len(vc) <= 6:
                continue
            for keys in keysets:
                # The flag must change within entities (not a fixed store attribute).
                if keys and WU.groupby(keys[0], sort=False)[f].nunique().mean() < 1.5:
                    continue
                for lv, share in list(vc.items())[:max_levels]:
                    if not 0.001 < share < 0.999:
                        continue
                    lv = lv if f in self.cat_cols_ else float(lv)
                    for kind in ("prev", "next"):
                        add(EventRecency(keys, t, f, lv, kind))

    def _cell_gain(self, cell, ncell, g, h, lam, fold, K, y, margin, base):
        """Cross-fitted Newton gain of a lookup table over integer cells."""
        idx = fold * ncell + cell
        if g.ndim == 1:
            return base - _loss(self.task_, y, margin + _fold_steps(idx, K, ncell, g, h, lam)[fold, cell])
        return base - _stepped_class_loss(y, margin, idx, ncell, K, g, h, lam)

    def _fast_interactions(self, W, y, margin, imp, n_cols=30, n_bins=8, n_levels=16):
        """FAST-style interaction detection on the current model's residuals.

        Every pair of the top columns is cut into a 2-D grid (quantile bins for
        numerics, frequent levels for keys) and scored by the cross-fitted
        Newton gain of the grid minus the better of its two 1-D gains: what a
        pairwise lookup adds that neither column explains alone, measured on
        rows the lookup did not see. Triples extend the best pairs the same way.
        """
        cols = [c for c in imp.sort_values(ascending=False).index if c in W.columns][:n_cols]
        if self.task_ == "multiclass" and self.time_budget:
            # Every grid is scored once per class: Expedia's 100 hotel clusters over 510k search rows
            # spent 1,985 s of wall in this scan before the search began. When the scan (singles,
            # pairs, up to 130 triples) would take more than its share of the work left, it reads a
            # fixed random subset of the rows instead.
            k = len(cols)
            est = _CLASS_CELL_S * len(y) * self.n_classes_ * (k + k * (k - 1) // 2 + 130)
            cap = _SCAN_SHARE * max(self._time_left(), 0.0)
            m = max(2_000, int(len(y) * cap / est)) if est > cap else len(y)
            if m < len(y):
                keep = np.sort(np.random.default_rng(self.random_state + 8).choice(len(y), m, replace=False))
                self._log(f"interaction scan: {est:.0f}s of work on {len(y)} rows over {self.n_classes_} classes; "
                          f"reading {m} of them")
                W, y = W.iloc[keep], np.asarray(y)[keep]
                margin = margin[keep] if np.ndim(margin) else margin
        n = len(y)
        K = 5
        fold = np.random.default_rng(self.random_state + 7).permutation(n) % K
        g, h = self._grad_hess(y, margin)
        lam = 10.0 * float(h.mean())
        base = _loss(self.task_, y, margin)
        codes, sizes = {}, {}
        for c in cols:
            if c in self.key_cols_ and (c in self.cat_cols_ or W[c].nunique() <= n_levels):
                v = pd.Series(_key_values(W[c]))
                top = v.value_counts().index[:n_levels]
                code = pd.Categorical(v, categories=top).codes.astype(np.int64)
                code[code < 0] = len(top)
                codes[c], sizes[c] = code, len(top) + 1
            elif c not in self.cat_cols_:
                x = W[c].to_numpy(dtype=float)
                fin = np.isfinite(x)
                if fin.sum() < 20:
                    continue
                e = np.unique(np.nanquantile(x[fin], np.linspace(0, 1, n_bins + 1)[1:-1]))
                codes[c] = np.where(fin, np.searchsorted(e, np.where(fin, x, 0.0), side="right"), len(e) + 1)
                sizes[c] = len(e) + 2
        args = (g, h, lam, fold, K, y, margin, base)
        one = {c: self._cell_gain(codes[c], sizes[c], *args) for c in codes}
        pairs = {}
        for a, b in combinations(list(codes), 2):
            pairs[(a, b)] = self._cell_gain(codes[a] * sizes[b] + codes[b], sizes[a] * sizes[b], *args) \
                - max(one[a], one[b], 0.0)
        best = sorted(pairs, key=pairs.get, reverse=True)
        triples = {}
        for a, b in best[:10]:
            if pairs[(a, b)] <= 0:
                break
            ab = codes[a] * sizes[b] + codes[b]
            for c in list(codes)[:15]:
                if c in (a, b):
                    continue
                t = tuple(sorted((a, b, c)))
                if t in triples:
                    continue
                triples[t] = self._cell_gain(ab * sizes[c] + codes[c], sizes[a] * sizes[b] * sizes[c], *args) \
                    - max(pairs[(a, b)] + max(one[a], one[b], 0.0), 0.0)
        if self.task_ == "multiclass":
            self._charge(_CLASS_CELL_S * n * self.n_classes_ * (len(one) + len(pairs) + len(triples)))
        return ({k: v for k, v in pairs.items() if v > 0}, {k: v for k, v in triples.items() if v > 0})

    def _wide_pairs(self, W, y, margin, imp):
        """Differences and ratios from a vectorised scan of all strong and closely related
        numeric pairs (``pairscan``); Loan Default's f528 - f527 sits outside the top columns."""
        budget = min(120.0, 0.05 * self._time_left())
        if self.n_classes_ > 2 or budget < 1:
            return []
        from featureforge.pairscan import scan_pairs
        num = [c for c in self.raw_cols_ if c not in self.cat_cols_ and c in W.columns]
        strong = [c for c in imp.sort_values(ascending=False).index if c in num][:60]
        g, h = self._grad_hess(y, margin)
        t, spent = time.time(), [0.0]
        out = scan_pairs(W, num, strong, g, h, lambda m, rows: _loss(self.task_, y[rows], m), margin,
                         time_budget=budget, cell_s=_PAIR_CELL_S, spent=spent, seed=self.random_state)
        self._charge(spent[0])
        self._log(f"pair scan: {len(num)} numerics, {time.time() - t:.0f}s -> "
                  + (", ".join(f"{a} {op} {b} ({v:.4f})" for op, a, b, v in out[:6]) or "nothing"))
        return out

    def _interaction_candidates(self, W, add):
        """Turn tree-path interactions into explicit features of every applicable family."""
        if not self.n_interactions or not getattr(self, "pair_gain_", None):
            return
        is_key = lambda c: c in self.key_cols_
        dense = lambda c: c not in self.cat_cols_ and W[c].nunique() > 10
        half = self.n_interactions // 2
        fp, ft = getattr(self, "fast_pairs_", {}), getattr(self, "fast_triples_", {})
        pairs = list(dict.fromkeys(
            [tuple(sorted(p)) for p in sorted(fp, key=fp.get, reverse=True)[:half]]
            + sorted(self.pair_gain_, key=self.pair_gain_.get, reverse=True)))[:self.n_interactions]
        for a, b in pairs:
            if dense(a) and dense(b):
                for op in self.arith_ops:
                    add(Arith(op, a, b))
                add(BinnedPairTE(a, b, self.n_classes_))
                add(MixedCount([a, b], [a, b], 16))
            elif is_key(a) and is_key(b):
                add(Count([a, b]))
                add(TargetEnc([a, b], self.n_classes_))
            else:
                k, c = (a, b) if is_key(a) else (b, a)
                binned = [x for x in (a, b) if not is_key(x)]
                if not is_key(k) or any(x in self.cat_cols_ for x in binned):
                    continue
                add(MixedTE([k, c], binned, self.n_classes_, 8))
                add(MixedCount([k, c], binned, 8))
                if c not in self.cat_cols_:
                    for st in self.group_stats:
                        add(GroupStat(k, c, st))
        triples = list(dict.fromkeys(
            sorted(ft, key=ft.get, reverse=True)[:half // 2]
            + sorted(self.triple_gain_, key=self.triple_gain_.get, reverse=True)))[:half]
        for t in triples:
            binned = [c for c in t if not is_key(c) or (c not in self.cat_cols_ and W[c].nunique() > 16)]
            if any(c in self.cat_cols_ and c in binned for c in t):
                continue
            add(MixedTE(list(t), binned, self.n_classes_, 6))
            add(MixedCount(list(t), binned, 6))

    def _materialize(self, specs: List[Spec], W: pd.DataFrame, y, folds, keep_fn=None,
                     score_fn=None) -> Dict[str, np.ndarray]:
        """Values of the candidates that pass the validity checks and ``keep_fn``.

        Candidates are computed (and scored by ``score_fn``, which must not depend on
        other candidates) in batches on a thread pool, mostly numpy / pandas kernels that
        release the GIL; ``keep_fn(spec, values, out, score)`` then decides in the original
        order, so the result is the same as computing them one by one."""
        from concurrent.futures import ThreadPoolExecutor
        out = {}
        # Label-free statistics (counts, group statistics, ...) over every row whose
        # features are known: search rows plus gate and unlabeled rows, as at the end.
        WU = None
        raw = set(self.raw_cols_)
        if getattr(self, "WU_", None) is not None:
            WU = self.WU_
        spent: Dict[str, list] = {}      # family -> [candidates, thread seconds measured] (log only)
        # No family of candidates may take over a third of the work left for building them
        # (a few slow ones, neighbour features on large tables, otherwise starve the rest).
        # Screening, the joint ranking fit and the prefix ladder: eight CV fits, at most half
        # the work left (when the first round's work is capped by the budget).
        reserve = min(8 * getattr(self, "_cv_s", 0.0), 0.5 * max(self._time_left(), 0.0))
        n_threads = max(1, self._threads())
        # Thread work: candidates are built on several threads at once, counted as the reference
        # container's four whatever this machine has, so the same candidates are built everywhere.
        fam_cap = max(self._time_left() - reserve, 0.0) * self.family_time_share * _REF_THREADS

        def work(s):
            """A candidate's building cost on the work clock (thread seconds), from its family and rows."""
            rows = len(W) + (_WU_ROW_SHARE * len(WU) if (WU is not None and not s.target_dep and set(s.parents) <= raw) else 0)
            w = _SPEC_WORK.get(type(s).__name__, _SPEC_WORK_DEFAULT) * rows * _ROW_PASS_S
            if isinstance(s, (KNNTarget, KNNClassDist)):
                # Brute-force neighbours: every row against every row, over the subspace's columns
                # (thread seconds per candidate: IEEE-CIS 151k rows 45-59, Home Credit 246k rows 193).
                w *= rows / 150_000 * (len(s.parents) + 8) / 18
            if self.task_ == "multiclass":
                if isinstance(s, CrossLinearOOF):
                    # Multinomial SAGA updates every class's weights for each non-zero: its fit grows with
                    # the classes (209k rows, 8 keys and their pairs: 20 s binary, 22 s at 12 classes, 92 s
                    # at 100; single keys 6.5, 7 and 33 s). The family constant is the binary fit's.
                    w *= 1 + self.n_classes_ / 26
                # The residual probe scores every output column against every class.
                w += _CLASS_CELL_S * getattr(s, "n_out", 1) * self.n_classes_ * len(W)
            return w

        # scikit-learn's configuration is per thread: the guard's smaller distance blocks are set on each.
        wm = getattr(self, "_sk_wm", None)
        tn = getattr(self, "_text_nnz", None)

        def compute(s, WU=WU):
            t_s = time.time()
            if wm is not None:
                import sklearn
                with sklearn.config_context(working_memory=wm >> 20):
                    return _compute(s, WU, t_s)
            return _compute(s, WU, t_s)

        def _compute(s, WU, t_s):
            try:
                if s.target_dep:
                    v = s.fit_transform_oof(W, y, self.ctx_, folds)
                elif WU is not None and set(s.parents) <= raw:
                    v = s.fit(WU, None, self.ctx_).transform(W, self.ctx_)
                else:
                    v = s.fit(W, y, self.ctx_).transform(W, self.ctx_)
                v = np.asarray(v, dtype=np.float32)
                v2 = v if v.ndim == 2 else v[:, None]
                col = v2[:, 0]
                finite = np.isfinite(col)
                if finite.mean() < 0.05 or np.nanstd(np.where(finite, col, np.nan)) == 0:
                    return None
                return v, (score_fn(s, v) if score_fn is not None else None)
            except Exception:
                return None
            finally:
                st = spent.setdefault(type(s).__name__, [0, 0.0])
                st[0] += 1
                st[1] += time.time() - t_s

        # Which candidates get built is decided up front on the work clock, the same on every run
        # and machine: one of each family first, then family by family from the cheapest per
        # candidate; a family stops at its share, and building stops where the ranking's reserve
        # begins.
        def fam(sp):
            return getattr(sp, "family_", type(sp).__name__)
        fams: Dict[str, list] = {}
        for sp in specs:
            fams.setdefault(fam(sp), []).append(sp)
        order = {sp.name: i for i, sp in enumerate(specs)}
        n_all = len(specs)
        plan, fam_work, capped, total = [], {}, set(), 0.0
        firsts = [g[0] for g in fams.values()]
        rest = [sp for f in sorted(fams, key=lambda f: work(fams[f][0])) for sp in fams[f][1:]]
        # When every candidate fits, none is skipped: a family's share only matters when the
        # families compete for the work (IEEE-CIS's seven neighbour encodings fit with the rest).
        everything = sum(work(sp) for sp in specs)
        fits_all = not self.time_budget or self._time_left() - everything / _REF_THREADS >= reserve
        for sp in (specs if fits_all else firsts + rest):
            if fits_all:
                total += work(sp)
                plan.append(sp)
                continue
            f = fam(sp)
            if f in capped:
                continue
            w = work(sp)
            # A candidate that would cut into the reserve is skipped, not the end of the plan:
            # cheaper ones after it (another family's) may still fit.
            if self.time_budget and self._time_left() - (total + w) / _REF_THREADS < reserve:
                continue
            if self.time_budget and fam_work.get(f, 0.0) + w > fam_cap and f in fam_work:
                capped.add(f)
                self._log(f"budget: {f} candidates would take {fam_work[f] + w:.0f}s of work; the rest of them skipped")
                continue
            fam_work[f] = fam_work.get(f, 0.0) + w
            total += w
            plan.append(sp)
        if self.time_budget:
            self._log(f"budget: {self._time_left():.0f}s of work left; building takes {total / _REF_THREADS:.0f}s")
        if len(plan) < n_all:
            self._log(f"budget: built {len(plan)} of {n_all} candidates")
        self._charge(total / _REF_THREADS)
        # A candidate starts only while the memory left could hold its working arrays twice over, or when
        # no other is running: four threads each building a statistic over every row took Instacart's
        # 8.5 million rows from 10.5 to 13 GB of a 14.3 GB limit. Which candidates are built, and each
        # one's values, do not depend on how many run at once; fewer run when memory is short.
        gate = threading.Condition()
        running = [0]
        # The working sets of the candidates already running, which they may not have allocated yet: a candidate
        # starts only while the memory free less those could hold its own twice over (MCTS under a 3.5 GB limit:
        # each admitted on what was free when it started, together past the limit).
        held = [0]

        def gated(s):
            on_wu = WU is not None and not s.target_dep and set(s.parents) <= raw
            rows = len(WU) if on_wu else len(W)
            need = 2 * _spec_bytes(s, rows, wm, tn)
            with gate:
                while running[0] and available_bytes() - held[0] < need:
                    gate.wait(0.2)
                running[0] += 1
                held[0] += need // 2
                free = available_bytes()
            try:
                if need <= 2 * free:
                    return compute(s)
                # Memory guard: even alone it would not fit. A label-free statistic is fitted on a seeded sample
                # of the rows it reads; any other candidate is not built.
                if on_wu:
                    k = int(0.5 * free / max(_spec_bytes(s, 1, wm, tn), 1))
                    if k >= 1000:
                        pick = np.sort(np.random.default_rng(self.random_state).choice(len(WU), k, replace=False))
                        self._log(f"memory guard: {s.name} fitted on {k} of {len(WU)} rows")
                        return compute(s, _renumbered(WU.iloc[pick]))
                self._log(f"memory guard: {s.name} not built ({need / 2 / 1e9:.1f} GB, {free / 1e9:.1f} GB free)")
                return None
            finally:
                with gate:
                    running[0] -= 1
                    held[0] -= need // 2
                    gate.notify_all()

        batch = 4 * n_threads
        with ThreadPoolExecutor(n_threads) as pool:
            for i in range(0, len(plan), batch):
                chunk = plan[i:i + batch]
                for s, r in zip(chunk, pool.map(gated, chunk)):
                    if r is None:
                        _release(s)
                        continue
                    v, score = r
                    if keep_fn is not None and not keep_fn(s, v, out, score):
                        _release(s)
                        continue
                    out[s.name] = v
        # Back in generation order: families are built cheapest first, so the order they
        # finished in varied from run to run, and with it the joint model's
        # column order, its split-gain ranking and the round's picks (Airline: the same commit
        # kept 41 features or none on alternate runs).
        out = dict(sorted(out.items(), key=lambda kv: order.get(kv[0], len(order))))
        _CODE_CACHE.clear()  # factorised keys of large frames add up to gigabytes
        _COL_CACHE.clear()
        _GROUP_CACHE.clear()
        top = sorted(spent.items(), key=lambda kv: -kv[1][1])[:6]
        self._log("  materialised (thread time): " + ", ".join(f"{k} {n}x {t:.0f}s" for k, (n, t) in top)
                  + f" ({self._work:.0f}s of work)")
        return out

    # ------------------------------------------------------------ screening
    def _grad_hess(self, y, margin):
        if self.task_ == "regression":
            return margin - y, np.ones_like(margin)
        if self.task_ == "binary":
            p = 1 / (1 + np.exp(-margin))
            return p - y, np.maximum(p * (1 - p), 1e-6)
        m = margin - margin.max(axis=1, keepdims=True)
        P = np.exp(m)
        P /= P.sum(axis=1, keepdims=True)
        Y = np.eye(self.n_classes_)[y.astype(int)]
        return P - Y, np.maximum(P * (1 - P), 1e-6)

    def _residual_gain(self, x, g, h, lam, n_bins) -> float:
        """Cross-fitted loss reduction from a histogram Newton correction on one feature.

        The candidate is cut into quantile bins (missing values get their own
        bin). For each fold, every bin receives the regularised Newton step of
        the current model's gradients computed on the *other* folds, i.e. a
        one-feature, ``n_bins``-leaf tree fitted to the residuals; the step is
        applied to the held-out fold and the loss is measured on all rows.
        """
        finite = np.isfinite(x)
        if finite.sum() < 20:
            return -np.inf
        # From one argsort of the finite values: their quantiles (taken from the sorted values: the same order
        # statistics, so the same edges, without numpy's partition, 3x slower here) and their bins, read off
        # where each edge falls in the sorted values (a row's bin is the number of edges at or below it, as
        # searchsorted(side="right") counts). Non-finite values get the last bin.
        pos = np.flatnonzero(finite) if not finite.all() else None
        xf = x if pos is None else x[pos]
        order = np.argsort(xf)
        xs = xf[order]
        edges = np.unique(np.quantile(xs, np.linspace(0, 1, n_bins + 1)[1:-1]))
        nb = len(edges) + 2
        cuts = np.searchsorted(xs, edges, side="left")
        bins = np.full(len(x), nb - 1, dtype=np.intp)
        bins[order if pos is None else pos[order]] = np.repeat(np.arange(len(edges) + 1), np.diff(np.r_[0, cuts, len(xs)]))
        K = self._n_probe_folds
        cell = self._probe_fold * nb + bins
        if g.ndim == 2:
            return self._probe_base_loss - _stepped_class_loss(self._probe_y, self._probe_margin, cell, nb, K,
                                                               g, h, lam)
        # The step of row r is steps[fold[r], bins[r]], i.e. the flat table at its cell.
        steps = _fold_steps(cell, K, nb, g, h, lam).ravel()
        return self._probe_base_loss - _stepped_loss(self.task_, self._probe_y, self._probe_margin, steps, cell)

    def _probe_setup(self, W, y, margin):
        """Cross-fitted residual probe for the current margin, plus each column's own gain."""
        n = len(y)
        self._n_probe_folds = 5
        self._probe_fold = np.random.default_rng(self.random_state).permutation(n) % self._n_probe_folds
        self._probe_y, self._probe_margin = y, margin
        self._probe_base_loss = _loss(self.task_, y, margin)
        g, h = self._grad_hess(y, margin)
        lam = 10.0 * float(h.mean())
        n_bins = int(np.clip(n // 50, 8, 64))

        def probe(v):
            v = np.asarray(v)
            cols = [v] if v.ndim == 1 else [v[:, j] for j in range(v.shape[1])]
            return max(self._residual_gain(c.astype(float), g, h, lam, n_bins) for c in cols)

        def own_gain(c):
            col = W[c]
            x = pd.factorize(col)[0].astype(float) if c in self.cat_cols_ else col.to_numpy(dtype=float)
            return max(probe(x), 0.0)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, self._threads())) as pool:
            parent_gain = dict(zip(W.columns, pool.map(own_gain, list(W.columns))))
        return probe, parent_gain

    def _spec_novelty(self, spec, v, probe, parent_gain):
        """(gain, novel gain) of a candidate under the residual probe."""
        if getattr(spec, "paired", False):
            # Block of per-column transforms: each output against its own parent, summed.
            gs = [probe(v[:, j]) for j in range(v.shape[1])]
            src = getattr(spec, "paired_parents", spec.parents)
            nov = sum(max(g - self.novelty_slack * parent_gain.get(p, 0.0), 0.0) for g, p in zip(gs, src))
            return sum(max(g, 0.0) for g in gs), nov
        g = probe(v)
        parents = getattr(spec, "novelty_parents", None) or spec.parents
        return g, g - self.novelty_slack * max(parent_gain.get(p, 0.0) for p in parents)

    def _screen(self, values: Dict[str, np.ndarray], specs: Dict[str, Spec], W: pd.DataFrame,
                y, margin, idx_a, idx_b, keep: int, scores=None) -> List[str]:
        """Rank candidates by *novel* residual gain.

        A candidate's raw residual gain is compared with the gain of its own
        parent columns under the same probe: an early-stopped model leaves some
        residual signal in existing columns, and any re-expression of such a
        column (``a + const``-like transforms, group statistics of it, ...) would
        otherwise look useful. Only gain beyond the best parent counts.
        """
        n = len(y)
        scores = dict(scores or {})
        missing = [nm for nm in values if nm not in scores]
        if missing:
            probe, parent_gain = self._probe_setup(W, y, margin)
            for name in missing:
                scores[name] = self._spec_novelty(specs[name], values[name], probe, parent_gain)
        gains = {nm: g for nm, (g, _) in scores.items()}
        novelty = {nm: v for nm, (_, v) in scores.items()}
        ranked = sorted([nm for nm in novelty if nm in values], key=novelty.get, reverse=True)
        alive = [nm for nm in ranked if novelty[nm] > 0 and not getattr(specs[nm], "screen_exempt", False)][:keep]
        alive = [nm for nm in ranked if getattr(specs[nm], "screen_exempt", False)] + alive
        self._log(f"  screened {len(novelty)} candidates on {n} rows (5-fold cross-fitted) "
                  f"-> {sum(g > 0 for g in novelty.values())} novel, {len(alive)} kept")
        self._last_gains, self._last_novelty = gains, novelty
        return alive

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y, X_unlabeled: Optional[pd.DataFrame] = None):
        """Search features on ``(X, y)``.

        ``X_unlabeled`` (e.g. the competition test set) is optional: when given,
        the final count and group statistics of the selected features are
        computed over training plus unlabeled rows, a standard transductive
        contest trick. Target statistics only ever use labeled rows.
        """
        with _one_blas_thread():
            # Below 200 rows the 20% gate is not held out at all (too few rows), so the search's picks went in
            # unjudged (Sonar, 166 rows: held-out log loss +32%); the cross-validated gate judges them there too.
            if self.cv_gate and self.gate_frac and 60 <= len(X) <= self.cv_gate_rows:
                done = self._fit_cv_gated(X, y, X_unlabeled)
                if done is not None:
                    return done
            self._fit(X, y, X_unlabeled)
            # Up to 10,000 rows the 20% gate holds 600-2,000 rows and needs z >= 1.65 below 1,000; when it turns
            # the search down, the blocks alone get the cross-validated gate, where every row is a gate row
            # (Phoneme: gate z=+1.60 for the blocks the other tools gained 10-12% with).
            if (self.cv_gate and self.gate_frac and self.cv_gate_rows < len(X) <= 10_000 and not self.gate_passed_
                    and self.time_col_ is None and self.group_col_ is None):
                keep = dict(self.__dict__)
                if self._fit_cv_gated(X, y, X_unlabeled, second=True) is None:
                    self.__dict__.clear()
                    self.__dict__.update(keep)
            return self

    def _params(self):
        import inspect
        return {k: getattr(self, k) for k in inspect.signature(type(self).__init__).parameters if k != "self"}

    def _fit_cv_gated(self, X, y, X_unlabeled, second=False):
        """Small tables: the gate as 5-fold cross-validation of the whole search.

        A 20% gate on a few hundred rows cannot tell a real gain from noise (breast cancer, 559 rows:
        gate loss 0.120 -> 0.100 at z=+0.44 against the 1.645 required). Here each fold runs the search
        on the other four folds (no gate of its own) and scores raw columns against its features on the
        fold's rows, so every row is a gate row and the gain is judged on the procedure, not on features
        the search already saw. When the pooled paired gain clears the gate's z, the features of a search
        on all rows are kept. Otherwise this returns None and the usual search and gate run. Tables with a
        time or group structure (found by the first fold's search) take the usual path too."""
        from featureforge.utils.task import infer_task
        self._t0 = time.time()
        self._cells = X.shape[0] * X.shape[1]
        n = len(X)
        Xr = _renumbered(self._named(X))
        y_arr = np.asarray(y)
        task = self.task or infer_task(pd.Series(y_arr))
        K = 5
        if task == "regression":
            folds = list(KFold(K, shuffle=True, random_state=self.random_state + 7).split(np.zeros(n)))
        else:
            folds = list(StratifiedKFold(K, shuffle=True, random_state=self.random_state + 7).split(np.zeros(n), y_arr))
        # The search's rounds are left out here: on a few hundred rows their picks overfit the search's own CV
        # (German credit, Pima diabetes: held out worse on every one of five splits, z -3 to -5); the blocks,
        # a handful of fixed-width CV fits, are what carries over (Balance Scale, Tic-Tac-Toe, vehicles).
        base = {**self._params(), "gate_frac": 0.0, "time_budget": self.time_budget / (K + 1),
                "verbose": False, "n_rounds": 0}
        d = np.zeros(n)
        t0 = time.time()
        for i, (tr, va) in enumerate(folds):
            sub = FeatureForge(**base)
            sub.cv_gate, sub.block_numeric = False, self.block_numeric
            Xtr, Xva = Xr.iloc[tr].reset_index(drop=True), Xr.iloc[va].reset_index(drop=True)
            U = Xva if X_unlabeled is None else pd.concat([Xva, self._named(X_unlabeled)], ignore_index=True)
            sub._fit(Xtr, y_arr[tr], U)
            if sub.time_col_ is not None or sub.group_col_ is not None:
                self._log("cross-validated gate: time or group structure found -> usual gate")
                return None
            if not sub.new_columns_ and not sub.recode_:
                continue
            yt = y_arr.astype(float)
            if task == "regression" and self.log_target:
                yt = np.log1p(np.clip(yt, 0, None))
            raw = _holdout_rows(task, Xtr, yt[tr], Xva, yt[va], self.random_state, self._threads())
            fe = _holdout_rows(task, sub.transform_train(Xtr), yt[tr], sub.transform(Xva), yt[va],
                               self.random_state, self._threads())
            d[va] = raw - fe
            self._log(f"cross-validated gate fold {i + 1}: {len(sub.new_columns_)} features, held-out loss "
                      f"{raw.mean():.6f} -> {fe.mean():.6f}")
        dw = _winsor(d, None if task == "regression" else y_arr)
        z = float(dw.mean() / (dw.std(ddof=1) / np.sqrt(n) + 1e-300)) if dw.std() > 0 else 0.0
        # Every row is a gate row here, so the usual gate's z applies: its stricter small-gate rule was for a fifth
        # of the rows standing in for all of them.
        z_needed = self.gate_z
        ok = d.mean() > 0 and z >= z_needed
        self._log(f"cross-validated gate over {n} rows: mean gain {d.mean():+.6f} (z={z:+.2f}, need {z_needed:.2f}) "
                  f"-> {'PASS' if ok else 'usual gate'} ({time.time() - t0:.0f}s)")
        if not ok and second:
            return None
        if not ok:
            # The whole procedure was judged on every row and did not beat the raw columns: the usual
            # 20% gate, with a fifth of the rows, is not asked again (breast cancer: its gate passed
            # features on two of five splits and both lost held out, 0.061 -> 0.089 and 0.163 -> 0.205).
            self._gate_veto = True
            try:
                return self._fit(X, y, X_unlabeled)
            finally:
                self._gate_veto = False
        final = FeatureForge(**{**base, "verbose": self.verbose})
        final.block_numeric = self.block_numeric
        final._fit(X, y, X_unlabeled)
        params = self._params()
        self.__dict__.update(final.__dict__)
        self.__dict__.update(params)
        self.cv_gate = True
        self.gate_passed_ = True
        self.cv_gate_z_ = z
        return self

    def _fit(self, X: pd.DataFrame, y, X_unlabeled: Optional[pd.DataFrame] = None):
        self._t0 = time.time()
        self._work = 0.0
        self._deadline = None
        self.addons_ = {}
        # Column names as unique strings (integer names broke sorting, a repeated name made pandas return a frame
        # where a column was expected); transform renames new rows' columns the same way.
        X = _renumbered(self._named(X))
        self._cells = X.shape[0] * X.shape[1]
        if X_unlabeled is not None:
            X_unlabeled = self._named(X_unlabeled)
        # The caller's frames, which stay in memory through the fit (see _mem_rows).
        self._in_bytes = sum(float(F.memory_usage(index=True, deep=True).sum())
                             for F in (X, X_unlabeled) if F is not None)
        y = pd.Series(np.asarray(y))
        self.model_width_ = X.shape[1] + 3 * self.max_new_features   # widest frame a model sees
        self.n_synthetic_ = 0
        if X_unlabeled is not None and self.drop_synthetic:
            fake = synthetic_rows(X, X_unlabeled)
            if fake.any():
                self.n_synthetic_ = int(fake.sum())
                self._log(f"unlabeled rows: {self.n_synthetic_} of {len(fake)} look synthetic "
                          "(no value unique among all rows); left out of label-free statistics")
                X_unlabeled = X_unlabeled[~fake]
        # Fields the unlabeled rows lack: almost constant there on a value training rows (almost) never
        # hold, while varied in training (PLAsTiCC's spectroscopic redshift: -9 on 97% of test objects,
        # measured on 70% of training ones). A model trained on them meets a value it never saw.
        self.shifted_ = [] if X_unlabeled is None else _absent_in_test(X, X_unlabeled)
        self.has_test_ = X_unlabeled is not None and len(X_unlabeled) > 0
        if self.shifted_:
            self._log(f"left out, almost constant in the unlabeled rows on a value training rarely holds: {self.shifted_}")
            X = X.drop(columns=self.shifted_)
            X_unlabeled = X_unlabeled.drop(columns=[c for c in self.shifted_ if c in X_unlabeled.columns])
        if self.task is None:
            from featureforge.utils.task import infer_task
            self.task_ = infer_task(y)
        else:
            self.task_ = self.task
        if self.task_ == "regression":
            self.n_classes_ = 0
            y_np = y.to_numpy(dtype=float)
            if self.log_target:
                # log1p of a small positive target (a volatility of 0.003) is the target itself: the
                # search would chase absolute errors. log keeps it a relative-error search.
                small = np.nanmin(y_np) > 0 and np.nanmedian(y_np) < 1
                y_np = np.log(y_np) if small else np.log1p(y_np)
        else:
            self.classes_, y_np = np.unique(y.to_numpy(), return_inverse=True)
            self.n_classes_ = len(self.classes_)
        self.date_cols_ = [c for c in X.columns if _is_date_like(X[c])]
        for c in self.date_cols_:
            X[c] = _to_days(X[c])
        self.cat_cols_ = [c for c in X.columns if _is_cat(X[c])]
        for c in self.cat_cols_:
            X[c] = _as_str_frame_col(X[c])
        self.cat_levels_ = {c: _warm(pd.Index(sorted(pd.unique(X[c])))) for c in self.cat_cols_}
        self.text_cols_ = [c for c in self.cat_cols_ if _is_text(X[c])] if self.text else []
        self._text_nnz = {c: _text_nnz(X, c, self.random_state) for c in self.text_cols_}
        # Structured strings (molecules, sequences, segmented codes): read character by character.
        from featureforge.specs.sequences import is_seq_string, is_smiles
        self.seq_cols_ = [c for c in self.cat_cols_ if c not in self.text_cols_ and is_seq_string(X[c])] if self.text else []
        self.smiles_cols_ = [c for c in self.seq_cols_ if is_smiles(X[c])]
        X = self._compact(X)
        # Coded strings that carry a number ("location 118") are offered as that number.
        self.numbered_ = [c for c in self.cat_cols_ if c not in self.text_cols_ and _numbered_codes(X[c])]
        # Native categorical splits on many-level columns overfit; frequency-rank codes
        # are the usual contest alternative. Which one wins is measured, not assumed.
        self.hc_cols_ = [c for c in self.cat_cols_ if X[c].nunique() > self.hc_threshold]
        self.recode_ = False
        self.rank_maps_ = {}
        self.base_cols_ = list(X.columns)
        self.id_cols_ = self._id_columns(X)
        self.echo_cols_ = self._label_echo_cols(X, y_np) if self.label_echo else set()
        self.echo_cols_ |= {c for c in self.label_cols if c in X.columns}
        if self.echo_cols_:
            self._log(f"label-echo columns (kept out of cross-row features): {sorted(self.echo_cols_)}")
        self.anchors_ = self._find_anchors(X) if self.entities else []
        X = self._add_anchors(X)
        self.raw_cols_ = list(X.columns)
        self.key_cols_ = [c for c in X.columns
                          if 2 <= X[c].nunique() <= (0.5 * len(X) if c in self.cat_cols_
                                                     else self.max_key_cardinality)]
        # Integer codes with many levels (Amazon's RESOURCE and MGR_ID, card numbers) are
        # categories written as numbers: their crosses and target maps are what contest
        # winners built, so they join the keys while staying numeric for the trees.
        self.code_cols_ = [c for c in self._code_columns(X) if c not in self.key_cols_]
        if self.code_cols_:
            self._log(f"integer codes used as keys: {self.code_cols_}")
        self.key_cols_ += self.code_cols_
        self.key_cols_ += [a[3] for a in self.anchors_]
        self.key_cols_ = [c for c in self.key_cols_ if c not in self.text_cols_]
        self._charge(_SETUP_COL_S * X.shape[1] * min(len(X), 300_000) / 300_000)
        if self.text_cols_:
            self._log(f"free text: {', '.join(self.text_cols_)}")
        if self.seq_cols_:
            self._log(f"structured strings: {', '.join(self.seq_cols_)}"
                      + (f" (molecules: {', '.join(self.smiles_cols_)})" if self.smiles_cols_ else ""))
        self.ctx_ = Context(self.task_, self.n_classes_, self.random_state)

        # Gate split: these rows never influence the search.
        n = len(X)
        idx = np.arange(n)
        strat = y_np if self.task_ != "regression" else None
        self.time_col_ = self._detect_time(X, X_unlabeled)
        if (self.time_col_ is None and self.time_col == "auto" and X_unlabeled is not None
                and self._row_ordered(X, y_np, X_unlabeled) and self._detect_group(X, X_unlabeled) is None):
            # Rows in file order form a series (each row nearly repeats the one before: minute bars
            # without their timestamps, as DRW's crypto file came): random gate rows and folds would
            # sit between their neighbours, and neighbour / target features pass on a gate the later
            # test never matches (DRW: gate +41%, held-out Pearson 0.073 -> 0.016). The file position
            # orders the gate and folds as a time column would; no feature is built from it.
            X[ROW_TIME] = np.arange(n, dtype=float)
            X_unlabeled = X_unlabeled.copy()
            X_unlabeled[ROW_TIME] = n + np.arange(len(X_unlabeled), dtype=float)
            self.time_col_ = ROW_TIME
            self._log("rows in file order form a series: gate and folds follow file order")
        self.horizon_ = 0.0
        self.embargo_ = np.array([], dtype=int)
        self._groups = {}
        self.group_col_ = None if self.time_col_ is not None else self._detect_group(X, X_unlabeled)
        if self.group_col_ is not None:
            self._log(f"test rows are new {self.group_col_} groups: grouped gate, folds and screening")
        self.seq_order_ = None if self.group_col_ is None else self._detect_order(X, self.group_col_)
        if self.seq_order_ is not None:
            self._log(f"{self.group_col_} groups are sequences ordered by {self.seq_order_}: lags, leads, running sums")
        if self.time_col_ is not None:
            # Levels of the time column never recur later: no target maps over it (label-
            # free counts and group statistics per date still carry over to test dates).
            self.id_cols_ = [c for c in self.id_cols_ if c != self.time_col_]
            self._log(f"time-ordered gate on {self.time_col_}")
        if self.gate_frac and n >= 200 and self.time_col_ is not None:
            tx = X[self.time_col_].to_numpy(dtype=float)
            order = np.argsort(tx, kind="stable")
            n_gate = int(round(self.gate_frac * n))
            if X_unlabeled is not None and self.time_col_ in X_unlabeled:
                # Hold out the latest rows covering the test period's horizon: a gate
                # that reaches much further back can hinge on one season (a holiday
                # peak just before it) that the real test period never sees.
                tu = self._prep(X_unlabeled)[self.time_col_].to_numpy(dtype=float)
                horizon = np.nanmax(tu) - np.nanmax(tx)
                self.horizon_ = float(horizon) if np.isfinite(horizon) else 0.0
                if np.isfinite(horizon) and horizon > 0:
                    n_h = int(np.sum(tx > np.nanmax(tx) - horizon))
                    n_gate = int(np.clip(n_h, 0.05 * n, n_gate))
            idx_sel, idx_gate = np.sort(order[:n - n_gate]), np.sort(order[n - n_gate:])
            if X_unlabeled is not None and self.time_col_ in X_unlabeled:
                # The test begins some time after the last labelled row: the gate starts the
                # same time after the search rows, so features that lean on rows just before
                # (neighbours' labels, a client's last few outcomes) are judged as the test
                # will see them. The skipped rows still feed label-free statistics.
                gap = np.nanmin(tu) - np.nanmax(tx)
                step = np.diff(np.unique(tx[np.isfinite(tx)]))
                # A real gap: well beyond the time column's own spacing (daily data whose test
                # starts the next day has none) and at least 2% of the training span.
                big = (len(step) and gap > 2 * np.median(step)
                       and gap >= 0.02 * (np.nanmax(tx) - np.nanmin(tx)))
                if np.isfinite(gap) and big and len(idx_gate):
                    # The gate sits as far after the search rows, on average, as the test sits
                    # after training: the gap plus half the difference between the test's span
                    # and the gate's (IEEE-CIS: a month's gap, then six months of test rows,
                    # where neighbour-label features that still help a month out stop helping).
                    # The search keeps at least a quarter of the rows.
                    t_g = np.nanmin(tx[idx_gate])
                    span_u = np.nanmax(tu) - np.nanmin(tu)
                    span_g = np.nanmax(tx[idx_gate]) - t_g
                    emb_len = gap + max(0.0, (span_u - span_g) / 2)
                    emb = idx_sel[tx[idx_sel] > t_g - emb_len]
                    room = len(idx_sel) - int(0.25 * n)
                    if len(emb) > room:
                        emb = np.sort(emb[np.argsort(tx[emb], kind="stable")[len(emb) - max(room, 0):]])
                    if len(emb):
                        self.embargo_ = emb
                        idx_sel = np.setdiff1d(idx_sel, emb)
                        self._log(f"gate starts {np.nanmin(tx[idx_gate]) - np.nanmax(tx[idx_sel]):.0f} after the search rows, "
                                  f"as the test sits {gap + span_u / 2:.0f} after training on average ({len(emb)} rows "
                                  f"between them feed only label-free statistics)")
        elif self.gate_frac and n >= 200 and self.group_col_ is not None:
            gx = X[self.group_col_].to_numpy()
            levels, codes = np.unique(as_text(pd.Series(gx)).to_numpy() if gx.dtype == object else gx, return_inverse=True)
            in_gate = np.random.default_rng(self.random_state).random(len(levels)) < self.gate_frac
            idx_sel, idx_gate = np.flatnonzero(~in_gate[codes]), np.flatnonzero(in_gate[codes])
        elif self.gate_frac and n >= 200:
            try:
                idx_sel, idx_gate = train_test_split(idx, test_size=self.gate_frac,
                                                     random_state=self.random_state, stratify=strat)
            except ValueError:
                idx_sel, idx_gate = train_test_split(idx, test_size=self.gate_frac,
                                                     random_state=self.random_state)
            idx_sel, idx_gate = np.sort(idx_sel), np.sort(idx_gate)
        else:
            idx_sel, idx_gate = idx, np.array([], dtype=int)

        # Time-ordered or new-group tests: outside proposals are judged on this same split, not on
        # random folds (IEEE-CIS: proposals that cleared random folds at z=+3.4 lost held out).
        self.holdout_ = None
        if len(idx_gate) and (self.time_col_ is not None or self.group_col_ is not None):
            hs, hg = np.asarray(idx_sel), np.asarray(idx_gate)
            if self.time_col_ is not None:
                tt = X[self.time_col_].to_numpy(dtype=float)
                hs, hg = hs[np.argsort(tt[hs], kind="stable")], hg[np.argsort(tt[hg], kind="stable")]
            self.holdout_ = (hs, hg)

        # Large tables: the search runs on the latest rows (time-ordered data), whole random
        # groups (new-entity tests) or a random sample, so that its CV fits leave room in the
        # budget for several rounds; the rows left out still feed label-free statistics.
        idx_out = self.embargo_
        m_cap = None
        # Memory: each round holds up to four times ``keep`` candidate columns over the search
        # rows next to the model's matrix. When labels carry little signal, fits stop early and
        # the time rule above never shrinks the search (Malware's 5.9M search rows with shuffled
        # labels: killed at 14 GB in the first round), so the rows are also capped by memory.
        # The frame of every row (search, held-out and unlabeled; see _set_wu) is built for each round
        # whatever the search rows: its size comes off the memory the search rows may use.
        mu = X.memory_usage(index=False, deep=False)
        all_rows = float(mu[[c for c in self.raw_cols_ if c in mu.index]].sum()) * (
            1 + (0 if X_unlabeled is None else len(X_unlabeled) / max(len(X), 1)))
        mem_cap = self._mem_rows(len(self.raw_cols_), fixed=all_rows,
                                 held=self._in_bytes + float(X.memory_usage(index=True, deep=False).sum()))
        n_wu = len(X) + (0 if X_unlabeled is None else len(X_unlabeled))
        u_bytes = all_rows * (0 if X_unlabeled is None else len(X_unlabeled)) / max(n_wu, 1)
        guard = self._memory_guard(min(mem_cap or len(idx_sel), len(idx_sel)), n_wu, all_rows, u_bytes,
                                   self._in_bytes + float(X.memory_usage(index=True, deep=False).sum()))
        self._wu_keep = None
        if guard is not None:
            mem_cap, self._wu_keep = guard
        if mem_cap is not None and mem_cap < len(idx_sel):
            self._log(f"memory: searching at most {mem_cap} rows")
        for attempt in range(3):
            m = min(m_cap or len(idx_sel), mem_cap or len(idx_sel))
            if len(idx_sel) > m:
                rng = np.random.default_rng(self.random_state)
                if self.time_col_ is not None:
                    keep = idx_sel[np.argsort(X[self.time_col_].to_numpy(dtype=float)[idx_sel], kind="stable")[-m:]]
                elif self.group_col_ is not None:
                    g = pd.factorize(as_text(X[self.group_col_]).to_numpy()[idx_sel])[0]
                    perm = rng.permutation(g.max() + 1)
                    r = perm[g]
                    keep = idx_sel[np.argsort(r, kind="stable")[:m]]
                else:
                    keep = rng.choice(idx_sel, m, replace=False)
                keep = np.sort(keep)
                idx_out = np.union1d(idx_out, np.setdiff1d(idx_sel, keep))
                self._log(f"search on {m} of {len(idx_sel)} rows")
                idx_sel = keep
            W = _renumbered(X.iloc[idx_sel])
            yW = y_np[idx_sel]
            if self.group_col_ is not None:
                gx = as_text(X[self.group_col_]).to_numpy()
                self._groups = {len(W): gx[idx_sel], n: gx}
            self.U_search_ = self.WU_ = None
            extra = np.concatenate([idx_out, idx_gate]).astype(int)
            self._wu_src = None if X_unlabeled is None and not len(idx_out) else \
                self._wu_guard(X, W, extra, X_unlabeled)
            self.ctx_.extra_rows = None
            # Small tables get repeated CV so that selection is not driven by fold noise.
            n_rep = int(np.clip(round(12_000 / max(len(W), 1)), 1, 3))
            folds = [self._folds(len(W), yW, self.cv, self.random_state + 100 * r) for r in range(n_rep)]
            te_folds = self._te_folds(W, yW, self.random_state + 1)
            # Screening split (A trains the residual boosters, B measures them).
            try:
                idx_a, idx_b = train_test_split(np.arange(len(W)), test_size=0.3, random_state=self.random_state,
                                                stratify=yW if self.task_ != "regression" else None)
            except ValueError:
                idx_a, idx_b = train_test_split(np.arange(len(W)), test_size=0.3, random_state=self.random_state)
            if self.group_col_ is not None:
                f0 = self._folds(len(W), yW, 10, self.random_state + 3)
                idx_b = np.sort(np.concatenate([va for _, va in f0[:3]]))
                idx_a = np.setdiff1d(np.arange(len(W)), idx_b)
            if self.time_col_ is not None and self.time_cv:
                # Rows close in time share their period's level, so random folds reward
                # features that interpolate between neighbouring days. Contiguous time
                # blocks as folds, and screening measured on the latest rows, reward only
                # what carries over to other periods, as the test period needs.
                order = np.argsort(W[self.time_col_].to_numpy(dtype=float), kind="stable")
                blocks = np.array_split(order, self.cv)
                folds = [[(np.sort(np.concatenate(blocks[:i] + blocks[i + 1:])), np.sort(b))
                          for i, b in enumerate(blocks)]]
                n_b = int(round(0.3 * len(W)))
                idx_a, idx_b = np.sort(order[:-n_b]), np.sort(order[-n_b:])

            selected: List[Spec] = []
            self.history_ = []
            self._nested_cache = {}
            # The model's view of the search rows as one float32 matrix (``_lgb_matrix``), with
            # its categorical positions and column names; rounds append the chosen features.
            Mw = (*self._lgb_matrix(self._model_frame(W)), list(W.columns))
            # The first round is promised sixteen CV fits; when they would take more than three times
            # the budget on the work clock, the search runs on fewer rows (not below 100k).
            room = 3 * self.time_budget
            shrinkable = attempt < 2 and self.time_budget and len(W) > 150_000
            out = self._cv(None, yW, folds, mine=self.n_interactions > 0, mat=Mw,
                           give_up=room / 16 if shrinkable else None)
            if out is None:
                self._cv_s = self._cv_projected
            else:
                margin, cur_loss, imp = out
                # One CV fit's work: the search stops when fewer than two are left in the budget.
                self._cv_s = self._last_cv_cost
            if shrinkable and 16 * self._cv_s > room:
                m_cap = max(100_000, int(len(W) * room / (16 * self._cv_s) * 0.8))
                if m_cap < len(W):
                    self._log(f"budget: one CV fit costs {self._cv_s:.0f}s of work on {len(W)} rows; searching fewer rows")
                    continue
            break
        self._cv_w = max(len(Mw[2]), 1)  # the width that one CV fit's work was measured at
        if self.n_interactions:
            self.fast_pairs_, self.fast_triples_ = self._fast_interactions(W, yW, margin, imp)
        self.wide_pairs_ = self._wide_pairs(W, yW, margin, imp) if self.pair_scan else []
        self._yW = yW  # search rows' labels, for checks that keep labels out of label-free specs
        self.base_cv_loss_ = cur_loss
        if self.hc_cols_:
            # Where later rows bring many levels the search rows never saw (new devices and cards
            # in a click log's or IEEE-CIS's later period), ranks count every row whose features are
            # known, as the final fit does; else those rows go unranked and the gate judges a recode
            # the transform never produces. Otherwise they are fitted on the search rows: over the
            # test file too, a large test of known levels (West Nile's is 11x the training file)
            # set the ranks and the search found less.
            U_ = None if X_unlabeled is None else self._prep(X_unlabeled)
            later = U_ if U_ is not None else X.drop(index=W.index, errors="ignore")
            new_share = max((~_as_str(later[c]).isin(set(_as_str(W[c])))).mean() if len(later) else 0.0
                            for c in self.hc_cols_)
            if new_share > 0.1:
                self._fit_rank_maps(X, *([] if U_ is None else [U_]))
            else:
                self._fit_rank_maps(W)
            self._log(f"frequency ranks: {100 * new_share:.0f}% of later rows hold levels the search rows "
                      f"lack -> fitted on {'all known rows' if new_share > 0.1 else 'the search rows'}")
            Wr = (*self._lgb_matrix(self._model_frame(W, recode=True)), list(W.columns))
            m_r, loss_r, imp_r = self._cv(None, yW, folds, mat=Wr)
            self._log(f"high-cardinality recode of {len(self.hc_cols_)} columns: CV loss "
                      f"{cur_loss:.6f} -> {loss_r:.6f} ({100 * (cur_loss - loss_r) / cur_loss:+.2f}%)")
            if loss_r < cur_loss * (1 - self.min_rel_gain):
                self.recode_ = True
                Mw, margin, cur_loss, imp = Wr, m_r, loss_r, imp_r
            # Mw alone holds the model matrix: the joint ranking releases it, and the round's
            # largest matrices are built while it would otherwise linger here.
            Wr = U_ = later = None
        self._log(f"task={self.task_} rows={n} (search {len(W)}, gate {len(idx_gate)}) "
                  f"cols={X.shape[1]} base CV loss={cur_loss:.6f} (one CV fit {self._cv_s:.0f}s of work)")

        for r in range(0 if getattr(self, "_gate_veto", False) else self.n_rounds):
            if len(selected) >= self.max_new_features:
                break
            # Memory guard: the least a round needs (the fewest search rows the guard ever leaves, 200 a CV fold,
            # their candidates' values and a LightGBM fit over them) against the memory free now. When even that
            # does not fit, the search stops and keeps what earlier rounds accepted (a 100-class table under 70% of
            # its peak: the base model's fits left no memory free, and the round's candidates were killed).
            m_min = min(len(W), 200 * self.cv)
            width = min(80, 3 * self.max_new_features) + len(self.raw_cols_)
            need = m_min * self._round_row_bytes() + self._lgb_fit_bytes(m_min, width)
            free = available_bytes()
            if need > free:
                self._log(f"memory guard: search stopped before round {r + 1} (needs about {need / 1e9:.2f} GB, "
                          f"{free / 1e9:.2f} GB free); earlier rounds' features stand")
                break
            # The first round always gets room for sixteen CV fits (building, then eight for
            # screening and the prefix ladder); later rounds start only while ten fit in the budget.
            self._deadline = None
            if r == 0 and self._time_left() < 16 * self._cv_s:
                floor = min(16 * self._cv_s, 3 * self.time_budget)
                self._deadline = self._work + floor
                self._log(f"budget: first round given {floor:.0f}s")
            if r > 0 and self._time_left() < 10 * self._cv_s:
                self._log(f"budget: {max(self._time_left(), 0):.0f}s left, one CV fit takes {self._cv_s:.0f}s; "
                          f"search stops")
                break
            values = None  # release the previous round's candidates first
            self._set_wu(W)
            cands = self._generate(W, imp, selected, r)
            keep = min(80, 3 * self.max_new_features)
            spec_by_name = {s.name: s for s in cands}
            # Screen while materialising: only candidates with novel residual gain are
            # kept in memory (thousands of full-length columns otherwise).
            probe, parent_gain = self._probe_setup(W, yW, margin)
            scores = {}

            def score_fn(spec, v):
                return self._spec_novelty(spec, v, probe, parent_gain)

            def keep_fn(spec, v, out, score):
                g, nov = score
                scores[spec.name] = (g, nov)
                if getattr(spec, "gate_addon", False) and spec.name not in self.addons_:
                    self.addons_[spec.name] = (spec, v)
                if getattr(spec, "screen_exempt", False):
                    return True
                if nov <= 0:
                    return False
                rivals = [nm for nm in out if not getattr(spec_by_name[nm], "screen_exempt", False)]
                if len(rivals) >= 4 * keep:
                    worst = min(rivals, key=lambda nm: scores[nm][1])
                    if scores[worst][1] >= nov:
                        return False
                    del out[worst]
                    _release(spec_by_name[worst])
                return True
            values = self._materialize(cands, W, yW, te_folds, keep_fn=keep_fn, score_fn=score_fn)
            self._set_wu(W, on=False)
            _trim_heap()
            self._log(f"round {r + 1}: {len(cands)} candidates generated, {len(scores)} valid")
            if not values:
                break
            spec_by_name = {s.name: s for s in cands}
            survivors = self._screen(values, spec_by_name, W, yW, margin, idx_a, idx_b,
                                     keep=keep, scores=scores)
            for nm in set(values) - set(survivors):
                _release(spec_by_name[nm])
            n_values = len(values)
            values = {nm: values[nm] for nm in survivors}
            if not survivors:
                break
            # Joint model ranks survivors by split gain alongside current features.
            new = [(col, values[nm] if values[nm].ndim == 1 else values[nm][:, j])
                   for nm in survivors for j, col in enumerate(spec_by_name[nm].out_names())]
            new = list(dict(new).items())  # one column per name, last wins (as column assignment)
            base_cols = Mw[2]
            jcols = base_cols + [c for c, _ in new if c not in set(base_cols)]
            # Memory guard: the joint matrix and its LightGBM fit (below), priced against the memory free now. When
            # they would not fit, the search stops here and keeps what earlier rounds accepted (Expedia's 12 classes
            # under a 335 MB limit, about 0.1 GB free after screening: killed building this Dataset).
            need = 4 * len(W) * len(jcols) + self._lgb_fit_bytes(len(W), len(jcols))
            free = available_bytes()
            if need > free:
                self._log(f"memory guard: search stopped before the joint model of {len(survivors)} survivors "
                          f"(needs about {need / 1e9:.2f} GB, {free / 1e9:.2f} GB free); earlier rounds' features stand")
                for nm in survivors:
                    _release(spec_by_name[nm])
                values = new = None
                break
            # Column-major: a run of its columns is a matrix LightGBM reads without a copy (see the
            # prefix ladder below).
            Mj = np.empty((len(W), len(jcols)), dtype=np.float32, order="F")
            Mj[:, :len(base_cols)] = Mw[0]
            jpos = {c: i for i, c in enumerate(jcols)}
            for c, v in new:
                Mj[:, jpos[c]] = v
            cats_j = set(Mw[1])
            Mw = None  # its columns lead the joint matrix
            # The joint matrix holds every survivor's values: keep only what it cannot
            # (values of another dtype), so they are not held twice from here on.
            values = {nm: v for nm, v in values.items() if v.dtype != np.float32}
            new = None
            import lightgbm as lgb
            # Not keeping the raw matrix: the row subsets would each copy their rows of it.
            Dj = lgb.Dataset(Mj, yW, categorical_feature=sorted(cats_j),
                             params=self._sample_cap(self._lgb_params(), *Mj.shape)).construct()
            b = self._fit_eval_rows(Dj, idx_a, idx_b)
            del Dj
            gain = pd.Series(b.feature_importance("gain"), index=jcols)

            def sub(cols, copy=True):
                idx = [jpos[c] for c in cols]
                cats = [i for i, j in enumerate(idx) if j in cats_j]
                return (Mj[:, idx], cats, list(cols)) if copy else (Mj, cats, list(cols), idx)

            rank = sorted(survivors, key=lambda nm: -sum(gain.get(c, 0) for c in spec_by_name[nm].out_names()))
            rank = [nm for nm in rank if sum(gain.get(c, 0) for c in spec_by_name[nm].out_names()) > 0]
            # Survivors' columns in gain order right after the current features: every gain prefix is
            # then a run of the joint matrix's columns, which the ladder's fits read in place instead
            # of each copying their columns (5,700 columns over 92,800 rows on a wide table: 2.1 GB
            # per fit, next to the joint matrix itself).
            lead = list(dict.fromkeys(list(base_cols) + [c for nm in rank for c in spec_by_name[nm].out_names()]))
            order_j = lead + [c for c in jcols if c not in set(lead)]
            _permute_columns(Mj, [jpos[c] for c in order_j])
            jpos = {c: i for i, c in enumerate(order_j)}

            nested = None
            if self.nested_cv is True or (self.nested_cv == "auto" and self.time_col_ is not None and self.time_cv):
                te_specs = [sp for sp in list(selected) + [spec_by_name[nm] for nm in survivors]
                            if isinstance(sp, TargetEnc)]
                # Features built on those encodings (a sum or product of two) are recomputed per fold
                # from the fold's encodings too: kept at their values over the whole search frame, whose
                # folds are not the CV's, they carry the validation rows' labels (Avazu round 2: search CV
                # +6.3%, gate +0.13%).
                tdep = {c for sp in te_specs for c in sp.out_names()}
                derived = []
                for sp in list(selected) + [spec_by_name[nm] for nm in survivors]:
                    if not isinstance(sp, TargetEnc) and tdep & set(sp.parents):
                        derived.append(sp)
                        tdep |= set(sp.out_names())
                nested = {"specs": te_specs, "derived": derived, "W": W}
            room = self.max_new_features - len(selected)
            # Two orderings: joint-model split gain, and novel residual gain from
            # screening. Wide candidate pools can push strong but narrow features
            # (a single pairwise interaction) down the split-gain order.
            novel_rank = sorted(survivors, key=lambda nm: -self._last_novelty.get(nm, 0.0))
            orders = [("gain", rank, (3, 6, 12, 25, 50, 80)), ("novelty", novel_rank, (3, 6, 12))]
            best_k, best_loss, best_fit, best_rank = 0, cur_loss, None, rank
            tried = set()
            for label, order, steps in orders:
                ladder = [k for k in steps if k < min(len(order), room)]
                if label == "gain":
                    ladder.append(min(len(order), room))
                for k in sorted(set(ladder)):
                    if k <= 0:
                        continue
                    key = frozenset(order[:k])
                    cols = list(base_cols) + [c for nm in order[:k] for c in spec_by_name[nm].out_names()]
                    # A step costs its width: priced at the base CV's width, a 320-column step on
                    # Expedia 3% (100 classes) was started with a 23-column fit's work left and took
                    # 1,362 s.
                    if key in tried or (tried and self._time_left() < self._cv_s * len(cols) / self._cv_w):
                        continue
                    tried.add(key)
                    oof_k, loss_k, imp_k = self._cv(None, yW, folds, nested=nested, mat=sub(cols, copy=False))
                    self._log(f"  {label} top-{k:<3d} CV loss={loss_k:.6f} ({100 * (cur_loss - loss_k) / cur_loss:+.2f}%)")
                    if loss_k < best_loss:
                        best_k, best_loss, best_fit, best_rank = k, loss_k, (oof_k, imp_k, cols), order
            rank = best_rank
            if best_k == 0 or (cur_loss - best_loss) / cur_loss < self.min_rel_gain:
                self._log(f"round {r + 1}: no prefix beats current CV loss by {self.min_rel_gain:.2%}; stopping")
                break
            chosen = [spec_by_name[nm] for nm in rank[:best_k]]
            for sp in chosen:
                sp.round_ = r
            selected.extend(chosen)
            Mw = sub(best_fit[2])
            for s in chosen:
                for j, col in enumerate(s.out_names()):
                    v = values.get(s.name)
                    W[col] = Mj[:, jpos[col]].copy() if v is None else v if v.ndim == 1 else v[:, j]
            Mj = None
            margin, imp = best_fit[0], best_fit[1]
            self.history_.append(dict(round=r + 1, n_candidates=n_values, n_added=best_k,
                                      cv_loss_before=cur_loss, cv_loss_after=best_loss))
            self._log(f"round {r + 1}: +{best_k} features, CV loss {cur_loss:.6f} -> {best_loss:.6f} ({self._work:.0f}s of work)")
            if chosen:
                self._log(f"  round {r + 1} picks: {', '.join(sp.name for sp in chosen)[:600]}")
            cur_loss = best_loss

        cur_loss = self._arith_blocks(W, yW, folds, imp, selected, cur_loss)
        self._deadline = None
        self.search_cv_loss_ = cur_loss
        last_r = max((sp.round_ for sp in selected), default=None)
        chosen_names = {sp.name for sp in selected}
        self.addons_ = {nm: av for nm, av in self.addons_.items() if nm not in chosen_names}
        for sp, v in self.addons_.values():
            sp.round_ = last_r if last_r is not None else 0
            for j, col in enumerate(sp.out_names()):
                W[col] = v if v.ndim == 1 else v[:, j]
        self.selected_ = selected
        # Candidate values of the last round can be gigabytes on large tables.
        values = cands = Mw = None
        _trim_heap()
        self.gate_passed_ = None
        if (selected or self.recode_ or self.anchors_) and len(idx_gate):
            # Memory guard: the gate's widest model (raw columns and every pick), priced against the memory free
            # now, at the larger of its two phases: building the search rows' dataset (their float32 matrix and
            # LightGBM's bin sample, the matrix freed once binned), then fitting it and scoring the gate rows (the
            # binned rows, the gate rows' frame and matrix, the histogram pool and per-row gradients). Priced as
            # their sum, Springleaf's 1,935-column gate came to 7.6 GB against 6.5 GB free at the normal limit and
            # was skipped, though the whole run had peaked at 12.4 GB of 14.3 with it. When it would not fit,
            # nothing can be shown to beat the raw columns: they are kept, as when the gate rejects.
            width = len(self.base_cols_) + sum(len(sp.out_names()) for sp in selected)
            n_s, n_g = len(idx_sel), len(idx_gate)
            p = self._sample_cap(self._lgb_params(), n_s, width)
            build = 4 * n_s * width + 13 * min(p.get("bin_construct_sample_cnt", 200_000), n_s) * width
            fit = (n_s + 8 * n_g) * width + 24 * 255 * 31 * width + 24 * n_s * max(self.n_classes_, 1)
            need = max(build, fit)
            free = available_bytes()
            if need > free:
                self.gate_passed_ = False
                self.gate_raw_loss_ = self.gate_fe_loss_ = float("nan")
                self.addons_ = {}
                self.recode_ = False
                self._log(f"memory guard: gate skipped (needs about {need / 1e9:.2f} GB, {free / 1e9:.2f} GB free) "
                          f"-> raw columns kept")
            else:
                best_set, self.gate_raw_loss_, self.gate_fe_loss_ = self._gate(X, y_np, idx_sel, idx_gate, W)
                self.gate_passed_ = best_set is not None
                self.addons_ = {}
                if self.gate_passed_:
                    self.selected_ = list(best_set)
                else:
                    self.recode_ = False
                self._log(f"gate: raw={self.gate_raw_loss_:.6f} fe={self.gate_fe_loss_:.6f} "
                          f"({100 * (self.gate_raw_loss_ - self.gate_fe_loss_) / self.gate_raw_loss_:+.2f}%) "
                          f"-> {'PASS' if self.gate_passed_ else 'REJECT'}")
            if not self.gate_passed_:
                self.selected_ = []
                dropped = {a[3] for a in self.anchors_}
                self.raw_cols_ = [c for c in self.raw_cols_ if c not in dropped]
                self.anchors_ = []

        if getattr(self, "_gate_veto", False):
            self._log("cross-validated gate rejected the search -> raw columns kept")
            self.gate_passed_, self.selected_, self.recode_, self.addons_ = False, [], False, {}
            dropped = {a[3] for a in self.anchors_}
            self.raw_cols_ = [c for c in self.raw_cols_ if c not in dropped]
            self.anchors_ = []
        # Refit every spec's statistics on all training rows.
        self.U_search_ = self.WU_ = self._wu_src = None
        self._fit_full(X, y_np, None if X_unlabeled is None else self._prep(X_unlabeled))
        # Read by fits only: the unlabeled rows' raw columns (a copy) and every row's group labels
        # (Instacart's 8.5 million rows: about 1.2 GB held through the transforms after the fit).
        self.ctx_.extra_rows = None
        self._groups = {}
        _CODE_CACHE.clear()
        _COL_CACHE.clear()
        _GROUP_CACHE.clear()
        self.elapsed_ = time.time() - self._t0
        self._log(f"done: {len(self.selected_)} features added in {self.elapsed_:.1f}s ({self._work:.0f}s of work)")
        return self

    def _mem_rows(self, n_cols, share=0.3, fixed=0.0, held=0.0):
        """Search rows whose candidate columns, their build-time copies and the model's matrix
        (float32) fit in ``share`` of the memory left under the limit (the container's cgroup
        limit when set), or None when it is unknown. Malware: about 6 KB per search row in the
        first round on top of the 7 GB its tables already take. ``fixed``: bytes needed whatever
        the search rows. ``held``: bytes the data already takes (the caller's frames and the
        fit's own), which with ``_BASE_RSS`` for the interpreter and its libraries stands in for
        the memory in use: read live, it moved the cap, and every pick after it, from run to run
        of one commit (TalkingData: 919,814 vs 920,499 search rows, 40 minutes apart). The
        result is rounded down to two significant digits. The live reading remains a guard:
        when the rows would take more than twice ``share`` of what is actually free, they are
        cut to that, and the log says so."""
        lim = []
        for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                v = open(f).read().strip()
                if v.isdigit() and int(v) < 1 << 60:
                    lim.append(int(v))
            except OSError:
                pass
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            lim.append(page * os.sysconf("SC_PHYS_PAGES"))
            rss = int(open("/proc/self/statm").read().split()[1]) * page
        except (ValueError, OSError, AttributeError, IndexError):
            rss = 0
        if not lim:
            return None
        keep = min(80, 3 * self.max_new_features)
        row = 4 * (16 * keep + 2 * n_cols + 16)
        n = int(share * max(min(lim) - _BASE_RSS - held - fixed, 0) / row)
        if n >= 100:
            step = 10 ** (len(str(n)) - 2)
            n = n // step * step
        live = int(2 * share * max(min(lim) - rss - fixed, 0) / row) if rss else n
        if live < n:
            self._log(f"memory guard: {n} search rows would not fit in the memory free now; cut to {live} "
                      f"(this run's rows, and the picks after them, depend on the memory in use)")
            n = live
        return n

    def _round_row_bytes(self):
        """What a search round holds per search row (see _memory_guard)."""
        keep = min(80, 3 * self.max_new_features)
        row = 4 * (16 * keep + 2 * len(self.raw_cols_) + 16)
        # LightGBM's binned dataset of the joint matrix (a byte a value), and for several classes its gradients and
        # hessians (float32), scores and the folds' predictions (float64) for every class (Expedia's 12 and 100
        # classes under 70% of their peak: killed in the gain ladder's fits, with no term for either).
        row += keep + len(self.raw_cols_)
        if self.task_ == "multiclass":
            row += 24 * self.n_classes_
        return row

    def _memory_guard(self, n_search, n_wu, wu_bytes, u_bytes, held):
        """None when a search round fits in the memory free now (the machine's, or this process's memory cgroups'
        when less, which also counts the other processes in them): what the round adds to what this process
        already holds, from the frames' actual sizes, with no fixed part (the interpreter and the frames read so
        far are in use already):
          - the frame of every row the label-free statistics read, ``wu_bytes`` over ``n_wu`` rows, and the
            unlabeled rows' prepared copy, ``u_bytes``;
          - one candidate's working arrays over each of those rows, 2 x _SPEC_BYTES_PER_ROW a row;
          - per search row, float32 values of the candidates alive at once: the screen's pool of 4 x ``keep``
            columns and the joint matrix's survivors, about 16 x ``keep`` values, next to the model's matrix of
            the raw columns (two copies) and 16 more for folds and predictions; the binned dataset, a byte a
            column; and for a multiclass task 24 bytes a class.
        (0.8 GB fixed and the limit in place of the memory free fired on 493-row ICR under a 189 MB limit, whose
        whole run peaks at 0.27 GB.) When it fires: all of both kinds of rows if they fit after all, else the
        most search rows that fit next to every label-free row, else both cut in proportion, with at least 200
        search rows for each CV fold (a fold's loss on fewer moved by more than the gains it is to measure).
        Neighbour queries also take smaller distance blocks."""
        keep = min(80, 3 * self.max_new_features)
        row = self._round_row_bytes()
        f_wu = wu_bytes + u_bytes
        # One candidate's working arrays over every row, or one fit's histogram pool (a histogram of 255 bins of
        # gradient, hessian and count, float64, for each column of the joint matrix and each of 31 leaves).
        work = max(2 * _SPEC_BYTES_PER_ROW * n_wu, 24 * 255 * 31 * (keep + len(self.raw_cols_)))
        need = f_wu + work + n_search * row
        free = available_bytes()
        if need <= free:
            return None
        try:
            rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            rss = 0
        import sklearn
        sklearn.set_config(working_memory=128)
        self._sk_wm = 128 << 20
        avail = max(free - work, 0)
        m_min = min(n_search, 200 * self.cv)
        m, k = n_search, None
        if f_wu + n_search * row > avail:
            m = int((avail - f_wu) / row)
            if m < m_min:
                s = avail / (f_wu + n_search * row)
                m = max(int(s * n_search), m_min)
                k = int(max(s * n_wu - m, 0))
        m = min(m, n_search)
        self._log(f"memory guard: a search round would add {need / 1e9:.2f} GB to the {rss / 1e9:.2f} GB in use, "
                  f"{free / 1e9:.2f} GB free; searching at most {m} rows, label-free statistics read "
                  f"{'every row' if k is None else f'{k} rows'} beyond them")
        return m, k

    def _wu_guard(self, X, W, extra, U):
        """The rows the label-free statistics read beyond the search rows (``extra`` rows of X, and U),
        sampled (seeded) to the count the memory guard set; all of them when it did not fire."""
        k = getattr(self, "_wu_keep", None)
        n_x = len(extra) + (0 if U is None else len(U))
        if k is None or k >= n_x:
            return X, extra, U
        if k == 0:
            return None
        rng = np.random.default_rng(self.random_state + 7)
        pick = np.sort(rng.choice(n_x, k, replace=False))
        extra_k = extra[pick[pick < len(extra)]]
        u_k = pick[pick >= len(extra)] - len(extra)
        U_k = None if U is None else _renumbered(U.iloc[u_k])
        return X, extra_k, U_k

    def _set_wu(self, W, on=True):
        """Search rows, then gate and unlabeled rows: every row whose raw features are known,
        for the label-free statistics of candidates (the rows beyond the search rows are a
        view, not a second copy). Built for each round's candidates and dropped while models
        are fitted, when it would only add gigabytes to the peak."""
        self.U_search_ = self.WU_ = None
        if on and getattr(self, "_wu_src", None) is not None:
            X, idx_gate, U = self._wu_src
            Up = None if U is None else self._prep(U)
            cols, pos = list(self.raw_cols_), X.columns.get_indexer(self.raw_cols_)
            # Sixteen columns at a time: the parts' copies of all the rows were as large as the frame itself, on
            # top of it (Bosch, 4,264 columns: past the 15 GB limit; Instacart, where the rows beyond the search
            # rows are 8 million of 78 columns: 4 GB at the search's peak). Joined column by column (each a
            # view of its chunk), not consolidated: consolidating copied it all again.
            step = 16
            out = {}
            for i in range(0, len(cols), step):
                cc = cols[i:i + step]
                parts = [W[cc], X.iloc[idx_gate, pos[i:i + step]]] + ([] if Up is None else [Up[cc]])
                chunk = _concat_cats(parts)
                del parts
                out.update({c: chunk[c] for c in cc})
            del Up
            self.WU_ = pd.DataFrame(out, copy=False) if len(cols) > step else chunk
            del out
            self.U_search_ = _renumbered(self.WU_.iloc[len(W):])
        self.ctx_.extra_rows = self.U_search_

    def _gate(self, X, y, idx_sel, idx_gate, W):
        """Score raw vs. engineered feature sets on the held-out gate rows.

        ``W`` holds the search rows with the selected features as computed during
        the search (target features out of fold); every selected spec is still
        fitted on exactly those rows, so the gate rows only need ``transform``.
        """
        # The gate rows once: features are added to a shallow copy (columns inserted, never
        # written into), not to a second full copy of a wide table.
        Xg = _renumbered(X.iloc[idx_gate])
        ys, yg = y[idx_sel], y[idx_gate]
        Fs, Fg = W, Xg.copy(deep=False)
        addons = [sp for sp, _ in getattr(self, "addons_", {}).values()]
        for s in list(self.selected_) + addons:
            if all(c in Fg.columns for c in s.out_names()):
                continue
            vg = s.transform(Fg, self.ctx_)
            for j, col in enumerate(s.out_names()):
                Fg[col] = vg if np.ndim(vg) == 1 else vg[:, j]
        import lightgbm as lgb
        es_tr, es_va = self._folds(len(Fs), ys, 5, self.random_state + 11)[0]
        if self.time_col_ is not None:
            # Early-stop on the latest search rows too, so the gate's models are tuned
            # for later periods as the competition's are.
            order = np.argsort(Fs[self.time_col_].to_numpy(dtype=float), kind="stable")
            n_va = max(1, len(order) // 5)
            es_tr, es_va = np.sort(order[:-n_va]), np.sort(order[-n_va:])

        def gate_loss(cols, recode=None):
            D = self._lgb_data(self._model_frame(_columns(Fs, cols), recode), ys, lr=0.05)
            G, _ = self._lgb_matrix(self._model_frame(_columns(Fg, cols), recode))
            b = self._fit_eval_rows(D, es_tr, es_va, lr=0.05, es=100)
            # Refit on all search rows at the early-stopped size, then score the gate rows.
            full = lgb.train(self._train_params(D, 0.05, seed=self.random_state), D, max(1, b.best_iteration))
            margin = full.predict(G, raw_score=True)
            iters = b.current_iteration() + full.current_iteration()
            # The fit's work, as a CV fold is charged (from the iterations it trained).
            self._gate_fit_s = _TREE_CELL_S * iters * len(ys) * len(cols) * (
                self.n_classes_ if self.task_ == "multiclass" else 1)
            D = G = b = full = None
            _trim_heap()
            return _row_loss(self.task_, yg, margin)

        # Rows of one period share its shocks, so their paired differences are not
        # independent: the period-level z, logged next to the row-level one, treats each period
        # (gate time value, or one of 30 equal time blocks when values are finer) as one observation.
        blocks = None
        if self.time_col_ is not None and self.time_col_ in Xg.columns:
            tg = Xg[self.time_col_].to_numpy(dtype=float)
            u, codes = np.unique(tg, return_inverse=True)
            if len(u) > 60:
                order = np.argsort(tg, kind="stable")
                codes = np.empty(len(tg), dtype=int)
                codes[order] = np.arange(len(tg)) * 30 // len(tg)
            blocks = codes if codes.max() >= 4 else None
        yc = yg if self.task_ != "regression" else None
        tg_all = (Xg[self.time_col_].to_numpy(dtype=float)
                  if self.time_col_ is not None and self.time_col_ in Xg.columns else None)
        self._gate_yt = (yc, tg_all)
        # The baseline is always the raw columns as given (native categoricals).
        raw_rows = gate_loss(self.base_cols_, recode=False)
        raw_l = float(raw_rows.mean())
        # Small gates are noisy: demand 95% one-sided confidence below 1000 rows.
        z_needed = self.gate_z if len(yg) >= 1000 or self.gate_z < 0 else max(self.gate_z, 1.645)
        # Candidate feature sets are the cumulative rounds; the best one on the gate is
        # kept only if its paired per-row improvement clears ``gate_z`` standard errors.
        best_set, best_l, best_z = None, raw_l, 0.0
        # Candidate sets: the cumulative rounds; round -1 is the high-cardinality recode alone.
        cands = [("anchors", [])] if self.anchors_ else []
        for r in ([-1] if self.recode_ else []) + sorted({s.round_ for s in self.selected_}):
            cands.append((f"rounds<={r + 1}", [s for s in self.selected_ if s.round_ <= r]))
        # Picks that skipped the residual screen (kNN class distances, numbered codes) reach
        # the gate untested on their own: each set is also offered without each such family,
        # so one that does not carry over cannot ride on the others (Telstra: kNN distances
        # passed with the location number and cost 0.025 log loss held-out).
        for label, specs in list(cands):
            for f in sorted({type(sp).__name__ for sp in specs if getattr(sp, "screen_exempt", False)}):
                gone, rest = set(), []
                for sp in specs:
                    if type(sp).__name__ == f or any(p in gone for p in sp.parents):
                        gone.update(sp.out_names())
                    else:
                        rest.append(sp)
                if rest:
                    cands.append((f"{label} without {f}", rest))
        # Numbered codes the screen passed over (their worth is in interactions) are offered on
        # top of each set.
        if addons:
            cands += [(f"{label} + {', '.join(sp.name for sp in addons)}", list(specs) + addons)
                      for label, specs in list(cands) if specs and not label.endswith(" without StrNum")]
        # The recode was chosen on the search rows; each feature set is also offered on the
        # native categoricals, so features that need them are not sunk by it (Telstra: the
        # location's number lost with "location" rank-coded, gained with it native).
        if self.recode_:
            cands += [(label + " native", specs, False) for label, specs in cands if specs]
        best_recode = self.recode_
        full_ids, full_ok = {id(s) for s in self.selected_}, False
        for label, specs, *rc in cands:
            rc = rc[0] if rc else None
            cols = self.raw_cols_ + [c for s in specs for c in s.out_names()]
            rows = gate_loss(cols, rc)
            d = raw_rows - rows
            dw = _winsor(d, yc)
            z = float(dw.mean() / (dw.std(ddof=1) / np.sqrt(len(dw)) + 1e-300))
            zb = _block_z(dw, blocks) if blocks is not None else None
            late = _later_half(dw, tg_all) if tg_all is not None else None
            self._log(f"  gate {label}{' (recoded)' if self.recode_ and rc is None else ''}: loss={rows.mean():.6f} vs raw {raw_l:.6f} "
                      f"(z={z:+.2f}" + (f", by period {zb:+.2f}" if zb is not None else "")
                      + (f", later half {late:+.2e}" if late is not None else "") + f", need {z_needed:.2f})")
            if specs and {id(s) for s in specs} == full_ids and z >= z_needed and (late is None or late > 0):
                full_ok = True
            # Time-ordered gates: the gain must still hold over the later half of the gate rows
            # (Hull Tactical, 1,086 daily gate rows: z=+1.59 overall, +2.47 / -1.35 by half;
            # held-out Pearson 0.072 -> 0.038).
            if rows.mean() < best_l and z >= z_needed and (late is None or late > 0):
                best_set, best_l, best_z = specs, float(rows.mean()), z
                best_recode = self.recode_ if rc is None else rc
        self.recode_ = best_recode
        fam_on = self.gate_families is True or (self.gate_families == "auto" and (
            len(getattr(self, "embargo_", ())) or (len(idx_gate) >= 20_000 and self._sparse_te(X))))
        # Under an embargo the families are checked even when the whole set passes: on
        # IEEE-CIS the set passed while dropping kNN and cross-linear target features
        # cut the gate loss further (0.1152 -> 0.1081) and lifted held-out AUC 0.926 -> 0.937.
        if fam_on and self.selected_:
            best_set, best_l = self._gate_families(gate_loss, raw_rows, z_needed, best_set, best_l,
                                                   budgeted=full_ok)
        return best_set, raw_l, best_l

    def _arith_blocks(self, W, yW, folds, imp, selected, cur_loss):
        """Dense arithmetic blocks: one operation over every pair of the leading numeric columns,
        tried as a whole. The residual screen scores one candidate at a time, so it misses
        features whose worth is in helping the trees split (Jungle Chess: piece-file and rank
        differences, held out 0.264 -> 0.176 log loss with all pairwise differences, while every
        single difference screened below its parents). Blocks are added greedily while each one
        lowers the CV loss by ``min_rel_gain``; the gate then judges them as one more round."""
        if self.time_col_ is not None or getattr(self, "_gate_veto", False):
            return cur_loss
        num_cols = [c for c in self.raw_cols_ if c in W.columns and c not in self.cat_cols_]
        lead = [c for c in imp.sort_values(ascending=False).index if c in num_cols]
        # As many leading numerics as keep a block's matrix near 20 million cells (80 MB of float32): every
        # pair among 18 vehicle-shape measures carried over (held out -24% log loss) where the leading 8 lost.
        m = self.block_numeric
        while m > 3 and m * (m - 1) / 2 * len(W) > 2e7:
            m -= 1
        lead = [c for c in lead if W[c].nunique() > 2][:m]
        have = {sp.name for sp in selected}
        blocks = []
        if len(lead) >= 3:
            for op in ("sub", "add", "div", "mul"):
                specs = [sp for sp in (Arith(op, a, b) for a, b in combinations(lead, 2)) if sp.name not in have]
                if specs:
                    blocks.append((op, specs))
        # Every low-cardinality categorical as its frequency: a number the trees split freely, where
        # LightGBM's categorical splits need min_data_per_group rows a level (Tic-Tac-Toe's nine squares,
        # 766 rows: held-out log loss -40% with the counts).
        low = [c for c in self.raw_cols_ if c in self.cat_cols_ and c in W.columns and c not in self.hc_cols_]
        counts = [sp for sp in (Count([c]) for c in low) if sp.name not in have]
        if len(counts) >= 2:
            blocks.append(("count", counts))
        # Each leading numeric against the mean of its group, for the leading low-cardinality keys
        # (Deep Feature Synthesis's per-category aggregates, which Featuretools won small tables with).
        keys = [c for c in imp.sort_values(ascending=False).index
                if c in low or (c in num_cols and 3 <= W[c].nunique() <= 32)][:4]
        vals_ = [c for c in lead if W[c].nunique() > 10][:6]
        groups = [sp for sp in (GroupStat(k, v, st) for k in keys for v in vals_ if v != k for st in ("mean", "dev"))
                  if sp.name not in have]
        if groups:
            blocks.append(("group", groups))
        if not blocks:
            return cur_loss
        r_next = max((sp.round_ for sp in selected), default=-1) + 1
        cols = list(W.columns)
        F = W
        best = cur_loss
        for op, specs in blocks:
            width = len(cols) + len(specs)
            if self._time_left() < self._cv_s * width / max(self._cv_w, 1):
                self._log(f"arithmetic blocks: budget left too small for the {op} block; stopping")
                break
            self._set_wu(F)
            src = F if self.WU_ is None else self.WU_  # label-free: over every row whose features are known
            vals = {sp.name: sp.fit(src, None, self.ctx_).transform(F, self.ctx_) for sp in specs}
            self._set_wu(F, on=False)
            G = F.assign(**vals)
            M = (*self._lgb_matrix(self._model_frame(G)), list(G.columns))
            _, loss, _ = self._cv(None, yW, folds, mat=M)
            self._log(f"block {op} ({len(specs)} columns): CV loss "
                      f"{best:.6f} -> {loss:.6f} ({100 * (best - loss) / best:+.2f}%)")
            if (best - loss) / best >= self.min_rel_gain:
                for sp in specs:
                    sp.round_ = r_next
                    W[sp.name] = vals[sp.name]
                selected.extend(specs)
                F, cols, best = W, list(W.columns), loss
                r_next += 1
        return best

    def _sparse_te(self, X) -> bool:
        """A selected target statistic keyed on a thin key (under 20 rows a level). Its
        out-of-fold values carry each training row's own label against its group's (KDD Cup
        2012: within-group correlation -0.16 to -0.32), which group statistics on the same
        key let the model read; held-out rows have no such signal (0.7500 -> 0.7476 AUC with
        both families). The family pass finds the set without the pair. Only on large gates:
        on Amazon's 5k gate rows it dropped families on noise (held-out -0.003 / -0.001 AUC)."""
        keys = set(self.key_cols_)
        for sp in self.selected_:
            if sp.target_dep:
                for p in sp.parents:
                    if p in keys and p in X.columns and len(X) / max(X[p].nunique(), 1) < 20:
                        return True
        return False

    def _gate_families(self, gate_loss, raw_rows, z_needed, best_set, best_l, budgeted=False):
        """When the whole set fails the gate (or always, under an embargo), drop feature families (target encodings, group
        statistics, event recencies, ...) one at a time, each time the one whose removal helps
        the gate rows most, and keep the best remaining set if it clears the gate by one more
        standard error than usual (the families were chosen on the gate rows). One family
        that does not carry over to later periods, e.g. Rossmann's target encodings, no
        longer sinks the others."""
        def fam(sp):
            return type(sp).__name__

        def closed(specs):
            # Keep only specs whose inputs are raw columns or outputs of kept specs.
            have = set(self.raw_cols_)
            out = []
            for sp in specs:
                if all(p in have for p in sp.parents):
                    out.append(sp)
                    have.update(sp.out_names())
            return out

        spent = 0.0

        def score(specs):
            nonlocal spent
            rows = gate_loss(self.raw_cols_ + [c for sp in specs for c in sp.out_names()])
            spent += getattr(self, "_gate_fit_s", 0.0)
            d = raw_rows - rows
            yc, tg = getattr(self, "_gate_yt", (None, None))
            dw = _winsor(d, yc)
            z = float(dw.mean() / (dw.std(ddof=1) / np.sqrt(len(dw)) + 1e-300))
            if tg is not None and _later_half(dw, tg) <= 0:
                z = min(z, 0.0)   # reversed over the later gate rows: never clears the gate
            return float(rows.mean()), z

        cur = closed(list(self.selected_))
        # A rescue the budget cannot pay for: the greedy pass costs about F(F+1)/2 gate fits (IEEE-CIS
        # full data: 27 fits, 3,710 s of wall for a 900 s budget). Target statistics are the features
        # whose search CV can be optimistic, so the label-free families are tried first, in one fit;
        # when they clear the gate they are kept and the pass goes on as a refinement under the
        # budget rule below. Otherwise the greedy pass runs to the end, as when it fits the budget.
        fams = {fam(sp) for sp in cur}
        est = sum(range(2, len(fams) + 1)) * getattr(self, "_gate_fit_s", 0.0)
        if not budgeted and self.time_budget and est > self.time_budget:
            free = closed([sp for sp in cur if not sp.target_dep])
            if free and len(free) < len(cur):
                l, z = score(free)
                self._log(f"  gate families: the pass takes about {est:.0f}s of work; label-free families first: "
                          f"loss={l:.6f} vs raw {raw_rows.mean():.6f} (z={z:+.2f}, need {z_needed + 1:.2f}): "
                          + ", ".join(sorted({fam(sp) for sp in free})))
                if l < best_l and z >= z_needed + 1:
                    best_set, best_l, cur, budgeted = list(free), l, free, True
        # The pass refits the gate models once per family per step (about F^2/2 fits): on a wide
        # table each fit is minutes of work (a held-back 1,930-column contest: 1,400 s for the first
        # step alone). When the whole set already passed the gate the pass only refines it, and it
        # gets one search budget of work: a step starts only when its fits, at the cost of the last
        # gate fit, still fit in what is left. When the whole set failed, the pass is a rescue
        # (IEEE-CIS keeps nothing but anchors without it) and runs to the end.
        while len({fam(sp) for sp in cur}) > 1:
            step = len({fam(sp) for sp in cur}) * getattr(self, "_gate_fit_s", 0.0)
            if budgeted and self.time_budget and spent + step > self.time_budget:
                self._log(f"  gate families: next step takes {step:.0f}s of work, {max(self.time_budget - spent, 0):.0f}s "
                          f"left of {self.time_budget:.0f}s; pass stops")
                break
            trials, over = [], False
            fams_now = sorted({fam(sp) for sp in cur})
            for f in fams_now:
                rest = closed([sp for sp in cur if fam(sp) != f])
                if rest:
                    trials.append((score(rest), f, rest))
                    # The step's estimate is the last gate fit's cost a family; dropping a family can make the fits
                    # dearer (a held-back 1,935-column contest: a step estimated within the budget took 1,560 s of
                    # wall, then the pass stopped with 0 s left). A refinement stops at the fit that runs it past
                    # its budget, so it overruns by one fit at most; a step that ran within it is never stopped.
                    if budgeted and self.time_budget and spent > self.time_budget:
                        over = True
                        break
            if not trials:
                break
            (l, z), f, rest = min(trials, key=lambda t: t[0][0])
            self._log(f"  gate without {f}: loss={l:.6f} vs raw {raw_rows.mean():.6f} (z={z:+.2f}, need {z_needed + 1:.2f})"
                      + (f" (of {len(trials)} of {len(fams_now)} families tried)" if over else ""))
            cur = rest
            # Stopped partway, the best set tried still counts under the same stricter rule: kept only when it clears
            # the gate by one more standard error and beats the best so far; else the best so far stands (a set that
            # passed the gate, never worse than the raw columns).
            if l < best_l and z >= z_needed + 1:
                best_set, best_l = list(cur), l
            if over:
                self._log(f"  gate families: {spent:.0f}s of work spent of {self.time_budget:.0f}s; pass stops")
                break
        return best_set, best_l

    def _fit_full(self, X, y, U=None):
        if self.recode_:
            self._fit_rank_maps(X, *([] if U is None else [U]))
        folds = self._te_folds(X, y, self.random_state + 1)
        # Shallow copies: the features below are added as new columns, never written into the
        # raw ones, so one copy of the raw columns (the context's) serves both frames.
        self.ctx_.extra_rows = None if U is None else U[self.raw_cols_]
        F = X.copy(deep=False)
        U = None if U is None else self.ctx_.extra_rows.copy(deep=False)
        # A feature's values on the unlabeled rows are only needed here when a later feature is
        # built from it; transform() computes them for whichever rows it is given. Neighbour
        # features queried every unlabeled row against every training row for nothing
        # (Santander Transaction: 335 of the final fit's 620 s, then again for the test file).
        used = {c for s in self.selected_ for c in s.parents}
        self.new_columns_ = [a[3] for a in self.anchors_] + [c for s in self.selected_ for c in s.out_names()]
        # Kept features are kept as float32 columns, each its own array (selecting them from F and casting
        # them were two more copies of them all: Instacart's 6.8 million training rows, killed at 13 GB; as
        # one matrix they could not leave memory a column at a time, see transform_train); only those later
        # features are built from stay in F as well (F's copy of them all was the final fit's peak under a
        # 10 GB limit).
        kept = {a[3]: np.array(F[a[3]].to_numpy(dtype=np.float32), copy=True) for a in self.anchors_}
        # Memory guard: each feature's fit over every row, priced against the memory free when it starts (the
        # search priced it over its search rows only), over the training and unlabeled rows both (a target model's
        # transform of new rows is fitted on every training row). One that would not fit is left out, with the
        # features built from it (Avito's 886k rows under 70% of its peak passed the gate, then were killed after).
        n_all = len(F) + (0 if U is None else len(U))
        wm, tn = getattr(self, "_sk_wm", None), getattr(self, "_text_nnz", None)
        left_out = set()
        for s in self.selected_:
            if left_out.intersection(s.parents):
                left_out.update(s.out_names())
                continue
            need = _spec_bytes(s, n_all, wm, tn) + 4 * len(F) * len(s.out_names())
            free = available_bytes()
            if need > free:
                self._log(f"memory guard: {s.name} left out of the final fit (needs about {need / 1e9:.2f} GB, "
                          f"{free / 1e9:.2f} GB free)")
                left_out.update(s.out_names())
                continue
            if s.target_dep:
                v = s.fit_transform_oof(F, y, self.ctx_, folds)
            elif U is not None:
                # Only the spec's columns: a copy of the whole growing frame per feature is
                # gigabytes at a time on Microsoft Malware's 9M rows.
                cols = list(dict.fromkeys(s.parents))
                s.fit(_concat_cats([F[cols], U[cols]]), None, self.ctx_)
                v = s.transform(F, self.ctx_)
            else:
                v = s.fit(F, y, self.ctx_).transform(F, self.ctx_)
            if U is not None and used.intersection(s.out_names()):
                vu = s.transform(U, self.ctx_)
                vu = np.asarray(vu, dtype=np.float32)
                for i, col in enumerate(s.out_names()):
                    U[col] = vu if np.ndim(vu) == 1 else vu[:, i]
            # float32, as the search saw them (and half the memory of the growing frame).
            v = np.asarray(v, dtype=np.float32)
            for i, col in enumerate(s.out_names()):
                vc = np.ascontiguousarray(v if np.ndim(v) == 1 else v[:, i])
                kept[col] = vc
                if col in used:
                    F[col] = vc
        if left_out:
            self.selected_ = [s for s in self.selected_ if not left_out.intersection(s.out_names())]
            self.new_columns_ = [c for c in self.new_columns_ if c not in left_out]
        self._train_cols = kept
        self._n_train = len(F)

    # ------------------------------------------------------------- transform
    def _prep(self, X):
        drop = [c for c in getattr(self, "shifted_", []) if c in X.columns]
        X = _renumbered(X.drop(columns=drop) if drop else X)
        for c in getattr(self, "date_cols_", []):
            X[c] = _to_days(X[c])
        for c in self.cat_cols_:
            X[c] = _as_str_frame_col(X[c])
        return self._add_anchors(self._compact(X))

    def _compact(self, X):
        """Short-text categoricals held as ``category`` (a code per row) rather than one string per
        row: the same values, an eighth of the memory (Microsoft Malware's 30 string columns over
        9M rows were 5 GB per copy of the frame)."""
        for c in self.cat_cols_:
            if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
                # Free text too when it repeats (MCTS: a game's rules, thousands of characters,
                # on each of its matches; one string per row was gigabytes per frame copy).
                if c not in self.text_cols_ or X[c].nunique() <= 0.5 * len(X):
                    X[c] = X[c].astype("category")
        return X

    def _detect_group(self, X, U):
        """A column whose test values are (almost) all unseen in training while it
        repeats within training: users, sessions or patients, when the test set holds
        new ones. Validation must then split whole groups, and target maps over the
        column cannot help."""
        if self.group_col != "auto":
            return self.group_col if self.group_col in X.columns else None
        if U is None or len(U) < 50:
            return None
        Up = self._prep(U)
        n, best, best_ov = len(X), None, 0.2
        for c in X.columns:
            # A column built from labels (an outcome history's counts) is no group, however new its test values.
            if c not in Up.columns or c in self.text_cols_ or c in getattr(self, "echo_cols_", ()):
                continue
            nu = X[c].nunique()
            if not (50 <= nu <= n / 2):
                continue
            if c not in self.cat_cols_:
                x = X[c].to_numpy(dtype=float)
                fin = x[np.isfinite(x)]
                if len(fin) < 0.9 * n or np.any(fin != np.round(fin)):
                    continue
            # A missing test value counts as seen when training has missing values too, as isin did under
            # pandas 3; pandas 2's isin never matched the set's NaN, so the choice depended on the version
            # (Riiid: a label history half missing in the test rows, 0.58 seen under pandas 3, 0.07 under 2).
            seen = X[c].dropna().unique()
            ov = float((Up[c].isin(seen) | (Up[c].isna() & X[c].isna().any())).mean())
            if ov < best_ov:
                best, best_ov = c, ov
        return best

    def _detect_order(self, X, key):
        """A numeric column that orders each group's rows: distinct within nearly every group
        and rising in row order (time steps of a breath, event times of a session)."""
        g = pd.factorize(X[key].to_numpy(), use_na_sentinel=False)[0]   # (missing keys: one group, not code -1)
        sizes = np.bincount(g)
        if np.median(sizes) < 5:
            return None
        rng = np.random.default_rng(0)
        pick = rng.choice(g.max() + 1, min(2000, g.max() + 1), replace=False)
        rows = np.flatnonzero(np.isin(g, pick))
        gs = g[rows]
        for c in X.columns:
            if c == key or c in self.cat_cols_ or not pd.api.types.is_numeric_dtype(X[c]):
                continue
            x = X[c].to_numpy(dtype=float)[rows]
            if not np.isfinite(x).all():
                continue
            same = gs[1:] == gs[:-1]
            rising = (np.diff(x) > 0)[same].mean() if same.any() else 0.0
            if rising > 0.98:
                return c
        return None

    def _row_ordered(self, X, y, U=None, max_rows=200_000) -> bool:
        """Most numeric columns carry over from each row to the next: their lag-1 correlation
        is beyond chance (z > 4, which a shuffled file passes in about 3 columns per 100,000),
        the labels are not sorted, and the test rows lie beyond the training rows rather than
        among them. Shuffled files never qualify; time-ordered ones do. A file kept in some other
        order (Forest Cover's, by area) whose test rows were drawn from among its rows does not."""
        num = [c for c in self.base_cols_ if c not in self.cat_cols_ and X[c].nunique() > 10][:200]
        if len(num) < 3 or len(X) < 1000:
            return False
        yy = np.asarray(y, dtype=float)[:max_rows]
        if np.mean(np.diff(yy) >= 0) > 0.99 or np.mean(np.diff(yy) <= 0) > 0.99:
            return False
        z = []
        for c in num:
            x = X[c].to_numpy(dtype=float)[:max_rows]
            a, b = x[:-1], x[1:]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 100 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                z.append(np.corrcoef(a[ok], b[ok])[0, 1] * np.sqrt(ok.sum()))
        if not (len(z) >= 3 and float(np.mean(np.array(z) > 4)) > 0.5):
            return False
        return U is None or self._test_beyond(X, U, num)

    def _test_beyond(self, X, U, num, m=2000, seed=0) -> bool:
        """Test rows sit farther from the training rows than training rows sit from each other.
        Equal-sized samples of training and test rows find their nearest neighbour among the other
        training rows (columns as ranks); a test drawn from among the training rows (a random
        holdout, however the file is ordered) gives the two the same distances, a later period
        gives the test larger ones. One-sided rank test, z > 4."""
        from scipy.stats import mannwhitneyu
        from sklearn.neighbors import NearestNeighbors
        cols = [c for c in num if c in U.columns][:50]
        if len(cols) < 3 or len(U) < 100:
            return True
        # Ranked one column at a time into single precision: the whole frame's float64 ranks, and its
        # interleaved copy, were 16 bytes per cell (Instacart's 8.5 million rows: 7 GB).
        A = np.empty((len(X) + len(U), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            s = pd.to_numeric(pd.concat([X[c], U[c]], ignore_index=True), errors="coerce").to_frame()
            A[:, j] = s.rank(pct=True).fillna(0.5).to_numpy(dtype=np.float32)[:, 0]  # (a frame's rank: as before on nullable ints)
        tr, te = A[:len(X)], A[len(X):]
        rng = np.random.default_rng(seed)
        k = min(m, len(te), len(tr) // 4)
        perm = rng.permutation(len(tr))
        q_tr, ref = tr[perm[:k]], tr[perm[k:]]
        if len(ref) > 100_000:
            ref = ref[rng.choice(len(ref), 100_000, replace=False)]
        q_te = te[rng.choice(len(te), k, replace=False)]
        nn = NearestNeighbors(n_neighbors=1).fit(ref)
        d_tr, d_te = nn.kneighbors(q_tr)[0][:, 0], nn.kneighbors(q_te)[0][:, 0]
        u = mannwhitneyu(d_te, d_tr, alternative="greater").statistic
        z = (u - k * k / 2) / np.sqrt(k * k * (2 * k + 1) / 12)
        self._log(f"file-order check: test rows {'beyond' if z > 4 else 'among'} the training rows "
                  f"(nearest-row distance z={z:+.1f})")
        return bool(z > 4)

    def _detect_time(self, X, U):
        if self.time_col != "auto":
            return self.time_col if self.time_col in X.columns else None
        if U is None or len(U) < 50:
            return None
        best, best_frac = None, 0.0
        Up = self._prep(U)
        self.time_like_ = []
        for c in X.columns:
            if c in self.cat_cols_ or c not in Up.columns:
                continue
            # Dates qualify at any granularity (many rows share a day); other numerics
            # need many levels, or test values entirely beyond the training range.
            nu = X[c].nunique()
            dense = c in self.date_cols_ or nu >= 0.05 * len(X)
            if not dense and nu < 20:
                continue
            x, u = X[c].to_numpy(dtype=float), Up[c].to_numpy(dtype=float)
            if np.isfinite(x).mean() < 0.99 or np.isfinite(u).mean() < 0.99:
                continue
            # Test rows later than (almost) every training row.
            frac = float(np.mean(u > np.nanquantile(x, 0.99)))
            if frac > 0.9:
                self.time_like_.append(c)  # week numbers, period ids: test levels are all new
            if frac > (0.9 if dense else 0.98) and (frac, c in self.date_cols_) > (best_frac, False):
                best, best_frac = c, frac
        return best

    # ------------------------------------------------------------- entities
    def _label_echo_cols(self, X, y, min_corr=0.2, max_lag=5, max_rows=300_000):
        """Columns that carry earlier rows' labels. Rows of an entity (an ID-like column, or a
        pair of them such as student and question) are taken in file order, which event logs
        keep chronological. Two signatures:

        - running statistic: the change from a row to the entity's next row tracks the first
          row's label (running mean, count or accuracy of past answers);
        - lagged label: the value ``k`` rows later (k <= ``max_lag``) tracks this row's label
          far more than the row's own value does (the previous, 2nd previous ... answer).

        Statistics over an entity's other rows would hand a row its own label through the
        later rows' values, so these columns stay out of every cross-row feature."""
        if y is None or self.n_classes_ > 2 or not self.id_cols_:
            return set()
        yv = np.asarray(y, dtype=float)
        n = len(X)
        num = [c for c in X.columns if c not in self.cat_cols_ and c not in self.id_cols_]
        if not num:
            return set()
        ids = list(self.id_cols_[:3])
        codes = {k: pd.factorize(X[k])[0] for k in ids}
        groups = [codes[k] for k in ids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                gi, gj = codes[ids[i]], codes[ids[j]]
                pair = pd.factorize(gi.astype(np.int64) * (gj.max() + 2) + gj)[0]
                pair[(gi < 0) | (gj < 0)] = -1
                groups.append(pair)
        rng = np.random.default_rng(0)

        def corr(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 200 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
                return 0.0
            r = np.corrcoef(a[m], b[m])[0, 1]
            return float(r) if np.isfinite(r) else 0.0

        echo = set()
        for g in groups:
            # next occurrence of the same entity, rows in file order
            order = np.argsort(g, kind="stable")
            nxt = np.full(n, -1)
            same = (g[order[1:]] == g[order[:-1]]) & (g[order[:-1]] >= 0)
            nxt[order[:-1][same]] = order[1:][same]
            rows = np.flatnonzero(nxt >= 0)
            if len(rows) < 200:
                continue
            if len(rows) > max_rows:
                rows = np.sort(rng.choice(rows, max_rows, replace=False))
            # k-th next occurrence for the sampled rows
            ahead = [rows]
            for _ in range(max_lag):
                prev = ahead[-1]
                cur = np.where(prev >= 0, nxt[np.maximum(prev, 0)], -1)
                ahead.append(cur)
            r1, y0 = ahead[1], yv[rows]
            y1 = yv[r1]
            for c in num:
                if c in echo:
                    continue
                v = X[c].to_numpy(dtype=float)  # one column at a time: all at once were 8 bytes per cell
                # running statistic of past labels (sign too: a running mean over many
                # earlier answers moves by little, but always toward the latest answer)
                d = v[r1] - v[rows]
                for dd in (d, np.sign(d)):
                    r = corr(dd, y0)
                    if abs(r) >= min_corr and abs(r) >= 3 * abs(corr(dd, y1)):
                        echo.add(c)
                        break
                if c in echo:
                    continue
                # lagged label: the entity's k-th next row holds this row's label
                r_own = abs(corr(v[rows], y0))
                for k in range(1, max_lag + 1):
                    rk = ahead[k]
                    ok = rk >= 0
                    if ok.sum() < 200:
                        break
                    r = abs(corr(v[rk[ok]], y0[ok]))
                    if r >= min_corr and r >= 3 * r_own:
                        echo.add(c)
                        break
        return echo

    def _id_columns(self, X, cap=8):
        """Integer or categorical columns with many levels that repeat: card numbers,
        addresses, customer or store ids. Most levels first."""
        n, out = len(X), []
        for c in X.columns:
            if c in getattr(self, "date_cols_", []) or c in getattr(self, "text_cols_", []):
                continue
            nu = X[c].nunique()
            if nu < 100:
                continue
            vc = X[c].value_counts(normalize=True)
            # Over n/5 levels it is still an id when most rows sit in levels seen three or more
            # times: click logs' device IPs (Avazu: a quarter of rows are distinct levels, yet 78%
            # of rows belong to repeating devices). Near-unique row ids fail this.
            if nu > n / 5 and vc[vc * n >= 3].sum() < 0.5:
                continue
            if vc.iloc[0] > 0.5:  # mostly one value: a count or flag, not an id
                continue
            if c not in self.cat_cols_:
                x = X[c].to_numpy(dtype=float)
                fin = x[np.isfinite(x)]
                if len(fin) < 0.5 * n or np.any(fin != np.round(fin)):
                    continue
                # Counts and amounts get rarer as they grow; codes do not.
                if pd.Series(vc.index.to_numpy(dtype=float)).corr(pd.Series(vc.to_numpy(dtype=float)),
                                                                  method="spearman") < -0.3:
                    continue
            out.append((nu, c))
        return [c for _, c in sorted(out, reverse=True)][:cap]

    def _code_columns(self, X, cap=12):
        """Integer columns past the key cardinality whose levels behave like codes, not
        quantities: thinly spread over a wide range (Amazon's ROLE_TITLE: 343 levels
        between 117,000 and 311,000), not piled up at the low end as counts and amounts
        are, and not rounded to tens as prices and areas written by people are."""
        n, out = len(X), []
        for c in X.columns:
            if c in self.cat_cols_ or c in getattr(self, "date_cols_", []):
                continue
            nu = X[c].nunique()
            # From 300 levels: Kick's WarrantyCost (281 prices of warranty plans) lost
            # 0.4% log loss as a key on all three seeds.
            if not (max(300, self.max_key_cardinality) <= nu <= n / 3):
                continue
            x = X[c].to_numpy(dtype=float)
            fin = x[np.isfinite(x)]
            if len(fin) < 0.5 * n or np.any(fin != np.round(fin)):
                continue
            if X[c].value_counts(normalize=True).iloc[0] > 0.5:
                continue
            u = np.unique(fin)
            if np.gcd.reduce(np.diff(u).astype(np.int64)) > 1:  # quantised: a measurement in steps
                continue
            span = u[-1] - u[0]
            if span / len(u) < 10 or np.quantile(u, 0.9) - u[0] < 0.2 * span or np.mean(u % 10 == 0) >= 0.2:
                continue
            out.append((nu, c))
        return [c for _, c in sorted(out, reverse=True)][:cap]

    def _find_anchors(self, X, max_anchors=3, max_ratio=0.8):
        """Find (time, scale, delta) triples where ``floor(time / scale) - delta`` is
        constant within entities, as an account-opening day is while "days since
        opening" keeps growing. Test: grouped by an ID-like column, ``t - delta``
        takes clearly fewer distinct values than the control ``t + delta``; for a
        delta unrelated to time the two counts match."""
        n = len(X)
        if n < 2000 or not self.id_cols_:
            return []
        self._t0 = getattr(self, "_t0", time.time())
        S = X.sample(min(n, 100_000), random_state=self.random_state) if n > 100_000 else X
        num = [c for c in X.columns if c not in self.cat_cols_ and c not in set(self.label_cols)]
        vals = {c: S[c].to_numpy(dtype=float) for c in num}
        def is_int(v):
            f = v[np.isfinite(v)]
            return len(f) > 0 and not np.any(f != np.round(f))
        # Wide anonymous tables have hundreds of time-like columns: try the most distinct
        # ones, within a minute of work (Loan Default's 700 columns took hours otherwise).
        times = sorted([c for c in num if X[c].nunique() > 0.2 * n and np.nanmin(vals[c]) >= 0],
                       key=lambda c: -X[c].nunique())[:20]
        key_work = [0.0]   # work-clock seconds spent counting keys
        deltas = sorted([c for c in num if c not in getattr(self, "echo_cols_", ()) and is_int(vals[c]) and 50 <= X[c].nunique() <= 0.2 * n],
                        key=lambda c: -X[c].nunique())[:80]
        ids = []
        for c in self.id_cols_[:3]:
            v = S[c]
            ids.append(pd.factorize(v)[0].astype(np.int64))
        def n_pairs(idc, v):
            ok = np.isfinite(v)
            if ok.sum() < 0.3 * len(v):
                return 0
            # Codes, not values: columns of huge magnitude overflow int64.
            code = idc[ok] * (1 << 32) + pd.factorize(v[ok])[0].astype(np.int64)
            key_work[0] += _KEY_ROW_S * len(code)
            return len(np.unique(code))
        found = []
        for t in times:
            if key_work[0] > 60:
                break
            tv = vals[t]
            for s in (1, 60, 3600, 86400, 604800):
                span = (np.nanmax(tv) - np.nanmin(tv)) / s
                if span < 10:
                    continue
                for d in deltas:
                    if d == t:
                        continue
                    dv = vals[d]
                    if np.nanmax(dv) - np.nanmin(dv) < 0.5 * span:
                        continue
                    a = np.floor(tv / s) - dv
                    ratios = []
                    for idc in ids:
                        nd = n_pairs(idc, a + 2 * dv)
                        if nd:
                            ratios.append(n_pairs(idc, a) / nd)
                    if ratios and min(ratios) < max_ratio:
                        found.append((min(ratios) / np.isfinite(dv).mean(), t, s, d))
        found.sort()
        self.anchor_found_ = list(found)
        out, used = [], set()
        for r, t, s, d in found:
            if d in used:
                continue
            used.add(d)
            out.append((t, s, d, f"anchor__{d}__{t}_{s}"))
            self._log(f"entity anchor: floor({t}/{s}) - {d} (score {r:.2f})")
            if len(out) >= max_anchors:
                break
        return out

    def _add_anchors(self, X):
        for t, s, d, name in getattr(self, "anchors_", []):
            X[name] = np.floor(X[t].to_numpy(dtype=float) / s) - X[d].to_numpy(dtype=float)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add features to new rows using statistics from all training rows."""
        with _one_blas_thread():
            return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._named(X)
        F = self._prep(X)
        for s in self.selected_:
            v = s.transform(F, self.ctx_)
            for j, col in enumerate(s.out_names()):
                F[col] = v if np.ndim(v) == 1 else v[:, j]
        out = self._recode_out(X)
        for c in self.new_columns_:
            out[c] = F[c].astype(np.float32).to_numpy()
        return out

    @staticmethod
    def _named(X: pd.DataFrame) -> pd.DataFrame:
        names = _unique_names(X.columns)
        return X if list(X.columns) == names else X.set_axis(names, axis=1)

    def _recode_out(self, X):
        # Without copying the data: reset_index and drop each made a deep copy of the whole frame
        # (two extra copies of a wide float64 table, the 15 GB limit on a 1,900-column contest).
        out = _renumbered(X)
        for c in getattr(self, "shifted_", []):
            if c in out.columns:
                del out[c]
        if self.recode_:
            for c in self.rank_maps_:
                out[c] = self._rank(out[c], c)
        return out

    def transform_train(self, X: pd.DataFrame, release: bool = False) -> pd.DataFrame:
        """Training rows with features as fitted (target encodings out-of-fold). ``release``: each fitted
        column leaves this object once copied into the output (the output and a full copy of the fitted
        columns next to it were 1.6 GB more on Instacart's 6.8 million rows, killed under a 10 GB limit);
        transform_train cannot be called again after it."""
        out = self._recode_out(self._named(X))
        if len(out) != self._n_train:
            raise ValueError("transform_train expects the exact frame passed to fit")
        for c in dict.fromkeys(self.new_columns_):
            out[c] = self._train_cols[c]
            if release:
                del self._train_cols[c]
        return out

    def fit_transform(self, X, y, X_unlabeled=None):
        return self.fit(X, y, X_unlabeled).transform_train(X)

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)


def _holdout_rows(task, Ftr, ytr, Fva, yva, seed, threads, rounds=3000):
    """Per-row loss on ``Fva`` of a LightGBM fitted on ``Ftr`` (early-stopped on a fifth of it, then refitted)."""
    import lightgbm as lgb
    Ftr, Fva = Ftr.copy(), Fva.copy()
    for c in Ftr.columns:
        if not pd.api.types.is_numeric_dtype(Ftr[c]) or pd.api.types.is_bool_dtype(Ftr[c]):
            a, b = (v.astype("string").fillna("__na__").astype(object) for v in (Ftr[c], Fva[c]))
            levels = pd.Index(pd.unique(a))
            Ftr[c], Fva[c] = pd.Categorical(a, categories=levels), pd.Categorical(b, categories=levels)
    Ftr.columns = Fva.columns = [f"f{i}" for i in range(Ftr.shape[1])]
    k = int(max(ytr.max(), yva.max())) + 1 if task == "multiclass" else 0
    P = dict(objective={"binary": "binary", "multiclass": "multiclass", "regression": "regression"}[task],
             learning_rate=0.05, num_leaves=31, min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, num_threads=threads, verbose=-1, seed=seed, max_cat_to_onehot=8,
             cat_smooth=20, **({"num_class": k} if k else {}))
    perm = np.random.default_rng(seed).permutation(len(Ftr))
    cut = int(0.8 * len(Ftr))
    a, b = np.sort(perm[:cut]), np.sort(perm[cut:])
    m = lgb.train(P, lgb.Dataset(Ftr.iloc[a], ytr[a]), rounds, valid_sets=[lgb.Dataset(Ftr.iloc[b], ytr[b])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    full = lgb.train(P, lgb.Dataset(Ftr, ytr), max(1, int(m.best_iteration * 1.1)))
    return _row_loss(task, yva, full.predict(Fva, raw_score=True))
