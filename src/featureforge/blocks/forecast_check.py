"""Held-out check of the forecasting block before it replaces FeatureForge's search.

The last labelled periods play the test file; the block is kept only when a quick model with it
beats one without it on those rows by the gate's paired z rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from featureforge.utils.text import as_text


_CHECK_CHUNK_ROWS = 250_000   # rows the forecasting block is transformed at a time (check and output)


def _check_matrix(frames: list[pd.DataFrame], picks: list[tuple[int, np.ndarray | None]],
                  out: list[np.ndarray] | None = None, at: int = 0) -> list[np.ndarray]:
    """Rows of the frames as float32 matrices for a quick model: dates as days, text and categories as codes
    shared by all the frames. ``picks`` lists (frame, rows) per matrix; rows None takes the whole frame.
    With ``out``, the columns are written into those matrices from column ``at`` on."""
    cols = list(frames[0].columns)
    if out is None:
        out = [np.empty((len(frames[f]) if r is None else len(r), len(cols)), dtype=np.float32) for f, r in picks]
    for j, c in enumerate(cols):
        ss = [F[c] for F in frames]
        if pd.api.types.is_datetime64_any_dtype(ss[0]):
            naive = [s_.dt.tz_convert(None) if getattr(s_.dt, "tz", None) is not None else s_ for s_ in ss]
            vs = [((s_ - pd.Timestamp("2000-01-01")).dt.total_seconds() / 86400).to_numpy(dtype=np.float32) for s_ in naive]
        elif pd.api.types.is_numeric_dtype(ss[0]) and not isinstance(ss[0].dtype, pd.CategoricalDtype):
            vs = [s_.to_numpy(dtype=np.float32, na_value=np.nan) for s_ in ss]
        elif all(isinstance(s_.dtype, (pd.StringDtype, pd.CategoricalDtype)) for s_ in ss):
            # Each frame's distinct values are coded once and their text sorted across frames: the codes the
            # rows' text gives below, without a Python string per row (the check's raw matrices were Rohlik's
            # peak, 6.1 GB of its 6.5 GB container, in this conversion).
            parts = [pd.factorize(s_, use_na_sentinel=True) for s_ in ss]
            # (a missing value's text is "nan", as as_text writes it; code -1 reads the last entry)
            texts = [np.asarray(as_text(pd.Series(u)), dtype=object).tolist() + (["nan"] if (k < 0).any() else [])
                     for k, u in parts]
            order = pd.Index(sorted(set().union(*texts)))
            vs = [order.get_indexer(np.array(t_ + ["nan"], dtype=object)).astype(np.float32)[k]
                  for (k, _), t_ in zip(parts, texts)]
        else:
            t = [as_text(s_) for s_ in ss]
            codes, _ = pd.factorize(pd.concat(t, ignore_index=True), sort=True)
            codes = np.where(codes < 0, np.nan, codes).astype(np.float32)
            vs, i = [], 0
            for t_ in t:
                vs.append(codes[i:i + len(t_)]); i += len(t_)
        for M, (f, r) in zip(out, picks):
            M[:, at + j] = vs[f] if r is None else vs[f][r]
    return out


def forecast_check(ff, X: pd.DataFrame, y: np.ndarray, kw: dict, task: str | None, log_target: bool,
                   z_needed: float = 1.0, max_rows: int = 2_000_000) -> tuple[bool, str]:
    """Family-level held-out check of the forecasting block before it replaces the search.

    The last labelled periods play the test file: an origin as far before the last labelled
    period as the test's longest horizon, and the rows at the test's horizons after it held
    out. The family is fitted again on the rows up to that origin, with the held-out rows as its
    unlabeled rows, exactly as on the real test; one quick model on the raw columns and one on
    the raw columns plus the block are fitted on the same earlier rows (early-stopped on their
    latest fifth) and scored on the held-out rows. The block is kept when its paired per-row gain
    clears ``z_needed`` standard errors (FeatureForge's gate rule); otherwise the normal search
    runs. Returns (keep, note); a window the data cannot hold keeps the block, as before."""
    import lightgbm as lgb
    from featureforge.blocks.forecast import ForecastFeatures
    p = ff._periods(X)
    o = ff.T_ - ff.hmax_
    fit = np.isfinite(p) & (p <= o)
    held = np.isfinite(p) & (p >= o + ff.hmin_) & (p <= ff.T_)
    if fit.sum() < 1000 or held.sum() < 200 or len(np.unique(p[fit])) < ff.hmax_:
        return True, f"window too short to check (history {len(np.unique(p[fit]))} periods, horizon {ff.hmax_})"
    Xf, Xh = X[fit].reset_index(drop=True), X[held].reset_index(drop=True)
    yf, yh = y[fit], y[held]
    probe = ForecastFeatures(**{**kw, "verbose": False}).fit(Xf, yf, Xh)
    if not probe.active_ or probe.entity_ != ff.entity_:
        return True, f"family not found on the earlier rows ({getattr(probe, 'reason_', 'other entity')}); kept"
    if task is None:
        task = "binary" if len(np.unique(y[np.isfinite(y)])) == 2 else "regression"
    yt = lambda v: np.log1p(np.maximum(v, 0)) if log_target else v
    tf, th = yt(yf.astype(float)), yt(yh.astype(float))
    order = np.argsort(p[fit], kind="stable")
    if len(order) > max_rows:   # the latest rows before the origin
        order = order[-max_rows:]
    n_va = max(1, len(order) // 5)
    es_tr, es_va = np.sort(order[:-n_va]), np.sort(order[-n_va:])
    params = dict(objective={"binary": "binary", "multiclass": "multiclass"}.get(task, "regression"),
                  learning_rate=0.1, num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=100, verbose=-1, num_threads=4, seed=0, deterministic=True, force_col_wise=True)
    if task == "multiclass":
        params["num_class"] = int(np.nanmax(y)) + 1
    # The raw columns are converted once and only the rows each quick model reads are built; the block
    # arm's matrices are the raw arm's with the block's columns written after them. Each model's
    # training rows live only as LightGBM's bins while it trains.
    picks = [(0, es_tr), (0, es_va), (1, None)]
    raw = _check_matrix([Xf, Xh], picks)
    # The block's inputs from the earlier rows are read now, so their frame is not held through both arms.
    block_rows = [_CHECK_CHUNK_ROWS]
    chunks = probe.transform_chunks(Xf, lambda: block_rows[0])
    del Xf
    losses = {}
    for arm in ("raw", "block"):
        if arm == "block":
            # The block's columns are all float32; the earlier rows are transformed a block of rows at a
            # time straight into the rows each model reads (Store Sales: the whole block for 2.9M rows
            # was the check's peak).
            # (the held-out rows' block too, a block of rows at a time: whole, its working columns were the
            # step at M5's 4.6 GB kill on pandas 3)
            try:
                _, Fh = probe.transform_float32(Xh, _CHECK_CHUNK_ROWS, guard=True, log=None)
            except MemoryError as e:
                return False, f"memory guard: {e} (the check's held-out rows); the block is not used"
            del Xh
            nr, nf = raw[0].shape[1], Fh.shape[1]
            # A memory guard: the models' matrices (each the raw columns and the block's, made while its raw
            # matrix is let go) and one block of the earlier rows' working columns against the memory free now,
            # with fewer rows a block when short, and the block not used when the matrices would not fit (the
            # check was Rohlik's peak: 6.24 GB of its 6.5 GB container).
            from featureforge.utils.memory import available_bytes
            per_row = nf * probe._BLOCK_BYTES_PER_CELL + probe._BLOCK_BYTES_PER_ROW
            need = sum(len(R) for R in raw) * nf * 4 + max(R.nbytes for R in raw)
            free = available_bytes()
            if free - need < 10_000 * per_row:
                return False, (f"memory guard: the check's matrices need {need / 1e9:.2f} GB more, "
                               f"{free / 1e9:.2f} GB free; the block is not used")
            fit_rows = int((free - need) // 2 // per_row)
            if fit_rows < block_rows[0] and len(es_tr) + len(es_va) > fit_rows:
                block_rows[0] = max(10_000, fit_rows)
                print(f"memory guard: forecasting check builds {block_rows[0]} earlier rows at a time "
                      f"(matrices {need / 1e9:.2f} GB, {free / 1e9:.2f} GB free)", flush=True)
            wide = []
            for i in range(len(raw)):
                W = np.empty((len(raw[i]), nr + nf), dtype=np.float32)
                W[:, :nr] = raw[i]
                raw[i] = None
                wide.append(W)
            del W
            raw = None
            wide[2][:, nr:] = Fh
            del Fh
            for a0, Fc in chunks:
                V = Fc.to_numpy(dtype=np.float32)
                for W, rows in ((wide[0], es_tr), (wide[1], es_va)):
                    lo, hi = np.searchsorted(rows, [a0, a0 + len(Fc)])
                    W[lo:hi, nr:] = V[rows[lo:hi] - a0]
                del V, Fc
            del chunks, probe
            mats = wide
        else:
            mats = raw
        dtr = lgb.Dataset(mats[0], tf[es_tr], params=params).construct()
        dva = lgb.Dataset(mats[1], tf[es_va], reference=dtr, params=params).construct()
        Mh = mats[2]
        if arm == "block":
            del mats, wide
        b = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        q = b.predict(Mh, num_iteration=b.best_iteration)
        if task == "regression":
            losses[arm] = (th - q) ** 2
        elif task == "binary":
            q = np.clip(q, 1e-7, 1 - 1e-7)
            losses[arm] = -(th * np.log(q) + (1 - th) * np.log(1 - q))
        else:
            losses[arm] = -np.log(np.clip(q[np.arange(len(q)), th.astype(int)], 1e-7, 1))
        del dtr, dva, Mh, b
    d = losses["raw"] - losses["block"]
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-300))
    keep = bool(d.mean() > 0 and z >= z_needed)
    note = (f"held-out check on {int(held.sum())} rows after origin period {o} ({int(fit.sum())} earlier rows): "
            f"loss raw {losses['raw'].mean():.6f} vs with the block {losses['block'].mean():.6f} (z={z:+.2f}, need {z_needed:.2f})")
    return keep, note
