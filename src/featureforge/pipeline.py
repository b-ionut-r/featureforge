"""End-to-end feature pipeline: related tables, structure-triggered blocks and the FeatureForge search.

``build_features`` reads a training and a test table (plus any child tables), and writes
``train_features.parquet`` and ``test_features.parquet``: the original columns, every block that
switched on from the data's structure, and the columns FeatureForge's search kept. Features only;
train any model on the outputs.

Stages, each on only when the data has the structure it needs:

1. **Related tables.** Each child table (``tables``) is aggregated per key (``RelatedTables``), or as of
   each main row when the main table repeats the key and both carry a comparable time column
   (``asof_features``); price paths, latest-state history, outcome history of event logs, same-code
   matches and lookups follow. Out-of-fold child-row models are added when a quick CV says they help.
   A table too big for memory is read a key range at a time.
2. **Lists, JSON and per-unit amounts** of the main table, then **intraday panels**.
3. **Forecasting.** With a date, repeating entities and test rows after the training period, the
   forecasting block is built and checked on a held-out window shaped like the test; when kept it
   replaces the search.
4. **FeatureForge search** on the main columns plus the related columns a quick model ranks highest.

Every stage has a memory guard: when its working set would not fit in the memory free now it is
skipped (or run in a cheaper, output-identical mode) with a log line, never killed.

From the command line, ``featureforge --train train.csv --test test.csv --target y`` (see ``cli.py``)::

    featureforge --train application_train.csv --test application_test.csv \\
        --target TARGET --id SK_ID_CURR \\
        --table bureau=bureau.csv --table prev=previous_application.csv --out-dir features/

Each table is ``name=path[:key[:time]]``. The key defaults to the main table's id column (or the one
column it shares with the main table); the time column defaults to the first column named like a date
or a day/month count.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from featureforge.blocks.forecast_check import _CHECK_CHUNK_ROWS, forecast_check
from featureforge.blocks.history import recent_window_features, event_log_features, history_features, repeats
from featureforge.blocks.lists import as_lists, json_fields, list_features, unit_ratio_features
from featureforge.blocks.panel import panel_features, panel_helps
from featureforge.blocks.relational import (Child, code_target_features, RelatedTables, bag_features, asof_features, child_model_features, lookup_features,
                                            match_features, time_values)
from featureforge.blocks.returns import return_features
from featureforge.forge import FeatureForge
from featureforge.io import normalise, read, release_arrow
from featureforge.utils.memory import _trim_heap, aggregation_bytes, available_bytes, chunks, memory_bytes
from featureforge.utils.text import as_text

TIME_HINTS = ("days", "day", "month", "date", "time", "week", "year")


def log_y(y: pd.Series) -> np.ndarray:
    """The search's log target: log of a small positive target (FeatureForge does the same), else log1p."""
    v = y.to_numpy(dtype=float)
    return np.log(v) if np.nanmin(v) > 0 and np.nanmedian(v) < 1 else np.log1p(v)


def parse_table(spec: str, main: pd.DataFrame, id_col: str | None) -> Child:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    key = parts[1] if len(parts) > 1 and parts[1] else None
    need = aggregation_bytes(parts[0], key) if key else None
    big = need is not None and need > STREAM_SHARE * memory_bytes() and stream_rows(parts[0]) > STREAM_MIN_ROWS
    if big and key in main.columns and main[key].is_unique:
        # (one main row per key only: a key repeating in the main table is an event log aggregated as of each row)
        # Too big to aggregate whole (Optiver's 160M-row order book): read a key range at a time.
        from featureforge.utils.memory import _dataset
        df = normalise(_dataset(parts[0]).head(200_000).to_pandas())
        tcol = parts[2] if len(parts) > 2 else next(
            (c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])
             and any(h in c.lower() for h in TIME_HINTS)), None)
        drop = [c for c in df.columns if c != key and (c.upper().startswith(("SK_ID", "ID_")) or c.lower().endswith("_id"))]
        print(f"table {name}: {_dataset(parts[0]).count_rows()} rows, key={key}, read by key range "
              f"(whole: {need / 2**30:.0f} GB to aggregate)", flush=True)
        return Child(name, df, key=key, time=tcol, drop=drop, source=parts[0])
    df = read(parts[0])
    if key is None:
        shared = [c for c in df.columns if c in main.columns]
        key = id_col if id_col in df.columns else (shared[0] if len(shared) == 1 else None)
    if key is None:
        # No shared column: a lookup table (components, products) whose first column's values
        # appear in main-table columns, possibly numbered slots (component_id_1..8).
        print(f"table {name}: {df.shape}, lookup on {df.columns[0]}", flush=True)
        return Child(name, df, key=df.columns[0], drop=["__lookup__"])
    tcol = parts[2] if len(parts) > 2 else next(
        (c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])
         and any(h in c.lower() for h in TIME_HINTS)), None)
    # Other id-like columns (foreign keys to further tables) are not features.
    drop = [c for c in df.columns if c != key and (c.upper().startswith(("SK_ID", "ID_")) or c.lower().endswith("_id"))]
    print(f"table {name}: {df.shape}, key={key}, time={tcol}, ignored ids={drop}", flush=True)
    return Child(name, df, key=key, time=tcol, drop=[c for c in drop if c != key])


STREAM_SHARE, STREAM_BUDGET_SHARE = 0.4, 0.2   # of the machine's memory
# Whole tables aggregated fine up to Elo's 29M transactions (and Amex's 5.5M wide statements); Optiver's book ran a
# 15 GB machine out of memory from 34M rows on.
STREAM_MIN_ROWS = 30_000_000


def stream_rows(path: str) -> int:
    from featureforge.utils.memory import _dataset
    return _dataset(path).count_rows()


# pandas 2 copies unless told not to; pandas 3 never does (copy-on-write) and deprecates the keyword.
_NOCOPY = {"copy": False} if int(pd.__version__.split(".")[0]) < 3 else {}


def _rows(F: pd.DataFrame, a: int = 0, b: int | None = None) -> pd.DataFrame:
    """Rows a..b of F (all by default) numbered from 0, sharing F's blocks: reset_index copied, and consolidated,
    the whole frame (Store Sales' and M5's output: their pipeline's peak)."""
    b = len(F) if b is None else b
    v = F if (a, b) == (0, len(F)) else F.iloc[a:b]
    if not v.index.equals(pd.RangeIndex(b - a)):
        v = v.set_axis(pd.RangeIndex(b - a), axis=0, **_NOCOPY)
    return v


def streamed_features(ch: Child, history: bool) -> list[pd.DataFrame]:
    """RelatedTables, price-path and latest-state history features of a streamed child, a key range at a time
    (each key's rows sit in one range, so this equals aggregating the table whole)."""
    rt, plan, hist = RelatedTables([], drop_const=False), {}, None
    F, R, H = [], [], []
    t = time.time()
    for i, df in enumerate(chunks(ch.source, ch.key, max(1, int(STREAM_BUDGET_SHARE * memory_bytes())), normalise)):
        c = Child(ch.name, df, key=ch.key, time=ch.time, drop=ch.drop)
        F.append(rt._aggregate(c, ch.key))
        R.append(return_features(df, ch.key, ch.name, ch.time, plan=plan, ignore=ch.drop))
        hist = (history and repeats(c)) if hist is None else hist
        if hist:
            H.append(history_features(c))
        print(f"table {ch.name}: key range {i + 1} ({len(df)} rows) aggregated, {time.time() - t:.0f}s", flush=True)
        del df, c
    F = pd.concat(F)
    F = F[[c for c in F.columns if F[c].notna().mean() > 0.01 and F[c].nunique() > 1]]
    return [x for x in (F, pd.concat(R) if R and R[0].shape[1] else None, pd.concat(H) if H else None) if x is not None]


def date_col(df: pd.DataFrame) -> str | None:
    """The first text column holding calendar dates (2013-04-22, 2016-01-01 19:00:00)."""
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = as_text(df[c].dropna().head(1000))
        if len(v) and v.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.95:
            return c
    return None


def match_cols(main: pd.DataFrame, ch: Child) -> list[str]:
    """Code columns a child shares with the main table besides the key (category, brand, company):
    integer or text codes with 20+ levels whose main-table values mostly occur in the child."""
    out = []
    for c in ch.df.columns:
        # An id the main row also carries (Instacart's product_id) is matched on, though never aggregated as a number.
        if c == ch.key or c == ch.time or c not in main.columns or any(h in c.lower() for h in TIME_HINTS):
            continue
        s = ch.df[c]
        if pd.api.types.is_float_dtype(s) and not np.all(np.mod(s.dropna().to_numpy()[:10000], 1) == 0):
            continue
        if s.nunique() < 20:
            continue
        vals = pd.unique(as_text(main[c].dropna()))
        if len(vals) and np.isin(vals, pd.unique(as_text(s.dropna()))).mean() >= 0.5:
            out.append(c)
    return out


def main_time(main: pd.DataFrame, ch: Child, given: str | None) -> str | None:
    """The main table's column that is comparable with the child's time column."""
    if given:
        return given
    if ch.time in main.columns:
        return ch.time
    cands = [c for c in main.columns if pd.api.types.is_numeric_dtype(main[c])
             and any(h in c.lower() for h in TIME_HINTS)]
    return cands[0] if len(cands) == 1 else None


CM_CHECK_PARENTS, CM_CHECK_ROWS = 60_000, 300_000


def cut_related(A: list, y: np.ndarray, n_train: int, task: Optional[str], room: float) -> list:
    """The related blocks cut to the columns a quick model uses most, when all of them would not fit in ``room``
    bytes: a LightGBM fit on up to 100,000 training rows of every block ranks the columns by gain, and the best
    are kept while they fit. Fewer related columns, never a kill (Home Credit Risk Stability: 1,189 related
    columns over 1.5M rows, 7.3 GB, killed joining them on the 15 GB bench). A run whose blocks fit never reads
    this."""
    import lightgbm as lgb
    from featureforge._lgb import _lgb_objective
    need = sum(frame_bytes(F) for F in A)
    if need <= room:
        return A
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(n_train, min(n_train, 100_000), replace=False))
    names = [c for F in A for c in F.columns]
    num = {c for F in A for c in F.columns if pd.api.types.is_numeric_dtype(F[c])}
    M = np.empty((len(rows), len(names)), np.float32)
    j = 0
    for F in A:
        for c in F.columns:
            M[:, j] = F[c].to_numpy(dtype=np.float32, na_value=np.nan)[rows] if c in num else np.nan
            j += 1
    yy = np.asarray(y)[rows]
    task = task or ("binary" if len(np.unique(yy)) == 2 else "regression")
    k = int(np.nanmax(yy)) + 1 if task == "multiclass" else 0
    p = dict(_lgb_objective(task, k), learning_rate=0.1, num_leaves=31, min_child_samples=50, feature_fraction=0.5,
             verbose=-1, seed=0, deterministic=True, force_col_wise=True)
    gain = lgb.train(p, lgb.Dataset(M, yy, free_raw_data=True), 200).feature_importance("gain")
    del M
    size = {c: frame_bytes(F[[c]]) for F in A for c in F.columns}
    keep, used = set(), 0.0
    for i in np.argsort(-gain, kind="stable"):
        c = names[i]
        if c not in num:      # (text kept as it was: the search reads it as categories)
            keep.add(c)
            used += size[c]
    for i in np.argsort(-gain, kind="stable"):
        c = names[i]
        if c in num and gain[i] > 0 and used + size[c] <= room:
            keep.add(c)
            used += size[c]
    print(f"memory guard: related columns cut to the {len(keep)} of {len(names)} a quick model uses most "
          f"(whole: {need / 1e9:.1f} GB, room for {room / 1e9:.1f} GB)", flush=True)
    out = []
    while A:
        F = A.pop(0)
        cols = [c for c in F.columns if c in keep]
        if cols:
            out.append(F[cols])
        del F
        _trim_heap()
    return out


def join_columns(A: list) -> pd.DataFrame:
    """pd.concat(A, axis=1) of frames on one index, emptying A. When the copy does not fit next to the parts and
    every column is float32, the parts are copied one at a time into one float32 block, each dropped once copied:
    the same frame, at the parts plus the largest one instead of twice the parts (Home Credit under 5.6 GiB: killed
    joining its related columns)."""
    need = sum(frame_bytes(F) for F in A)
    free = available_bytes()
    same = all(F.index.equals(A[0].index) and (F.dtypes == np.float32).all() for F in A)
    if need <= free or not same:
        out = pd.concat(A, axis=1)
        A.clear()
        return out
    print(f"memory guard: related columns joined a table at a time (whole: {need / 1e9:.1f} GB, "
          f"{free / 1e9:.1f} GB free)", flush=True)
    index, columns = A[0].index, A[0].columns.append([F.columns for F in A[1:]])
    arr = np.empty((len(index), len(columns)), np.float32, order="F")
    j = 0
    while A:
        F = A.pop(0)
        arr[:, j:j + F.shape[1]] = F.to_numpy()
        j += F.shape[1]
        del F
        _trim_heap()
    return pd.DataFrame(arr, index=index, columns=columns, copy=False)


def stage_fits(name: str, need: float) -> bool:
    """Memory guard for a stage whose output the pipeline can do without: whether its working set (``need``
    bytes, estimated from the frames it reads) fits in the memory free now. When it does not, the log says
    the stage is skipped and the pipeline carries on: fewer columns, never a kill (the gate keeps the result
    no worse than raw columns). A run that fits never reads this: the same stages run as before."""
    free = available_bytes()
    if need <= free:
        return True
    print(f"memory guard: {name} skipped (needs about {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free)", flush=True)
    return False


def frame_bytes(df: pd.DataFrame) -> float:
    return float(df.memory_usage(index=True, deep=False).sum())


def child_model_bytes(size, rows=None) -> float:
    """child_model_features' working set on a child table of ``size`` (rows, bytes): the table, its copy of the
    columns, up to 24 float32 row-pair columns and their concatenation onto the copy; then the table and that
    matrix, next to the rows of one fit (at most 3 million) copied for LightGBM and binned. (Home Credit's 10M POS
    rows under 5.6 GiB, priced at 1.8 GB with 12 row pairs counted once: killed building its matrix.)"""
    n = size[0] if rows is None else rows
    b = size[1] / max(size[0], 1)
    w = b + 4 * 24
    return max(n * (3 * b + 4 * 48), n * (b + w) + 2 * min(n, 3_000_000) * w)


def child_models_help(main, rel, cm, y, task, max_rows=60_000, seed=0, what="child models", min_gain=0.0):
    """Keep a block (child-row models, code bags) only when it adds to the aggregates: a quick 3-fold LightGBM on
    training rows with and without it (child models: Home Credit +0.28 AUC points; on Elo they cost hours and
    lowered the score, the card-level label being noise for each transaction. Code bags: Walmart's trip types
    gained, Home Credit's day counts read as codes cost 0.001 AUC)."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold, StratifiedKFold
    t = time.time()
    base = pd.concat([main.reset_index(drop=True)] + rel, axis=1)
    full = pd.concat([base] + cm, axis=1)
    for X in (base, full):
        for c in X.columns:
            if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
                X[c] = as_text(X[c]).astype("category")
    task = task or ("binary" if y.nunique() == 2 else "regression")
    if task == "multiclass":
        yv = pd.factorize(y, sort=True)[0].astype(float)
    else:
        yv = y.to_numpy(dtype=float)
    rows = np.arange(len(yv))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    obj = {"binary": "binary", "multiclass": "multiclass"}.get(task, "regression")
    metric = {"binary": "binary_logloss", "multiclass": "multi_logloss"}.get(task, "l2")
    # (multiclass at 0.1: one of Walmart's 38-class folds came out 32% worse with the bags, the other two near even)
    P = dict(objective=obj, learning_rate=0.05 if task == "multiclass" else 0.1, num_leaves=31,
             min_child_samples=50, feature_fraction=0.5, verbose=-1, num_threads=4, seed=seed,
             deterministic=True, force_col_wise=True)  # histogram mode not chosen by timing (load)
    if task == "multiclass":
        P["num_class"] = int(yv.max()) + 1
    split = (StratifiedKFold(3, shuffle=True, random_state=seed).split(rows, yv[rows]) if task != "regression"
             else KFold(3, shuffle=True, random_state=seed).split(rows))
    gains = []
    for fi, vi in split:
        fi, vi = rows[fi], rows[vi]
        loss = []
        for X in (base, full):
            b = lgb.train(P, lgb.Dataset(X.iloc[fi], yv[fi]), 1000, valid_sets=[lgb.Dataset(X.iloc[vi], yv[vi])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            loss.append(b.best_score["valid_0"][metric])
        gains.append((loss[0] - loss[1]) / loss[0])
    keep = float(np.mean(gains)) > min_gain and sum(g > 0 for g in gains) >= 2
    print(f"{what}: {100 * np.mean(gains):+.2f}% CV loss over the aggregates "
          f"({', '.join(f'{100 * g:+.2f}%' for g in gains)}) -> {'kept' if keep else 'dropped'} "
          f"in {time.time() - t:.0f}s", flush=True)
    return keep


@dataclass
class PipelineConfig:
    """Every option of ``build_features``; the command line's flags map one to one."""
    train: Union[str, pd.DataFrame]
    test: Union[str, pd.DataFrame]
    target: str
    id: Optional[str] = None
    table: List[str] = field(default_factory=list)
    time: Optional[str] = None
    child_models: bool = True
    task: Optional[str] = None
    log_target: bool = False
    budget: float = 900
    top_related: int = 200
    out_dir: str = "features"
    related_cache: Optional[str] = None
    forge_kw: dict = field(default_factory=dict)
    forecast: str = "auto"
    forecast_kw: dict = field(default_factory=dict)
    history: str = "auto"
    panel: str = "auto"
    lists: str = "auto"
    proposals: Optional[str] = None
    llm: str = "auto"
    description: Optional[str] = None
    llm_model: Optional[str] = None


def _load(src) -> pd.DataFrame:
    """A table from a CSV or parquet path, or a copy of a DataFrame, normalised as files are."""
    if isinstance(src, pd.DataFrame):
        return normalise(src.copy())
    return read(str(src))


def _table_specs(tables) -> List[str]:
    """``tables`` as ``name=path[:key[:time]]`` strings: a list of such strings, or a mapping of name to a
    path, a ``(path, key)`` / ``(path, key, time)`` tuple or a dict with ``path``, ``key`` and ``time``."""
    if not tables:
        return []
    if not isinstance(tables, dict):
        return [str(s) for s in tables]
    out = []
    for name, v in tables.items():
        if isinstance(v, dict):
            v = (v["path"], v.get("key"), v.get("time"))
        elif not isinstance(v, (tuple, list)):
            v = (v,)
        parts = [str(v[0])] + ["" if x is None else str(x) for x in v[1:]]
        while len(parts) > 1 and parts[-1] == "":
            parts.pop()
        out.append(f"{name}=" + ":".join(parts))
    return out


def _forecast_room() -> float:
    """Memory the forecasting block's training rows may take (with their raw columns): under half of what is free."""
    return 0.45 * available_bytes()


def build_features(train, test, target: str, *, id: Optional[str] = None, tables=None, time_col: Optional[str] = None,
                   child_models: bool = True, task: Optional[str] = None, log_target: bool = False,
                   budget: float = 900, top_related: int = 200, out_dir: str = "features",
                   related_cache: Optional[str] = None, forge_kw: Optional[dict] = None, forecast: str = "auto",
                   forecast_kw: Optional[dict] = None, history: str = "auto", panel: str = "auto",
                   lists: str = "auto", proposals: Optional[str] = None, llm: str = "auto",
                   description: Optional[str] = None, llm_model: Optional[str] = None) -> Tuple[Path, Path]:
    """Build features for a contest and write them to ``out_dir``.

    ``train`` and ``test`` are CSV/parquet paths or DataFrames; ``train`` holds ``target``. ``test`` is the
    unlabeled rows the features will be used on (the contest's test file): label-free statistics read them
    and the search's checks are built to resemble them. ``tables`` are child tables (see ``_table_specs``).
    ``time_col`` is the main table's time column for as-of aggregation (default: found when a key repeats).
    ``budget`` is FeatureForge's search budget in seconds of a 4-core reference machine; ``forge_kw`` and
    ``forecast_kw`` pass extra arguments to ``FeatureForge`` and ``ForecastFeatures``. ``forecast``,
    ``history``, ``panel`` and ``lists`` are ``"auto"`` (on from the data's structure) or ``"off"``.
    ``proposals`` is a Python file of domain features written outside the search (``PROPOSALS = [(label, fn)]``,
    e.g. by an LLM from the schema); each is kept only if it passes a cross-validated check on top of the
    search's output (see ``featureforge.proposals``). Without one, ``llm="auto"`` asks Claude to write it from
    the schema when ``ANTHROPIC_API_KEY`` is set (``"on"`` / ``"off"`` force it; ``description`` is the data's
    description for the prompt; see ``featureforge.llm``). Without a key the output is unchanged.

    Returns the paths of ``train_features.parquet`` (with the target) and ``test_features.parquet``.
    """
    return run(PipelineConfig(train=train, test=test, target=target, id=id, table=_table_specs(tables),
                              time=time_col, child_models=child_models, task=task, log_target=log_target,
                              budget=budget, top_related=top_related, out_dir=str(out_dir),
                              related_cache=related_cache, forge_kw=dict(forge_kw or {}), forecast=forecast,
                              forecast_kw=dict(forecast_kw or {}), history=history, panel=panel, lists=lists,
                              proposals=proposals, llm=llm, description=description, llm_model=llm_model))


def _add_proposals(cfg, out_tr, y, frames, test_path, holdout=None):
    """Proposals that pass the check on top of the training output, added to both outputs."""
    from featureforge import proposals as P
    from featureforge.utils.task import infer_task
    task = cfg.task or infer_task(y)
    yy = np.asarray(y, dtype=float)
    if task == "regression" and cfg.log_target:
        yy = np.log1p(np.clip(yy, 0, None))
    kept = P.gate(out_tr, yy, task, frames, holdout=holdout)
    if not kept:
        return out_tr
    n = len(out_tr)
    cols = dict(frames)
    te = pd.read_parquet(test_path)
    for label, names in kept:
        v = cols[label]
        for c in names:
            out_tr[c] = v[c].iloc[:n].to_numpy()
            te[c] = v[c].iloc[n:].to_numpy()
    te.to_parquet(test_path)
    print(f"proposals kept: {', '.join(l for l, _ in kept)}", flush=True)
    return out_tr


def run(cfg: PipelineConfig) -> Tuple[Path, Path]:
    """``build_features`` from a ``PipelineConfig``."""
    t0 = time.time()
    tr, te = _load(cfg.train), _load(cfg.test)
    y = tr.pop(cfg.target)
    labels = None
    if cfg.task == "multiclass" or (cfg.task is None and not pd.api.types.is_numeric_dtype(y) and y.nunique() > 2):
        # Class names (Santander's product codes) become 0..K-1 for the search; written back as given.
        codes, labels = pd.factorize(as_text(y), sort=True)
        y, cfg.task = pd.Series(codes, name=cfg.target), "multiclass"
    prop_frames = None
    if not cfg.proposals and cfg.llm != "off":
        from featureforge import llm as L
        if L.wanted(cfg.llm):
            from featureforge.utils.task import infer_task
            path = Path(cfg.out_dir) / L.FILE_NAME
            raw = [F.drop(columns=[cfg.id]) if cfg.id and cfg.id in F.columns else F for F in (tr, te)]
            if path.exists():
                print(f"llm proposals: reusing {path}", flush=True)
            else:
                name = Path(str(cfg.train)).stem if not isinstance(cfg.train, pd.DataFrame) else "data"
                L.write(*raw, cfg.task or infer_task(y), path, name=name, description=cfg.description,
                        model=cfg.llm_model)
            if path.exists():
                cfg.proposals = str(path)
    if cfg.proposals:
        from featureforge import proposals as P
        raw = [F.drop(columns=[cfg.id]) if cfg.id and cfg.id in F.columns else F for F in (tr, te)]
        prop_frames = P.compute(P.load(cfg.proposals), *raw)
        print(f"proposals: {len(prop_frames)} usable from {cfg.proposals}", flush=True)
    ids_tr = tr[cfg.id].to_numpy() if cfg.id else np.arange(len(tr))
    ids_te = te[cfg.id].to_numpy() if cfg.id else np.arange(len(tr), len(tr) + len(te))
    rel_tr = rel_te = None
    label_cols: list = []
    cache = Path(cfg.related_cache) if cfg.related_cache else None
    if cache is not None and (cache / "rel_train.parquet").exists():
        rel_tr, rel_te = pd.read_parquet(cache / "rel_train.parquet"), pd.read_parquet(cache / "rel_test.parquet")
        if (cache / "label_cols.json").exists():
            label_cols = json.loads((cache / "label_cols.json").read_text())
        tr, te = pd.read_parquet(cache / "main_train.parquet"), pd.read_parquet(cache / "main_test.parquet")
        print(f"related tables: {rel_tr.shape[1]} columns from {cache}", flush=True)
    elif cfg.table:
        children = [parse_table(s, tr, cfg.id) for s in cfg.table]
        src = {ch.name: s.split("=", 1)[1].split(":")[0] for s, ch in zip(cfg.table, children)}
        lookups = [ch for ch in children if "__lookup__" in ch.drop]
        children = [ch for ch in children if ch not in lookups]
        # One row per key: attributes of the main row (a tube's dimensions, its bill of
        # materials), joined as columns before anything is aggregated or looked up.
        for ch in [ch for ch in children if ch.source is None and ch.key in tr.columns
                   and not ch.df[ch.key].duplicated().any()]:
            cols = [c for c in ch.df.columns if c == ch.key or c not in tr.columns]
            tr = tr.merge(ch.df[cols], on=ch.key, how="left")
            te = te.merge(ch.df[cols], on=ch.key, how="left")
            children.remove(ch)
            print(f"table {ch.name}: one row per {ch.key}, joined as columns", flush=True)
        both_main = pd.concat([tr, te[[c for c in tr.columns if c in te.columns]]], ignore_index=True)
        asof = {ch.name: main_time(tr, ch, cfg.time) for ch in children
                if ch.source is None and ch.time is not None and ch.key in tr.columns and (tr[ch.key].duplicated().any() or cfg.time)}
        asof = {k: v for k, v in asof.items() if v is not None}
        keyed = [ch for ch in children if ch.name not in asof]
        F = pd.DataFrame()
        A = []
        bags = []   # code bags, kept below when they add to the aggregates
        recent = []   # recent-window summaries, likewise
        for ch in children:
            if ch.name in asof:
                print(f"table {ch.name}: as of {asof[ch.name]} per main row", flush=True)
                A.append(asof_features(both_main, ch.key, asof[ch.name], ch))
                if cfg.history == "auto" and cfg.target in ch.df.columns:
                    # The log carries the outcome itself (earlier answers): outcome history strictly before
                    # each row, overall and on the row's own item (the shared id-like column with most levels).
                    shared = [c for c in both_main.columns if c in ch.df.columns and c not in (ch.key, asof[ch.name])
                              and both_main[c].nunique() > 50]
                    item = max(shared, key=lambda c: both_main[c].nunique()) if shared else None
                    t = time.time()
                    El = event_log_features(both_main, ch, cfg.target, asof[ch.name], item=item,
                                            outcome_values=sorted(pd.unique(y.dropna())))
                    print(f"table {ch.name}: outcome history (item={item}), {El.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                    A.append(El)
                    label_cols += list(El.columns)
        cm_tables = []
        if cfg.child_models and keyed:
            # as-of children would see later rows' outcomes; labels per key are ambiguous
            # when the key repeats
            # (not for a table read by key range: its rows never sit in memory together)
            cm_tables = [ch for ch in keyed if tr[ch.key].is_unique and ch.source is None]

        # Tables dropped to save memory are written once as they were parsed (categories stay categories) and read
        # back from that copy: the same frame, read at about twice its size instead of the source's parse (Home
        # Credit Risk Stability's 6.5M previous applications: 0.8 GB as a frame, 4.2 GB to parse from its parquet,
        # killed beside 7.3 GB of related columns; 1.7 GB read back from the copy).
        spill: dict = {}
        spill_dir = []

        def keep_copy(name, df):
            # (written a row group at a time: about 0.1 GB more for a 0.8 GB frame)
            if name in spill or df is None or available_bytes() < 0.5 * frame_bytes(df):
                return
            if not spill_dir:
                import tempfile
                spill_dir.append(Path(tempfile.mkdtemp(prefix="featureforge_tables_")))
            path = spill_dir[0] / f"{len(spill)}.parquet"
            try:
                df.to_parquet(path)
                spill[name] = path
            except Exception:  # noqa: BLE001  (no copy: read from the source again)
                path.unlink(missing_ok=True)

        def reread(name):
            if name in spill:
                df = pd.read_parquet(spill[name])
                release_arrow()
                return df
            return read(src[name])

        def table(ch):
            return Child(ch.name, reread(ch.name), key=ch.key, time=ch.time, children=ch.children, drop=ch.drop)
        # Keyed tables stay held while memory allows. When the free memory is short of twice the other tables held
        # and four times the table at hand, the others are dropped and each is read again at its own step, for its
        # matches and for its child-row models: the same file gives the same frame (all five of Home Credit's, about
        # 3 GB, sat next to the aggregates: under 5.6 GiB, killed aggregating the fourth; dropped and read again
        # each time, a 2x Student Game Play stand-in's events table took two more minutes to read).
        sizes = {ch.name: (len(ch.df), frame_bytes(ch.df)) for ch in keyed if ch.source is None}
        one_at_a_time = False
        for ch in keyed:
            if ch.source is None:
                others = [o for o in keyed if o is not ch and o.source is None and o.df is not None]
                held = sum(sizes[o.name][1] for o in others)
                if not one_at_a_time and others and available_bytes() < 2 * held + 4 * sizes[ch.name][1]:
                    # From here on each table is read for its own step and dropped after it, the table at hand
                    # included (kept from the parse, it left the stages after it less memory: under 5.6 GiB, Home
                    # Credit's first two latest-state histories were skipped).
                    print(f"memory guard: related tables held one at a time from table {ch.name} "
                          f"({held / 1e9:.1f} GB of others dropped, read again when needed)", flush=True)
                    one_at_a_time = True
                    for o in keyed:
                        if o.source is None:
                            keep_copy(o.name, o.df)
                            o.df = None
                            _trim_heap()
                if ch.df is None:
                    ch.df = reread(ch.name)
                    if one_at_a_time:
                        keep_copy(ch.name, ch.df)
            if ch.source is not None:
                for j, Fs in enumerate(streamed_features(ch, cfg.history == "auto")):
                    Fs = Fs.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
                    for c in [c for c in Fs.columns if c.endswith("__count")]:
                        Fs[c] = Fs[c].fillna(0)
                    A.insert(j, Fs)
                continue
            try:
                Fk = RelatedTables([ch]).features().reindex(both_main[ch.key].to_numpy())
            except MemoryError as e:   # (its key ranges' guard: not even a small range fits)
                print(f"memory guard: table {ch.name} aggregates skipped ({e})", flush=True)
                Fk = None
                _trim_heap()
            if Fk is not None:
                Fk.index = both_main.index    # (set_index copied every column)
                for c in [c for c in Fk.columns if c.endswith("__count")]:
                    Fk[c] = Fk[c].fillna(0)
                A.insert(0, Fk)
                del Fk
            # Bags of the table's many-level categoricals and code columns, as latent components per parent.
            t = time.time()
            Bk = bag_features(ch) if stage_fits(f"table {ch.name}: code bags", 3 * frame_bytes(ch.df)) else pd.DataFrame()
            if Bk.shape[1]:
                bags.append(Bk.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                print(f"table {ch.name}: code bags, {Bk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            del Bk
            # Whole-history summaries (price paths, latest state) of an as-of table see only the rows before every
            # main row: a later snapshot would hold the very outcome of an earlier labelled row.
            if ch.name in asof:
                tmin = np.nanmin(time_values(both_main[asof[ch.name]]))
                ch = Child(ch.name, ch.df[time_values(ch.df[ch.time]) < tmin], key=ch.key, time=ch.time, drop=ch.drop)
            # Price paths (an order book, a trade log): realized volatility per parent.
            t = time.time()
            # (a sorted copy of the table's rows at most)
            Rk = return_features(ch.df, ch.key, ch.name, ch.time, ignore=ch.drop) \
                if stage_fits(f"table {ch.name}: price paths", frame_bytes(ch.df)) else pd.DataFrame()
            if Rk.shape[1]:
                A.insert(1, Rk.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                print(f"table {ch.name}: price paths, {Rk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            # (the key as text for the sort, about 64 bytes a row, and the sorted rows of the numeric columns, their
            # float32 copies and their shifted copy)
            if cfg.history == "auto" and repeats(ch) and stage_fits(
                    f"table {ch.name}: latest-state history",
                    len(ch.df) * (64 + 24.0 * (1 + sum(pd.api.types.is_numeric_dtype(t) for t in ch.df.dtypes)))):
                t = time.time()
                Hk = history_features(ch).reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
                print(f"table {ch.name}: latest-state history, {Hk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                A.insert(1, Hk)
                del Hk
                t = time.time()
                Wk = recent_window_features(ch)
                if Wk.shape[1]:
                    recent.append(Wk.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                    print(f"table {ch.name}: recent windows, {Wk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                del Wk
            Rk = None
            if one_at_a_time:
                ch.df = None
                _trim_heap()
        rel_tr, rel_te = pd.DataFrame(index=range(len(tr))), pd.DataFrame(index=range(len(te)))
        # Child rows that also share the main row's code values (a customer's purchases of the
        # offer's brand), counted before the main row's date when both tables carry dates.
        mt = date_col(both_main)
        for ch in [ch for ch in keyed if ch.source is None] + [ch for ch in children if ch.name in asof]:
            if ch.df is None:
                ch = table(ch)
            cols = match_cols(both_main, ch)
            if not cols:
                continue
            ct = ch.time if ch.time is not None else date_col(ch.df.drop(columns=[ch.key]))
            # An event log ordered by a time the main rows carry (order number, timestamp): count matches strictly
            # before each row in that time.
            ordinal = ch.name in asof and (mt is None or ct is None)
            if both_main[ch.key].duplicated().any() and not ordinal and (mt is None or ct is None):
                continue  # repeated keys without dates could match a row's own later events
            chm = Child(ch.name, ch.df, key=ch.key, time=ct, drop=ch.drop)
            t = time.time()
            Mf = match_features(both_main, ch.key, chm, cols, main_time=asof[ch.name] if ordinal else (mt if ct is not None else None),
                                ordinal=ordinal)
            print(f"table {ch.name}: same-{'/'.join(cols)} matches{' before ' + (asof[ch.name] if ordinal else mt) if (ordinal or (mt and ct)) else ''}: "
                  f"{Mf.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            A.append(Mf)
        for lk in lookups:
            L = lookup_features(both_main, lk.df, lk.key, lk.name)
            print(f"table {lk.name}: {L.shape[1]} lookup columns", flush=True)
            A.append(L)
        if recent:
            pos = np.arange(len(tr))
            if len(pos) > CM_CHECK_PARENTS:
                pos = np.sort(np.random.default_rng(0).choice(pos, CM_CHECK_PARENTS, replace=False))
            n_cols = len(tr.columns) + sum(F.shape[1] for F in A + recent)
            if stage_fits("recent windows check", 4 * 4.0 * len(pos) * n_cols) and child_models_help(
                    tr.drop(columns=[c for c in (cfg.id, cfg.target) if c and c in tr.columns]).iloc[pos],
                    [Fa.iloc[pos].reset_index(drop=True) for Fa in A], [R.iloc[pos].reset_index(drop=True) for R in recent],
                    y.iloc[pos].reset_index(drop=True), cfg.task, what="recent windows", min_gain=0.005):
                A.extend(recent)
            recent = []
        if bags:
            # (the check's frames of both arms on the sampled rows, float32 mostly, and one fit's copies)
            pos = np.arange(len(tr))
            if len(pos) > CM_CHECK_PARENTS:
                pos = np.sort(np.random.default_rng(0).choice(pos, CM_CHECK_PARENTS, replace=False))
            n_cols = len(tr.columns) + sum(F.shape[1] for F in A + bags)
            if stage_fits("code bags check", 4 * 4.0 * len(pos) * n_cols) and child_models_help(
                    tr.drop(columns=[c for c in (cfg.id, cfg.target) if c and c in tr.columns]).iloc[pos],
                    [Fa.iloc[pos].reset_index(drop=True) for Fa in A], [B.iloc[pos].reset_index(drop=True) for B in bags],
                    y.iloc[pos].reset_index(drop=True), cfg.task, what="code bags", min_gain=0.005):
                # (32 bag columns a table must earn 0.5%: Home Credit's day counts read as codes passed at +0.19%
                # and cost 0.0008 AUC; TalkingData mobile's apps earned +2.7%)
                A.extend(bags)
            bags = []
        # The child-row models read their tables again, one at a time (the as-of tables are dropped here too). What
        # the aggregation left behind (its key ranges' parts, freed arrays glibc keeps) goes back: the models' fits
        # are the pipeline's next peak.
        for ch in children:
            if ch.source is None:
                ch.df = None
        _trim_heap()
        if cm_tables and cfg.task != "multiclass":
            # The check's quick models read every related column on the sampled parents (float32 mostly): the
            # frames of both arms and one fit's training and validation rows of each, next to its child-row models
            # on those parents' rows.
            n_cols = len(tr.columns) + sum(Fa.shape[1] for Fa in A) + 5 * len(cm_tables)
            n_par = min(len(tr), CM_CHECK_PARENTS)
            need = 4 * 4.0 * n_par * n_cols + max(
                child_model_bytes(sizes[ch.name], min(sizes[ch.name][0], CM_CHECK_ROWS)) for ch in cm_tables)
            if not stage_fits("child-row models", need):
                cm_tables = []
        if cm_tables and cfg.task != "multiclass":   # (child-row models are binary or regression)
            task = cfg.task or ("binary" if y.nunique() == 2 else "regression")
            obj = "binary" if task == "binary" else "regression"
            # Decide on a sample of training parents with few child rows per fit first: on
            # Elo (29M transactions) the full models took over an hour and lowered the score.
            rng = np.random.default_rng(0)
            pos = np.arange(len(tr))
            if len(pos) > CM_CHECK_PARENTS:
                pos = np.sort(rng.choice(pos, CM_CHECK_PARENTS, replace=False))
            t = time.time()
            cm_s = []
            for ch in cm_tables:
                ks = tr[ch.key].to_numpy()[pos]
                df = table(ch).df
                sub = Child(ch.name, df[df[ch.key].isin(set(ks))], key=ch.key, time=ch.time,
                            children=ch.children, drop=ch.drop)
                del df
                Ms = child_model_features(sub, pd.Series(y.to_numpy()[pos], index=ks), [], task=obj,
                                          max_rows=CM_CHECK_ROWS)
                cm_s.append(Ms.reindex(ks).reset_index(drop=True))
            print(f"child models on {len(pos)} sampled parents: {time.time() - t:.0f}s", flush=True)
            keep = child_models_help(tr.drop(columns=[c for c in (cfg.id, cfg.target) if c and c in tr.columns]).iloc[pos],
                                     [Fa.iloc[pos].reset_index(drop=True) for Fa in A], cm_s,
                                     y.iloc[pos].reset_index(drop=True), cfg.task)
            for ch in cm_tables if keep else []:
                _trim_heap()
                if not stage_fits(f"child model {ch.name}", child_model_bytes(sizes[ch.name])):
                    continue
                t = time.time()
                yk = pd.Series(y.to_numpy(), index=tr[ch.key].to_numpy())
                M = child_model_features(table(ch), yk, te[ch.key].to_numpy(), task=obj)
                A.append(M.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                label_cols += list(M.columns)
                print(f"child model {ch.name}: {time.time() - t:.0f}s", flush=True)
        # What each parent's codes say about the target (out of fold), kept when the quick check finds it adds 0.5%.
        if cm_tables:
            task_ = cfg.task or ("binary" if y.nunique() == 2 else "regression")
            k_ = y.nunique() if task_ == "multiclass" else 1
            cte = []
            for ch in cm_tables:
                if not stage_fits(f"table {ch.name}: code target maps", 16.0 * sizes[ch.name][0] * (k_ + 3)):
                    continue
                t = time.time()
                yk = pd.Series(y.to_numpy(), index=tr[ch.key].to_numpy())
                C = code_target_features(table(ch), yk, te[ch.key].to_numpy(), task_)
                if C.shape[1]:
                    cte.append(C.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                    print(f"table {ch.name}: code target maps, {C.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                del C
            if cte:
                pos = np.arange(len(tr))
                if len(pos) > CM_CHECK_PARENTS:
                    pos = np.sort(np.random.default_rng(0).choice(pos, CM_CHECK_PARENTS, replace=False))
                if child_models_help(tr.drop(columns=[c for c in (cfg.id, cfg.target) if c and c in tr.columns]).iloc[pos],
                                     [Fa.iloc[pos].reset_index(drop=True) for Fa in A],
                                     [C.iloc[pos].reset_index(drop=True) for C in cte],
                                     y.iloc[pos].reset_index(drop=True), cfg.task, what="code target maps", min_gain=0.005):
                    A.extend(cte)
                    label_cols += [c for C in cte for c in C.columns]
            cte = []
        # One concat, not one per block (each pairwise concat copied every column so far), then each side as views
        # of its rows: a concat per side of row slices copied every block twice more, consolidated and joined
        # (Home Credit's 1,200 related columns: the pipeline's peak before the search).
        if A:
            # (room: 40% of what would be free without them, for the joined block next to its kept parts, then the
            # search's copies)
            A = cut_related(A, y.to_numpy(), len(tr), cfg.task,
                            0.4 * (available_bytes() + sum(frame_bytes(F) for F in A)))
            rel = join_columns(A)
            rel_tr, rel_te = _rows(rel, 0, len(tr)), _rows(rel, len(tr), len(rel))
            del rel
        if spill_dir:
            shutil.rmtree(spill_dir[0], ignore_errors=True)
        print(f"related tables: {rel_tr.shape[1]} columns in {time.time() - t0:.0f}s", flush=True)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            rel_tr.to_parquet(cache / "rel_train.parquet")
            (cache / "label_cols.json").write_text(json.dumps(label_cols))
            rel_te.to_parquet(cache / "rel_test.parquet")
            tr.to_parquet(cache / "main_train.parquet")  # with any one-row-per-key tables joined
            te.to_parquet(cache / "main_test.parquet")
        del A, both_main, children, keyed
        ch = chm = lookups = cm_tables = None  # the last table's rows otherwise stay referenced (Instacart's log: 1.8 GB)

    Xtr = tr.drop(columns=[cfg.id]) if cfg.id else tr
    Xte = te.drop(columns=[cfg.id]) if cfg.id else te
    Xte = Xte[Xtr.columns]
    n_main = tr.shape[1]
    if cfg.lists == "auto":
        t = time.time()
        Jtr, Jte, jfound = json_fields(Xtr, Xte)
        if jfound:
            Xtr = pd.concat([Xtr.drop(columns=jfound).reset_index(drop=True), Jtr], axis=1)
            Xte = pd.concat([Xte.drop(columns=jfound).reset_index(drop=True), Jte], axis=1)
            print(f"JSON fields: {jfound} -> {Jtr.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            del Jtr, Jte
        Ltr, Lte, found = list_features(Xtr, Xte)
        for c in found:  # downstream families read them as text: items joined
            J = as_lists(pd.concat([Xtr[c], Xte[c]], ignore_index=True)).map(" ; ".join).to_numpy()
            Xtr[c], Xte[c] = J[:len(Xtr)], J[len(Xtr):]
        Utr, Ute = unit_ratio_features(Xtr, Xte)
        if Ltr.shape[1] or Utr.shape[1]:
            Xtr = pd.concat([Xtr.reset_index(drop=True), Ltr, Utr], axis=1)
            Xte = pd.concat([Xte.reset_index(drop=True), Lte, Ute], axis=1)
            print(f"lists: {found} -> {Ltr.shape[1]} columns; per-unit amounts: {Utr.shape[1]} columns "
                  f"in {time.time() - t:.0f}s", flush=True)
    if cfg.panel == "auto":
        t = time.time()
        Ptr, Pte, P = panel_features(Xtr, Xte, y=y if cfg.task != "multiclass" else None)
        if P is not None and Ptr.shape[1]:
            keep, note = panel_helps(Xtr, Ptr, y, P["day"], cfg.task)
            print(f"panel check: {note} -> {'kept' if keep else 'dropped'} ({time.time() - t:.0f}s)", flush=True)
            if not keep:
                P = None
        if P is not None:
            Xtr = pd.concat([Xtr.reset_index(drop=True), Ptr], axis=1)
            Xte = pd.concat([Xte.reset_index(drop=True), Pte], axis=1)
            print(f"panel: {P} -> {Ptr.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
    del tr, te  # a second copy of both tables is gigabytes on IEEE-CIS
    release_arrow()
    forecasting = False
    if cfg.forecast == "auto":
        from featureforge.blocks.forecast import ForecastFeatures
        t = time.time()
        fkw = dict(cfg.forecast_kw)
        ff = ForecastFeatures(**fkw).fit(Xtr, y.to_numpy(), Xte)
        keep = ff.active_
        if keep:
            tc = time.time()
            keep, note = forecast_check(ff, Xtr, y.to_numpy(dtype=float), fkw, cfg.task, cfg.log_target)
            print(f"forecast: {note} -> {'kept' if keep else 'rejected; FeatureForge search runs'} "
                  f"({time.time() - tc:.0f}s)", flush=True)
        if keep:
            # Each block a block of rows at a time into one float32 matrix, with fewer rows a block when memory
            # is short (the whole transform's working columns were Store Sales' peak, and killed Rohlik in
            # 6.5 GB); when even the matrix would not fit, the block is skipped and the search runs, as when
            # the check rejects it.
            def block(X):
                if not len(X):
                    return ff.transform(X)
                names, M = ff.transform_float32(X, _CHECK_CHUNK_ROWS, guard=True)
                return pd.DataFrame(M, columns=names, index=X.index, copy=False)
            # A long history (the whole sales file of a large panel) feeds the block in full, but its block for
            # every training row would not fit: the training rows are cut to the latest periods that do, and the
            # earlier ones only feed the block (Favorita: 112 days x 210k series, 23.6M rows, skipped the block
            # whole and fell back to the search).
            if len(Xtr) > 100_000:
                names, _ = ff.transform_float32(Xtr.iloc[:2000], _CHECK_CHUNK_ROWS)
                row = 4.0 * len(names) + Xtr.memory_usage(deep=True).sum() / len(Xtr)
                room = _forecast_room() - 4.0 * len(names) * len(Xte)
                fit_rows = int(max(room, 0) / (1.1 * row))
                if fit_rows < len(Xtr):
                    pt = ff._periods(Xtr)
                    order = np.sort(pt[np.isfinite(pt)])[::-1]
                    cut = order[min(fit_rows, len(order)) - 1] if fit_rows > 0 else np.inf
                    sel = np.isfinite(pt) & (pt > cut) if (pt >= cut).sum() > fit_rows else np.isfinite(pt) & (pt >= cut)
                    if sel.sum() >= 0.05 * len(Xtr):
                        n_p = len(np.unique(pt[sel]))
                        print(f"forecast: long history: training rows cut to the latest {n_p} periods ({int(sel.sum())} of "
                              f"{len(Xtr)} rows) to fit memory; earlier periods feed the block only", flush=True)
                        Xtr = Xtr[sel].reset_index(drop=True)
                        y = y[sel].reset_index(drop=True)
                        ids_tr = ids_tr[sel]
                        if rel_tr is not None:
                            rel_tr = rel_tr[sel].reset_index(drop=True)
                        release_arrow()
            try:
                Ftr = block(Xtr)
                Fte = block(Xte)
                # The training rows joined to their block are a copy of both next to the originals, and writing
                # them out converts that copy again (Rossmann at 1.5 GB: killed there, after its block was built).
                need = int(1.1 * (Xtr.memory_usage(deep=True).sum() + Ftr.memory_usage(deep=True).sum()))
                free = available_bytes()
                if need > free:
                    Ftr = Fte = None
                    raise MemoryError(f"joining the forecasting block to the training rows needs {need / 2**30:.2f} GB, "
                                      f"{free / 2**30:.2f} GB free")
            except MemoryError as e:
                Ftr = Fte = None
                keep = False
                print(f"memory guard: {e}; forecasting block skipped, FeatureForge search runs", flush=True)
        if keep:
            forecasting = True
            Xtr = pd.concat([_rows(Xtr), _rows(Ftr)], axis=1)
            Xte = pd.concat([_rows(Xte), _rows(Fte)], axis=1)
            print(f"forecast: {Ftr.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            del Ftr, Fte
    if rel_tr is not None:
        # The search sees the related columns a quick model uses most; all are written out.
        import lightgbm as lgb
        # The matrix LightGBM would make of main + related columns, filled column by column: the
        # concatenated frame, its category-code copy and LightGBM's own conversion were three
        # copies of 1,330 columns at once (Home Credit: 13.4 GB, killed on the 15 GB bench).
        Xtr = _rows(Xtr)
        for c in Xtr.columns:
            if not (pd.api.types.is_numeric_dtype(Xtr[c]) or isinstance(Xtr[c].dtype, pd.CategoricalDtype)):
                Xtr[c] = as_text(Xtr[c]).astype("category")
        # (rel_tr itself is written out as it is; the search's copies of its text columns are categories)
        conv = {c: as_text(rel_tr[c]).astype("category") for c in rel_tr.columns
                if not (pd.api.types.is_numeric_dtype(rel_tr[c]) or isinstance(rel_tr[c].dtype, pd.CategoricalDtype))}
        names = list(Xtr.columns) + list(rel_tr.columns)
        srcs = [Xtr[c] for c in Xtr.columns] + [conv.get(c, rel_tr[c]) for c in rel_tr.columns]
        cats = [c for c, s_ in zip(names, srcs) if isinstance(s_.dtype, pd.CategoricalDtype) and not s_.cat.ordered]
        # Single precision when every value is exactly a float32 (float32 columns, category codes,
        # integers within 2**24): LightGBM bins the same numbers, from half the matrix. Home
        # Credit's 40 integer columns made it float64, 3.3 GB.
        def exact32(s_):
            if isinstance(s_.dtype, pd.CategoricalDtype) or s_.dtype in (np.float32, np.float16, bool):
                return True
            v = s_.to_numpy(dtype=np.float64, na_value=np.nan)
            return bool(np.all((v.astype(np.float32).astype(np.float64) == v) | np.isnan(v)))
        dt = np.float32 if all(exact32(s_) for s_ in srcs) else np.float64
        # Memory guard: the matrix, LightGBM's binned copy of it (about a byte per cell) and the sample it bins from
        # (up to 200,000 rows' values as doubles with their row numbers, 12 bytes a value: 3 GB on Home Credit's
        # 1,243 columns, killed under 5.6 GiB with the matrix priced alone) must fit in what is free now (the
        # machine's, or this process's memory cgroup's when less); when they would not, the quick model reads a
        # row sample.
        rows = None
        room, per, samp = 0.9 * available_bytes(), (np.dtype(dt).itemsize + 1) * len(names), 12 * len(names)
        fit_rows = int(room / (per + samp))
        if fit_rows > 200_000:
            fit_rows = int((room - 200_000 * samp) / per)
        if fit_rows < len(Xtr):
            rows = np.sort(np.random.default_rng(0).choice(len(Xtr), max(fit_rows, 1000), replace=False))
            print(f"memory guard: related columns ranked on {len(rows)} of {len(Xtr)} rows", flush=True)
        M = np.empty((len(Xtr) if rows is None else len(rows), len(names)), dtype=dt)
        for j, s_ in enumerate(srcs):
            if isinstance(s_.dtype, pd.CategoricalDtype):
                k = s_.cat.codes.to_numpy()
                v = np.where(k < 0, np.nan, k)
            else:
                v = s_.to_numpy(dtype=dt, na_value=np.nan)
            M[:, j] = v if rows is None else v[rows]
        del srcs
        obj = dict(objective="multiclass", num_class=len(labels)) if labels is not None else \
            dict(objective="binary" if y.nunique() == 2 else "regression")
        params = dict(obj, learning_rate=0.1, num_leaves=31, feature_fraction=0.5, verbose=-1,
                      deterministic=True, force_col_wise=True)  # histogram mode not chosen by timing (load)
        # Bin first and drop the matrix before boosting: LightGBM keeps only its binned copy.
        yy = log_y(y) if cfg.log_target else y
        ds = lgb.Dataset(M, yy if rows is None else np.asarray(yy)[rows], feature_name=names, categorical_feature=cats,
                         params=params).construct()
        del M
        b = lgb.train(params, ds, 300)
        del ds
        gain = pd.Series(b.feature_importance("gain"), index=names)[rel_tr.columns]
        top = list(gain.sort_values(ascending=False).index[:cfg.top_related])
        Xtr = pd.concat([Xtr, pd.DataFrame({c: conv.get(c, rel_tr[c]) for c in top})], axis=1)
        del conv
        Xte = pd.concat([Xte.reset_index(drop=True), rel_te[top]], axis=1)
        del b
    spill = None
    if rel_tr is not None:
        # Written aside while the search runs and read back for the output: held, the related columns were
        # memory the search could not use (Instacart's 8.5 million rows: 2.4 GB).
        spill = Path(cfg.out_dir) / ".related_columns"
        spill.mkdir(parents=True, exist_ok=True)
        rel_tr.to_parquet(spill / "train.parquet")
        rel_te.to_parquet(spill / "test.parquet")
        rel_cols = list(rel_tr.columns)
        del rel_tr, rel_te
        release_arrow()
    if forecasting:
        # On top of the forecasting columns the search's gate rows stop resembling the test
        # horizon: its picks lost held-out (Favorita 0.669 -> 0.689, Recruit 0.519 -> 0.525,
        # Rossmann kept none), so a forecasting contest gets the forecasting columns alone.
        print("forecasting family active: FeatureForge search skipped", flush=True)
        # The outputs are these frames themselves: a copy (which also consolidated every column into one
        # block) held the whole output twice (Store Sales: 5.4 GB peak, the run's highest).
        out_tr, out_te = (F if F.index.equals(pd.RangeIndex(len(F))) else F.reset_index(drop=True) for F in (Xtr, Xte))
        Xtr = Xte = None
    else:
        # Columns computed from the labels (child-row model scores, outcome histories) stay out of
        # FeatureForge's cross-row features: Home Credit's kept cell count over gender x binned
        # bureau model score read other rows' scores, which were fitted on this row's label.
        kw = dict(label_cols=[c for c in label_cols if c in Xtr.columns], **cfg.forge_kw)
        forge = FeatureForge(task=cfg.task, time_budget=cfg.budget, log_target=cfg.log_target, **kw).fit(Xtr, y, X_unlabeled=Xte)
        out_te = forge.transform(Xte)
    # The test rows are transformed, finished and written before the training rows are transformed: held,
    # their output and the fit's frames were 13.2 GB on Instacart's 8.5 million rows (the 14.3 GB limit).
    # Related columns are added a column at a time: a concat consolidates the whole output into a copy next
    # to it (Instacart: killed at 14 GB).
    rest = [c for c in rel_cols if c not in out_te.columns] if spill is not None else []

    def related(o, f):
        for c in rest:
            o[c] = pd.read_parquet(spill / f, columns=[c])[c]
        release_arrow()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    related(out_te, "test.parquet")
    if cfg.id:
        out_te.insert(0, cfg.id, ids_te)
    out_te.to_parquet(out / "test_features.parquet")
    del out_te, Xte
    release_arrow()
    holdout = None
    if not forecasting:
        out_tr = forge.transform_train(Xtr, release=True)
        holdout = getattr(forge, "holdout_", None)
        del forge
    related(out_tr, "train.parquet")
    if spill is not None:
        shutil.rmtree(spill, ignore_errors=True)
    if prop_frames and forecasting:
        print("proposals: skipped (the forecasting block has no held-out window to judge them on)", flush=True)
    elif prop_frames:
        out_tr = _add_proposals(cfg, out_tr, y, prop_frames, out / "test_features.parquet", holdout)
    out_tr[cfg.target] = labels[y.to_numpy()] if labels is not None else y.to_numpy()
    if cfg.id:
        out_tr.insert(0, cfg.id, ids_tr)
    out_tr.to_parquet(out / "train_features.parquet")
    print(f"done in {time.time() - t0:.0f}s: {out_tr.shape[1] - n_main - 1} columns added -> {out}/")
    return out / "train_features.parquet", out / "test_features.parquet"

