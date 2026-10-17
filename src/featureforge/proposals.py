"""Domain features proposed from outside the search (written by a person or an LLM), kept only if they carry over.

A proposals file is a Python module with ``PROPOSALS = [(label, fn), ...]``; ``fn`` takes the raw main table
(training rows then test rows, as loaded) and returns a Series or DataFrame with one row per input row. This
is how CAAFE-style LLM features reach FeatureForge: the LLM writes the file from the column names, types,
sample rows and the data's description, before any score is seen.

CAAFE keeps a proposal when it lowers a cross-validated loss at all; on the head-to-head bench that kept
spambase's guessed word-frequency features, which lost 10.5% held out. Here each proposal, in the order
written, is judged like FeatureForge's own features: 5-fold cross-validation over the training rows, raw plus
FeatureForge's output with and without it, paired per-row losses, kept only when the mean gain is positive
and clears one standard error (z >= 1). Proposals are judged on top of FeatureForge's output, so a proposal
that only repeats what the search found is not added.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from featureforge.screens import _winsor

MAX_ROWS = 20_000  # rows the check fits on: a uniform sample past this


def load(path) -> list:
    spec = importlib.util.spec_from_file_location("ff_proposals", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "PROPOSALS", []))


def compute(props, raw_tr: pd.DataFrame, raw_te: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """Each proposal over training then test rows; a proposal that fails, or returns the wrong length or
    nothing numeric, is dropped with a log line."""
    allx = pd.concat([raw_tr, raw_te], ignore_index=True)
    out = []
    for label, fn in props:
        try:
            v = fn(allx.copy())
            v = v.to_frame(label) if isinstance(v, pd.Series) else pd.DataFrame(v)
            v = v.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
            v = v.loc[:, v.nunique() > 1]
            if len(v) != len(allx) or not v.shape[1]:
                raise ValueError(f"{len(v)} rows x {v.shape[1]} usable columns")
            v.columns = [f"prop__{label}__{c}" for c in v.columns]
            out.append((label, v.reset_index(drop=True)))
        except Exception as e:  # noqa: BLE001  a proposal is outside code: any failure drops it
            print(f"proposal {label}: dropped ({repr(e)[:160]})", flush=True)
    return out


def _oof_rows(task, F, y, folds, seed, threads):
    from featureforge.forge import _holdout_rows
    rows = np.zeros(len(F))
    for tr, va in folds:
        rows[va] = _holdout_rows(task, F.iloc[tr].reset_index(drop=True), y[tr],
                                 F.iloc[va].reset_index(drop=True), y[va], seed, threads)
    return rows


def gate(out_tr: pd.DataFrame, y, task: str, frames, seed: int = 0, threads: int = 4, z_needed: float = 1.0):
    """The proposals kept, in order: [(label, columns)]. ``out_tr`` is the training rows' feature frame,
    ``frames`` the output of ``compute`` (training rows first)."""
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(out_tr)
    y = np.asarray(y, dtype=float)
    idx = np.arange(n)
    if n > MAX_ROWS:
        idx = np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS, replace=False))
    yy = y[idx]
    split = (KFold(5, shuffle=True, random_state=seed + 11) if task == "regression"
             else StratifiedKFold(5, shuffle=True, random_state=seed + 11))
    folds = list(split.split(np.zeros(len(idx)), None if task == "regression" else yy))
    base = out_tr.iloc[idx].reset_index(drop=True)
    if len(idx) * (base.shape[1] + 20) <= 5_000_000:
        threads = 1  # small fits: extra threads only contend (10x slower on a busy machine)
    cur = _oof_rows(task, base, yy, folds, seed, threads)
    kept = []
    for label, v in frames:
        cand = pd.concat([base, v.iloc[:n].iloc[idx].reset_index(drop=True)], axis=1)
        rows = _oof_rows(task, cand, yy, folds, seed, threads)
        d = _winsor(cur - rows, None if task == "regression" else yy)
        z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-300)) if d.std() > 0 else 0.0
        ok = (cur - rows).mean() > 0 and z >= z_needed
        print(f"proposal {label} ({v.shape[1]} columns): CV loss {cur.mean():.6f} -> {rows.mean():.6f} "
              f"(z={z:+.2f}) -> {'kept' if ok else 'dropped'}", flush=True)
        if ok:
            base, cur = cand, rows
            kept.append((label, list(v.columns)))
    return kept
