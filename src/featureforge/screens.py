"""Column screens run before and during the search: column families (``px1..px784``), numbered slot
families, latitude/longitude pairs, synthetic unlabeled rows, fields absent from the test rows,
and the robust statistics (winsorised gains, later-half and block z) the gate reads.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from featureforge._frame import _as_str
from featureforge.specs.dates import _is_date_like, _to_days
from featureforge.utils.text import as_text


def column_families(cols: Sequence[str], min_size: int = 3) -> Dict[str, List[str]]:
    """Group columns sharing a name stem that ends in a running index (``Soil_Type_12``, ``px3``)."""
    import re
    fam: Dict[str, List[str]] = {}
    idx: Dict[str, int] = {}
    for c in cols:
        m = re.match(r"^(.*?)[_.\-]?(\d+)$", str(c))
        if m and m.group(1):
            fam.setdefault(m.group(1), []).append(c)
            idx[c] = int(m.group(2))
    return {k: sorted(v, key=idx.get) for k, v in fam.items() if len(v) >= min_size}


def _winsor(d, y=None, q=0.01):
    """Clip paired per-row loss differences at their ``q`` / ``1 - q`` quantiles, so a handful
    of extreme rows cannot carry a gate decision. For class labels the quantiles are taken
    within each class: with rare positives every positive row sat in the global tails (H&M,
    0.4% positives: a set with a worse mean loss scored z=+190 once they were clipped)."""
    d = np.asarray(d, dtype=float)
    if y is None:
        lo, hi = np.quantile(d, [q, 1 - q])
        return np.clip(d, lo, hi)
    out = d.copy()
    y = np.asarray(y)
    for k in np.unique(y):
        m = y == k
        lo, hi = np.quantile(d[m], [q, 1 - q])
        out[m] = np.clip(d[m], lo, hi)
    return out


def _later_half(dw, t):
    """Mean of the paired differences over the later half of the gate rows by time: the test
    rows come after the gate, so a gain that has already reversed there does not carry."""
    order = np.argsort(np.asarray(t, dtype=float), kind="stable")
    return float(dw[order[len(order) // 2:]].mean())


def _block_z(d, blocks):
    """One-sided z of the mean paired difference with each block (period) as one observation."""
    m = np.bincount(blocks, weights=d) / np.maximum(np.bincount(blocks), 1)
    m = m[np.bincount(blocks) > 0]
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)) + 1e-300))


def synthetic_rows(X: pd.DataFrame, U: pd.DataFrame, min_rate: float = 0.9, min_cols: int = 20) -> np.ndarray:
    """Unlabeled rows that look generated from other rows' values (Santander 2019's
    fake test rows): no value of theirs is unique among all known rows, while
    almost every labelled row has one. Label-free; only tables with many
    continuous columns qualify, and only when the unlabeled set has clearly more
    such rows than the labelled set. Returns a boolean mask over ``U``'s rows."""
    none = np.zeros(len(U), dtype=bool)
    cols = [c for c in X.columns if c in U.columns and pd.api.types.is_float_dtype(X[c])
            and pd.api.types.is_numeric_dtype(U[c]) and X[c].nunique() >= 0.05 * len(X)]
    if len(cols) < min_cols or len(U) < 1000:
        return none
    ux, uu = np.zeros(len(X), dtype=bool), np.zeros(len(U), dtype=bool)
    for c in cols:
        a, b = X[c].to_numpy(dtype=float), U[c].to_numpy(dtype=float)
        _, inv, cnt = np.unique(np.concatenate([a, b]), return_inverse=True, return_counts=True)
        once = cnt[inv] == 1
        ux |= once[:len(a)]
        uu |= once[len(a):]
    r_x, miss_u = ux.mean(), 1.0 - uu.mean()
    if r_x < min_rate or miss_u < 2 * (1.0 - r_x) + 0.05:
        return none
    return ~uu


def slot_families(df: pd.DataFrame, cat_cols: Sequence[str]) -> List[tuple]:
    """Categorical column families that hold one multi-valued field in numbered slots
    (``component_id_1..8``, ``spec1..10``), each with its paired numeric family on the
    same slot numbers when there is one (``quantity_1..8``)."""
    import re
    cats = column_families([c for c in df.columns if c in cat_cols])
    nums = column_families([c for c in df.columns if c not in cat_cols])
    idx = lambda c: re.match(r"^.*?(\d+)$", str(c)).group(1)
    out = []
    for stem, cols in cats.items():
        # Slots of one field fill left to right (later slots mostly empty) and share their
        # values; a family of distinct anonymised variables (cat1..cat116) does neither.
        na = [df[c].isna().mean() + (_as_str(df[c]) == "__NA__").mean() for c in cols]
        if na[-1] < na[0] + 0.2:
            continue
        sets = [set(pd.unique(as_text(df[c].dropna()))) - {"__NA__"} for c in cols[:3]]
        if len(sets[0]) < 5 or len(sets[0] & sets[1]) < 0.2 * min(len(sets[0]), len(sets[1]) or 1):
            continue
        slots = [idx(c) for c in cols]
        pair = next((v for v in nums.values() if [idx(c) for c in v] == slots), None)
        out.append((stem, cols, pair))
    return out


def _absent_in_test(X: pd.DataFrame, U: pd.DataFrame) -> List[str]:
    """Columns at least 80% one value in ``U`` that at most 5% of ``X`` holds, while ``X`` has 10+ values."""
    out = []
    for c in X.columns:
        if c not in U.columns or len(U) < 50:
            continue
        vb = U[c].value_counts(dropna=False, normalize=True)
        if not len(vb) or vb.iloc[0] < 0.8 or X[c].nunique(dropna=False) < 10:
            continue
        v = vb.index[0]
        s_tr = float(X[c].isna().mean()) if pd.isna(v) else float((X[c] == v).mean())
        if s_tr <= 0.05 and not _later_than_train(X[c], U[c]):
            out.append(c)
    return out


def _later_than_train(x: pd.Series, u: pd.Series) -> bool:
    """``u`` (nearly) all beyond every value of ``x``: a test that is one later period (a single day or
    month) is constant on its date or period id. Such a column orders the test after training, and
    time detection needs it; it is not a field the test lacks."""
    if _is_date_like(x) and _is_date_like(u):
        a, b = _to_days(x), _to_days(u)
    elif pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(u) and not pd.api.types.is_bool_dtype(x):
        a, b = x.to_numpy(dtype=float, na_value=np.nan), u.to_numpy(dtype=float, na_value=np.nan)
    else:
        return False
    if not np.isfinite(a).any():
        return False
    return bool(np.mean(b > np.nanmax(a)) >= 0.98)


def _numbered_codes(s: pd.Series) -> bool:
    """A string column whose values are one shared prefix and a number."""
    v = pd.Series(pd.unique(as_text(s)))
    v = v[v != "__NA__"]
    if len(v) <= 10:
        return False
    m = v.str.extract(r"^(\D*?)\s*(\d+)\s*$")
    ok = m[1].notna()
    pre = m.loc[ok, 0]
    return bool(ok.mean() >= 0.95 and pre.nunique() == 1 and pre.iloc[0].strip().isalpha())


def geo_points(cols: Sequence[str]) -> List[tuple]:
    """(lat, lon) column pairs found by name: ``*lat*`` matched with the same name using ``lon``/``lng``."""
    import re
    out = []
    low = {c.lower(): c for c in cols}
    for c in cols:
        lc = c.lower()
        if "lat" not in lc:
            continue
        for rep in ("lon", "lng", "long"):
            for pat in ("latitude", "lat"):
                if pat in lc:
                    cand = lc.replace(pat, "longitude" if rep == "lon" and pat == "latitude" else rep)
                    if cand in low and low[cand] != c:
                        out.append((c, low[cand]))
                        break
            else:
                continue
            break
    return list(dict.fromkeys(out))
