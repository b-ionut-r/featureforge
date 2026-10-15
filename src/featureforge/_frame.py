"""Column helpers shared by the search and the feature specs: text views, category codes cached
per frame, the shared ``Context`` handed to every spec, and ``F[cols]`` without copies.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from featureforge.utils.memory import available_bytes


TE_SMOOTHING = 20.0
GROUP_STATS = ("mean", "std", "min", "max", "dev", "z")
ARITH_OPS = ("add", "sub", "mul", "div", "rdiv")


def _unique_names(cols) -> list:
    """Column names as strings, a repeated name numbered (``a``, ``a.1``) as pandas numbers a CSV's."""
    out, seen = [], set()
    for c in cols:
        name, k = str(c), 0
        while name in seen:
            k += 1
            name = f"{c}.{k}"
        seen.add(name)
        out.append(name)
    return out


def _is_cat(s: pd.Series) -> bool:
    return (s.dtype == object or s.dtype == bool or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s))


def _as_str(s: pd.Series) -> pd.Series:
    """String view of a column with missing values mapped to an explicit level."""
    return s.astype(object).where(s.notna(), "__NA__").astype(str)


def _as_str_frame_col(s: pd.Series) -> pd.Series:
    """``_as_str`` for a column of a frame, as the ``category`` column ``_compact`` would make of it
    when it already is one (sorted string levels, missing as "__NA__"): one string per row of
    Bosch's 2,140 categorical fields was 9 GB on 240k rows."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        names = s.cat.categories.astype(object).map(str)
        if names.is_unique:
            s = s.cat.rename_categories(names)
            if s.isna().any():
                if "__NA__" not in names:
                    s = s.cat.add_categories("__NA__")
                s = s.fillna("__NA__")
            s = s.cat.remove_unused_categories()
            return s.cat.reorder_categories(sorted(s.cat.categories))
    return _as_str(s)


def _permute_columns(M: np.ndarray, perm) -> None:
    """Reorder a column-major matrix's columns in place (column i takes old column perm[i]),
    one column of scratch memory: cycle by cycle, each column is one contiguous copy."""
    done = np.zeros(M.shape[1], dtype=bool)
    tmp = np.empty(M.shape[0], dtype=M.dtype)
    for s in range(M.shape[1]):
        if done[s] or perm[s] == s:
            done[s] = True
            continue
        tmp[:] = M[:, s]
        i = s
        while True:
            done[i] = True
            j = perm[i]
            if j == s:
                M[:, i] = tmp
                break
            M[:, i] = M[:, j]
            i = j


def _renumbered(X: pd.DataFrame) -> pd.DataFrame:
    """``X.reset_index(drop=True)`` without copying the data: a shallow copy whose columns are
    replaced, never written into, by what follows (Bosch: a second copy of a 4,264-column frame
    held through the whole search took the build past the container's 15 GB)."""
    X = X.copy(deep=False)
    X.index = pd.RangeIndex(len(X))
    return X


def _concat_cats(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Row-wise concat that keeps ``category`` columns as categories (on the union of the parts'
    levels): pandas otherwise falls back to one string object per row, gigabytes on 9M rows."""
    parts = [p.copy(deep=False) for p in parts]
    for c in parts[0].columns:
        if all(c in p.columns and isinstance(p[c].dtype, pd.CategoricalDtype) for p in parts):
            cats = parts[0][c].cat.categories
            for p in parts[1:]:
                cats = cats.union(p[c].cat.categories)
            for p in parts:
                if not p[c].cat.categories.equals(cats):
                    p[c] = p[c].cat.set_categories(cats)
    return pd.concat(parts, ignore_index=True)


def _key_values(s: pd.Series) -> np.ndarray:
    """Hashable, NaN-safe values used to index key categories."""
    if _is_cat(s):
        return _as_str(s).to_numpy()
    return s.astype(float).fillna(-1.234567e300).to_numpy()


ROW_TIME = "__row__"


class Context:
    """Shared task info passed to specs."""

    def __init__(self, task: str, n_classes: int, seed: int):
        self.task = task
        self.n_classes = n_classes
        self.seed = seed


# Each column of a frame is factorised once (codes + levels in order of appearance);
# vocabularies and combined key codes are then built from its few levels instead of
# re-hashing a million strings for every candidate that uses the column.
_COL_CACHE: dict = {}
# Caches are capped by size, oldest entries out first: on a million-row frame a few
# hundred cached code arrays are gigabytes.
_CACHE_BYTES = 1 << 30
_CACHE_LOCK = __import__("threading").Lock()
# scikit-learn's liblinear draws from one process-wide random generator, reseeded at the start of
# each fit: two fits on different threads interleave its draws, so a dual solve's values changed
# from run to run (IEEE-CIS xlin__keys; Amazon's picks). Liblinear fits take turns.
_LIBLINEAR_LOCK = __import__("threading").Lock()


def _warm(ix: pd.Index) -> pd.Index:
    """``ix`` with its hash table built. pandas builds an index's table on its first lookup, and
    threads looking up in the same new index at once can find it half built: ``get_indexer`` then
    takes the unique index for a non-unique one and raises. The level vocabularies cached below are
    shared by every candidate on their key (ICR: one to three candidates lost per round, a different
    count from run to run under load, and with them the picks)."""
    ix.is_unique
    return ix


def _cache_put(cache: dict, key, df, *entry):
    import weakref
    try:
        ref = weakref.ref(df)
    except TypeError:
        return
    size = sum(getattr(e, "nbytes", 0) for e in entry)
    # Smaller when memory runs short: three full caches were 3 GB of Instacart's search over 8.5 million
    # rows, where the 14 GB limit was reached (a dropped entry is only computed again).
    cap = min(_CACHE_BYTES, available_bytes() // 8)
    with _CACHE_LOCK:
        cache[key] = (ref, *entry, len(df), size)
        total = sum(v[-1] for v in cache.values())
        while total > cap and len(cache) > 1:
            old = cache.pop(next(iter(cache)))
            total -= old[-1]


def _col_codes(df: pd.DataFrame, c: str):
    hit = _COL_CACHE.get((id(df), c))
    if hit is not None and hit[0]() is df and hit[3] == len(df):
        return hit[1], hit[2]
    codes, levels = pd.factorize(_key_values(df[c]))
    levels = _warm(pd.Index(levels))
    codes = codes.astype(np.int32) if len(levels) < 2 ** 31 else codes
    _cache_put(_COL_CACHE, (id(df), c), df, codes, levels)
    return codes, levels


def _fit_vocab(df: pd.DataFrame, cols: Sequence[str]) -> List[pd.Index]:
    return [_col_codes(df, c)[1] for c in cols]


def _codes(df: pd.DataFrame, cols: Sequence[str], vocabs: List[pd.Index]) -> np.ndarray:
    """Combined int64 code of key columns; values unseen at fit time get their own code."""
    out = np.zeros(len(df), dtype=np.int64)
    for c, vocab in zip(cols, vocabs):
        codes, levels = _col_codes(df, c)
        pos = vocab.get_indexer(levels)  # each of the frame's levels in the vocabulary
        pos[pos < 0] = len(vocab)
        out = out * (len(vocab) + 1) + pos[codes]
    return out


# Many candidates share the same key columns (every group statistic of key k,
# every count / encoding of k); factorising k once per frame instead of twice
# per candidate removes most of their materialisation time.
_CODE_CACHE: dict = {}


def _code_cache_get(df, cols):
    hit = _CODE_CACHE.get((id(df), tuple(cols)))
    if hit is None or hit[0]() is not df or hit[3] != len(df):
        return None
    return hit[1], hit[2]


def _code_cache_put(df, cols, vocabs, codes):
    _cache_put(_CODE_CACHE, (id(df), tuple(cols)), df, vocabs, codes)


_GROUP_CACHE: dict = {}


def _grouper(df, cols, k):
    """``k`` (combined key codes of ``df[cols]``) as a categorical over its sorted distinct
    values: group statistics of many columns by the same key hash the key once instead of
    once per statistic. Groups, their order and the rows in each are those of ``groupby(k)``."""
    hit = _GROUP_CACHE.get((id(df), tuple(cols)))
    if hit is not None and hit[0]() is df and hit[3] == len(df):
        return hit[1], hit[2]
    codes, levels = pd.factorize(k, sort=True)
    g = pd.Categorical.from_codes(codes.astype(np.int32), categories=pd.Index(levels), validate=False)
    _warm(g.categories)
    _cache_put(_GROUP_CACHE, (id(df), tuple(cols)), df, g, levels)
    return g, levels


def X_nunique(df: pd.DataFrame, c: str) -> int:
    return int(df[c].nunique(dropna=False))


def _columns(F: pd.DataFrame, cols) -> pd.DataFrame:
    """``F[cols]`` as views of F's columns: selecting copies every column, which on a wide
    table is gigabytes per gate fit."""
    if not F.columns.is_unique or len(set(cols)) != len(cols):
        return F[cols]
    return pd.DataFrame({c: F[c] for c in cols}, copy=False)
