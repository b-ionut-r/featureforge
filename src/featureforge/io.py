"""Reading tables the way every block expects them.

Text becomes categories on pandas 2 and 3 alike (so the output does not depend on the pandas version),
mixed ``0`` / ``"0"`` cells one level, list cells joined items, and float64 float32.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from featureforge.utils.text import as_text


def read(path: str) -> pd.DataFrame:
    df = normalise(pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False))
    release_arrow()
    return df


def release_arrow():
    """Hand back the memory pyarrow's allocator keeps after a parquet read (Bosch: 2.6 GB of a
    2.6 GB table, held through the whole search)."""
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if isinstance(df[c].dtype, pd.StringDtype):
            # pandas 3 reads text as its str dtype, pandas 2 as object: text becomes the categorical the object path
            # below gives, so the output does not depend on the pandas version (Rossmann's Date stayed text under
            # pandas 3 and the panel family, which reads categories as levels, did not see it). Coded from the
            # column's distinct values: a Python string a row, next to the column's own buffers, was killed on a
            # 13.7M-row event log's text columns under its memory limit. (Text holds no list cells.)
            s = df[c]
            codes, uniq = pd.factorize(s, sort=True)
            cats = pd.Series(np.asarray(uniq, dtype=object), dtype=object).astype("category").cat.categories
            remap = np.append(cats.get_indexer(np.asarray(uniq, dtype=object)), -1)   # (-1, missing, stays -1)
            codes = remap[codes]
            df[c] = pd.Series(pd.Categorical.from_codes(codes, dtype=pd.CategoricalDtype(cats)), index=s.index, name=c)
            del s, codes
            continue
        if df[c].dtype == object and isinstance(df[c].dropna().iloc[:1].tolist()[0] if df[c].notna().any() else None,
                                                (list, tuple, np.ndarray)):
            # List cells (amenities, photo URLs): items joined, so they stay readable as a list column.
            df[c] = df[c].map(lambda v: " ; ".join(str(i).replace(";", ",") for i in v)
                              if isinstance(v, (list, tuple, np.ndarray)) else v)
        if df[c].dtype == object:
            # Mixed 0 / "0" (pandas reads a CSV column chunk by chunk) become one level.
            df[c] = df[c].where(df[c].isna(), as_text(df[c])).astype("category")
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df
