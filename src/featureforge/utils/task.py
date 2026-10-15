"""Task inference from the target column."""
from __future__ import annotations

import pandas as pd


def infer_task(y: pd.Series) -> str:
    """``"binary"``, ``"multiclass"`` or ``"regression"``, read from the target's values and dtype."""
    y = pd.Series(y)
    if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or y.dtype == bool:
        return "binary" if y.nunique() == 2 else "multiclass"
    n_unique = y.nunique()
    if n_unique == 2:
        return "binary"
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "multiclass"
    return "regression"
