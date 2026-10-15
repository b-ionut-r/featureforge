"""LightGBM objectives and the losses the search scores with: per-row losses for the gate's paired
test, and the stepped (one Newton step per cell) losses of the residual probes.
"""
from __future__ import annotations

import numpy as np


def _lgb_objective(task: str, n_classes: int) -> dict:
    if task == "regression":
        return {"objective": "regression", "metric": "l2"}
    if task == "binary":
        return {"objective": "binary", "metric": "binary_logloss"}
    return {"objective": "multiclass", "metric": "multi_logloss", "num_class": n_classes}


def _row_loss(task, y, margin) -> np.ndarray:
    """Per-row loss of raw margins in the LightGBM objective's own units."""
    y = np.asarray(y)
    if task == "regression":
        return (y - margin) ** 2
    if task == "binary":
        p = 1 / (1 + np.exp(-margin))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return _class_row_loss(y, margin - margin.max(axis=1, keepdims=True))


def _class_row_loss(y, m) -> np.ndarray:
    """Softmax log loss per row from margins ``m`` already shifted by their row maximum. ``m`` is
    overwritten: rows x classes arrays of half a million rows take 400 MB a copy."""
    picked = m[np.arange(len(y)), y.astype(int)]
    np.exp(m, out=m)
    return -(picked - np.log(m.sum(axis=1)))


def _loss(task, y, margin) -> float:
    return float(np.mean(_row_loss(task, y, margin)))


def _stepped_loss(task, y, margin, steps, cell) -> float:
    """``_loss(task, y, margin + steps[cell])`` for a binary or regression target, a few thousand rows at a time
    in a buffer that stays in cache, every operation in place: the same elementwise operations in the same
    order, then the mean over all rows, so the same value. Spelled out, each step allocated a full row vector
    (half a million rows: about ten passes through memory per probe)."""
    y, out = np.asarray(y), np.empty(len(y))
    for i in range(0, len(y), 8192):
        m = steps[cell[i:i + 8192]]
        m += margin[i:i + 8192]
        yc = y[i:i + 8192]
        if task == "regression":
            np.subtract(yc, m, out=m)
            np.square(m, out=out[i:i + 8192])
            continue
        np.negative(m, out=m)
        np.exp(m, out=m)
        m += 1
        np.divide(1, m, out=m)
        np.clip(m, 1e-15, 1 - 1e-15, out=m)
        a = np.log(m)
        a *= yc
        np.subtract(1, m, out=m)
        np.log(m, out=m)
        m *= 1 - yc
        a += m
        np.negative(a, out=out[i:i + 8192])
    return float(np.mean(out))


def _fold_steps(idx, n_folds, n_cells, g, h, lam) -> np.ndarray:
    """Regularised Newton step of every cell from the other folds' gradients: (folds, cells[, classes]).

    ``idx`` is fold * n_cells + cell per row. With many classes, one (cells x rows) indicator times the
    (rows x classes) gradients adds each cell's rows in row order, as one bincount per class does, so the
    steps are bit-identical without a strided copy of every class column.
    """
    if g.ndim == 1:
        G = np.bincount(idx, g, minlength=n_folds * n_cells).reshape(n_folds, n_cells)
        H = np.bincount(idx, h, minlength=n_folds * n_cells).reshape(n_folds, n_cells)
        return -(G.sum(0) - G) / (H.sum(0) - H + lam)
    if g.shape[1] < 8:
        return np.stack([_fold_steps(idx, n_folds, n_cells, g[:, k], h[:, k], lam) for k in range(g.shape[1])], 2)
    from scipy import sparse
    A = sparse.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(n_folds * n_cells, len(idx)))
    G = (A @ g).reshape(n_folds, n_cells, -1)
    H = (A @ h).reshape(n_folds, n_cells, -1)
    return -(G.sum(0) - G) / (H.sum(0) - H + lam)


def _stepped_class_loss(y, margin, idx, n_cells, K, g, h, lam) -> float:
    """Multiclass loss of ``margin`` plus, per class, the cross-fitted Newton step of each row's
    cell: rows of fold f in cell c (``idx = f * n_cells + c``) get the step fitted on the other
    folds' rows of c. The steps are gathered from the small (fold, cell) x classes table, a few
    thousand rows at a time, into a buffer that stays in cache while it is shifted and
    exponentiated in place: a whole rows x classes copy is 400 MB on the Expedia stand-in (510k
    rows x 100 classes), with four probes at a time. Same values as adding the per-class
    corrections column by column: every step is per row."""
    steps = _fold_steps(idx, K, n_cells, g, h, lam).reshape(K * n_cells, -1)
    y, out = np.asarray(y), np.empty(len(y))
    for i in range(0, len(y), 4096):
        x = steps[idx[i:i + 4096]]
        x += margin[i:i + 4096]
        x -= x.max(axis=1, keepdims=True)
        out[i:i + 4096] = _class_row_loss(y[i:i + 4096], x)
    return float(np.mean(out))
