"""Neighbour and linear meta-features: out-of-fold k-nearest-neighbour target means and per-class
distances, an out-of-fold spline-additive ridge model, and PCA / PLS projections.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from featureforge.specs.base import Spec, _OneHot
from featureforge.specs.encoding import _class_labels


def _one_blas_thread():
    """BLAS on one thread: a ridge fit or a PLS projection sums in an order set by the BLAS thread
    count, so its last bits (and the picks they tip) followed the machine's core count (PLS on
    128,000 x 24: four different results on 1, 2, 3 and 4 threads). Candidates are built on
    several threads at once anyway."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1, user_api="blas")


def _nn_algo(M) -> str:
    """kd-trees degrade sharply past ~5 dimensions; blocked brute force does not."""
    return "brute" if M.shape[1] > 5 else "auto"


class KNNTarget(Spec):
    """Mean target of the k nearest training rows in a standardised numeric subspace.

    A classic contest meta-feature: it injects local, smooth structure that
    axis-aligned trees approximate poorly. Training rows are encoded out of
    fold (a row is never its own neighbour); new rows query all training rows.
    Multiclass targets give per-class neighbour frequencies.
    """
    target_dep = True

    def __init__(self, cols: Sequence[str], ks: Sequence[int], n_classes: int, label: str,
                 weights: Optional[Sequence[float]] = None, rank: bool = False):
        super().__init__(cols)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.rank = rank
        self.ks = tuple(ks)
        self.n_classes = n_classes
        self.per = n_classes if n_classes > 2 else 1
        self.n_out = len(self.ks) * self.per
        self.name = f"knn__{label}"

    def _matrix(self, df, fit=False):
        M = df[self.parents].to_numpy(dtype=float)
        if getattr(self, "rank", False):
            # Rank-gauss: skewed / heavy-tailed axes get comparable spread.
            from scipy.special import ndtri
            if fit:
                self.qgrid_ = [np.unique(np.nanquantile(M[:, j], np.linspace(0, 1, 201)))
                               if np.isfinite(M[:, j]).any() else np.array([0.0])
                               for j in range(M.shape[1])]
            M = np.column_stack([
                ndtri(np.clip(np.interp(M[:, j], g, np.linspace(0, 1, len(g))) if len(g) > 1
                              else np.full(len(M), 0.5), 0.005, 0.995))
                for j, g in enumerate(self.qgrid_)])
            M = np.where(np.isfinite(df[self.parents].to_numpy(dtype=float)), M, np.nan)
        if fit:
            lo, hi = np.nanpercentile(M, 1, axis=0), np.nanpercentile(M, 99, axis=0)
            self.lo_, self.hi_ = lo, hi
            Mc = np.clip(M, lo, hi)
            self.med_ = np.nanmedian(Mc, axis=0)
            Mc = np.where(np.isnan(Mc), self.med_, Mc)
            self.mu_, self.sd_ = Mc.mean(0), Mc.std(0) + 1e-12
        Mc = np.clip(M, self.lo_, self.hi_)
        Mc = np.where(np.isnan(Mc), self.med_, Mc)
        Z = (Mc - self.mu_) / self.sd_
        if self.weights is not None:
            Z = Z * self.weights
        return Z.astype(np.float32)

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        # Classes kept as labels: the neighbour families only read which class a row is
        # (np.eye(100)[y] over 637k rows was 510 MB, and each fold took a copy of its rows).
        return _OneHot(y.astype(int), self.n_classes) if self.per > 1 else y[:, None]

    def _query(self, M_ref, Y_ref, M_q):
        from sklearn.neighbors import NearestNeighbors
        kmax = min(max(self.ks), len(M_ref))
        nn = NearestNeighbors(n_neighbors=kmax, algorithm=_nn_algo(M_ref)).fit(M_ref)
        _, idx = nn.kneighbors(M_q)
        out = np.empty((len(M_q), self.n_out))
        if self.per > 1:
            # Class counts among the first k neighbours, without the (rows x k x classes) one-hot
            # gather: 80k query rows x 50 neighbours x 100 classes is 3.2 GB twice over. Counts
            # are exact integers, so the frequencies equal the one-hot cumulative sums.
            lab = _class_labels(Y_ref)[idx]  # (n, kmax)
            base = np.arange(len(M_q))[:, None] * self.per
            for i, k in enumerate(self.ks):
                k = min(k, kmax)
                cnt = np.bincount((base + lab[:, :k]).ravel(), minlength=len(M_q) * self.per)
                out[:, i * self.per:(i + 1) * self.per] = cnt.reshape(len(M_q), self.per) / k
            return out
        csum = np.cumsum(Y_ref[idx], axis=1)  # (n, kmax, 1)
        for i, k in enumerate(self.ks):
            k = min(k, kmax)
            out[:, i * self.per:(i + 1) * self.per] = csum[:, k - 1, :] / k
        return out

    def fit(self, df, y, ctx):
        self.M_ = self._matrix(df, fit=True)
        self.Y_ = self._targets(y)
        return self

    def transform(self, df, ctx):
        return self._query(self.M_, self.Y_, self._matrix(df))

    def fit_transform_oof(self, df, y, ctx, folds):
        self.fit(df, y, ctx)
        out = np.empty((len(df), self.n_out))
        for tr, va in folds:
            out[va] = self._query(self.M_[tr], self.Y_[tr], self.M_[va])
        return out


class KNNClassDist(KNNTarget):
    """Mean distance to the k nearest training rows of each class (k = 1, 2, 4, ...).

    The kNN distance features of top Otto / contest solutions: how close a row
    sits to each class's manifold, a margin-like signal trees cannot form
    from raw axes. Out of fold on training rows.
    """
    def __init__(self, cols, ks, n_classes, label, weights=None):
        super().__init__(cols, ks, n_classes, label, weights)
        self.n_out = len(self.ks) * n_classes
        self.name = f"knnd__{label}"
        # With several classes its signal is which class sits nearest, read across the
        # per-class columns; one column at a time the residual probe scores it near zero, so
        # the screen dropped it by chance (wine_white: +12% CV in the joint model, novelty
        # -0.003 to +0.001 between runs). Binary targets keep the screen (churn lost 2-8%).
        self.screen_exempt = n_classes > 2

    def fit_transform_oof(self, df, y, ctx, folds):
        self.folds_ = [np.asarray(tr) for tr, _ in folds]
        return super().fit_transform_oof(df, y, ctx, folds)

    def transform(self, df, ctx):
        # Distances shrink as the reference grows: a training row's out-of-fold distances
        # come from (k-1)/k of the rows, so new rows are measured against the same fold
        # references and averaged. Measured against all rows, they sat nearer every class
        # than any training row (Telstra held-out: +0.004 / +0.028 log loss with them).
        folds = getattr(self, "folds_", None)
        if not folds:
            return super().transform(df, ctx)
        M_q = self._matrix(df)
        return sum(self._query(self.M_[tr], self.Y_[tr], M_q) for tr in folds) / len(folds)

    def _query(self, M_ref, Y_ref, M_q):
        from sklearn.neighbors import NearestNeighbors
        yc = _class_labels(Y_ref) if isinstance(Y_ref, _OneHot) or Y_ref.shape[1] > 1 else Y_ref[:, 0].astype(int)
        out = np.zeros((len(M_q), self.n_out))
        kmax = max(self.ks)
        for c in range(self.n_classes):
            R = M_ref[yc == c]
            if len(R) == 0:
                out[:, c * len(self.ks):(c + 1) * len(self.ks)] = np.nan
                continue
            kk = min(kmax, len(R))
            d, _ = NearestNeighbors(n_neighbors=kk, algorithm=_nn_algo(R)).fit(R).kneighbors(M_q)
            cs = np.cumsum(d, axis=1)
            for i, k in enumerate(self.ks):
                k = min(k, kk)
                out[:, c * len(self.ks) + i] = cs[:, k - 1] / k
        return out


class LinearOOF(KNNTarget):
    """Out-of-fold prediction of a spline-additive linear model (a ridge GAM).

    Smooth additive structure spread over many numerics (``sum_i f_i(x_i)``)
    costs a tree ensemble many splits; one stacked GAM margin hands it over
    in a single column. Training rows are encoded out of fold.
    """

    def __init__(self, cols: Sequence[str], n_classes: int, label: str):
        super().__init__(cols, (1,), n_classes, label)
        self.name = f"gam__{label}"

    def _model(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler
        est = Ridge(alpha=10.0) if self.n_classes == 0 else LogisticRegression(C=0.3, max_iter=300)
        return make_pipeline(SplineTransformer(n_knots=6, degree=3), StandardScaler(), est)

    def _query(self, M_ref, Y_ref, M_q):
        m = self._model()
        if self.n_classes == 0:
            return m.fit(M_ref, Y_ref[:, 0]).predict(M_q)[:, None]
        yc = _class_labels(Y_ref) if self.per > 1 else Y_ref[:, 0].astype(int)
        m.fit(M_ref, yc)
        if self.per == 1:
            return m.decision_function(M_q)[:, None]
        P = np.full((len(M_q), self.per), 1e-6)
        P[:, m.classes_] = m.predict_proba(M_q)
        return np.log(P)


class Projection(KNNTarget):
    """Linear projections of standardised numerics: PCA (unsupervised) or PLS.

    Trees split on one axis at a time; a rotated basis turns oblique
    structure (``lat + lon``-like directions across many columns) into
    single splits. PLS directions are fitted out of fold on training rows.
    """

    def __init__(self, cols: Sequence[str], kind: str, n_comp: int, n_classes: int, label: str):
        super().__init__(cols, (1,), n_classes, label)
        self.kind = kind
        self.n_out = min(n_comp, len(cols))
        self.target_dep = kind == "pls"
        self.name = f"{kind}__{label}"

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        # PLS regresses on the one-hot matrix itself.
        return np.eye(self.n_classes)[y.astype(int)] if self.per > 1 else y[:, None]

    def _proj(self, M_ref, Y_ref):
        if self.kind == "pca":
            from sklearn.decomposition import PCA
            return PCA(self.n_out, random_state=0).fit(M_ref)
        from sklearn.cross_decomposition import PLSRegression
        return PLSRegression(self.n_out, scale=False).fit(M_ref, Y_ref)

    def _query(self, M_ref, Y_ref, M_q):
        return self._proj(M_ref, Y_ref).transform(M_q)

    def fit(self, df, y, ctx):
        super().fit(df, y if y is not None else np.zeros(len(df)), ctx)
        self.model_ = self._proj(self.M_, self.Y_)
        return self

    def transform(self, df, ctx):
        return self.model_.transform(self._matrix(df))
