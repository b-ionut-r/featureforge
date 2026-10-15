"""Free-text features: how a field is written, TF-IDF SVD components, and out-of-fold sparse linear
models on words and character n-grams, or on one-hot keys and all their pairwise crosses.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from featureforge._frame import _LIBLINEAR_LOCK, _codes, _fit_vocab, _is_cat
from featureforge.specs.base import Spec
from featureforge.utils.text import as_text, as_text_frame


def _is_text(s: pd.Series, n_sample: int = 20_000) -> bool:
    """Free text: a string column of mostly distinct values written as several words
    (titles, descriptions, comments), as opposed to codes, ids or category labels."""
    if not _is_cat(s) or len(s) < 100:
        return False
    v = as_text(s.dropna())
    if len(v) > n_sample:
        v = v.sample(n_sample, random_state=0)
    if len(v) < 50 or v.nunique() < max(50, 0.05 * len(v)):
        return False
    # Several space-separated words, not one code made of letters and digits.
    return (float(v.str.count(r"[^\W\d_]{2,}").mean()) >= 2.0
            and float(v.str.split().str.len().mean()) >= 2.5)


def _text_values(df, c, clip: Optional[int] = None) -> pd.Series:
    t = as_text(df[c])
    t = t.where(t != "__NA__", "")
    # Long fields (review dumps) cost n-gram time quadratically in nothing but their tails.
    return t if clip is None else t.str.slice(0, clip)


def _text_parts(df, c, clip: Optional[int] = None):
    """``_text_values`` as its distinct texts and each row's index into them: texts repeated over
    many rows (a game's rules on each of its matches) are transformed once, not once per row."""
    s = df[c]
    if isinstance(s.dtype, pd.CategoricalDtype):
        codes, u = s.cat.codes.to_numpy().astype(np.int64), pd.Series(s.cat.categories.astype(object))
    else:
        codes, u = pd.factorize(s)
        u = pd.Series(np.asarray(u, dtype=object))
    u = pd.concat([u, pd.Series([np.nan], dtype=object)], ignore_index=True)
    return _text_values(pd.DataFrame({c: u}), c, clip), np.where(codes < 0, len(u) - 1, codes)


TEXT_CLIP = 1500


def _text_nnz(df, c, seed=0, n=300):
    """A text column's nonzeros a row in the text models' matrices: distinct words and word pairs, and when its
    texts average 80 characters or fewer (as TextLinearOOF decides) distinct character 2-4-grams, over a seeded
    sample of rows. For pricing only: what the models learn does not depend on it."""
    t, k = _text_parts(df, c, TEXT_CLIP)
    if not len(k):
        return 0.0, 0.0
    pick = t.to_numpy()[k[np.random.default_rng(seed).choice(len(k), min(n, len(k)), replace=False)]]
    pick = ["" if not isinstance(d, str) else d for d in pick]
    words = float(np.mean([len(set(_tfidf("word", 1).build_analyzer()(d))) for d in pick]))
    chars = 0.0
    if np.mean([len(d) for d in pick]) <= 80:
        chars = float(np.mean([len(set(_tfidf("char", 1).build_analyzer()(d))) for d in pick]))
    return words, chars


class TextStats(Spec):
    """How a free-text field is written: characters, words, digits, capitals,
    punctuation, distinct-word share and emptiness. Label-free."""
    n_out = 7

    def __init__(self, col: str):
        super().__init__([col])
        self.name = f"txtstat__{col}"

    def transform(self, df, ctx):
        t, codes = _text_parts(df, self.parents[0])
        n_ch = t.str.len().to_numpy(dtype=float)
        words = t.str.lower().str.findall(r"\w+")
        n_w = words.str.len().to_numpy(dtype=float)
        n_u = words.map(lambda w: len(set(w))).to_numpy(dtype=float)
        dig = t.str.count(r"\d").to_numpy(dtype=float)
        up = t.str.count(r"[A-Z]").to_numpy(dtype=float)
        pun = t.str.count(r"[^\w\s]").to_numpy(dtype=float)
        den = np.maximum(n_ch, 1.0)
        return np.column_stack([n_ch, n_w, dig / den, up / den, pun / den,
                                n_u / np.maximum(n_w, 1.0), (n_ch == 0).astype(float)]).astype(np.float32)[codes]


def _tfidf(kind: str, max_features: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=max_features,
                               sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float32)


def _doc_parts(df, extra, c, clip):
    """``_text_values`` of ``df`` then ``extra`` (when it has the column) as distinct texts and each
    row's index into them."""
    t, k = _text_parts(df, c, clip)
    if extra is not None and c in extra:
        t2, k2 = _text_parts(extra, c, clip)
        t, k = pd.concat([t, t2], ignore_index=True), np.concatenate([k, k2 + len(t)])
    return t, k


def _sample_rows(k, n):
    """The rows ``Series.sample(n, random_state=0)`` picks from a column of len(k) values."""
    return pd.Series(k).sample(n, random_state=0).to_numpy() if len(k) > n else k


def _fit_counted(vec, texts, k):
    """``vec.fit(texts[k])`` from the distinct texts alone: the vocabulary and idf ``fit`` learns on
    the repeated corpus, each distinct text analysed once (MCTS: a game's rules, thousands of
    characters, on each of its matches; analysing every copy was gigabytes and minutes per
    vectoriser). Exact while every term's total count stays below 2**24, the float32 sums'
    exact range; otherwise, and for options this does not reproduce, ``fit`` on the copies."""
    from collections import Counter
    from numbers import Integral
    from sklearn.feature_extraction.text import TfidfTransformer
    if (vec.vocabulary is not None or vec.binary or not vec.use_idf or vec.max_features is None
            or np.dtype(vec.dtype) != np.float32):
        return vec.fit(texts.to_numpy()[k])
    order = pd.unique(k)                              # distinct texts in order of first use
    mult = np.bincount(k, minlength=len(texts))[order]
    analyze = vec.build_analyzer()
    dfc, tfc = {}, {}
    for d, m in zip(texts.to_numpy()[order], mult):
        for term, n in Counter(analyze(d)).items():
            dfc[term] = dfc.get(term, 0) + int(m)
            tfc[term] = tfc.get(term, 0) + int(m) * n
    if not dfc or max(tfc.values()) >= 1 << 24:
        return vec.fit(texts.to_numpy()[k])
    n_doc = len(k)
    terms = sorted(dfc)
    dfs = np.array([dfc[t] for t in terms], dtype=np.int64)
    high = vec.max_df if isinstance(vec.max_df, Integral) else vec.max_df * n_doc
    low = vec.min_df if isinstance(vec.min_df, Integral) else vec.min_df * n_doc
    if high < low:
        raise ValueError("max_df corresponds to < documents than min_df")
    mask = (dfs <= high) & (dfs >= low)
    if mask.sum() > vec.max_features:                 # as CountVectorizer._limit_features
        tfs = np.array([tfc[t] for t in terms], dtype=np.float32)
        inds = (-tfs[mask]).argsort()[:vec.max_features]
        keep = np.zeros(len(dfs), dtype=bool)
        keep[np.where(mask)[0][inds]] = True
        mask = keep
    if not mask.any():
        raise ValueError("After pruning, no terms remain. Try a lower min_df or a higher max_df.")
    new = np.cumsum(mask) - 1
    pos = {t: i for i, t in enumerate(terms)}
    seen = dict.fromkeys(t for d in texts.to_numpy()[order] for t in analyze(d))   # first-use order
    vec.fixed_vocabulary_ = False
    vec.vocabulary_ = {t: new[pos[t]] for t in seen if mask[pos[t]]}
    df = dfs[mask].astype(np.float32)                 # as TfidfTransformer.fit
    df += float(vec.smooth_idf)
    idf = np.full_like(df, fill_value=n_doc + int(vec.smooth_idf), dtype=np.float32)
    idf /= df
    np.log(idf, out=idf)
    idf += 1.0
    vec._tfidf = TfidfTransformer(norm=vec.norm, use_idf=vec.use_idf, smooth_idf=vec.smooth_idf,
                                  sublinear_tf=vec.sublinear_tf)
    vec._tfidf.idf_ = idf
    vec._tfidf.n_features_in_ = len(vec.vocabulary_)
    return vec


class TextSVD(Spec):
    """Latent semantic components of a free-text field: TF-IDF of words and word
    pairs reduced by truncated SVD, fitted label-free on every row whose text is
    known. Topic-like directions a tree can split on."""

    def __init__(self, col: str, n_comp: int = 16, max_fit_rows: int = 200_000):
        super().__init__([col])
        self.n_out = n_comp
        self.max_fit_rows = max_fit_rows
        self.name = f"txtsvd__{col}"

    def fit(self, df, y, ctx):
        from sklearn.decomposition import TruncatedSVD
        c = self.parents[0]
        texts, k = _doc_parts(df, getattr(ctx, "extra_rows", None), c, TEXT_CLIP)
        k = _sample_rows(k, self.max_fit_rows)
        self.vec_ = _fit_counted(_tfidf("word", 50_000), texts, k)
        T = self.vec_.transform(texts)[k]
        self.n_out = int(min(self.n_out, max(1, T.shape[1] - 1)))
        self.svd_ = TruncatedSVD(self.n_out, random_state=0, algorithm="randomized", n_iter=4).fit(T)
        return self

    def transform(self, df, ctx):
        t, codes = _text_parts(df, self.parents[0], TEXT_CLIP)
        return self.svd_.transform(self.vec_.transform(t)).astype(np.float32)[codes]


class TextLinearOOF(Spec):
    """Out-of-fold sparse linear model on the words (and character n-grams) of
    free-text fields, optionally with one-hot keys alongside: the ridge / logistic
    model behind most text-tabular contest wins (Mercari, Avito), as one column a
    tree ensemble can use. Vocabularies are fitted label-free on every row whose
    text is known; the model's training rows are encoded out of fold."""
    target_dep = True

    def __init__(self, cols: Sequence[str], n_classes: int, keys: Sequence[str] = (), label: str = "",
                 chars: bool = True):
        super().__init__(list(cols) + list(keys))
        self.text_cols = list(cols)
        self.keys = list(keys)
        self.n_classes = n_classes
        self.chars = chars
        self.n_out = n_classes if n_classes > 2 else 1
        self.name = f"txtlin__{label or '+'.join(cols)}"
        self.novelty_parents = list(cols)

    def _vocab(self, df, ctx):
        extra = getattr(ctx, "extra_rows", None)
        self.vecs_ = []
        for c in self.text_cols:
            texts, k = _doc_parts(df, extra, c, TEXT_CLIP)
            k = _sample_rows(k, 300_000)
            vs = [_fit_counted(_tfidf("word", 200_000), texts, k)]
            if self.chars and pd.Series(texts.str.len().to_numpy()[k]).mean() <= 80:
                vs.append(_fit_counted(_tfidf("char", 100_000), texts, k))
            self.vecs_.append(vs)
        if self.keys:
            from sklearn.preprocessing import OneHotEncoder
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", min_frequency=2, dtype=np.float32).fit(
                as_text_frame(df[self.keys]))

    def _design(self, df):
        from scipy import sparse
        blocks = []
        for c, vs in zip(self.text_cols, self.vecs_):
            t, codes = _text_parts(df, c, TEXT_CLIP)
            blocks += [v.transform(t)[codes] for v in vs]
        if self.keys:
            blocks.append(self.ohe_.transform(as_text_frame(df[self.keys])))
        return sparse.hstack(blocks, format="csr")

    def _query(self, A, y, B):
        from sklearn.linear_model import LogisticRegression, Ridge
        if self.n_classes == 0:
            mu = y.mean()
            return (Ridge(alpha=3.0, solver="sparse_cg").fit(A, y - mu).predict(B) + mu)[:, None]
        if self.n_classes == 2:
            with _LIBLINEAR_LOCK:
                m = LogisticRegression(C=1.0, solver="liblinear", tol=1e-3, random_state=0).fit(A, y.astype(int))
            return m.decision_function(B)[:, None]
        Y = np.eye(self.n_classes)[y.astype(int)]
        return Ridge(alpha=3.0, solver="sparse_cg").fit(A, Y - Y.mean(0)).predict(B) + Y.mean(0)

    def fit(self, df, y, ctx):
        self._vocab(df, ctx)
        self.A_ = self._design(df)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def transform(self, df, ctx):
        r = self._query(self.A_, self.y_, self._design(df))
        return r[:, 0] if self.n_out == 1 else r

    def fit_transform_oof(self, df, y, ctx, folds):
        self.fit(df, y, ctx)
        out = np.empty((len(df), self.n_out))
        for tr, va in folds:
            out[va] = self._query(self.A_[tr], self.y_[tr], self.A_[va])
        return out[:, 0] if self.n_out == 1 else out


class CrossLinearOOF(Spec):
    """Out-of-fold sparse linear model on one-hot keys and all their pairwise crosses.

    The classic winning recipe on high-cardinality categorical contests (Amazon
    access, click-through, insurance): a regularised logistic / ridge model over
    one-hot features of every key and key pair learns thousands of interaction
    effects with shrinkage, which a tree ensemble can then use as one column.
    Training rows are encoded out of fold.
    """
    target_dep = True

    def __init__(self, keys: Sequence[str], n_classes: int, pairs: bool = True, label: str = "keys"):
        super().__init__(keys)
        self.n_classes = n_classes
        self.pairs = pairs
        self.n_out = n_classes if n_classes > 2 else 1
        self.name = f"xlin__{label}" + ("" if pairs else "_1")

    def _codes_matrix(self, df) -> np.ndarray:
        """int64 code per (row, block): each key, then each key pair."""
        cols = [_codes(df, [c], [v]) for c, v in zip(self.parents, self.vocabs_)]
        out = list(cols)
        if self.pairs:
            for i, j in combinations(range(len(cols)), 2):
                out.append(cols[i] * (len(self.vocabs_[j]) + 1) + cols[j])
        return np.stack(out, axis=1)

    @staticmethod
    def _design(C_ref, C_q):
        """Sparse one-hot of ``C_q`` over the levels seen at least twice per block of ``C_ref``."""
        from scipy import sparse
        rows, cols, off = [], [], 0
        n = len(C_q)
        for b in range(C_ref.shape[1]):
            lv, cnt = np.unique(C_ref[:, b], return_counts=True)
            lv = lv[cnt >= 2]
            pos = np.searchsorted(lv, C_q[:, b])
            pos = np.minimum(pos, max(len(lv) - 1, 0))
            hit = (len(lv) > 0) & (lv[pos] == C_q[:, b]) if len(lv) else np.zeros(n, bool)
            rows.append(np.nonzero(hit)[0])
            cols.append(pos[hit] + off)
            off += len(lv)
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, max(off, 1)))

    def _model(self, shape):
        from sklearn.linear_model import LogisticRegression, Ridge
        if self.n_classes == 0:
            return Ridge(alpha=3.0, solver="sparse_cg")
        if self.n_classes > 2:
            # Multinomial lbfgs costs a full pass over rows x classes per iteration and needs hundreds of them on a
            # wide one-hot design (100 classes: ~9 s an iteration on 170k rows); a few seeded SAGA epochs reach a
            # lower held-out loss at a fixed cost.
            return LogisticRegression(C=0.5, solver="saga", tol=1e-3, max_iter=10, random_state=0)
        # Wide one-hot designs solve much faster in the dual.
        return LogisticRegression(C=0.5, solver="liblinear", tol=1e-2, dual=shape[1] > shape[0], random_state=0)

    def _query(self, S_ref, y_ref, S_q):
        A, B = self._design(S_ref, S_ref), self._design(S_ref, S_q)
        m = self._model(A.shape)
        if self.n_classes == 0:
            mu = y_ref.mean()
            m.fit(A, y_ref - mu)
            return (m.predict(B) + mu)[:, None]
        if self.n_classes > 2:
            m.fit(A, y_ref.astype(int))
        else:
            with _LIBLINEAR_LOCK:
                m.fit(A, y_ref.astype(int))
        if self.n_out == 1:
            return m.decision_function(B)[:, None]
        P = np.full((B.shape[0], self.n_out), 1e-6)
        P[:, m.classes_] = m.predict_proba(B)
        return np.log(P)

    def fit(self, df, y, ctx):
        self.vocabs_ = _fit_vocab(df, self.parents)
        self.S_ = self._codes_matrix(df)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def transform(self, df, ctx):
        r = self._query(self.S_, self.y_, self._codes_matrix(df))
        return r[:, 0] if self.n_out == 1 else r

    def fit_transform_oof(self, df, y, ctx, folds):
        self.fit(df, y, ctx)
        out = np.empty((len(df), self.n_out))
        for tr, va in folds:
            out[va] = self._query(self.S_[tr], self.y_[tr], self.S_[va])
        return out[:, 0] if self.n_out == 1 else out
