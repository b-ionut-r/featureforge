"""Favorita held-out bench: the contest's last 16 training days held out, built like the contest test.

    python fav_bench.py prep                       # slim long panel of the test's store x item pairs
    python fav_bench.py arm --label h56r14 --hist 56 --rows 14 --src SRC
    python fav_bench.py judge --label h56r14

Pairs: the store x item pairs of the contest's test file; every day zero-filled (the train file lists only
days with sales), returns clipped at 0; onpromotion read from train (0 where the day has no row), as the
contest's test file gives it for its days. Held out: 2017-07-31 .. 2017-08-15 (16 days, the test's length).
Metric: NWRMSLE (perishable items weighted 1.25), on log1p units.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(os.environ.get("FAV_DATA", "/root/data/fav"))
R = Path(os.environ.get("FAV_RUNS", "/root/fav_runs"))
CONTEST = bool(int(os.environ.get("FAV_CONTEST", "0")))  # the real test file: features + a submission
HOLD0 = pd.Timestamp("2017-08-16" if CONTEST else "2017-07-31")
Y = "unit_sales"
PROMO = bool(int(os.environ.get("FAV_PROMO", "0")))


def prep():
    R.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(D / "train_2016-06.parquet")
    te = pd.read_csv(D / "test.csv", usecols=["store_nbr", "item_nbr"])
    pairs = te.drop_duplicates().reset_index(drop=True)
    pairs["pair"] = np.arange(len(pairs), dtype=np.int32)
    days = pd.date_range(tr.date.min(), tr.date.max(), freq="D")
    tr = tr.merge(pairs, on=["store_nbr", "item_nbr"], how="inner")
    E, P = len(pairs), len(days)
    di = ((tr.date - days[0]).dt.days).to_numpy()
    S = np.zeros((E, P), np.float32); PR = np.zeros((E, P), np.float32)
    S[tr.pair.to_numpy(), di] = np.clip(tr[Y].to_numpy(), 0, None)
    PR[tr.pair.to_numpy(), di] = tr.onpromotion.fillna(0).to_numpy()
    np.save(R / "sales.npy", S); np.save(R / "promo.npy", PR)
    pairs.to_parquet(R / "pairs.parquet"); pd.Series(days).to_frame("date").to_parquet(R / "days.parquet")
    print("pairs", E, "days", P, days[0], days[-1], "nonzero share", (S > 0).mean())


def long(first, last, with_y=True):
    """Long rows of every pair over days [first, last]."""
    S, PR = np.load(R / "sales.npy", mmap_mode="r"), np.load(R / "promo.npy", mmap_mode="r")
    pairs = pd.read_parquet(R / "pairs.parquet"); days = pd.read_parquet(R / "days.parquet").date
    a, b = int((first - days[0]).days), int((last - days[0]).days)
    n = b - a + 1; E = len(pairs)
    out = pd.DataFrame({
        "date": np.repeat(days.iloc[a:b + 1].to_numpy(), E),
        "store_nbr": np.tile(pairs.store_nbr.to_numpy().astype(np.int16), n),
        "item_nbr": np.tile(pairs.item_nbr.to_numpy().astype(np.int32), n),
        "onpromotion": np.asarray(PR[:, a:b + 1]).T.reshape(-1),
    })
    if with_y:
        out[Y] = np.asarray(S[:, a:b + 1]).T.reshape(-1)
    if not PROMO:  # recorded in train only on days with sales: held-out days would leak them
        out = out.drop(columns=["onpromotion"])
    return out


def statics():
    it = pd.read_csv(D / "items.csv"); st = pd.read_csv(D / "stores.csv")
    return it, st


def arm(label, hist, rows, src, extra):
    sys.path.insert(0, src)
    from featureforge.blocks.forecast import ForecastFeatures
    t0 = time.time()
    hist_tr = long(HOLD0 - pd.Timedelta(days=hist), HOLD0 - pd.Timedelta(days=1))
    if CONTEST:
        hold = pd.read_csv(D / "test.csv", parse_dates=["date"])
        hold["onpromotion"] = hold.onpromotion.astype(np.float32)
        hold["store_nbr"] = hold.store_nbr.astype(np.int16); hold["item_nbr"] = hold.item_nbr.astype(np.int32)
        if not PROMO:
            hold = hold.drop(columns=["onpromotion"])
    else:
        hold = long(HOLD0, HOLD0 + pd.Timedelta(days=15), with_y=False)
    it, st = statics()
    for df in (hist_tr, hold):
        for c, src_ in (("family", it), ("class", it), ("perishable", it)):
            df[c] = pd.Categorical(df.item_nbr.map(src_.set_index("item_nbr")[c]).to_numpy())
        for c in ("city", "state", "type", "cluster"):
            df[c] = pd.Categorical(df.store_nbr.map(st.set_index("store_nbr")[c]).to_numpy())
    y = hist_tr.pop(Y).to_numpy(dtype=float)
    kw = json.loads(extra) if extra else {}
    ids = hold.pop("id").to_numpy() if "id" in hold.columns else None
    ff = ForecastFeatures(**kw).fit(hist_tr, y, hold)
    assert ff.active_, ff.reason_
    win = (hist_tr.date >= HOLD0 - pd.Timedelta(days=rows)).to_numpy()
    Xw = hist_tr[win].reset_index(drop=True)
    yw = y[win]
    del hist_tr
    n1, M1 = ff.transform_float32(Xw, 250_000)
    n2, M2 = ff.transform_float32(hold, 250_000)
    o = R / label; o.mkdir(parents=True, exist_ok=True)
    A = pd.concat([Xw, pd.DataFrame(M1, columns=n1)], axis=1); A[Y] = yw
    B = pd.concat([hold.reset_index(drop=True), pd.DataFrame(M2, columns=n2)], axis=1)
    A.to_parquet(o / "train.parquet"); B.to_parquet(o / "hold.parquet")
    if ids is not None:
        np.save(o / "ids.npy", ids)
    json.dump(dict(seconds=round(time.time() - t0), cols=len(n1), train_rows=len(A)), open(o / "build.json", "w"))
    print(f"{label}: {len(n1)} columns, {len(A)} training rows, {time.time() - t0:.0f}s")


def judge(label, seeds=1):
    """One LightGBM on log1p sales (perishables weighted 1.25): early stopping on the latest quarter of the
    training days, then refit on all of them. Built lean: one float32 matrix, one binned dataset."""
    import gc
    import lightgbm as lgb
    o = R / label
    import pyarrow.parquet as pq
    class _Cols:   # a parquet file read a column at a time (the whole training frame next to its matrix: 13.5 GB)
        def __init__(self, f):
            self.f = pq.ParquetFile(f); self.columns = [c for c in self.f.schema_arrow.names if not c.startswith("__") and c != "id"]
        def __getitem__(self, c):
            return self.f.read(columns=[c]).to_pandas()[c]
        def __len__(self):
            return self.f.metadata.num_rows
        def pop(self, c):
            self.columns.remove(c); return self[c]
    A = _Cols(o / "train.parquet"); B = pd.read_parquet(o / "hold.parquet")
    if not CONTEST:
        S = np.load(R / "sales.npy", mmap_mode="r"); days = pd.read_parquet(R / "days.parquet").date
        a = int((HOLD0 - days[0]).days)
        yh = np.asarray(S[:, a:a + 16]).T.reshape(-1)
    y = np.log1p(np.clip(A.pop(Y).to_numpy(), 0, None)).astype(np.float32)
    cols = [c for c in A.columns if c != Y]; cats = []
    def mat(X, ref=None):
        M = np.empty((len(X), len(cols) + 1), np.float32)
        for j, c in enumerate(cols):
            v = X[c]
            if c == "date":
                M[:, j] = (v - pd.Timestamp("2016-01-01")).dt.days.to_numpy()
            elif isinstance(v.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(v):
                lv = ref[c] if ref is not None else pd.Index(sorted(v.astype(str).unique()))
                M[:, j] = lv.get_indexer(v.astype(str)).astype(np.float32)
                if ref is None:
                    cats.append(j); levels[c] = lv
            else:
                M[:, j] = v.to_numpy(dtype=np.float32)
        M[:, -1] = (M[:, cols.index("date")] + 5) % 7   # weekday (2016-01-01 was a Friday)
        return M
    levels = {}
    w = np.where(A["perishable"].astype(float).to_numpy() == 1, 1.25, 1.0).astype(np.float32)
    wh = np.where(B["perishable"].astype(float).to_numpy() == 1, 1.25, 1.0)
    MA = mat(A); del A; gc.collect()
    MB = mat(B, levels); del B; gc.collect()
    d = MA[:, cols.index("date")]; last, first = d.max(), d.min()
    va = d > last - max(3, (last - first + 1) // 4)
    p = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, max_bin=255)
    t0 = time.time()
    full = lgb.Dataset(MA, y, weight=w, categorical_feature=cats, free_raw_data=False, params={"max_bin": 255}).construct()
    tr_i, va_i = np.flatnonzero(~va), np.flatnonzero(va)
    m = lgb.train(dict(p, seed=0), full.subset(tr_i), 2000, valid_sets=[full.subset(va_i)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    it = m.best_iteration
    del m; gc.collect()
    pred = np.zeros(len(MB))
    for s_ in range(seeds):
        mf = lgb.train(dict(p, seed=s_), full, int(it * 1.05))
        pred += mf.predict(MB) / seeds
    if CONTEST:
        sub = pd.DataFrame({"id": np.load(o / "ids.npy"), "unit_sales": np.expm1(np.clip(pred, 0, None))})
        sub.to_csv(o / "submission.csv", index=False)
        print(f"submission {o / 'submission.csv'}: {len(sub)} rows, iterations {it}, {time.time() - t0:.0f}s")
        return
    err = np.sqrt(np.sum(wh * (np.clip(pred, 0, None) - np.log1p(yh)) ** 2) / wh.sum())
    res = dict(label=label, nwrmsle=float(err), iters=it, judge_s=round(time.time() - t0), **json.load(open(o / "build.json")))
    print(json.dumps(res))
    with open(Path(__file__).parent / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--label"); ap.add_argument("--hist", type=int, default=56)
    ap.add_argument("--rows", type=int, default=14); ap.add_argument("--src", default="/home/user/featureforge/src")
    ap.add_argument("--kw", default=""); ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    {"prep": prep, "arm": lambda: arm(a.label, a.hist, a.rows, a.src, a.kw),
     "judge": lambda: judge(a.label, a.seeds)}[a.cmd]()
