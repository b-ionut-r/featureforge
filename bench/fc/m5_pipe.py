"""M5 Accuracy through build_features, as a Kaggle entry: the last N days of sales_train_evaluation as the
training file, the 28 evaluation days as the test file; one LightGBM; the wide submission (validation rows 0).

    python m5_pipe.py build --days 450 --out /root/m5_runs/long     # build_features (FeatureForge picks the rows)
    python m5_pipe.py build --days 120 --out /root/m5_runs/d120      # the earlier entry's training window
    python m5_pipe.py fit --out /root/m5_runs/long                   # judge -> submission.csv
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
D = Path(os.environ.get("M5_DATA", "/root/data/m5"))


def prep(n_days, end=1941):
    s = pd.read_csv(D / "sales_train_evaluation.csv")
    cal = pd.read_csv(D / "calendar.csv"); pr = pd.read_csv(D / "sell_prices.csv")
    keys = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for k in keys:
        s[k] = s[k].astype("category")
    days = [f"d_{i}" for i in range(end + 1 - n_days, end + 1)]
    tr = s[keys + days].melt(id_vars=keys, var_name="d", value_name="sales")
    tr["sales"] = tr.sales.astype(np.float32)
    te = s[keys].merge(pd.DataFrame({"d": [f"d_{i}" for i in range(end + 1, end + 29)]}), how="cross")
    calc = ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
    out = []
    for df in (tr, te):
        df = df.merge(cal[calc], on="d", how="left").merge(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        df["F"] = df["d"].str[2:].astype(int) - end
        for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2", "store_id", "item_id", "dept_id", "cat_id", "state_id"):
            df[c] = df[c].astype("category")
        df["sell_price"] = df.sell_price.astype(np.float32)
        out.append(df)
    tr, te = out
    tr = tr[tr.sell_price.notna()]  # not on sale yet
    te["Id"] = te["id"].astype(str) + "|" + te["F"].astype(str)
    tr = tr.drop(columns=["id", "d", "F"]).sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    tr.insert(0, "Id", [f"t{i}" for i in range(len(tr))])
    te = te.drop(columns=["id", "d", "F"])
    return tr, te


def build(days, out, end=1941):
    sys.path.insert(0, os.environ.get("FF_SRC", "/home/user/featureforge/src"))
    from featureforge import build_features
    t = time.time()
    tr, te = prep(days, end)
    print("train", tr.shape, "test", te.shape, f"{time.time() - t:.0f}s", flush=True)
    build_features(tr, te, "sales", id="Id", budget=900, out_dir=out)
    json.dump(dict(build_s=round(time.time() - t), days=days, end=end), open(Path(out) / "build.json", "w"))


def fit(out, last=0, tag=""):
    import lightgbm as lgb, pyarrow.parquet as pq
    o = Path(out)
    f = pq.ParquetFile(o / "train_features.parquet")
    cols = [c for c in f.schema_arrow.names if c not in ("Id", "sales")]
    n = f.metadata.num_rows
    B = pd.read_parquet(o / "test_features.parquet")
    ids = B.pop("Id").to_numpy()
    M = np.empty((n, len(cols)), np.float32); MB = np.empty((len(B), len(cols)), np.float32); cats = []
    for j, c in enumerate(cols):
        v = f.read(columns=[c]).to_pandas()[c]; w = B[c]
        if c == "date":
            v = pd.to_datetime(v.astype(str)); w = pd.to_datetime(w.astype(str))
            M[:, j] = (v - pd.Timestamp("2011-01-01")).dt.days; MB[:, j] = (w - pd.Timestamp("2011-01-01")).dt.days
        elif isinstance(v.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(v):
            lv = pd.Index(pd.unique(pd.concat([v.astype(str), w.astype(str)], ignore_index=True)))
            M[:, j] = lv.get_indexer(v.astype(str)); MB[:, j] = lv.get_indexer(w.astype(str)); cats.append(j)
        else:
            M[:, j] = v.to_numpy(dtype=np.float32); MB[:, j] = w.to_numpy(dtype=np.float32)
        del v
    y = f.read(columns=["sales"]).to_pandas()["sales"].to_numpy(dtype=np.float32)
    del B; gc.collect()
    d = M[:, cols.index("date")]
    if last:  # train on the latest `last` days only (the features still come from the full history)
        k = d > d.max() - last
        M, y, d = M[k], y[k], d[k]; n = len(y); gc.collect()
    last, first = d.max(), d.min()
    va = d > last - max(7, (last - first + 1) // 8)
    p = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.1, verbose=-1, num_threads=4, max_bin=255)
    t0 = time.time()
    full = lgb.Dataset(M, y, categorical_feature=cats, free_raw_data=False).construct()
    m = lgb.train(dict(p, seed=0), full.subset(np.flatnonzero(~va)), 3000, valid_sets=[full.subset(np.flatnonzero(va))],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    it = m.best_iteration; del m; gc.collect()
    mf = lgb.train(dict(p, seed=0), full, int(it * 1.05))
    pred = np.clip(mf.predict(MB), 0, None)
    sub = pd.DataFrame({"Id": ids, "sales": pred})
    sub[["id", "F"]] = sub["Id"].str.split("|", expand=True)
    w = sub.pivot(index="id", columns="F", values="sales")[[str(k) for k in range(1, 29)]]
    w.columns = [f"F{k}" for k in range(1, 29)]
    v = w.copy() * 0; v.index = v.index.str.replace("_evaluation", "_validation")
    out_ = pd.concat([v, w]).reset_index().rename(columns={"index": "id"})
    out_.to_csv(o / f"submission{tag}.csv", index=False)
    r = dict(out=str(o), iters=it, train_rows=n, cols=len(cols), judge_s=round(time.time() - t0), **json.load(open(o / "build.json")))
    print(json.dumps(r))


def wrmsse(sub_path, end):
    """M5's metric on days end+1..end+28 (labels from sales_train_evaluation), evaluation rows of a submission."""
    s = pd.read_csv(D / "sales_train_evaluation.csv")
    cal = pd.read_csv(D / "calendar.csv"); pr = pd.read_csv(D / "sell_prices.csv")
    keys = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    hist = s[[f"d_{i}" for i in range(1, end + 1)]].to_numpy(np.float64)
    truth = s[[f"d_{i}" for i in range(end + 1, end + 29)]].to_numpy(np.float64)
    sub = pd.read_csv(sub_path).set_index("id").loc[s["id"]]
    pred = sub[[f"F{k}" for k in range(1, 29)]].to_numpy(np.float64)
    last = cal.set_index("d").loc[[f"d_{i}" for i in range(end - 27, end + 1)], "wm_yr_wk"].to_numpy()
    p = pr.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    P = p.reindex(pd.MultiIndex.from_frame(s[["store_id", "item_id"]]))[last].to_numpy()
    dollars = np.nansum(hist[:, -28:] * P, axis=1)
    levels = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"], ["state_id", "cat_id"], ["state_id", "dept_id"],
              ["store_id", "cat_id"], ["store_id", "dept_id"], ["item_id"], ["item_id", "state_id"], ["item_id", "store_id"]]
    tot = 0.0
    for lv in levels:
        g = np.zeros(len(s), int) if not lv else s.groupby(lv, observed=True).ngroup().to_numpy()
        k = g.max() + 1
        agg = lambda a: np.stack([np.bincount(g, a[:, j], k) for j in range(a.shape[1])], 1)
        H, T, Pr, w = agg(hist), agg(truth), agg(pred), np.bincount(g, dollars, k)
        sc = np.empty(k)
        for i in range(k):
            h = H[i]; nz = np.flatnonzero(h != 0); h = h[nz[0]:] if len(nz) else h
            sc[i] = np.mean(np.diff(h) ** 2) if len(h) > 1 else np.nan
        rmsse = np.sqrt(np.mean((T - Pr) ** 2, 1) / sc)
        ok = np.isfinite(rmsse)
        tot += np.sum(w[ok] / w.sum() * rmsse[ok]) / 12
    return tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--days", type=int, default=450); ap.add_argument("--out"); ap.add_argument("--last", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--end", type=int, default=1941)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.days, a.out, a.end)
    elif a.cmd == "score":
        print(json.dumps(dict(out=a.out, tag=a.tag, end=a.end, wrmsse=round(wrmsse(Path(a.out) / f"submission{a.tag}.csv", a.end), 5))))
    else:
        fit(a.out, a.last, a.tag)
