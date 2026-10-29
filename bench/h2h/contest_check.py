"""Contest regression check: one Kaggle contest built like its entry (child tables, unlabeled rows), a held-out
split shaped like the contest's test, FeatureForge from two source trees, one fixed judge (run.judge).

    python contest_check.py split --case hc
    python contest_check.py arm --case hc --label main --src /path/to/src [--budget 900]
    python contest_check.py judge --case hc --label main

hc: Home Credit, stratified 80/20 of application_train (seed 0); unlabeled = held-out rows + application_test.
ieee: IEEE-CIS, latest 40% of transactions by time, last 20% of that window held out; unlabeled = held-out rows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("H2H_DATA", Path.home() / "data" / "h2h")) / "kaggle"
ROOT = Path.home() / "h2h_runs" / "contests"
Y = "__y__"
HC_TABLES = [("bureau", "bureau.csv"), ("prev", "previous_application.csv"), ("pos", "POS_CASH_balance.csv"),
             ("inst", "installments_payments.csv"), ("cc", "credit_card_balance.csv")]


def split(case):
    d = ROOT / case
    d.mkdir(parents=True, exist_ok=True)
    if case == "hc":
        src = DATA / "hc"
        tr = pd.read_csv(src / "application_train.csv")
        te_real = pd.read_csv(src / "application_test.csv")
        from sklearn.model_selection import train_test_split
        a, b = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=0, stratify=tr.TARGET)
        fit, hold = tr.iloc[np.sort(a)], tr.iloc[np.sort(b)]
        np.save(d / "y_test.npy", hold.TARGET.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"TARGET": Y}).to_parquet(d / "train.parquet")
        pd.concat([hold.drop(columns=["TARGET"]), te_real], ignore_index=True).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="SK_ID_CURR", tables={n: str(src / f) for n, f in HC_TABLES})
    elif case == "hcs":
        src = Path(os.environ.get("HCS_DATA", Path.home() / "data" / "hcs"))
        rd = lambda *n: pd.concat([pd.read_parquet(src / f"train_{x}.parquet") for x in n], ignore_index=True)
        base = rd("base").merge(rd("static_0_0", "static_0_1"), on="case_id", how="left")
        base = base.merge(rd("static_cb_0"), on="case_id", how="left")
        kids = {"applprev": ("applprev_1_0", "applprev_1_1"), "person": ("person_1",), "cbb": ("credit_bureau_b_1",),
                "taxa": ("tax_registry_a_1",), "taxb": ("tax_registry_b_1",), "taxc": ("tax_registry_c_1",),
                "debit": ("debitcard_1",), "deposit": ("deposit_1",), "other": ("other_1",)}
        tables = {}
        for name, parts in kids.items():
            out = src / f"kid_{name}.parquet"
            if not out.exists():
                rd(*parts).to_parquet(out)
            tables[name] = str(out)
        # the contest's test is later weeks: hold out the latest 20% of weeks
        cut = np.quantile(base.WEEK_NUM, 0.8)
        fit, hold = base[base.WEEK_NUM < cut], base[base.WEEK_NUM >= cut]
        np.save(d / "y_test.npy", hold.target.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        np.save(d / "week_test.npy", hold.WEEK_NUM.to_numpy())
        fit.rename(columns={"target": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["target"]).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="case_id", tables=tables)
    elif case == "td":
        # TalkingData: every click of a 10% sample of ips (hash of ip), Nov 7-8 to train on, Nov 9 04:00-15:59 held out
        df = pd.read_parquet(Path(os.environ.get("TD_DATA", Path.home() / "data" / "talkingdata")) / "td_ip10.parquet")
        df = df.sort_values("click_time", kind="stable").reset_index(drop=True)
        df.insert(0, "click_id", np.arange(len(df)))
        day, hour = df.click_time.dt.day, df.click_time.dt.hour
        fit, hold = df[day.isin([7, 8])], df[(day == 9) & hour.between(4, 15)]
        np.save(d / "y_test.npy", hold.is_attributed.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"is_attributed": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["is_attributed"]).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="click_id", tables={})
    elif case == "ieee":
        src = DATA / "ieee"
        t, i = pd.read_csv(src / "train_transaction.csv"), pd.read_csv(src / "train_identity.csv")
        i.columns = [c.replace("-", "_") for c in i.columns]
        df = t.merge(i, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)
        df = df.iloc[int(0.6 * len(df)):].reset_index(drop=True)
        cut = int(0.8 * len(df))
        fit, hold = df.iloc[:cut], df.iloc[cut:]
        np.save(d / "y_test.npy", hold.isFraud.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"isFraud": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["isFraud"]).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="TransactionID", tables={})
    elif case == "ieeefull":
        # IEEE-CIS, every transaction; the latest 20% by time held out (the contest's test is later months)
        src = DATA / "ieee"
        t, i = pd.read_csv(src / "train_transaction.csv"), pd.read_csv(src / "train_identity.csv")
        i.columns = [c.replace("-", "_") for c in i.columns]
        df = t.merge(i, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)
        cut = int(0.8 * len(df))
        fit, hold = df.iloc[:cut], df.iloc[cut:]
        np.save(d / "y_test.npy", hold.isFraud.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"isFraud": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["isFraud"]).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="TransactionID", tables={})
    elif case == "porto":
        # Porto Seguro, every training row; stratified 80/20 (seed 0), as its test is a random draw
        df = pd.read_csv(DATA / "porto-seguro-safe-driver-prediction_train.csv.zip")
        from sklearn.model_selection import train_test_split
        a, b = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=df.target)
        fit, hold = df.iloc[np.sort(a)], df.iloc[np.sort(b)]
        np.save(d / "y_test.npy", hold.target.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"target": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["target"]).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="id", tables={})
    elif case == "paysim":
        # PaySim mobile-money fraud (Kaggle ealaxi/paysim1, 6.4M transactions, 0.13% fraud): the latest 20% of
        # hours held out; step (hour) as a timestamp; the bank's own rule flag dropped
        src = Path(os.environ.get("PAYSIM_DATA", Path.home() / "data" / "paysim"))
        df = pd.read_csv(next(src.glob("PS_*.csv"))).drop(columns=["isFlaggedFraud"])
        df.insert(0, "tx_id", np.arange(len(df)))
        df["step"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(df.step, unit="h")
        cut = df.step.quantile(0.8)
        fit, hold = df[df.step < cut], df[df.step >= cut]
        np.save(d / "y_test.npy", hold.isFraud.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"isFraud": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["isFraud"]).to_parquet(d / "test.parquet")
        meta = dict(task="binary", id="tx_id", tables={})
    elif case == "instacart":
        # Instacart Market Basket ($25k): which previously bought products 15,000 users (seed 0) reorder in their
        # next order. Rows = (user, product) pairs; 20% of users held out. Related tables, all from prior orders:
        # each pair's purchase history, each user's purchases (products, aisles, departments), each product's.
        src = Path(os.environ.get("FRESH_C", Path.home() / "data" / "fresh_c")) / "instacart"
        orders = pd.read_csv(src / "orders.csv")
        rng = np.random.default_rng(0)
        users = rng.choice(orders.loc[orders.eval_set == "train", "user_id"].unique(), 15000, replace=False)
        orders = orders[orders.user_id.isin(users)].copy()
        orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0)
        prior_o, train_o = orders[orders.eval_set == "prior"], orders[orders.eval_set == "train"]
        prod = pd.read_csv(src / "products.csv")[["product_id", "aisle_id", "department_id"]]
        op = pd.read_csv(src / "order_products__prior.csv")
        op = op[op.order_id.isin(prior_o.order_id)].merge(
            prior_o[["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]], on="order_id")
        op = op.merge(prod, on="product_id", how="left")
        opt = pd.read_csv(src / "order_products__train.csv")
        opt = opt[opt.order_id.isin(train_o.order_id)].merge(train_o[["order_id", "user_id"]], on="order_id")
        pairs = op[["user_id", "product_id"]].drop_duplicates().merge(
            train_o[["user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]], on="user_id")
        pairs = pairs.merge(prod, on="product_id", how="left")
        lab = set(zip(opt.user_id, opt.product_id))
        pairs[Y] = [int((u, p) in lab) for u, p in zip(pairs.user_id, pairs.product_id)]
        pairs.insert(0, "up", pairs.user_id.astype(np.int64) * 1_000_000 + pairs.product_id)
        op["up"] = op.user_id.astype(np.int64) * 1_000_000 + op.product_id
        hold_u = set(np.random.default_rng(1).choice(users, 3000, replace=False))
        m = pairs.user_id.isin(hold_u).to_numpy()
        fit, hold = pairs[~m], pairs[m]
        np.save(d / "y_test.npy", hold[Y].to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.to_parquet(d / "train.parquet")
        hold.drop(columns=[Y]).to_parquet(d / "test.parquet")
        hist = ["order_number", "add_to_cart_order", "reordered", "order_dow", "order_hour_of_day", "days_since_prior_order"]
        op[["up"] + hist].to_parquet(d / "up_hist.parquet")
        op[["user_id", "product_id", "aisle_id", "department_id"] + hist].to_parquet(d / "user_items.parquet")
        op[["product_id", "add_to_cart_order", "reordered", "order_number", "days_since_prior_order"]].to_parquet(d / "prod_hist.parquet")
        meta = dict(task="binary", id="up", tables={"up_hist": str(d / "up_hist.parquet") + ":up:order_number",
                                                     "user_items": str(d / "user_items.parquet") + ":user_id:order_number",
                                                     "prod_hist": str(d / "prod_hist.parquet") + ":product_id"})
    elif case == "tdmob":
        # TalkingData Mobile User Demographics ($25k): a phone's gender-age group (12 classes) from its brand, model and
        # apps. 20% of training devices held out (seed 0), the real test devices unlabeled too. Related tables, keyed by
        # device: its events (time, place), its apps (events and active events per app, from the event log) and its
        # apps' labels.
        src = Path(os.environ.get("FRESH_C", Path.home() / "data" / "fresh_c")) / "tdmob"
        tr = pd.read_csv(src / "gender_age_train.csv")[["device_id", "group"]]
        te_real = pd.read_csv(src / "gender_age_test.csv")
        ph = pd.read_csv(src / "phone_brand_device_model.csv").drop_duplicates("device_id")
        tr["group"] = pd.factorize(tr.group, sort=True)[0]
        hold_ids = tr.device_id.sample(frac=0.2, random_state=0)
        m = tr.device_id.isin(hold_ids)
        fit, hold = tr[~m].merge(ph, on="device_id", how="left"), tr[m].merge(ph, on="device_id", how="left")
        np.save(d / "y_test.npy", hold.group.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"group": Y}).to_parquet(d / "train.parquet")
        pd.concat([hold.drop(columns=["group"]), te_real.merge(ph, on="device_id", how="left")],
                  ignore_index=True).to_parquet(d / "test.parquet")
        ev = pd.read_csv(src / "events.csv", parse_dates=["timestamp"])
        ev[["device_id", "timestamp", "longitude", "latitude"]].to_parquet(d / "events.parquet")
        ae = pd.read_csv(src / "app_events.csv", usecols=["event_id", "app_id", "is_active"])
        ae = ae.merge(ev[["event_id", "device_id"]], on="event_id")
        da = ae.groupby(["device_id", "app_id"], as_index=False).agg(n_events=("event_id", "size"), n_active=("is_active", "sum"))
        del ae
        da.to_parquet(d / "device_apps.parquet")
        al = pd.read_csv(src / "app_labels.csv").drop_duplicates()
        dl = da[["device_id", "app_id"]].merge(al, on="app_id")[["device_id", "label_id"]]
        dl.groupby(["device_id", "label_id"], as_index=False).size().rename(columns={"size": "n_apps"}).to_parquet(d / "device_labels.parquet")
        meta = dict(task="multiclass", id="device_id", tables={"events": str(d / "events.parquet") + ":device_id:timestamp",
                                                                "apps": str(d / "device_apps.parquet") + ":device_id",
                                                                "labels": str(d / "device_labels.parquet") + ":device_id"})
    elif case == "jane":
        # Jane Street Real-Time Market Data ($120k): responder_6 of 39 symbols, from the mirror's last partition: its
        # last 59 days, every 4th time step (each step's cross-section whole). The latest 12 days held out. Each row
        # carries the previous day's responders at its step and symbol, as the contest serves them.
        src = Path(os.environ.get("FRESH_C", Path.home() / "data" / "fresh_c")) / "jane"
        df = pd.read_parquet(src / "part-9.parquet", filters=[("date_id", ">=", 1639)])
        df = df[df.time_id % 4 == 0].reset_index(drop=True)
        resp = [f"responder_{i}" for i in range(9)]
        lag = df[["date_id", "time_id", "symbol_id"] + resp].copy()
        lag["date_id"] += 1
        df = df.merge(lag.rename(columns={r: f"{r}_lag_1" for r in resp}), on=["date_id", "time_id", "symbol_id"], how="left")
        df = df[df.date_id >= 1640].sort_values(["date_id", "time_id", "symbol_id"]).reset_index(drop=True)
        df = df.drop(columns=[r for r in resp if r != "responder_6"])
        df.insert(0, "row_id", np.arange(len(df)))
        cut = sorted(df.date_id.unique())[-12]
        fit, hold = df[df.date_id < cut], df[df.date_id >= cut]
        np.save(d / "y_test.npy", hold.responder_6.to_numpy())
        np.save(d / "w_test.npy", hold.weight.to_numpy())
        np.save(d / "n_hold.npy", np.array([len(hold)]))
        fit.rename(columns={"responder_6": Y}).to_parquet(d / "train.parquet")
        hold.drop(columns=["responder_6"]).to_parquet(d / "test.parquet")
        meta = dict(task="regression", id="row_id", tables={})
    elif case in ("homesite", "walmart", "optiver", "enefit"):
        # Paid contests never run before (Kaggle dataset mirrors under FRESH_C), each split like its test.
        src = Path(os.environ.get("FRESH_C", Path.home() / "data" / "fresh_c"))
        if case == "homesite":
            # Homesite Quote Conversion: stratified 80/20; unlabeled = held-out rows + the real test file.
            tr = pd.read_csv(src / "homesite-quote-conversion" / "train.csv")
            te_real = pd.read_csv(src / "homesite-quote-conversion" / "test.csv")
            from sklearn.model_selection import train_test_split
            a, b = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=0, stratify=tr.QuoteConversion_Flag)
            fit, hold = tr.iloc[np.sort(a)], tr.iloc[np.sort(b)]
            np.save(d / "y_test.npy", hold.QuoteConversion_Flag.to_numpy())
            np.save(d / "n_hold.npy", np.array([len(hold)]))
            fit.rename(columns={"QuoteConversion_Flag": Y}).to_parquet(d / "train.parquet")
            pd.concat([hold.drop(columns=["QuoteConversion_Flag"]), te_real], ignore_index=True).to_parquet(d / "test.parquet")
            meta = dict(task="binary", id="QuoteNumber", tables={})
        elif case == "walmart":
            # Walmart Trip Type: one row per visit (weekday, trip type) and its scanned items as a related table;
            # 20% of visits held out (seed 0), the real test visits unlabeled too.
            w = src / "walmart-trip-type-classification"
            it_tr, it_te = pd.read_csv(w / "walmart_train.csv"), pd.read_csv(w / "walmart_test.csv")
            it_te["VisitNumber"] += 10_000_000  # the test file numbers its visits from 1 again
            vis = it_tr.groupby("VisitNumber", as_index=False).agg(Weekday=("Weekday", "first"), TripType=("TripType", "first"))
            vis["TripType"] = pd.factorize(vis.TripType, sort=True)[0]
            vte = it_te.groupby("VisitNumber", as_index=False).agg(Weekday=("Weekday", "first"))
            hold_ids = vis.VisitNumber.sample(frac=0.2, random_state=0)
            m = vis.VisitNumber.isin(hold_ids)
            fit, hold = vis[~m], vis[m]
            np.save(d / "y_test.npy", hold.TripType.to_numpy())
            np.save(d / "n_hold.npy", np.array([len(hold)]))
            fit.rename(columns={"TripType": Y}).to_parquet(d / "train.parquet")
            pd.concat([hold.drop(columns=["TripType"]), vte], ignore_index=True).to_parquet(d / "test.parquet")
            items = pd.concat([it_tr.drop(columns=["TripType", "Weekday"]), it_te.drop(columns=["Weekday"])], ignore_index=True)
            items.to_parquet(d / "items.parquet")
            meta = dict(task="multiclass", id="VisitNumber", tables={"items": str(d / "items.parquet")})
        elif case == "optiver":
            # Optiver Trading at the Close: the latest 20% of days held out; day + seconds as a timestamp.
            df = pd.read_csv(src / "optiver-trading-at-the-close" / "train.csv").drop(columns=["time_id"])
            df = df[df.target.notna()].reset_index(drop=True)
            df["ts"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(df.date_id, unit="D") + pd.to_timedelta(df.seconds_in_bucket, unit="s")
            cut = df.date_id.quantile(0.8)
            fit, hold = df[df.date_id < cut], df[df.date_id >= cut]
            np.save(d / "y_test.npy", hold.target.to_numpy())
            np.save(d / "n_hold.npy", np.array([len(hold)]))
            fit.rename(columns={"target": Y}).to_parquet(d / "train.parquet")
            hold.drop(columns=["target"]).to_parquet(d / "test.parquet")
            meta = dict(task="regression", id="row_id", tables={})
        else:
            # Enefit prosumers: the latest 20% of hours held out (the main table only).
            df = pd.read_csv(src / "predict-energy-behavior-of-prosumers" / "train.csv").drop(columns=["data_block_id"])
            df = df[df.target.notna()].reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df.datetime)
            cut = df.datetime.quantile(0.8)
            fit, hold = df[df.datetime < cut], df[df.datetime >= cut]
            np.save(d / "y_test.npy", hold.target.to_numpy())
            np.save(d / "n_hold.npy", np.array([len(hold)]))
            fit.rename(columns={"target": Y}).to_parquet(d / "train.parquet")
            hold.drop(columns=["target"]).to_parquet(d / "test.parquet")
            meta = dict(task="regression", id="row_id", tables={})
    (d / "meta.json").write_text(json.dumps(meta))


def arm(case, label, src, budget, proposals=None):
    d = ROOT / case
    meta = json.loads((d / "meta.json").read_text())
    sys.path.insert(0, src)
    from featureforge import build_features
    t0 = time.time()
    build_features(str(d / "train.parquet"), str(d / "test.parquet"), Y, id=meta["id"], tables=meta["tables"] or None,
                   task=meta["task"], budget=budget, out_dir=str(d / label),
                   **({"proposals": proposals} if proposals else {}))
    (d / label / "seconds.json").write_text(json.dumps(dict(seconds=time.time() - t0)))


def judge_cell(case, label):
    sys.path.insert(0, str(HERE))
    from run import judge
    d = ROOT / case
    meta = json.loads((d / "meta.json").read_text())
    y = np.load(d / "y_test.npy")
    n = len(y)
    if label == "raw":
        A = pd.read_parquet(d / "train.parquet")
        B = pd.read_parquet(d / "test.parquet").iloc[:n]
    else:
        A = pd.read_parquet(d / label / "train_features.parquet")
        B = pd.read_parquet(d / label / "test_features.parquet").iloc[:n]
    yt = A.pop(Y).to_numpy()
    if meta["task"] == "multiclass":  # class labels may come back as text
        yt = pd.to_numeric(pd.Series(yt)).to_numpy().astype(int)
    for X in (A, B):
        if meta["id"] in X.columns:
            X.drop(columns=[meta["id"]], inplace=True)
    for X in (A, B):   # timestamps as numbers (as text they would be one category per click)
        for c in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[c]):
                X[c] = X[c].astype("int64") / 1e9
    if os.environ.get("H2H_JUDGE_F32"):  # big cases: every numeric column as float32, for every arm alike
        for X in (A, B):
            for c in X.columns:
                if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]) and X[c].dtype != np.float32:
                    X[c] = X[c].astype(np.float32)
    preds = []
    r = judge(A, yt, B[A.columns], y, meta["task"], 0, preds=preds)
    if (d / "w_test.npy").exists() and preds:   # the contest's weighted zero-mean R2
        w = np.load(d / "w_test.npy")
        r["wr2"] = float(1 - np.sum(w * (y - preds[0]) ** 2) / np.sum(w * y ** 2))
    secs = json.loads((d / label / "seconds.json").read_text())["seconds"] if label != "raw" else 0
    r.update(case=case, label=label, n_cols=A.shape[1], build_seconds=secs)
    print(json.dumps(r))
    with open(HERE / "results" / "contests.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"), ap.add_argument("--case"), ap.add_argument("--label"), ap.add_argument("--src")
    ap.add_argument("--budget", type=float, default=900)
    ap.add_argument("--proposals", default=None)
    a = ap.parse_args()
    {"split": lambda: split(a.case), "arm": lambda: arm(a.case, a.label, a.src, a.budget, a.proposals),
     "judge": lambda: judge_cell(a.case, a.label)}[a.cmd]()
