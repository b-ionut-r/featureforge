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
    for X in (A, B):
        if meta["id"] in X.columns:
            X.drop(columns=[meta["id"]], inplace=True)
    r = judge(A, yt, B[A.columns], y, meta["task"], 0)
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
