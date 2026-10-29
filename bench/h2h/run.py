"""Head-to-head bench: every feature tool through one fixed bagged LightGBM judge on the same outer holdouts.

    python run.py --datasets ames,titanic --arms raw,featureforge,openfe --seeds 0,1,2 --log results.jsonl

For each dataset and seed: a stratified 80/20 split. The 20% is the holdout; its labels never reach a
tool, its feature rows are handed over as the unlabeled test file (as in a contest). Each tool runs in a
subprocess with a wall-clock cap; a crash or timeout is logged and scored as the raw columns. Done cells
are skipped, so the run resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from datasets import SPEC, SPECS, load  # noqa: E402

Y = "__y__"
COMP_PY = os.environ.get("H2H_COMP_PY", "/home/user/venv_comp/bin/python")  # OpenFE, AutoFeat, Featuretools
PY = {"featureforge": sys.executable, "llm": sys.executable}


def judge(Xtr, ytr, Xte, yte, task, seed, threads=4, preds=None):
    """The pinned judge: LightGBM, early stopping on a random 15% of training rows, then three seeds
    refit on all training rows at 1.1x the best iteration, predictions averaged."""
    lean = bool(os.environ.get("H2H_JUDGE_F32"))  # big cases: no copies, one binned dataset shared by every fit
    if not lean:
        Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if not pd.api.types.is_numeric_dtype(Xtr[c]) or pd.api.types.is_bool_dtype(Xtr[c]):
            a, b = (v.astype("string").fillna("__na__").astype(object) for v in (Xtr[c], Xte[c]))
            cats = pd.Index(pd.unique(a))
            Xtr[c], Xte[c] = pd.Categorical(a, categories=cats), pd.Categorical(b, categories=cats)
    Xtr.columns = Xte.columns = [f"f{i}" for i in range(Xtr.shape[1])]
    k = int(max(ytr.max(), yte.max())) + 1 if task == "multiclass" else 0
    n = len(Xtr)
    P = dict(objective={"binary": "binary", "multiclass": "multiclass", "regression": "regression"}[task],
             learning_rate=0.05, num_leaves=63, min_child_samples=100 if n >= 20000 else 20, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=threads, verbose=-1, max_bin=255,
             max_cat_to_onehot=8, cat_smooth=50, **({"num_class": k} if k else {}))
    perm = np.random.default_rng(seed + 1).permutation(n)
    fit, va = np.sort(perm[: int(0.85 * n)]), np.sort(perm[int(0.85 * n):])
    if lean:
        full = lgb.Dataset(Xtr, ytr, params=dict(P), free_raw_data=True).construct()
        b = lgb.train(dict(P, seed=0), full.subset(fit), 5000, valid_sets=[full.subset(va)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000,
                      valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    p = np.zeros((len(Xte), k)) if k else np.zeros(len(Xte))
    for s in range(3):
        ds = full if lean else lgb.Dataset(Xtr, ytr)
        p += lgb.train(dict(P, seed=s), ds, int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
    if task == "multiclass":
        pp = np.clip(p, 1e-15, 1)
        pp /= pp.sum(1, keepdims=True)
        return dict(loss=float(log_loss(yte, pp, labels=np.arange(k))), acc=float(np.mean(p.argmax(1) == yte)))
    if task == "binary":
        return dict(loss=float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))), auc=float(roc_auc_score(yte, p)))
    if preds is not None:
        preds.append(p)
    return dict(loss=float(mean_squared_error(yte, p) ** 0.5))


def make_split(name, seed, root):
    d = root / name / f"s{seed}"
    if (d / "meta.json").exists():
        return d
    X, y, task, desc = load(name)
    strat = y if task != "regression" else None
    if GROUP_SPLIT:  # repeated entities: whole entities are held out
        from sklearn.model_selection import GroupShuffleSplit
        itr, ite = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, groups=X[SPEC[name].entity]))
    elif TIME_SPLIT:  # time-ordered table: the last 20% in time is held out, as a later test would be
        order = np.lexsort([pd.to_numeric(X[c], errors="coerce").fillna(-1).to_numpy() if X[c].dtype != object
                            else pd.to_datetime(X[c], errors="coerce").astype("int64").to_numpy() for c in SPEC[name].time[::-1]])
        cut = int(0.8 * len(X))
        itr, ite = order[:cut], order[cut:]
    else:
        try:
            itr, ite = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=strat)
        except ValueError:  # a class with one row
            itr, ite = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    itr, ite = np.sort(itr), np.sort(ite)
    d.mkdir(parents=True, exist_ok=True)
    X.iloc[itr].reset_index(drop=True).assign(**{Y: y.iloc[itr].to_numpy()}).to_parquet(d / "train.parquet")
    X.iloc[ite].reset_index(drop=True).to_parquet(d / "test.parquet")
    np.save(d / "y_test.npy", y.iloc[ite].to_numpy())
    (d / "meta.json").write_text(json.dumps(dict(task=task, n=len(X), p=X.shape[1], desc=desc)))
    return d


# AutoFeat's feature selection fits its candidate pool in memory: past these sizes it timed out at the cap or
# crashed in every run, so it is not started (logged as "skipped", scored as the raw columns).
TIME_SPLIT = os.environ.get("H2H_TIME_SPLIT") == "1"
GROUP_SPLIT = os.environ.get("H2H_GROUP_SPLIT") == "1"
AUTOFEAT_MAX = dict(rows=5_000, cols=30)


def run_cell(name, seed, arm, root, budget, cap, threads, shuffle=False, raw_done=None):
    d = make_split(name, seed, root)
    meta = json.loads((d / "meta.json").read_text())
    if arm == "autofeat" and (meta["n"] * 0.8 > AUTOFEAT_MAX["rows"] or meta["p"] > AUTOFEAT_MAX["cols"]) \
            and raw_done is not None:
        return dict(raw_done, arm="autofeat", status="skipped", seconds=0.0, judge_seconds=0.0)
    task = meta["task"]
    tr, te = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")
    ytr, yte = tr.pop(Y).to_numpy(), np.load(d / "y_test.npy")
    rec = dict(dataset=name, group=SPEC[name].group, seed=seed, arm=arm + ("_shuffled" if shuffle else ""),
               task=task, n_train=len(tr), n_raw=tr.shape[1], status="ok", seconds=0.0, n_new=0)
    Xtr, Xte = tr, te
    if arm != "raw":
        work = d
        if shuffle:  # leak control: shuffled training labels, same holdout
            work = d / "shuffled"
            work.mkdir(exist_ok=True)
            for f in ("test.parquet", "meta.json"):
                shutil.copy(d / f, work / f)
            pd.read_parquet(d / "train.parquet").assign(
                **{Y: np.random.default_rng(0).permutation(ytr)}).to_parquet(work / "train.parquet")
            ytr = pd.read_parquet(work / "train.parquet")[Y].to_numpy()
        out = work / arm.replace(":", "_")
        shutil.rmtree(out, ignore_errors=True)
        cmd = [sys.executable if arm.startswith(("ff:", "llmff:", "ffp:", "ffpc:", "ffc:", "ffpm:")) else PY.get(arm, COMP_PY), str(HERE / "arms.py"), arm, str(work), "--budget", str(budget), "--threads", str(threads)]
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            so, se = p.communicate(timeout=cap)
            rec["seconds"] = time.time() - t0
            out.mkdir(exist_ok=True)
            (out / "log.txt").write_text((so or "")[-200000:] + "\n--- stderr\n" + (se or "")[-50000:])
            if p.returncode != 0:
                rec.update(status="crash", error=(se or so)[-600:])
        except subprocess.TimeoutExpired:  # kill the whole process group: tools fork worker pools
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            rec.update(status="timeout", seconds=float(cap))
        os.system(f"pkill -9 -s {p.pid} 2>/dev/null")  # any worker left in the session
        (out).mkdir(exist_ok=True)
        if rec["status"] == "ok":
            info = json.loads((out / "info.json").read_text())
            rec["n_new"], rec["fe_seconds"] = info.get("n_new", 0), info["seconds"]
            A, B = pd.read_parquet(out / "train.parquet"), pd.read_parquet(out / "test.parquet")
            if info.get("full"):
                Xtr, Xte = A, B[A.columns]
            else:
                Xtr = pd.concat([tr, A.set_axis(tr.index)], axis=1)
                Xte = pd.concat([te, B[A.columns].set_axis(te.index)], axis=1)
        for f in ("train.parquet", "test.parquet"):  # keep only info.json
            (out / f).unlink(missing_ok=True)
        shutil.rmtree(out / "ff_out", ignore_errors=True)
    rec["n_cols"] = int(Xtr.shape[1])
    t0 = time.time()
    rec.update(judge(Xtr, ytr, Xte, yte, task, seed))  # the judge always runs 4 threads: same trees for every arm
    rec["judge_seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="suite")
    ap.add_argument("--arms", default="raw,featureforge,openfe,autofeat,featuretools")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budget", type=int, default=900, help="each tool's own budget knob, where it has one")
    ap.add_argument("--cap", type=int, default=3600, help="wall-clock cap per tool run, seconds")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--root", default=str(Path.home() / "h2h_runs"))
    ap.add_argument("--log", default=str(HERE / "results" / "results.jsonl"))
    ap.add_argument("--shuffle", action="store_true", help="shuffled-label leak control")
    a = ap.parse_args()
    groups = {s.group for s in SPECS}
    names = [n for tok in a.datasets.split(",") for n in
             ([s.name for s in SPECS if s.group == tok] if tok in groups else [tok])]
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    done, raws = set(), {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["seed"], r["arm"]))
            if r["arm"] == "raw" and r.get("status") == "ok":
                raws[(r["dataset"], r["seed"])] = r
    for name in names:
        for seed in map(int, a.seeds.split(",")):
            for arm in a.arms.split(","):
                key = (name, seed, arm + ("_shuffled" if a.shuffle else ""))
                if key in done:
                    continue
                try:
                    rec = run_cell(name, seed, arm, Path(a.root), a.budget, a.cap, a.threads, a.shuffle,
                                   raw_done=raws.get((name, seed)))
                    if arm == "raw" and rec.get("status") == "ok":
                        raws[(name, seed)] = rec
                except Exception as e:  # data or judge failure: log and move on
                    rec = dict(dataset=name, seed=seed, arm=key[2], status="error", error=repr(e)[-600:])
                print(json.dumps({k: v for k, v in rec.items() if k != "error"}), flush=True)
                with log.open("a") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
