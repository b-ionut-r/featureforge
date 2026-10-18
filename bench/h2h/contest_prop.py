"""Proposals on a contest, on top of a finished FeatureForge build (the search's output does not depend on them).

    python contest_prop.py --case hc --base v10 --proposals llm_contest/hc.py --label v10p

Runs exactly the pipeline's last step (``proposals.compute`` on the raw main table, ``proposals.gate`` on the
training output) and writes ``<label>/`` for ``contest_check.py judge``. Build seconds = base + this step.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from featureforge import proposals as P  # noqa: E402

ROOT = Path.home() / "h2h_runs" / "contests"
Y = "__y__"

ap = argparse.ArgumentParser()
ap.add_argument("--case"), ap.add_argument("--base"), ap.add_argument("--proposals"), ap.add_argument("--label")
a = ap.parse_args()
d = ROOT / a.case
meta = json.loads((d / "meta.json").read_text())
t0 = time.time()
tr, te = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")
y = tr.pop(Y).to_numpy()
raw = [F.drop(columns=[meta["id"]]) for F in (tr, te)]
frames = P.compute(P.load(HERE / a.proposals), *raw)
print(f"proposals: {len(frames)} usable", flush=True)
A = pd.read_parquet(d / a.base / "train_features.parquet")
B = pd.read_parquet(d / a.base / "test_features.parquet")
feat = A.drop(columns=[Y, meta["id"]], errors="ignore")
kept = P.gate(feat, np.asarray(y, float), meta["task"], frames)
n = len(A)
cols = dict(frames)
for label, names in kept:
    for c in names:
        A[c] = cols[label][c].iloc[:n].to_numpy()
        B[c] = cols[label][c].iloc[n:].to_numpy()
out = d / a.label
out.mkdir(exist_ok=True)
A.to_parquet(out / "train_features.parquet")
B.to_parquet(out / "test_features.parquet")
base_s = json.loads((d / a.base / "seconds.json").read_text())["seconds"]
(out / "seconds.json").write_text(json.dumps(dict(seconds=base_s + time.time() - t0, proposals_seconds=time.time() - t0)))
print(f"kept: {[l for l, _ in kept]} in {time.time() - t0:.0f}s", flush=True)
