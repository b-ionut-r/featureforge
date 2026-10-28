"""Favorita end to end through build_features: the user's whole recent history as the training file."""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("FF_SRC", "/home/user/featureforge/src"))
import fav_bench as fb
days = int(sys.argv[1]); out = sys.argv[2]

tr = fb.long(fb.HOLD0 - pd.Timedelta(days=days), fb.HOLD0 - pd.Timedelta(days=1))
if fb.CONTEST:
    te = pd.read_csv(fb.D / "test.csv", parse_dates=["date"]); te["onpromotion"] = te.onpromotion.astype(np.float32)
    if not fb.PROMO: te = te.drop(columns=["onpromotion"])
else:
    te = fb.long(fb.HOLD0, fb.HOLD0 + pd.Timedelta(days=15), with_y=False)
    te.insert(0, "id", np.arange(len(te)))
tr.insert(0, "id", -np.arange(1, len(tr) + 1))
it, st = fb.statics()
for df in (tr, te):
    df["store_nbr"] = df.store_nbr.astype(np.int16); df["item_nbr"] = df.item_nbr.astype(np.int32)
    for c in ("family", "class", "perishable"):
        df[c] = pd.Categorical(df.item_nbr.map(it.set_index("item_nbr")[c]).to_numpy())
    for c in ("city", "state", "type", "cluster"):
        df[c] = pd.Categorical(df.store_nbr.map(st.set_index("store_nbr")[c]).to_numpy())
print("train", tr.shape, "test", te.shape, flush=True)
from featureforge import build_features
t = time.time()
build_features(tr, te, fb.Y, id="id", log_target=True, budget=900, out_dir=out)
print(f"build_features {time.time() - t:.0f}s", flush=True)
