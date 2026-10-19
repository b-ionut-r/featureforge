"""Summarize results.jsonl: lift over raw columns per tool, wins, average rank and runtime.

    python report.py [results.jsonl] [--md out.md]

Lift is the percent reduction of the holdout loss (log loss, or RMSE) against the raw columns on the same
split; a crash or timeout counts as the raw columns (lift 0) and is listed. Ranks are over the tools on each
dataset (1 = lowest loss), averaged over splits first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def table(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["status"] != "error"].drop_duplicates(["dataset", "seed", "arm"], keep="last")
    raw = df[df.arm == "raw"].set_index(["dataset", "seed"])["loss"]
    df["raw_loss"] = [raw.get((d, s), np.nan) for d, s in zip(df.dataset, df.seed)]
    df["lift"] = 100 * (df.raw_loss - df.loss) / df.raw_loss
    df["secs"] = df["seconds"].fillna(0)
    return df


def summarize(df, arms=None, min_seeds=1):
    arms = arms or [a for a in ["featureforge", "llm", "openfe", "featuretools", "autofeat"] if a in set(df.arm)]
    # only (dataset, seed) cells every compared arm has finished, so each tool is scored on the same splits
    cells = df[df.arm.isin(arms + ["raw"])].pivot_table(index=["dataset", "seed"], columns="arm", values="loss")
    cells = cells.dropna(subset=[a for a in arms + ["raw"] if a in cells.columns])
    keep = set(cells.index)
    d = df[[(a, b) in keep for a, b in zip(df.dataset, df.seed)]]
    per = d.groupby(["dataset", "arm"]).agg(lift=("lift", "mean"), loss=("loss", "mean"), secs=("secs", "mean"),
                                           fails=("status", lambda s: int((s != "ok").sum())),
                                           group=("group", "first")).reset_index()
    rank = per[per.arm.isin(arms + ["raw"])].copy()
    rank["rank"] = rank.groupby("dataset")["loss"].rank(method="average")
    out = []
    for a in arms:
        p = per[per.arm == a]
        out.append(dict(tool=a, datasets=len(p), mean_lift=p.lift.mean(), median_lift=p.lift.median(),
                        wins=int((p.lift > 0.1).sum()), ties=int((p.lift.abs() <= 0.1).sum()),
                        losses=int((p.lift < -0.1).sum()), worst=p.lift.min(),
                        avg_rank=rank[rank.arm == a]["rank"].mean(), failed_runs=int(p.fails.sum()),
                        mean_secs=p.secs.mean(), median_secs=p.secs.median()))
    out.append(dict(tool="raw", datasets=per[per.arm == "raw"].shape[0], avg_rank=rank[rank.arm == "raw"]["rank"].mean()))
    return pd.DataFrame(out), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(HERE / "results" / "results.jsonl"))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    df = table(a.log)
    arms = a.arms.split(",") if a.arms else None
    s, per = summarize(df, arms)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    wide = per.pivot_table(index=["group", "dataset"], columns="arm", values="lift").round(2)
    print(wide.to_string())
    if a.md:
        lines = ["| Tool | Datasets | Mean lift % | Median lift % | Wins / ties / losses vs raw | Worst % | Avg rank | Failed runs | Mean s |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in s.itertuples():
            if r.tool == "raw":
                lines.append(f"| raw columns | {r.datasets} | 0 | 0 | | | {r.avg_rank:.2f} | | |")
                continue
            lines.append(f"| {r.tool} | {r.datasets} | {r.mean_lift:+.2f} | {r.median_lift:+.2f} | {r.wins} / {r.ties} / {r.losses} | "
                         f"{r.worst:+.1f} | {r.avg_rank:.2f} | {r.failed_runs} | {r.mean_secs:.0f} |")
        Path(a.md).write_text("\n".join(lines) + "\n\n" + wide.to_markdown() + "\n")


if __name__ == "__main__":
    main()
