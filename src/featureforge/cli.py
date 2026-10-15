"""Command line: ``featureforge --train train.csv --test test.csv --target y [--table name=path[:key[:time]]] ...``

Writes ``train_features.parquet`` and ``test_features.parquet`` to ``--out-dir``. Every option is
``build_features``'s keyword of the same name (see ``featureforge.pipeline``).
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

from featureforge.pipeline import PipelineConfig, run


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="featureforge", description="Automated, leak-free feature engineering for tabular "
                                 "contests: related tables, structure-triggered blocks and a gated feature search.")
    ap.add_argument("--train", required=True, help="training table (CSV or parquet) holding the target")
    ap.add_argument("--test", required=True, help="the rows the features are for (the contest's test file)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", default=None, help="row id column, written to the outputs and never used as a feature")
    ap.add_argument("--table", action="append", default=[], help="child table: name=path[:key[:time]] (repeatable)")
    ap.add_argument("--time", default=None, help="main-table time column for as-of aggregation (default: auto, when a key repeats; given, every timed table is aggregated as of it, e.g. a snapshot date when train and test are different months)")
    ap.add_argument("--child-models", action=argparse.BooleanOptionalAction, default=True,
                    help="out-of-fold child-row model features for every keyed child table, kept when a quick CV "
                         "says they help (Home Credit 0.7976 -> 0.8004; --no-child-models turns them off)")
    ap.add_argument("--task", default=None, choices=["regression", "binary", "multiclass"])
    ap.add_argument("--log-target", action="store_true", help="search on log1p(target) (RMSLE-scored contests)")
    ap.add_argument("--budget", type=float, default=900, help="FeatureForge time budget (s of a 4-core reference machine)")
    ap.add_argument("--top-related", type=int, default=200,
                    help="related-table columns handed to FeatureForge's search (all are kept in the output)")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--related-cache", default=None,
                    help="directory to save the related-table features to, or load them from when present "
                         "(runs that differ only in FeatureForge settings share them)")
    ap.add_argument("--forge-kw", default="{}", help="JSON of extra FeatureForge arguments")
    ap.add_argument("--forecast", default="auto", choices=["auto", "off"],
                    help="forecasting block (featureforge.blocks.forecast): on when the data has a date, "
                         "repeating entity keys and test rows after the training period")
    ap.add_argument("--forecast-kw", default="{}", help="JSON of extra ForecastFeatures arguments")
    ap.add_argument("--history", default="auto", choices=["auto", "off"],
                    help="history blocks (featureforge.blocks.history), on from structure: latest state of keyed "
                         "child tables with a time column (last, last - mean, last - previous), and outcome history "
                         "of as-of event logs that carry the target (earlier outcomes, strictly before each row)")
    ap.add_argument("--panel", default="auto", choices=["auto", "off"],
                    help="intraday panels: each entity against all at the same moment and its own last steps of the "
                         "day (featureforge.blocks.panel); on from structure")
    ap.add_argument("--lists", default="auto", choices=["auto", "off"],
                    help="list columns (item counts and common-item indicators) and per-unit amounts (a skewed amount "
                         "over small counts and their total), featureforge.blocks.lists; on from structure")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    a = parser().parse_args(argv)
    run(PipelineConfig(train=a.train, test=a.test, target=a.target, id=a.id, table=a.table, time=a.time,
                       child_models=a.child_models, task=a.task, log_target=a.log_target, budget=a.budget,
                       top_related=a.top_related, out_dir=a.out_dir, related_cache=a.related_cache,
                       forge_kw=json.loads(a.forge_kw), forecast=a.forecast, forecast_kw=json.loads(a.forecast_kw),
                       history=a.history, panel=a.panel, lists=a.lists))


if __name__ == "__main__":
    main()
