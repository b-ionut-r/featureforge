"""Public tabular datasets for the head-to-head bench, loaded as (X, y, task).

Three groups:
  * ``suite``: the 20 public tables (AutoGluon/PMLB) FeatureForge's suites reported on.
  * ``openml``: 22 OpenML copies of Kaggle contests and Kaggle Playground source tables, plus 7
    tables with entity, ID or time structure.
  * ``papers``: the OpenML tables used by the OpenFE (ICML 2023) and CAAFE (NeurIPS 2023) papers
    that are not already above.
Regression targets scored by RMSLE are log1p-transformed here, so every arm sees the same target and
the judge scores RMSE. Files cache under $H2H_DATA (default ~/data/h2h).
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

AG = "https://autogluon.s3.amazonaws.com/datasets/"
PMLB = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"


@dataclass
class Spec:
    name: str
    url: str
    target: str = ""
    task: str = "binary"            # regression | binary | multiclass
    log: bool = False               # RMSLE contest metric: log1p the target
    drop: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    sep: str = ","
    max_rows: Optional[int] = 20_000
    group: str = "suite"


def O(did):
    return f"openml:{did}"


SPECS = [
    # --- suite: real categoricals / contest classics, then numeric PMLB tables
    Spec("ames", AG + "AmesHousingPriceRegression/train_data.csv", "SalePrice", "regression", True, ["Order", "PID"], ["MS.SubClass"]),
    Spec("titanic", AG + "titanic/train.csv", "Survived", "binary", drop=["PassengerId", "Name"]),
    Spec("adult", AG + "AdultIncomeBinaryClassification/train_data.csv", "class"),
    Spec("covertype", AG + "CoverTypeMulticlassClassification/train_data.csv", "Cover_Type", "multiclass"),
    Spec("fars", PMLB.format("fars"), "target", "multiclass", sep="\t"),
    Spec("churn", PMLB.format("churn"), "target", sep="\t"),
    Spec("coil2000", PMLB.format("coil2000"), "target", sep="\t"),
    Spec("sleep", PMLB.format("sleep"), "target", "multiclass", sep="\t"),
    Spec("houses", PMLB.format("537_houses"), "target", "regression", sep="\t"),
    Spec("fried", PMLB.format("564_fried"), "target", "regression", sep="\t"),
    Spec("house_16H", PMLB.format("574_house_16H"), "target", "regression", sep="\t"),
    Spec("pol", PMLB.format("201_pol"), "target", "regression", sep="\t"),
    Spec("wind", PMLB.format("503_wind"), "target", "regression", sep="\t"),
    Spec("puma8NH", PMLB.format("225_puma8NH"), "target", "regression", sep="\t"),
    Spec("cpu_act", PMLB.format("197_cpu_act"), "target", "regression", sep="\t"),
    Spec("magic", PMLB.format("magic"), "target", sep="\t"),
    Spec("phoneme", PMLB.format("phoneme"), "target", sep="\t"),
    Spec("spambase", PMLB.format("spambase"), "target", sep="\t"),
    Spec("wine_white", PMLB.format("wine_quality_white"), "target", "multiclass", sep="\t"),
    Spec("page_blocks", PMLB.format("page_blocks"), "target", "multiclass", sep="\t"),
    # --- openml: Kaggle contests and Playground source tables
    Spec("amazon", O(4135), max_rows=None, group="openml"),
    Spec("porto_seguro", O(42742), drop=["id"], max_rows=100_000, group="openml"),
    Spec("allstate", O(42571), task="regression", log=True, drop=["id"], max_rows=60_000, group="openml"),
    Spec("kdd_appetency", O(1111), max_rows=None, group="openml"),
    Spec("kdd_upselling", O(1114), max_rows=None, group="openml"),
    Spec("bank_marketing", O(1461), max_rows=None, group="openml"),
    Spec("bank_churn", O(46911), max_rows=None, group="openml"),
    Spec("give_me_credit", O(46929), max_rows=60_000, group="openml"),
    Spec("kick", O(41162), max_rows=None, group="openml"),
    Spec("bike_sharing", O(42712), task="regression", log=True, max_rows=None, group="openml"),
    Spec("credit_default", O(46919), max_rows=None, group="openml"),
    Spec("hr_analytics", O(46935), max_rows=None, group="openml"),
    Spec("coupon", O(46937), max_rows=None, group="openml"),
    Spec("abalone", O(46903), task="regression", log=True, max_rows=None, group="openml"),
    Spec("diamonds", O(42225), task="regression", log=True, max_rows=None, group="openml"),
    Spec("airline_satisfaction", O(46920), max_rows=60_000, group="openml"),
    Spec("food_delivery", O(46928), task="regression", max_rows=None, group="openml"),
    Spec("steel_plates", O(46959), task="multiclass", max_rows=None, group="openml"),
    Spec("click", O(42733), max_rows=None, group="openml"),
    Spec("house_prices", O(42563), task="regression", log=True, drop=["Id"], max_rows=None, group="openml"),
    Spec("mercedes", O(42570), task="regression", drop=["ID"], max_rows=None, group="openml"),
    Spec("miami_housing", O(46942), task="regression", log=True, max_rows=None, group="openml"),
    Spec("airlines", O(1169), cat_cols=["Flight"], max_rows=100_000, group="openml"),
    Spec("medical_charges", O(42130), task="regression", log=True,
         drop=["provider_name", "provider_street_address", "average_medicare_payments"],
         cat_cols=["provider_id", "provider_zip_code"], max_rows=100_000, group="openml"),
    Spec("nyc_taxi_tip", O(42729), task="regression", drop=["total_amount"], max_rows=100_000, group="openml"),
    Spec("sf_crime", O(42344), max_rows=100_000, group="openml"),
    Spec("kc_house", O(42731), task="regression", log=True, drop=["id"], cat_cols=["zipcode"], max_rows=None, group="openml"),
    Spec("wine_reviews", O(41275), task="regression", max_rows=100_000, group="openml"),
    Spec("zurich_delays", O(42495), task="regression", max_rows=None, group="openml"),
    # --- papers: CAAFE's OpenML tables, then OpenFE's
    Spec("balance_scale", O(11), task="multiclass", group="papers"),
    Spec("breast_w", O(15), group="papers"),
    Spec("cmc", O(23), task="multiclass", group="papers"),
    Spec("credit_g", O(31), group="papers"),
    Spec("diabetes", O(37), group="papers"),
    Spec("tic_tac_toe", O(50), group="papers"),
    Spec("eucalyptus", O(188), task="multiclass", group="papers"),
    Spec("pc1", O(1068), group="papers"),
    Spec("jungle_chess", O(41027), task="multiclass", max_rows=None, group="papers"),
    Spec("nomao", O(1486), max_rows=None, group="papers"),
    Spec("miniboone", O(41150), max_rows=100_000, group="papers"),
    Spec("diabetes130", O(4541), task="multiclass", max_rows=100_000, group="papers"),
    Spec("vehicle", O(54), task="multiclass", group="papers"),
    Spec("california", O(44090), task="regression", log=True, max_rows=None, group="papers"),
    # --- fresh: chosen after the new blocks and gates were designed, never run before the final check
    # (OpenML-CC18 and regression-benchmark tables; list fixed before any run)
    Spec("kr_vs_kp", O(3), group="fresh"),
    Spec("credit_approval", O(29), group="fresh"),
    Spec("ilpd", O(1480), group="fresh"),
    Spec("qsar_biodeg", O(1494), group="fresh"),
    Spec("wdbc", O(1510), group="fresh"),
    Spec("blood_transfusion", O(1464), group="fresh"),
    Spec("banknote", O(1462), group="fresh"),
    Spec("ozone", O(1487), group="fresh"),
    Spec("wilt", O(40983), group="fresh"),
    Spec("segment", O(36), task="multiclass", group="fresh"),
    Spec("satimage", O(182), task="multiclass", group="fresh"),
    Spec("letter", O(6), task="multiclass", group="fresh"),
    Spec("higgs", O(23512), group="fresh"),
    Spec("kin8nm", O(189), task="regression", group="fresh"),
    Spec("elevators", O(216), task="regression", group="fresh"),
    Spec("brazilian_houses", O(42688), task="regression", log=True, group="fresh"),
    Spec("cpu_small", O(227), task="regression", group="fresh"),
    Spec("wine_red", O(40691), task="multiclass", group="fresh"),
    Spec("mushroom", O(24), group="fresh"),
    Spec("sylvine", O(41146), group="fresh"),
    # --- kaggle: whole contest train files (no row cap), a random 80/20 split as these contests' tests were
    Spec("santander_full", "kaggle:santander-customer-transaction-prediction/train.csv", "target", drop=["ID_code"],
         max_rows=None, group="kaggle"),
    Spec("porto_full", "kaggle:porto-seguro-safe-driver-prediction/train.csv", "target", drop=["id"],
         max_rows=None, group="kaggle"),
]
SPEC = {s.name: s for s in SPECS}


def data_dir() -> Path:
    d = Path(os.environ.get("H2H_DATA", Path.home() / "data" / "h2h"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _openml(did: int):
    meta_f = data_dir() / f"openml_{did}.json"
    if not meta_f.exists():
        with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{did}", timeout=60) as r:
            meta_f.write_bytes(r.read())
    meta = json.loads(meta_f.read_text())["data_set_description"]
    pq = data_dir() / f"openml_{did}.pq"
    if not pq.exists():
        urllib.request.urlretrieve(meta["parquet_url"], pq)
    df = pd.read_parquet(pq)
    drop = []
    for key in ("ignore_attribute", "row_id_attribute"):
        v = meta.get(key)
        drop += v.split(",") if isinstance(v, str) else list(v or [])
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == bool:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df.drop(columns=[c for c in drop if c in df.columns]), meta["default_target_attribute"], meta


def load(name: str):
    """(X, y, task, description). The row cap's sample is fixed (seed 0) so every split sees one table."""
    s = SPEC[name]
    desc = ""
    if s.url.startswith("kaggle:"):  # a contest's train file, downloaded with the Kaggle CLI
        slug, fname = s.url[7:].split("/")
        f = data_dir() / "kaggle" / f"{slug}_{fname}.zip"
        if not f.exists():
            import subprocess
            f.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["kaggle", "competitions", "download", "-c", slug, "-f", fname, "-p", str(f.parent)], check=True)
            (f.parent / f"{fname}.zip").rename(f)
        df, target = pd.read_csv(f), s.target
    elif s.url.startswith("openml:"):
        df, target, meta = _openml(int(s.url.split(":")[1]))
        target = s.target or target
        desc = (meta.get("description") or "")[:3000]
    else:
        f = data_dir() / s.url.split("/datasets/")[-1].replace("/", "_")
        if not f.exists():
            urllib.request.urlretrieve(s.url, f)
        df, target = pd.read_csv(f, sep=s.sep), s.target
    df = df.drop(columns=[c for c in s.drop if c in df.columns])
    df = df[df[target].notna()].reset_index(drop=True)
    if s.max_rows and len(df) > s.max_rows:
        df = df.sample(s.max_rows, random_state=0).reset_index(drop=True)
    y = df.pop(target)
    for c in s.cat_cols:
        df[c] = df[c].astype(str)
    for c in df.columns:  # pandas 3 string dtype -> object, so every tool reads the same column types
        if pd.api.types.is_string_dtype(df[c]) and df[c].dtype != object:
            df[c] = df[c].astype(object)
    if s.task == "regression":
        y = pd.to_numeric(y, errors="coerce").astype(float)
        if s.log:
            y = np.log1p(y.clip(lower=0))
    else:
        y = pd.Series(pd.factorize(y.astype(str), sort=True)[0], name=target)
    return df, y, s.task, desc
