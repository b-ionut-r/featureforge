"""Domain features for kick, written from the schema and description only."""
import re

import numpy as np
import pandas as pd


def _p(df, c):
    # MMR prices of 0/1 are placeholders for "no guide price"
    v = pd.to_numeric(df[c], errors="coerce")
    return v.where(v > 1)


def _div(a, b):
    return a / b.where(b.abs() > 0)


def cost_vs_market(df):
    # paying above the MMR auction guide (or far below it) flags a bad buy / hidden problem
    cost = pd.to_numeric(df["VehBCost"], errors="coerce").where(lambda x: x > 1)
    out = pd.DataFrame(index=df.index)
    out["cost_over_acq_auc_avg"] = _div(cost, _p(df, "MMRAcquisitionAuctionAveragePrice"))
    out["cost_over_acq_auc_clean"] = _div(cost, _p(df, "MMRAcquisitionAuctionCleanPrice"))
    out["cost_over_cur_auc_avg"] = _div(cost, _p(df, "MMRCurrentAuctionAveragePrice"))
    out["cost_over_acq_retail"] = _div(cost, _p(df, "MMRAcquisitionRetailAveragePrice"))
    out["cost_minus_acq_auc_avg"] = cost - _p(df, "MMRAcquisitionAuctionAveragePrice")
    return out


def odometer_per_year(df):
    # miles per year of age: abnormally low (rolled-back) or high usage both raise kick risk
    age = pd.to_numeric(df["VehicleAge"], errors="coerce").astype(float)
    odo = pd.to_numeric(df["VehOdo"], errors="coerce")
    return pd.DataFrame({"odo_per_year": odo / (age + 0.5),
                         "odo_minus_make_age": odo - odo.groupby([df["Make"].astype(str), age]).transform("mean")},
                        index=df.index)


def price_trend(df):
    # change in market guide between acquisition and now, and clean-vs-average spread (condition uncertainty)
    out = pd.DataFrame(index=df.index)
    out["auc_trend"] = _div(_p(df, "MMRCurrentAuctionAveragePrice"), _p(df, "MMRAcquisitionAuctionAveragePrice"))
    out["retail_trend"] = _div(_p(df, "MMRCurrentRetailAveragePrice"), _p(df, "MMRAcquisitionRetailAveragePrice"))
    out["clean_spread"] = _div(_p(df, "MMRAcquisitionAuctionCleanPrice"), _p(df, "MMRAcquisitionAuctionAveragePrice"))
    out["retail_margin"] = _div(_p(df, "MMRAcquisitionRetailAveragePrice"), _p(df, "MMRAcquisitionAuctionAveragePrice"))
    out["n_missing_mmr"] = sum(_p(df, c).isna().astype(float) for c in df.columns if c.startswith("MMR"))
    return out


def warranty_ratio(df):
    # warranty cost relative to vehicle cost: insurer's view of mechanical risk per dollar
    return _div(pd.to_numeric(df["WarrantyCost"], errors="coerce"), pd.to_numeric(df["VehBCost"], errors="coerce"))


def parsed_model(df):
    # engine displacement, cylinders, drive and doors parsed from Model/SubModel strings
    txt = (df["Model"].astype(object).fillna("").astype(str) + " " +
           df["SubModel"].astype(object).fillna("").astype(str)).str.upper()

    def disp(s):
        m = re.search(r"(\d\.\d)\s*L", s)
        return float(m.group(1)) if m else np.nan

    def cyl(s):
        m = re.search(r"\bV(6|8|10)\b|\b(4|6|8)C\b|\bI-?(4|5|6)\b", s)
        if not m:
            return np.nan
        return float(next(g for g in m.groups() if g))

    def doors(s):
        m = re.search(r"\b(2|4|5)D\b", s)
        return float(m.group(1)) if m else np.nan
    return pd.DataFrame({"engine_l": txt.map(disp), "cylinders": txt.map(cyl), "doors": txt.map(doors),
                         "awd_4wd": txt.str.contains(r"4WD|AWD|4X4").astype(float),
                         "ffv": txt.str.contains("FFV").astype(float)}, index=df.index)


def purchase_date_parts(df):
    # purchase month/weekday/year: seasonality of auction quality and time trend
    d = pd.to_datetime(pd.to_numeric(df["PurchDate"], errors="coerce"), unit="s", errors="coerce")
    return pd.DataFrame({"purch_month": d.dt.month.astype(float), "purch_dow": d.dt.dayofweek.astype(float),
                         "purch_t": (d - pd.Timestamp("2009-01-01")).dt.days.astype(float)}, index=df.index)


def buyer_and_model_frequency(df):
    # buyer volume and how common the model is: rare models and occasional buyers are riskier
    out = pd.DataFrame(index=df.index)
    for c in ["BYRNO", "Model", "SubModel", "VNZIP1"]:
        s = df[c].astype(object).fillna("__nan__").astype(str)
        out["cnt_" + c] = s.groupby(s).transform("size").astype(float)
    return out


def cost_vs_model_peers(df):
    # price paid relative to same model & age peers: cheap-for-its-kind cars hide defects
    cost = pd.to_numeric(df["VehBCost"], errors="coerce")
    key = df["Model"].astype(object).fillna("").astype(str) + "_" + df["VehicleAge"].astype(str)
    return _div(cost, cost.groupby(key).transform("mean"))


PROPOSALS = [
    ("cost_vs_market", cost_vs_market),
    ("odometer_per_year", odometer_per_year),
    ("price_trend", price_trend),
    ("cost_vs_model_peers", cost_vs_model_peers),
    ("warranty_ratio", warranty_ratio),
    ("parsed_model", parsed_model),
    ("buyer_and_model_frequency", buyer_and_model_frequency),
    ("purchase_date_parts", purchase_date_parts),
]
