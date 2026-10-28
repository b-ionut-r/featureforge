"""Domain features for porto_seguro, written from the schema and description only."""
import numpy as np
import pandas as pd


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def missing_count(df):
    # number of missing fields per policy: a well-known strong signal in this competition (incomplete applications)
    cols = [c for c in df.columns if not c.startswith("ps_calc")]
    return df[cols].isna().sum(axis=1).astype(float)


def reg03_decoded(df):
    # ps_reg_03 = sqrt(k)/40 for an integer k (a de-obfuscated regional code); recover k and split into two digits groups
    k = (_num(df["ps_reg_03"]) * 40) ** 2
    k = k.round()
    return pd.DataFrame({"k": k, "k_mod27": k % 27, "k_div27": np.floor(k / 27)})


def car13_x_reg03(df):
    # interaction of the strongest car feature (vehicle value proxy) with regional risk
    return _num(df["ps_car_13"]) * _num(df["ps_reg_03"])


def car_decoded(df):
    # ps_car_12, ps_car_13, ps_car_15 are square roots of integer quantities; recover the underlying integers
    return pd.DataFrame({"car12_sq": (_num(df["ps_car_12"]) ** 2 * 10000).round(),
                         "car13_sq": (_num(df["ps_car_13"]) ** 2 * 48400).round(),
                         "car15_sq": (_num(df["ps_car_15"]) ** 2).round()})


def ind_bin_sum(df):
    # count of active individual flags (ps_ind_06..18 bins)
    cols = [c for c in df.columns if c.startswith("ps_ind") and c.endswith("_bin")]
    return df[cols].apply(_num).sum(axis=1)


def ind_06_09_onehot(df):
    # ps_ind_06..09 bins form a one-hot block; collapse to one categorical code
    cols = ["ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"]
    m = df[cols].apply(_num).fillna(0).values
    return pd.Series(np.where(m.sum(1) > 0, m.argmax(1), -1).astype(float), index=df.index)


def car11cat_count(df):
    # frequency of the high-cardinality car model code: common models behave differently from rare ones
    k = df["ps_car_11_cat"].astype(str)
    return k.groupby(k).transform("size").astype(float)


def car13_rel_car11cat(df):
    # vehicle value relative to the average for the same car model code
    v = _num(df["ps_car_13"])
    k = df["ps_car_11_cat"].astype(str)
    return v - v.groupby(k).transform("mean")


def reg_sum(df):
    # combined regional score
    return _num(df["ps_reg_01"]) + _num(df["ps_reg_02"]) + _num(df["ps_reg_03"]).fillna(0)


PROPOSALS = [("missing_count", missing_count), ("car13_x_reg03", car13_x_reg03), ("reg03_decoded", reg03_decoded),
             ("car_decoded", car_decoded), ("ind_bin_sum", ind_bin_sum), ("ind_06_09_onehot", ind_06_09_onehot),
             ("car13_rel_car11cat", car13_rel_car11cat), ("car11cat_count", car11cat_count), ("reg_sum", reg_sum)]
