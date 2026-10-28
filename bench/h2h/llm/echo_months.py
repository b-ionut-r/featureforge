"""Domain features for echo_months, written from the schema and description only."""
import numpy as np
import pandas as pd


def _b(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def followup_status(df):
    # still_alive + alive_at_1 jointly bracket survival (description): dead and not alive at 1 year => < 12 months
    sa = _b(df, "still_alive")
    a1 = _b(df, "alive_at_1")
    return pd.DataFrame({"still_alive": sa, "alive_at_1": a1,
                         "died_within_year": ((sa == 0) & (a1 == 0)).astype(float),
                         "alive_at_1_missing": a1.isna().astype(float)})


def lv_function(df):
    # LV systolic function: low fractional shortening and high E-point septal separation = poor contractility
    # OUTSIDE KNOWLEDGE: normal fractional shortening > ~0.30, EPSS > ~10 mm abnormal, LVDD > ~5.5 cm dilated
    return pd.DataFrame({"fs_low": (df["fractional"] < 0.30).astype(float), "epss_high": (df["epss"] > 10).astype(float),
                         "lv_dilated": (df["lvdd"] > 5.5).astype(float),
                         "epss_over_fs": df["epss"] / df["fractional"].replace(0, np.nan)})


def segments_seen(df):
    # wall-motion index = wall-motion score / number of segments seen (description): recover segment count
    return df["wall_score"] / df["wall_index"].replace(0, np.nan)


def dysfunction_score(df):
    # combined heart-failure burden: wall-motion index, dilation and reduced shortening
    z = (df["wall_index"].fillna(df["wall_index"].median()) - 1.0
         + (df["lvdd"] - 4.5).fillna(0) / 2.0 + (0.30 - df["fractional"]).fillna(0) * 5
         + _b(df, "pericardial").fillna(0))
    return z


def age_x_wall(df):
    # older patients with extensive infarction (high wall index) have the shortest survival
    return pd.DataFrame({"age_x_wmi": df["age"] * df["wall_index"], "age_over_fs": df["age"] * (1 - df["fractional"])})


def contractility_volume(df):
    # stroke-volume-like proxy: shortening times diastolic dimension (cubed for volume)
    return pd.DataFrame({"fs_x_lvdd3": df["fractional"] * df["lvdd"] ** 3, "lvdd3": df["lvdd"] ** 3})


PROPOSALS = [("followup_status", followup_status), ("lv_function", lv_function), ("segments_seen", segments_seen),
             ("dysfunction_score", dysfunction_score), ("age_x_wall", age_x_wall),
             ("contractility_volume", contractility_volume)]
