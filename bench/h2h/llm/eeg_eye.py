"""Domain features for eeg_eye, written from the schema and description only."""
import numpy as np
import pandas as pd

# Channel order (from description): V1 AF3, V2 F7, V3 F3, V4 FC5, V5 T7, V6 P(7), V7 O1,
# V8 O2, V9 P8, V10 T8, V11 FC6, V12 F4, V13 F8, V14 AF4.
COLS = ["V%d" % i for i in range(1, 15)]
PAIRS = [(1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)]


def _robust_z(df):
    x = df[COLS].astype(float)
    med = x.median()
    iqr = (x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
    return ((x - med) / iqr).clip(-10, 10)


# Left-right hemispheric differences of mirrored electrodes: eye closure changes symmetric patterns.
def hemispheric_asymmetry(df):
    out = {}
    for a, b in PAIRS:
        out["asym_%d_%d" % (a, b)] = df["V%d" % a] - df["V%d" % b]
    return pd.DataFrame(out, index=df.index)


# Frontal (AF3, AF4, F7, F8) channels pick up eye/blink artifacts; compare them with occipital (O1, O2).
# OUTSIDE KNOWLEDGE: eye movements and blinks dominate frontal EEG; alpha rises over occipital sites with eyes closed.
def frontal_vs_occipital(df):
    z = _robust_z(df)
    fr = z[["V1", "V14", "V2", "V13"]].mean(axis=1)
    oc = z[["V7", "V8"]].mean(axis=1)
    return pd.DataFrame({"frontal_z": fr, "occipital_z": oc, "front_minus_occ": fr - oc}, index=df.index)


# Robust standardized channel values (median/IQR) to tame the spike artifacts present in raw headset data.
def robust_channels(df):
    z = _robust_z(df)
    z.columns = ["rz_" + c for c in COLS]
    return z


# Common-average reference: subtracting the instantaneous mean across channels removes shared drift.
def common_average_reference(df):
    x = df[COLS].astype(float)
    car = x.sub(x.mean(axis=1), axis=0)
    car.columns = ["car_" + c for c in COLS]
    return car


# Spread across channels at an instant (global field power proxy) on standardized values.
def cross_channel_spread(df):
    z = _robust_z(df)
    return pd.DataFrame({
        "z_std": z.std(axis=1), "z_max_abs": z.abs().max(axis=1), "z_mean": z.mean(axis=1),
    }, index=df.index)


# Eye-blink proxy: horizontal EOG-like difference (F7 - F8) and vertical-like frontal mean (AF3 + AF4).
# OUTSIDE KNOWLEDGE: F7/F8 difference approximates horizontal eye movement, AF3/AF4 vertical/blink activity.
def eog_proxies(df):
    z = _robust_z(df)
    return pd.DataFrame({
        "heog": z["V2"] - z["V13"],
        "veog": (z["V1"] + z["V14"]) / 2 - (z["V3"] + z["V12"]) / 2,
    }, index=df.index)


PROPOSALS = [
    ("hemispheric_asymmetry", hemispheric_asymmetry),
    ("frontal_vs_occipital", frontal_vs_occipital),
    ("robust_channels", robust_channels),
    ("common_average_reference", common_average_reference),
    ("cross_channel_spread", cross_channel_spread),
    ("eog_proxies", eog_proxies),
]
