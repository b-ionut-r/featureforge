"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 14 channels of values ~4000-4600 and 14980 rows match the Emotiv EEG eye-state recording,
# channel order AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 (f_1..f_14). Target: eyes open/closed.
# Mirror pairs: (f_1,f_14) (f_2,f_13) (f_3,f_12) (f_4,f_11) (f_5,f_10) (f_6,f_9) (f_7,f_8).

_C = ["f_%d" % i for i in range(1, 15)]
_PAIRS = [(1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)]


def _x(df):
    return df[_C].astype(float)


def hemispheric_diffs(df):
    # left minus right for mirrored electrodes (lateralised activity, eye-movement artefacts)
    return pd.DataFrame({"d_%d_%d" % (a, b): df["f_%d" % a] - df["f_%d" % b] for a, b in _PAIRS})


def common_average_reference(df):
    # re-reference each channel to the row mean, removing global offsets/drifts common to all electrodes
    x = _x(df)
    car = x.sub(x.mean(axis=1), axis=0)
    car.columns = [c + "_car" for c in _C]
    return car


def row_summary(df):
    # global field strength and spread across the scalp
    x = _x(df)
    med = x.median(axis=1)
    return pd.DataFrame({
        "row_mean": x.mean(axis=1),
        "row_median": med,
        "row_std": x.std(axis=1),
        "row_range": x.max(axis=1) - x.min(axis=1),
        "row_mad": x.sub(med, axis=0).abs().median(axis=1),
    })


def region_contrasts(df):
    # frontal (blink / eye-movement) vs occipital (alpha when eyes closed) and temporal-parietal regions
    frontal = df[["f_1", "f_14", "f_2", "f_13", "f_3", "f_12"]].mean(axis=1)
    occip = df[["f_7", "f_8"]].mean(axis=1)
    temporal = df[["f_5", "f_10", "f_4", "f_11"]].mean(axis=1)
    pariet = df[["f_6", "f_9"]].mean(axis=1)
    return pd.DataFrame({
        "front_minus_occ": frontal - occip,
        "front_minus_temp": frontal - temporal,
        "occ_minus_par": occip - pariet,
        "prefrontal_mean": df[["f_1", "f_14"]].mean(axis=1),
        "prefrontal_minus_frontal": df[["f_1", "f_14"]].mean(axis=1) - df[["f_3", "f_12"]].mean(axis=1),
    })


def robust_channel_z(df):
    # channel value relative to its own robust (median/IQR) level - the recording has large spikes
    x = _x(df)
    med = x.median()
    iqr = (x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
    z = ((x - med) / iqr).clip(-20, 20)
    out = pd.DataFrame({"z_mean": z.mean(axis=1), "z_absmax": z.abs().max(axis=1),
                        "n_spike": (z.abs() > 5).sum(axis=1).astype(float)})
    zz = z.copy()
    zz.columns = [c + "_rz" for c in _C]
    return pd.concat([out, zz], axis=1)


PROPOSALS = [
    ("hemispheric_diffs", hemispheric_diffs),
    ("common_average_reference", common_average_reference),
    ("row_summary", row_summary),
    ("region_contrasts", region_contrasts),
    ("robust_channel_z", robust_channel_z),
]
