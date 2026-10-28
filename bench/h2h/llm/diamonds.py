"""Domain features for diamonds, written from the schema and description only."""
import numpy as np
import pandas as pd

CUT = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
COLOR = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
CLAR = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}


def _ord(df, c, m):
    return df[c].astype(str).str.strip().map(m).astype(float)


def face_up_spread(df):
    # visible diameter per cube-root of weight: stones that look bigger for their carat sell higher
    d = (df["x"].astype(float) + df["y"].astype(float)) / 2
    return d.replace(0, np.nan) / np.cbrt(df["carat"].astype(float).clip(lower=0.01))


def quality_grade(df):
    # combined colour+clarity grade; price per carat multiplies with both
    return _ord(df, "color", COLOR) + _ord(df, "clarity", CLAR)


def carat_x_quality(df):
    # size and grade interact multiplicatively in diamond pricing (rapaport grid)
    q = quality_grade(df) + _ord(df, "cut", CUT) / 2
    return np.log(df["carat"].astype(float).clip(lower=0.01)) + 0.08 * q


def volume(df):
    # physical volume x*y*z (0 dimensions are missing measurements)
    v = df["x"].astype(float) * df["y"].astype(float) * df["z"].astype(float)
    return v.replace(0, np.nan)


def density(df):
    # carat per mm^3: flags bad dimension records and odd proportions
    return df["carat"].astype(float) / volume(df)


def carat_vs_grade_group(df):
    # carat relative to stones of the same colour/clarity: bigger than peers commands a premium
    key = df["color"].astype(str) + "|" + df["clarity"].astype(str)
    c = df["carat"].astype(float)
    return c / c.groupby(key).transform("mean")


def magic_size_gap(df):
    # prices jump at psychological weights (0.5, 0.7, 1.0, 1.5, 2.0 ct): distance above the last threshold
    c = df["carat"].astype(float)
    th = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0])
    last = np.array([th[th <= v].max() if (th <= v).any() else 0.0 for v in c])
    return pd.Series(c.to_numpy() - last, index=df.index, dtype=float)


def depth_table_deviation(df):
    # distance from ideal proportions (depth ~61.5, table ~56): poor light return lowers value
    return (df["depth"].astype(float) - 61.5).abs() + (df["table"].astype(float) - 56.0).abs()


PROPOSALS = [
    ("carat_x_quality", carat_x_quality),
    ("face_up_spread", face_up_spread),
    ("quality_grade", quality_grade),
    ("carat_vs_grade_group", carat_vs_grade_group),
    ("magic_size_gap", magic_size_gap),
    ("volume", volume),
    ("density", density),
    ("depth_table_deviation", depth_table_deviation),
]
