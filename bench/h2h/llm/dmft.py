"""Domain features for dmft, written from the schema and description only."""
import numpy as np
import pandas as pd


def _bd(df):
    return (pd.to_numeric(df["DMFT.Begin"], errors="coerce").astype(float),
            pd.to_numeric(df["DMFT.End"], errors="coerce").astype(float))


def dmft_change(df):
    # absolute reduction in decayed/missing/filled teeth: how effective the assigned prevention was
    b, e = _bd(df)
    return pd.DataFrame({"dmft_drop": b - e})


def dmft_relative_change(df):
    # relative reduction: a drop of 2 from 3 is stronger evidence than 2 from 8
    b, e = _bd(df)
    return pd.DataFrame({"dmft_rel_drop": (b - e) / b.replace(0, np.nan), "dmft_end_over_begin": e / b.replace(0, np.nan)})


def dmft_flags(df):
    # caries-free at start/end, worsened, unchanged: discrete outcome states
    b, e = _bd(df)
    return pd.DataFrame({"begin_zero": (b == 0).astype(float), "end_zero": (e == 0).astype(float),
                         "worsened": (e > b).astype(float), "unchanged": (e == b).astype(float)})


def demographics(df):
    # numeric codes for gender and ethnicity (randomisation strata may differ across strategies)
    g = df["Gender"].astype(str)
    eth = df["Ethnic"].astype(str)
    return pd.DataFrame({"is_male": (g == "Male").astype(float), "eth_white": (eth == "White").astype(float),
                         "eth_black": (eth == "Black").astype(float), "eth_dark": (eth == "Dark").astype(float)})


def change_by_gender(df):
    # reduction split by gender to let a linear model see the interaction
    b, e = _bd(df)
    m = (df["Gender"].astype(str) == "Male").astype(float)
    return pd.DataFrame({"drop_male": (b - e) * m, "drop_female": (b - e) * (1 - m)})


def log_ratio(df):
    # log ratio with +1 smoothing: symmetric, defined at zero counts
    b, e = _bd(df)
    return np.log1p(e) - np.log1p(b)


PROPOSALS = [("dmft_change", dmft_change), ("dmft_relative_change", dmft_relative_change), ("dmft_flags", dmft_flags),
             ("demographics", demographics), ("change_by_gender", change_by_gender), ("log_ratio", log_ratio)]
