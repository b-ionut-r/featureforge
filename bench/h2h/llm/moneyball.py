"""Domain features for moneyball, written from the schema and description only."""
import numpy as np
import pandas as pd


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def ops(df):
    # on-base plus slugging: the description singles out OBP and SLG as what drives run scoring
    return _num(df, "OBP") + _num(df, "SLG")


def isolated_power(df):
    # SLG - BA: extra-base power; OBP - BA: walk/HBP contribution
    return pd.DataFrame({"iso": _num(df, "SLG") - _num(df, "BA"), "obp_minus_ba": _num(df, "OBP") - _num(df, "BA")})


def pythag_runs(df):
    # OUTSIDE KNOWLEDGE: Bill James Pythagorean expectation W/G = RS^2/(RS^2+RA^2) inverted to RS = RA*sqrt(W/(G-W))
    w, g, ra = _num(df, "W"), _num(df, "G"), _num(df, "RA")
    return ra * np.sqrt(w / (g - w).replace(0, np.nan))


def win_pct(df):
    # winning percentage and run-prevention rate per game
    g = _num(df, "G").replace(0, np.nan)
    return pd.DataFrame({"win_pct": _num(df, "W") / g, "ra_per_g": _num(df, "RA") / g})


def obp_slg_product(df):
    # runs ~ getting on base x advancing runners (Runs Created style OBP*SLG)
    return _num(df, "OBP") * _num(df, "SLG")


def era_flags(df):
    # league/era context: designated hitter in AL from 1973, offense levels shift by year
    # OUTSIDE KNOWLEDGE: AL adopted the DH in 1973
    y = _num(df, "Year")
    al = (df["League"].astype(str) == "AL").astype(float)
    return pd.DataFrame({"is_al": al, "dh": al * (y >= 1973), "year_c": y - 1990})


def opp_ops(df):
    # opponent OPS (pitching/defense quality), available in later seasons only
    return _num(df, "OOBP") + _num(df, "OSLG")


def playoffs_num(df):
    # playoff qualification as numeric
    return _num(df, "Playoffs")


PROPOSALS = [("ops", ops), ("isolated_power", isolated_power), ("pythag_runs", pythag_runs), ("win_pct", win_pct),
             ("obp_slg_product", obp_slg_product), ("era_flags", era_flags), ("opp_ops", opp_ops),
             ("playoffs_num", playoffs_num)]
