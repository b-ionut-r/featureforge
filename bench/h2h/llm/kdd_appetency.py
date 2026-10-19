"""Domain features for kdd_appetency, written from the schema and description only."""
import numpy as np
import pandas as pd

# Orange CRM data, anonymised: Var1-Var190 numeric, Var191-Var230 categorical.


def _num_cols(df):
    return [c for c in df.columns if c.startswith("Var") and c[3:].isdigit() and int(c[3:]) <= 190]


def _str(df, c):
    return df[c].astype(object).where(df[c].notna(), "__nan__").astype(str)


def count_encode_high_card(df):
    # frequency of high-cardinality codes (customer segment / offer / region ids): common vs rare profiles
    out = pd.DataFrame(index=df.index)
    for c in ["Var217", "Var202", "Var199", "Var192", "Var216", "Var198", "Var197", "Var204", "Var212"]:
        if c in df.columns:
            s = _str(df, c)
            out["cnt_" + c] = s.groupby(s).transform("size").astype(float)
    return out


def missing_profile(df):
    # how many of the numeric usage fields are filled: depth of the customer's product/usage history
    cols = _num_cols(df)
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"n_num_present": num.notna().sum(axis=1).astype(float),
                         "n_num_nonzero": (num.fillna(0) != 0).sum(axis=1).astype(float),
                         "n_cat_missing": sum(df[c].isna().astype(float) for c in df.columns
                                              if c not in cols)}, index=df.index)


def var126_vs_segment(df):
    # Var126 (a key score in KDD09) relative to the customer's Var206/Var218 segment
    v = pd.to_numeric(df["Var126"], errors="coerce")
    out = pd.DataFrame(index=df.index)
    for g in ["Var206", "Var218", "Var210"]:
        s = _str(df, g)
        out["v126_minus_" + g] = v - v.groupby(s).transform("mean")
    return out


def var113_vs_segment(df):
    # Var113 (heavy-tailed balance-like amount) relative to its segment mean and signed log
    v = pd.to_numeric(df["Var113"], errors="coerce")
    s = _str(df, "Var218")
    return pd.DataFrame({"v113_slog": np.sign(v) * np.log1p(v.abs()),
                         "v113_minus_seg": v - v.groupby(s).transform("mean")}, index=df.index)


def usage_ratios(df):
    # ratios of the dense, correlated amount fields (e.g. Var133/Var153, Var76/Var38): usage mix
    n = lambda c: pd.to_numeric(df[c], errors="coerce")
    d = lambda a, b: a / b.replace(0, np.nan)
    return pd.DataFrame({"r133_153": d(n("Var133"), n("Var153")),
                         "r81_113": d(n("Var81"), n("Var113").abs()), "r76_38": d(n("Var76"), n("Var38")),
                         "r6_13": d(n("Var6"), n("Var13"))}, index=df.index)


PROPOSALS = [
    ("count_encode_high_card", count_encode_high_card),
    ("missing_profile", missing_profile),
    ("var126_vs_segment", var126_vs_segment),
    ("usage_ratios", usage_ratios),
    ("var113_vs_segment", var113_vs_segment),
]
