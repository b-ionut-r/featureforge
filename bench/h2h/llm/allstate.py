"""Domain features for allstate, written from the schema and description only."""
import numpy as np
import pandas as pd

BIN = [f"cat{i}" for i in range(1, 73)]


def _lex(s):
    # lexical encoding of letter codes (A..Z, AA..ZZ) as base-26 numbers: the codes look ordered (alphabetic bins)
    s = s.astype("string").fillna("")
    def enc(v):
        r = 0
        for ch in str(v):
            r = r * 26 + (ord(ch) - ord("A") + 1)
        return float(r) if v else np.nan
    m = {v: enc(v) for v in s.unique()}
    return s.map(m).astype(float)


def lex_high_card(df):
    # the many-level cats (cat116, cat110, cat109, cat113, cat112) look like ordered binned codes; base-26 order lets trees split ranges
    return pd.DataFrame({c: _lex(df[c]) for c in ["cat116", "cat110", "cat109", "cat113", "cat112", "cat101", "cat100"]})


def combo_lex(df):
    # pairwise combinations of the most important cats (known strong interactions in claims severity) as ordered codes
    pairs = [("cat80", "cat79"), ("cat80", "cat12"), ("cat79", "cat57"), ("cat80", "cat87"), ("cat101", "cat79"),
             ("cat100", "cat80"), ("cat12", "cat81")]
    out = {}
    for a, b in pairs:
        out[f"{a}_{b}"] = _lex(df[a].astype(str) + df[b].astype(str))
    return pd.DataFrame(out)


def n_binary_B(df):
    # number of binary flags set to the rarer level: a crude "complexity of the claim" count
    cols = [c for c in BIN if c in df]
    return (df[cols].astype(str) == "B").sum(axis=1).astype(float)


def freq_high_card(df):
    # how common a claim's code is: rare codes (unusual injury/vehicle types) tend to be costlier
    out = {}
    for c in ["cat116", "cat110", "cat109", "cat113", "cat112"]:
        s = df[c].astype(str)
        out[c] = s.groupby(s).transform("size").astype(float)
    return pd.DataFrame(out)


def cont_diffs(df):
    # strongly correlated continuous pairs: their residual difference carries the independent signal
    return pd.DataFrame({"c12_m_c11": df["cont12"] - df["cont11"], "c1_m_c9": df["cont1"] - df["cont9"],
                         "c6_m_c10": df["cont6"] - df["cont10"], "c13_m_c6": df["cont13"] - df["cont6"]})


def cont_stats(df):
    # aggregate level of the anonymised continuous risk scores
    cc = df[[f"cont{i}" for i in range(1, 15)]].astype(float)
    return pd.DataFrame({"mean": cc.mean(axis=1), "std": cc.std(axis=1)})


def cont2_x_cont14(df):
    # cont2 is near-ordinal (29 levels) and cont14 the finest-grained score: their product mixes severity tiers
    return df["cont2"].astype(float) * df["cont14"].astype(float)


PROPOSALS = [("lex_high_card", lex_high_card), ("combo_lex", combo_lex), ("freq_high_card", freq_high_card),
             ("n_binary_B", n_binary_B), ("cont_diffs", cont_diffs), ("cont_stats", cont_stats),
             ("cont2_x_cont14", cont2_x_cont14)]
