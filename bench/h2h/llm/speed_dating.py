"""Domain features for speed_dating, written from the schema and description only."""
import numpy as np
import pandas as pd

ATTR_O = ["attractive_o", "sinsere_o", "intelligence_o", "funny_o", "ambitous_o", "shared_interests_o"]
ATTR_P = ["attractive_partner", "sincere_partner", "intelligence_partner", "funny_partner", "ambition_partner",
          "shared_interests_partner"]
IMP_SELF = ["attractive_important", "sincere_important", "intellicence_important", "funny_important",
            "ambtition_important", "shared_interests_important"]
IMP_O = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious",
         "pref_o_shared_interests"]
SELF = ["attractive", "sincere", "intelligence", "funny", "ambition"]


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def _m(df, cols):
    return pd.DataFrame({c: _f(df, c) for c in cols}, index=df.index)


def _norm(m):
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0)


def mutual_attraction(df):
    # a match needs both sides: my rating of partner and partner's rating of me, combined
    return pd.DataFrame({"attr_min": np.fmin(_f(df, "attractive_o"), _f(df, "attractive_partner")),
                         "attr_prod": _f(df, "attractive_o") * _f(df, "attractive_partner"),
                         "shared_prod": _f(df, "shared_interests_o") * _f(df, "shared_interests_partner")})


def weighted_score_by_partner(df):
    # how partner rated me, weighted by what the partner says they value (their utility for me)
    w = _norm(_m(df, IMP_O)).to_numpy()
    r = _m(df, ATTR_O).to_numpy()
    return pd.Series(np.nansum(w * r, axis=1) / np.where(np.isnan(r), 0, w).sum(1).clip(1e-9), index=df.index)


def weighted_score_by_self(df):
    # how I rated partner, weighted by my own stated preferences (my utility for them)
    w = _norm(_m(df, IMP_SELF)).to_numpy()
    r = _m(df, ATTR_P).to_numpy()
    return pd.Series(np.nansum(w * r, axis=1) / np.where(np.isnan(r), 0, w).sum(1).clip(1e-9), index=df.index)


def rating_means(df):
    # overall warmth on each side: mean rating given and received
    return pd.DataFrame({"mean_received": _m(df, ATTR_O).mean(axis=1), "mean_given": _m(df, ATTR_P).mean(axis=1)})


def like_and_guess(df):
    # my overall like and my guess they liked me: both high means I would say yes and expect reciprocity
    lk, g = _f(df, "like"), _f(df, "guess_prob_liked")
    return pd.DataFrame({"like_x_guess": lk * g, "like_min_guess": np.fmin(lk, g), "like_minus_mean_given":
                         lk - _m(df, ATTR_P).mean(axis=1)})


def pref_alignment(df):
    # similarity of the two people's preference profiles (L1 distance of normalised importance weights)
    a = _norm(_m(df, IMP_SELF)).to_numpy()
    b = _norm(_m(df, IMP_O)).to_numpy()
    return pd.Series(np.abs(a - b).sum(1), index=df.index)


def self_vs_perceived(df):
    # self-rating vs how the partner saw me (overconfidence) on attractiveness and on average
    s = _m(df, SELF).to_numpy()
    o = _m(df, ATTR_O[:5]).to_numpy()
    return pd.DataFrame({"attr_gap": _f(df, "attractive") - _f(df, "attractive_o"),
                         "self_minus_received": np.nanmean(s, 1) - np.nanmean(o, 1)}, index=df.index)


def similarity(df):
    # homophily cues: same race weighted by importance, interest correlation, age gap
    same = pd.to_numeric(df["samerace"].astype(object), errors="coerce").astype(float)
    return pd.DataFrame({"same_race_x_imp": same * _f(df, "importance_same_race"),
                         "interests_corr": _f(df, "interests_correlate"),
                         "age_gap_signed": _f(df, "age") - _f(df, "age_o"),
                         "met_before": (_f(df, "met").fillna(0) > 0).astype(float)})


def selectivity(df):
    # expected matches: low expectation = picky or pessimistic; gender mapped to number
    g = df["gender"].astype(object).astype(str).str.lower()
    return pd.DataFrame({"exp_matches": _f(df, "expected_num_matches"),
                         "exp_happy": _f(df, "expected_happy_with_sd_people"),
                         "male": np.where(g == "male", 1.0, np.where(g == "female", 0.0, np.nan))}, index=df.index)


PROPOSALS = [("mutual_attraction", mutual_attraction), ("weighted_score_by_partner", weighted_score_by_partner),
             ("weighted_score_by_self", weighted_score_by_self), ("rating_means", rating_means),
             ("like_and_guess", like_and_guess), ("pref_alignment", pref_alignment),
             ("self_vs_perceived", self_vs_perceived), ("similarity", similarity), ("selectivity", selectivity)]
