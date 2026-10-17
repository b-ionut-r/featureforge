"""Domain features for eucalyptus, written from the schema and description only."""
import numpy as np
import pandas as pd

SCORES = ["Vig", "Ins_res", "Stem_Fm", "Crown_Fm", "Brnch_Fm"]


def form_score_mean(df):
    # utility rating is a judgement over vigour, insect resistance and stem/crown/branch form
    return df[SCORES].astype(float).mean(axis=1)


def ht_vs_site(df):
    # height relative to other seedlots at the same trial site: removes site/age effects
    h = df["Ht"].astype(float)
    return h - h.groupby(df["Abbrev"].astype(str)).transform("mean")


def dbh_vs_site(df):
    # diameter relative to the site mean
    d = df["DBH"].astype(float)
    return d - d.groupby(df["Abbrev"].astype(str)).transform("mean")


def surv_vs_site(df):
    # survival relative to the site mean: species that survive where others die are good conservers
    s = df["Surv"].astype(float)
    return s - s.groupby(df["Abbrev"].astype(str)).transform("mean")


def stem_volume_index(df):
    # DBH^2 * height, a standard tree volume (biomass) index
    return df["DBH"].astype(float) ** 2 * df["Ht"].astype(float)


def slenderness(df):
    # height over diameter: wind-firmness and stem form
    return df["Ht"].astype(float) / df["DBH"].astype(float).replace(0, np.nan)


def form_vs_site(df):
    # mean form score relative to the site mean (scorers differ per site)
    f = form_score_mean(df)
    return f - f.groupby(df["Abbrev"].astype(str)).transform("mean")


def growth_per_year(df):
    # height per year since planting; trials were assessed at roughly the same calendar time
    yrs = (df["Year"].astype(float).max() + 1 - df["Year"].astype(float)).clip(lower=1)
    return df["Ht"].astype(float) / yrs


def species_ht_rank_at_site(df):
    # percentile rank of height within the site
    return df.groupby(df["Abbrev"].astype(str))["Ht"].rank(pct=True).astype(float)


def n_missing_scores(df):
    # missing assessments indicate dead trees or unrecorded plots (Morea station)
    return df[SCORES + ["Surv"]].isna().sum(axis=1).astype(float)


PROPOSALS = [
    ("form_score_mean", form_score_mean),
    ("ht_vs_site", ht_vs_site),
    ("species_ht_rank_at_site", species_ht_rank_at_site),
    ("surv_vs_site", surv_vs_site),
    ("dbh_vs_site", dbh_vs_site),
    ("form_vs_site", form_vs_site),
    ("stem_volume_index", stem_volume_index),
    ("slenderness", slenderness),
    ("growth_per_year", growth_per_year),
    ("n_missing_scores", n_missing_scores),
]
