"""Domain features for cmc, written from the schema and description only."""
import numpy as np
import pandas as pd


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def children_per_fertile_year(df):
    # fertility rate since ~15: women who already reached desired family size switch to (long-term) contraception
    return _n(df, "Number_of_children_ever_born") / (_n(df, "Wifes_age") - 14).clip(lower=1)


def children_vs_age_peers(df):
    # more children than women of the same age: completed family -> long-term method
    k = _n(df, "Number_of_children_ever_born")
    ab = _n(df, "Wifes_age") // 5
    return k - k.groupby(ab).transform("mean")


def no_children(df):
    # childless women rarely use contraception (trying to conceive)
    return (_n(df, "Number_of_children_ever_born") == 0).astype(float)


def edu_sum_and_gap(df):
    # couple education level drives method knowledge; wife's relative education reflects her autonomy
    w, h = _n(df, "Wifes_education"), _n(df, "Husbands_education")
    return pd.DataFrame({"edu_sum": w + h, "edu_gap": w - h})


def ses_index(df):
    # socio-economic status: wife education + living standard + good media exposure (0 = good in the UCI coding)
    return _n(df, "Wifes_education") + _n(df, "Standard-of-living_index") + 2 * (1 - _n(df, "Media_exposure"))


def age_x_education(df):
    # young educated women favour short-term methods, older ones long-term
    return _n(df, "Wifes_age") * _n(df, "Wifes_education")


PROPOSALS = [("children_per_fertile_year", children_per_fertile_year), ("children_vs_age_peers", children_vs_age_peers),
             ("edu_sum_and_gap", edu_sum_and_gap), ("no_children", no_children), ("ses_index", ses_index),
             ("age_x_education", age_x_education)]
