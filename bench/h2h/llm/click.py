"""Domain features for click, written from the schema and description only (KDD Cup 2012 track 2 search ads)."""
import numpy as np
import pandas as pd


def _size(s):
    s = s.astype(str)
    return s.groupby(s).transform("size").astype(float)


def id_counts(df):
    # frequency of each id: popular ads/advertisers/queries behave differently, rare ids act as a cold-start flag
    return pd.DataFrame({c: _size(df[c]) for c in ["ad_id", "advertiser_id", "keyword_id", "title_id",
                                                     "description_id", "user_id"]})


def relative_position(df):
    # (depth - position)/depth: being the top ad on a page with many ads gets most clicks
    d = pd.to_numeric(df["depth"], errors="coerce").astype(float)
    p = pd.to_numeric(df["position"], errors="coerce").astype(float)
    return (d - p) / d.replace(0, np.nan)


def anonymous_user(df):
    # user_id 0 = not logged in / unknown user
    return (df["user_id"].astype("string").str.strip() == "0").astype(float)


def ad_keyword_count(df):
    # ad shown for the same query many times = well-matched ad-query pair
    return _size(df["ad_id"].astype(str) + "_" + df["keyword_id"].astype(str))


def advertiser_breadth(df):
    # advertisers with many distinct ads/keywords are big professional advertisers
    a = df["advertiser_id"].astype(str)
    return pd.DataFrame({"n_ads": df.groupby(a)["ad_id"].transform("nunique").astype(float),
                         "n_kw": df.groupby(a)["keyword_id"].transform("nunique").astype(float)})


def id_magnitudes(df):
    # numeric id size: ids are assigned sequentially, so small keyword/title ids are common/old tokens
    return pd.DataFrame({c: np.log1p(pd.to_numeric(df[c], errors="coerce").astype(float))
                         for c in ["keyword_id", "title_id", "description_id", "ad_id"]})


def impression_vs_ad(df):
    # impressions aggregated in this row vs the ad's average
    i = pd.to_numeric(df["impression"], errors="coerce").astype(float)
    return i / i.groupby(df["ad_id"].astype(str)).transform("mean")


PROPOSALS = [("id_counts", id_counts), ("relative_position", relative_position), ("id_magnitudes", id_magnitudes),
             ("ad_keyword_count", ad_keyword_count), ("anonymous_user", anonymous_user),
             ("advertiser_breadth", advertiser_breadth), ("impression_vs_ad", impression_vs_ad)]
