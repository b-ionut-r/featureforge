"""Domain features for cloud, written from the schema and description only."""
import numpy as np
import pandas as pd


def _seeded(df):
    return (df["seeded"].astype(object) == "S").astype(float)


def control_mean(df):
    # target-area rain tracks the regional weather seen in nearby control areas: mean and log-mean of controls
    m = (df["NC"] + df["SC"] + df["NWC"]) / 3.0
    return pd.DataFrame({"ctrl_mean": m, "ctrl_logmean": np.log1p(df[["NC", "SC", "NWC"]]).mean(axis=1),
                         "ctrl_geo": np.exp(np.log(df[["NC", "SC", "NWC"]].clip(lower=0.01)).mean(axis=1))})


def seeding_effect(df):
    # seeding effect is multiplicative on the natural rainfall: seeded x control rainfall
    m = (df["NC"] + df["SC"] + df["NWC"]) / 3.0
    return pd.DataFrame({"seeded": _seeded(df), "seeded_x_ctrl": _seeded(df) * m})


def wind_direction(df):
    # NW control is upwind of the targets under westerly flow; its share signals storm track orientation
    # OUTSIDE KNOWLEDGE: Tasmania lies in the prevailing westerlies
    tot = (df["NC"] + df["SC"] + df["NWC"]).replace(0, np.nan)
    return pd.DataFrame({"nwc_share": df["NWC"] / tot, "nc_minus_sc": df["NC"] - df["SC"],
                         "nc_share": df["NC"] / tot})


def season_code(df):
    # season: rainfall regime differs (winter frontal vs summer convective); cyclic encoding
    idx = df["season"].astype(object).map({"SUMMER": 0, "AUTUMN": 1, "WINTER": 2, "SPRING": 3}).astype(float)
    ang = 2 * np.pi * idx / 4.0
    return pd.DataFrame({"season_sin": np.sin(ang), "season_cos": np.cos(ang)})


def seeded_by_season(df):
    # seeding works only with suitable cloud types; interact seeding with season
    s = _seeded(df)
    se = df["season"].astype(object)
    return pd.DataFrame({f"seed_{k.lower()}": s * (se == k).astype(float) for k in ["SUMMER", "AUTUMN", "WINTER", "SPRING"]})


def control_spread(df):
    # spatial variability of controls: uniform widespread rain vs localised showers
    c = df[["NC", "SC", "NWC"]]
    return c.std(axis=1) / c.mean(axis=1).replace(0, np.nan)


PROPOSALS = [("control_mean", control_mean), ("seeding_effect", seeding_effect), ("wind_direction", wind_direction),
             ("season_code", season_code), ("seeded_by_season", seeded_by_season), ("control_spread", control_spread)]
