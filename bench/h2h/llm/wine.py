"""Domain features for wine, written from the schema and description only."""
import numpy as np
import pandas as pd

OD = "OD280%2FOD315_of_diluted_wines"


def phenol_fractions(df):
    # share of phenols that are flavanoids vs non-flavanoids: cultivar-specific phenolic profile
    tp = df["Total_phenols"].replace(0, np.nan)
    return pd.DataFrame({"flav_frac": df["Flavanoids"] / tp, "nonflav_frac": df["Nonflavanoid_phenols"] / tp,
                         "flav_over_nonflav": df["Flavanoids"] / df["Nonflavanoid_phenols"].replace(0, np.nan)})


def color_hue(df):
    # colour intensity relative to hue: deep, purple-tinted (low hue) wines vs lighter, orange-tinted ones
    return pd.DataFrame({"color_over_hue": df["Color_intensity"] / df["Hue"].replace(0, np.nan),
                         "color_x_hue": df["Color_intensity"] * df["Hue"]})


def proline_alcohol(df):
    # proline (amino acid from ripe grapes) scales with alcohol/ripeness; log scale tames its range
    return pd.DataFrame({"log_proline": np.log(df["Proline"].astype(float).clip(lower=1)),
                         "proline_per_alcohol": df["Proline"] / df["Alcohol"].replace(0, np.nan)})


def protein_phenol(df):
    # OD280/OD315 measures protein/phenolic content; product with flavanoids gives total phenolic richness
    return pd.DataFrame({"od_x_flav": df[OD] * df["Flavanoids"],
                         "od_over_color": df[OD] / df["Color_intensity"].replace(0, np.nan)})


def ash_alkalinity(df):
    # alkalinity of ash relative to ash content: mineral buffering (soil/terroir)
    return df["Alcalinity_of_ash"] / df["Ash"].replace(0, np.nan)


def malic_ratio(df):
    # malic acid (sourness, cooler climate/less ripe) relative to alcohol (ripeness)
    return df["Malic_acid"] / df["Alcohol"].replace(0, np.nan)


def condensed_tannin(df):
    # proanthocyanins (tannins) relative to flavanoids and to colour: tannin structure per cultivar
    return pd.DataFrame({"proanth_over_flav": df["Proanthocyanins"] / df["Flavanoids"].replace(0, np.nan),
                         "proanth_x_color": df["Proanthocyanins"] * df["Color_intensity"]})


PROPOSALS = [("phenol_fractions", phenol_fractions), ("color_hue", color_hue), ("proline_alcohol", proline_alcohol),
             ("protein_phenol", protein_phenol), ("ash_alkalinity", ash_alkalinity), ("malic_ratio", malic_ratio),
             ("condensed_tannin", condensed_tannin)]
