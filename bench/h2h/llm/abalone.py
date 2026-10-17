"""Domain features for abalone, written from the schema and description only."""
import numpy as np
import pandas as pd


def shell_frac(df):
    # shell keeps growing (calcifying) with age while meat does not, so shell share of total weight rises with age
    return df["Shell_weight"] / df["Whole_weight"].replace(0, np.nan)


def shucked_to_shell(df):
    # meat-to-shell ratio falls as abalone age; old animals are mostly shell
    return df["Shucked_weight"] / df["Shell_weight"].replace(0, np.nan)


def density(df):
    # weight per bounding volume: older shells are thicker/denser for the same dimensions
    vol = df["Length"] * df["Diameter"] * df["Height"]
    return df["Whole_weight"] / vol.replace(0, np.nan)


def lost_weight(df):
    # whole minus parts = water/blood lost at shucking; relative to whole it varies with age/condition
    parts = df["Shucked_weight"] + df["Viscera_weight"] + df["Shell_weight"]
    return (df["Whole_weight"] - parts) / df["Whole_weight"].replace(0, np.nan)


def shell_vs_size_group(df):
    # shell weight relative to animals of similar length: heavier shell than peers of same size implies older
    lb = (df["Length"] * 40).round()
    return df["Shell_weight"] / df.groupby(lb)["Shell_weight"].transform("median").replace(0, np.nan)


def height_to_length(df):
    # growth in length slows with age while shell keeps thickening (height), so the shape changes
    return df["Height"] / df["Length"].replace(0, np.nan)


def log_shell_weight(df):
    # age grows roughly logarithmically with weight (von Bertalanffy growth), log linearises it
    return np.log1p(df["Shell_weight"] * 100)


PROPOSALS = [("shell_frac", shell_frac), ("shell_vs_size_group", shell_vs_size_group),
             ("shucked_to_shell", shucked_to_shell), ("density", density), ("lost_weight", lost_weight),
             ("height_to_length", height_to_length), ("log_shell_weight", log_shell_weight)]
