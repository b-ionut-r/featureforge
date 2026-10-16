"""Domain features for magic, written from the schema and description only."""
import numpy as np
import pandas as pd

# MAGIC gamma telescope Hillas parameters: FLength/FWidth (mm), FSize=log10(total charge),
# FConc/FConc1 (light share in brightest pixels), FAsym/FM3Long/FM3Trans (shower asymmetry),
# FAlpha (deg, angle of major axis to camera centre), FDist (mm, centre-of-gravity to camera centre).


def _div(a, b):
    return a / b.replace(0, np.nan)


def elongation(df):
    # gamma showers are narrow ellipses, hadron showers are wide and irregular
    return pd.DataFrame({"len_over_width": _div(df["FLength"], df["FWidth"]),
                         "log_area": np.log10(df["FLength"].clip(lower=1e-3) * df["FWidth"].clip(lower=1e-3))},
                        index=df.index)


def light_density(df):
    # photons per unit image area: hadron showers spread the same charge over a larger, patchier area
    area = df["FLength"].clip(lower=1e-3) * df["FWidth"].clip(lower=1e-3)
    return pd.DataFrame({"size_minus_logarea": df["FSize"] - np.log10(area),
                         "conc_x_size": df["FConc"] * 10 ** df["FSize"]}, index=df.index)


def miss_distance(df):
    # 'miss' = perpendicular distance of the shower axis from the source: small for gammas from the target
    a = np.radians(df["FAlpha"].abs())
    return pd.DataFrame({"miss": df["FDist"] * np.sin(a), "dist_along": df["FDist"] * np.cos(a)}, index=df.index)


def width_scaled_by_size(df):
    # gamma image width/length grow with energy (size); scaling removes this to expose hadron outliers
    return pd.DataFrame({"width_per_size": _div(df["FWidth"], df["FSize"]),
                         "length_per_size": _div(df["FLength"], df["FSize"]),
                         "length_per_dist": _div(df["FLength"], df["FDist"])}, index=df.index)


def asymmetry_normalised(df):
    # third moments/asymmetry relative to length: gammas point consistently head-tail toward the source
    return pd.DataFrame({"asym_over_len": _div(df["FAsym"], df["FLength"]),
                         "m3long_over_len": _div(df["FM3Long"], df["FLength"]),
                         "m3trans_over_width": _div(df["FM3Trans"], df["FWidth"]),
                         "abs_m3trans": df["FM3Trans"].abs()}, index=df.index)


def concentration_ratio(df):
    # share of light in the single brightest pixel vs top two: compact gamma cores
    return _div(df["FConc1"], df["FConc"])


PROPOSALS = [
    ("elongation", elongation),
    ("miss_distance", miss_distance),
    ("light_density", light_density),
    ("width_scaled_by_size", width_scaled_by_size),
    ("asymmetry_normalised", asymmetry_normalised),
    ("concentration_ratio", concentration_ratio),
]
