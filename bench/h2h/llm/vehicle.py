"""Domain features for vehicle, written from the schema and description only."""
import numpy as np
import pandas as pd


def variance_ratio(df):
    # minor/major scaled variance: how elongated the silhouette's mass is (bus long and boxy, cars squatter)
    return df["SCALED_VARIANCE_MINOR"] / df["SCALED_VARIANCE_MAJOR"].replace(0, np.nan)


def scatter_x_elong(df):
    # scatter ratio and elongatedness are near-inverse views of the same moment; their product/residual isolates shape detail
    return pd.DataFrame({"scatter_x_elong": df["SCATTER_RATIO"] * df["ELONGATEDNESS"],
                         "scatter_per_rect": df["SCATTER_RATIO"] / df["PR.AXIS_RECTANGULARITY"].replace(0, np.nan)})


def circ_ratios(df):
    # circularity vs compactness and distance circularity: van (boxy) vs saloon (curved) bodies
    return pd.DataFrame({"circ_over_comp": df["CIRCULARITY"] / df["COMPACTNESS"].replace(0, np.nan),
                         "dcirc_over_circ": df["DISTANCE_CIRCULARITY"] / df["CIRCULARITY"].replace(0, np.nan)})


def gyration_rel(df):
    # radius of gyration relative to major-axis variance: mass distribution along the body
    return df["SCALED_RADIUS_OF_GYRATION"] / np.sqrt(df["SCALED_VARIANCE_MAJOR"].clip(lower=1))


def hollows_minus_kurt(df):
    # hollows ratio vs minor-axis kurtosis: concavities (windows/wheel arches) separate Saab from Opel
    return df["HOLLOWS_RATIO"].astype(float) - df["KURTOSIS_ABOUT_MINOR"].astype(float)


def rect_ratio(df):
    # max-length rectangularity vs principal-axis rectangularity: view-angle-normalised boxiness
    return df["MAX.LENGTH_RECTANGULARITY"] / df["PR.AXIS_RECTANGULARITY"].replace(0, np.nan)


def aspect_ratio_ratio(df):
    # principal-axis vs max-length aspect ratio: separates side views from end views
    return df["PR.AXIS_ASPECT_RATIO"] / df["MAX.LENGTH_ASPECT_RATIO"].replace(0, np.nan)


def radius_over_scatter(df):
    # radius ratio relative to scatter ratio
    return df["RADIUS_RATIO"] / df["SCATTER_RATIO"].replace(0, np.nan)


PROPOSALS = [("variance_ratio", variance_ratio), ("circ_ratios", circ_ratios), ("hollows_minus_kurt", hollows_minus_kurt),
             ("rect_ratio", rect_ratio), ("scatter_x_elong", scatter_x_elong), ("gyration_rel", gyration_rel),
             ("aspect_ratio_ratio", aspect_ratio_ratio), ("radius_over_scatter", radius_over_scatter)]
