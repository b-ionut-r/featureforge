"""Domain features for breast_w, written from the schema and description only."""
import numpy as np
import pandas as pd

COLS = ["Clump_Thickness", "Cell_Size_Uniformity", "Cell_Shape_Uniformity", "Marginal_Adhesion",
        "Single_Epi_Cell_Size", "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses"]


def total_score(df):
    # cytologists sum abnormality grades: malignancy shows as uniformly elevated scores
    return df[COLS].astype(float).mean(axis=1) * len(COLS)


def n_abnormal(df):
    # number of features graded abnormal (>=4 on the 1-10 scale)
    return (df[COLS].astype(float) >= 4).sum(axis=1).astype(float)


def size_shape_nuclei(df):
    # the three strongest malignancy markers (non-uniform size, shape, bare nuclei) combined
    x = df[COLS].astype(float)
    return x["Cell_Size_Uniformity"] + x["Cell_Shape_Uniformity"] + x["Bare_Nuclei"].fillna(x["Cell_Size_Uniformity"])


def max_score(df):
    # worst single grade: any strongly abnormal feature raises suspicion
    return df[COLS].astype(float).max(axis=1)


PROPOSALS = [("total_score", total_score), ("size_shape_nuclei", size_shape_nuclei), ("n_abnormal", n_abnormal),
             ("max_score", max_score)]
