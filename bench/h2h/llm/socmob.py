"""Domain features for socmob, written from the schema and description only."""
import numpy as np
import pandas as pd

# occupation codes 1..17 as listed in the description (17 = Professional self-employed ... 1 = Farm laborer)
CODES = {"professional-self-employed": 17, "professional-salaried": 16, "manager": 15, "salesman-nonretail": 14,
         "proprietor": 13, "clerk": 12, "salesman-retail": 11, "craftsman-manufacturing": 10, "craftsmen-other": 9,
         "craftsman-construction": 8, "service-worker": 7, "operative-nonmanufacturing": 6,
         "operative-manufacturing": 5, "laborer-manufacturing": 4, "laborer-nonmanufacturing": 3,
         "farmer/farm-manager": 2, "farm-laborer": 1}


def _code(s):
    return s.astype(str).str.lower().str.replace("_", "-").str.replace(" ", "-").map(CODES).astype(float)


def _cls(code):
    # 3 = white collar (11-17), 2 = blue collar/service (3-10), 1 = farm (1-2)
    return pd.Series(np.select([code >= 11, code >= 3, code >= 1], [3, 2, 1], np.nan), index=code.index)


def occ_codes(df):
    # description's occupational ordering roughly tracks socioeconomic status
    return pd.DataFrame({"father_code": _code(df["fathers_occupation"]), "son_code": _code(df["sons_occupation"])})


def mobility(df):
    # intergenerational mobility: son's status minus father's, and its magnitude
    d = _code(df["sons_occupation"]) - _code(df["fathers_occupation"])
    return pd.DataFrame({"mobility": d, "abs_mobility": d.abs()})


def same_occupation(df):
    # occupational inheritance: mobility tables have heavy diagonals
    return (_code(df["sons_occupation"]) == _code(df["fathers_occupation"])).astype(float)


def class_mobility(df):
    # coarse class (farm / manual / white collar) of father and son and same-class flag
    fc, sc = _cls(_code(df["fathers_occupation"])), _cls(_code(df["sons_occupation"]))
    return pd.DataFrame({"father_class": fc, "son_class": sc, "same_class": (fc == sc).astype(float), "class_move": sc - fc})


def log_first_count(df):
    # counts are skewed; log of first-occupation count (current-occupation cell size tracks it)
    return np.log1p(df["counts_for_sons_first_occupation"].clip(lower=0))


def demographics(df):
    # family disruption and race indicators (the paper's focus) and their interaction
    non = (df["family_structure"].astype(str) == "nonintact").astype(float)
    blk = (df["race"].astype(str) == "black").astype(float)
    return pd.DataFrame({"nonintact": non, "black": blk, "nonintact_x_black": non * blk})


def count_x_group(df):
    # first-occupation count scaled by group (smaller groups, e.g. black nonintact, have small cells)
    lc = log_first_count(df)
    d = demographics(df)
    return pd.DataFrame({"logc_x_black": lc * d["black"], "logc_x_nonintact": lc * d["nonintact"],
                         "logc_x_diag": lc * same_occupation(df)})


PROPOSALS = [("occ_codes", occ_codes), ("mobility", mobility), ("same_occupation", same_occupation),
             ("class_mobility", class_mobility), ("log_first_count", log_first_count),
             ("demographics", demographics), ("count_x_group", count_x_group)]
