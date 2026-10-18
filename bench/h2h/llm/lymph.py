"""Domain features for lymph, written from the schema and description only."""
import numpy as np
import pandas as pd

_YN = ["block_of_affere", "bl_of_lymph_c", "bl_of_lymph_s", "by_pass", "extravasates",
       "regeneration_of", "early_uptake_in", "dislocation_of", "exclusion_of_no"]


def _yes(s):
    return s.astype(object).map({"yes": 1.0, "no": 0.0}).astype(float)


def blockage_count(df):
    # lymphographic signs of obstruction (blocks, by-pass, extravasation): count of flow abnormalities
    cols = ["block_of_affere", "bl_of_lymph_c", "bl_of_lymph_s", "by_pass", "extravasates"]
    return sum(_yes(df[c]).fillna(0) for c in cols)


def node_size_balance(df):
    # enlargement vs diminution of nodes: enlarged nodes suggest malignant/metastatic infiltration
    e = pd.to_numeric(df["lym_nodes_enlar"], errors="coerce").astype(float)
    d = pd.to_numeric(df["lym_nodes_dimin"], errors="coerce").astype(float)
    return pd.DataFrame({"enlar_minus_dimin": e - d, "enlar_x_nodes": e * pd.to_numeric(df["no_of_nodes_in"], errors="coerce")})


def lymphatics_ord(df):
    # severity ordering of lymphatic vessel appearance: normal < arched < deformed < displaced
    return df["lymphatics"].astype(object).map({"normal": 0.0, "arched": 1.0, "deformed": 2.0, "displaced": 3.0}).astype(float)


def defect_severity(df):
    # nodal filling defects / changes: none < lacunar < marginal lacunar < central lacunar (central = replaced node)
    m = {"no": 0.0, "lacunar": 1.0, "lac_margin": 2.0, "lac_central": 3.0}
    d = df["defect_in_node"].astype(object).map(m).astype(float)
    c = df["changes_in_node"].astype(object).map(m).astype(float)
    return pd.DataFrame({"defect_ord": d, "changes_ord": c, "defect_plus_changes": d + c})


def structure_flags(df):
    # node shape and internal structure: round nodes and vesicles/chalices are classic malignant patterns
    shape = df["changes_in_lym"].astype(object).map({"bean": 0.0, "oval": 1.0, "round": 2.0}).astype(float)
    sf = df["special_forms"].astype(object).map({"no": 0.0, "chalices": 1.0, "vesicles": 2.0}).astype(float)
    st = df["changes_in_stru"].astype(object).astype(str)
    return pd.DataFrame({"shape_ord": shape, "special_ord": sf,
                         "stru_no": (st == "no").astype(float)})


def abnormal_total(df):
    # global abnormality score: all yes-flags plus severity of defects and node count
    s = sum(_yes(df[c]).fillna(0) for c in _YN if c not in ("early_uptake_in",))
    return s + pd.to_numeric(df["no_of_nodes_in"], errors="coerce").astype(float) / 2.0


PROPOSALS = [("blockage_count", blockage_count), ("node_size_balance", node_size_balance),
             ("lymphatics_ord", lymphatics_ord), ("defect_severity", defect_severity),
             ("structure_flags", structure_flags), ("abnormal_total", abnormal_total)]
