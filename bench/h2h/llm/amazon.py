"""Domain features for amazon, written from the schema and description only."""
import numpy as np
import pandas as pd


def _k(df, cols):
    return df[list(cols)].astype(str).agg("_".join, axis=1)


def _size(key):
    return key.groupby(key).transform("size").astype(float)


def counts_single(df):
    # how often a resource / manager / role appears: popular resources are routinely granted
    return pd.DataFrame({c: _size(df[c].astype(str)) for c in
                         ["RESOURCE", "MGR_ID", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC"]})


def resource_share_in_dept(df):
    # share of a department's requests that are for this resource: a resource commonly used by your dept is approved
    pair = _size(_k(df, ["RESOURCE", "ROLE_DEPTNAME"]))
    return pair / _size(df["ROLE_DEPTNAME"].astype(str))


def resource_share_in_role(df):
    # same idea at the job-title level: peers with the same title already use this resource
    pair = _size(_k(df, ["RESOURCE", "ROLE_TITLE"]))
    return pair / _size(df["ROLE_TITLE"].astype(str))


def resource_share_under_mgr(df):
    # a manager whose team repeatedly requests the resource has likely approved it before
    pair = _size(_k(df, ["RESOURCE", "MGR_ID"]))
    return pair / _size(df["MGR_ID"].astype(str))


def pair_counts(df):
    # raw co-occurrence counts of resource with organisational context
    out = {}
    for c in ["ROLE_ROLLUP_2", "ROLE_FAMILY", "ROLE_FAMILY_DESC"]:
        out[c] = _size(_k(df, ["RESOURCE", c]))
    return pd.DataFrame(out)


def mgr_breadth(df):
    # number of distinct resources requested under a manager and distinct depts using a resource (generic vs niche)
    return pd.DataFrame({
        "mgr_nres": df.groupby(df["MGR_ID"].astype(str))["RESOURCE"].transform("nunique").astype(float),
        "res_ndept": df.groupby(df["RESOURCE"].astype(str))["ROLE_DEPTNAME"].transform("nunique").astype(float),
        "res_nmgr": df.groupby(df["RESOURCE"].astype(str))["MGR_ID"].transform("nunique").astype(float),
    })


PROPOSALS = [("resource_share_in_dept", resource_share_in_dept), ("resource_share_in_role", resource_share_in_role),
             ("resource_share_under_mgr", resource_share_under_mgr), ("counts_single", counts_single),
             ("pair_counts", pair_counts), ("mgr_breadth", mgr_breadth)]
