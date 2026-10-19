"""Domain features for kc2, written from the schema and description only."""
import numpy as np
import pandas as pd

CC = "lOCodeAndComment"


def _f(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def complexity_density(df):
    # cyclomatic complexity per line of code: branchy, dense modules are harder to get right
    loc = _f(df, "loc").replace(0, np.nan)
    return pd.DataFrame({"vg_per_loc": _f(df, "v(g)") / loc,
                         "branch_per_loc": _f(df, "branchCount") / loc})


def essential_ratio(df):
    # essential/cyclomatic complexity: share of unstructured control flow (gotos, multiple exits)
    vg = _f(df, "v(g)").replace(0, np.nan)
    return pd.DataFrame({"ev_over_vg": _f(df, "ev(g)") / vg,
                         "iv_over_vg": _f(df, "iv(g)") / vg})


def comment_ratio(df):
    # documentation density: comment lines (incl. mixed lines) relative to code lines
    code = _f(df, "lOCode")
    com = _f(df, "lOComment") + _f(df, CC)
    return pd.DataFrame({"comment_per_code": com / code.replace(0, np.nan),
                         "blank_share": _f(df, "lOBlank") / (code + com + _f(df, "lOBlank")).replace(0, np.nan)})


def vocabulary_reuse(df):
    # operand/operator reuse: total tokens per unique token; low reuse means many distinct names to track
    return pd.DataFrame({"opnd_reuse": _f(df, "total_Opnd") / _f(df, "uniq_Opnd").replace(0, np.nan),
                         "op_reuse": _f(df, "total_Op") / _f(df, "uniq_Op").replace(0, np.nan)})


def log_effort_size(df):
    # Halstead effort/volume span orders of magnitude; logs make them comparable, effort per line is the per-line mental load
    return pd.DataFrame({"log_e": np.log1p(_f(df, "e").clip(lower=0)),
                         "log_v": np.log1p(_f(df, "v").clip(lower=0)),
                         "e_per_loc": np.log1p((_f(df, "e") / _f(df, "loc").replace(0, np.nan)).clip(lower=0))})


def tokens_per_line(df):
    # Halstead length per line of code: how packed each line is
    return _f(df, "n") / _f(df, "lOCode").replace(0, np.nan)


def bugs_per_loc(df):
    # Halstead's estimated bug count (b) normalised by size: estimated defect density
    return _f(df, "b") / _f(df, "loc").replace(0, np.nan)


def difficulty_x_complexity(df):
    # interaction of Halstead difficulty and McCabe complexity: modules hard on both axes
    return np.log1p(_f(df, "d").clip(lower=0) * _f(df, "v(g)").clip(lower=0))


PROPOSALS = [("complexity_density", complexity_density), ("essential_ratio", essential_ratio),
             ("comment_ratio", comment_ratio), ("vocabulary_reuse", vocabulary_reuse),
             ("log_effort_size", log_effort_size), ("tokens_per_line", tokens_per_line),
             ("bugs_per_loc", bugs_per_loc), ("difficulty_x_complexity", difficulty_x_complexity)]
