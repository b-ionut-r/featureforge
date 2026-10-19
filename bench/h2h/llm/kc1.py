"""Domain features for kc1, written from the schema and description only."""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def comment_density(df):
    # share of commented lines: poorly documented modules are harder to maintain correctly
    code = _f(df, "lOCode")
    com = _f(df, "lOComment") + _f(df, "locCodeAndComment")
    return pd.DataFrame({"comment_per_code": com / code.replace(0, np.nan),
                         "blank_per_code": _f(df, "lOBlank") / code.replace(0, np.nan)})


def complexity_ratios(df):
    # essential/design complexity as a share of cyclomatic: unstructured control flow and coupling per decision
    vg = _f(df, "v(g)").replace(0, np.nan)
    return pd.DataFrame({"ev_over_vg": _f(df, "ev(g)") / vg, "iv_over_vg": _f(df, "iv(g)") / vg})


def complexity_density(df):
    # decisions and branches per line of code: dense branching is a classic defect driver
    loc = _f(df, "loc").replace(0, np.nan)
    return pd.DataFrame({"vg_per_loc": _f(df, "v(g)") / loc, "branch_per_loc": _f(df, "branchCount") / loc})


def halstead_vocab(df):
    # vocabulary and operand/operator reuse: repeated use of few names suggests tangled logic
    n1, n2 = _f(df, "uniq_Op"), _f(df, "uniq_Opnd")
    N1, N2 = _f(df, "total_Op"), _f(df, "total_Opnd")
    return pd.DataFrame({"vocab": n1 + n2, "opnd_reuse": N2 / n2.replace(0, np.nan),
                         "op_reuse": N1 / n1.replace(0, np.nan), "opnd_op_ratio": N2 / N1.replace(0, np.nan)})


def halstead_length_gap(df):
    # OUTSIDE KNOWLEDGE: Halstead estimated length N^ = n1*log2(n1) + n2*log2(n2); gap to actual length flags atypical code
    n1, n2 = _f(df, "uniq_Op").clip(lower=1), _f(df, "uniq_Opnd").clip(lower=1)
    est = n1 * np.log2(n1) + n2 * np.log2(n2)
    return np.log1p(est) - np.log1p(_f(df, "n"))


def log_size_effort(df):
    # size/effort metrics are heavy-tailed; logs make defect risk roughly linear in them
    return pd.DataFrame({"log_loc": np.log1p(_f(df, "loc")), "log_e": np.log1p(_f(df, "e")),
                         "log_v": np.log1p(_f(df, "v"))})


def effort_per_line(df):
    # Halstead effort per line of code: intrinsic difficulty independent of module size
    return np.log1p(_f(df, "e")) - np.log1p(_f(df, "loc"))


def difficulty_x_complexity(df):
    # modules that are both hard to read (d) and branchy (v(g)) are most error-prone
    return np.log1p(_f(df, "d")) * np.log1p(_f(df, "v(g)"))


PROPOSALS = [("comment_density", comment_density), ("complexity_ratios", complexity_ratios),
             ("complexity_density", complexity_density), ("halstead_vocab", halstead_vocab),
             ("halstead_length_gap", halstead_length_gap), ("log_size_effort", log_size_effort),
             ("effort_per_line", effort_per_line), ("difficulty_x_complexity", difficulty_x_complexity)]
