"""Domain features for pc3, written from the schema and description only."""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def comment_density(df):
    # comment and blank lines per executable line: documentation effort around the code
    ex = _f(df, "LOC_EXECUTABLE").replace(0, np.nan)
    com = _f(df, "LOC_COMMENTS") + _f(df, "LOC_CODE_AND_COMMENT")
    return pd.DataFrame({"comment_per_exec": com / ex, "blank_per_exec": _f(df, "LOC_BLANK") / ex})


def complexity_density(df):
    # decisions/conditions per executable line: branch-dense code is defect-prone
    ex = _f(df, "LOC_EXECUTABLE").replace(0, np.nan)
    return pd.DataFrame({"cc_per_exec": _f(df, "CYCLOMATIC_COMPLEXITY") / ex,
                         "cond_per_exec": _f(df, "CONDITION_COUNT") / ex,
                         "calls_per_exec": _f(df, "CALL_PAIRS") / ex})


def complexity_ratios(df):
    # essential/design complexity relative to cyclomatic: unstructuredness and coupling per path
    cc = _f(df, "CYCLOMATIC_COMPLEXITY").replace(0, np.nan)
    return pd.DataFrame({"ess_over_cc": _f(df, "ESSENTIAL_COMPLEXITY") / cc,
                         "design_over_cc": _f(df, "DESIGN_COMPLEXITY") / cc})


def condition_structure(df):
    # compound conditions (multiple/modified conditions per decision) are hard to test fully
    dec = _f(df, "DECISION_COUNT").replace(0, np.nan)
    return pd.DataFrame({"multi_cond_per_dec": _f(df, "MULTIPLE_CONDITION_COUNT") / dec,
                         "mod_cond_per_dec": _f(df, "MODIFIED_CONDITION_COUNT") / dec})


def graph_shape(df):
    # flow-graph edges vs nodes: extra edges beyond a chain = loops/branches (E - N is cyclomatic-like)
    e, n = _f(df, "EDGE_COUNT"), _f(df, "NODE_COUNT")
    return pd.DataFrame({"edge_minus_node": e - n, "edge_per_node": e / n.replace(0, np.nan)})


def halstead_vocab(df):
    # vocabulary and operand reuse: heavy reuse of few names suggests tangled logic
    n1, n2 = _f(df, "NUM_UNIQUE_OPERATORS"), _f(df, "NUM_UNIQUE_OPERANDS")
    N1, N2 = _f(df, "NUM_OPERATORS"), _f(df, "NUM_OPERANDS")
    return pd.DataFrame({"vocab": n1 + n2, "opnd_reuse": N2 / n2.replace(0, np.nan),
                         "op_reuse": N1 / n1.replace(0, np.nan)})


def halstead_length_gap(df):
    # OUTSIDE KNOWLEDGE: Halstead estimated length N^ = n1*log2(n1) + n2*log2(n2); gap to actual length flags atypical code
    n1 = _f(df, "NUM_UNIQUE_OPERATORS").clip(lower=1)
    n2 = _f(df, "NUM_UNIQUE_OPERANDS").clip(lower=1)
    est = n1 * np.log2(n1) + n2 * np.log2(n2)
    return np.log1p(est) - np.log1p(_f(df, "HALSTEAD_LENGTH"))


def log_size_effort(df):
    # heavy-tailed size/effort metrics on log scale
    return pd.DataFrame({"log_loc": np.log1p(_f(df, "LOC_TOTAL")), "log_effort": np.log1p(_f(df, "HALSTEAD_EFFORT")),
                         "effort_per_line": np.log1p(_f(df, "HALSTEAD_EFFORT")) - np.log1p(_f(df, "LOC_TOTAL"))})


def params_x_complexity(df):
    # wide interfaces combined with high complexity: many inputs interacting with many paths
    return _f(df, "PARAMETER_COUNT") * np.log1p(_f(df, "CYCLOMATIC_COMPLEXITY"))


PROPOSALS = [("comment_density", comment_density), ("complexity_density", complexity_density),
             ("complexity_ratios", complexity_ratios), ("condition_structure", condition_structure),
             ("graph_shape", graph_shape), ("halstead_vocab", halstead_vocab),
             ("halstead_length_gap", halstead_length_gap), ("log_size_effort", log_size_effort),
             ("params_x_complexity", params_x_complexity)]
