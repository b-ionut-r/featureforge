"""Domain features for pc1, written from the schema and description only."""
import numpy as np
import pandas as pd


def _div(a, b):
    return a / b.replace(0, np.nan)


def complexity_density(df):
    # cyclomatic complexity per line: dense branching code is harder to test and gets more defects
    return pd.DataFrame({"vg_per_loc": _div(df["v(g)"], df["loc"]),
                         "branch_per_loc": _div(df["branchCount"], df["loc"]),
                         "e_per_loc": _div(df["E"], df["loc"])}, index=df.index)


def structure_ratios(df):
    # essential/design complexity relative to cyclomatic: unstructured (goto-like) and highly coupled code
    return pd.DataFrame({"ev_over_vg": _div(df["ev(g)"], df["v(g)"]),
                         "iv_over_vg": _div(df["iv(G)"], df["v(g)"])}, index=df.index)


def comment_ratios(df):
    # comment density: under-documented or heavily-patched (comment-laden) modules
    code = df["lOCode"].astype(float)
    com = df["lOComment"].astype(float) + df["locCodeAndComment"].astype(float)
    return pd.DataFrame({"comment_ratio": _div(com, code + com),
                         "blank_ratio": _div(df["lOBlank"].astype(float), code + com + df["lOBlank"].astype(float))},
                        index=df.index)


def halstead_estimate_gap(df):
    # Halstead estimated length vs actual N: large gaps flag unusual/obfuscated vocabulary use
    u1, u2 = df["uniq_Op"].clip(lower=1), df["uniq_Opnd"].clip(lower=1)
    nhat = u1 * np.log2(u1) + u2 * np.log2(u2)
    return pd.DataFrame({"nhat_over_n": _div(nhat, df["N"]),
                         "log_vocab": np.log2(u1 + u2)}, index=df.index)


def operand_reuse(df):
    # how often each operand/operator is reused: high reuse = data-heavy, state-manipulating code
    return pd.DataFrame({"opnd_reuse": _div(df["total_Opnd"], df["uniq_Opnd"]),
                         "op_reuse": _div(df["total_Op"], df["uniq_Op"]),
                         "tokens_per_loc": _div(df["N"], df["loc"])}, index=df.index)


def log_size_effort(df):
    # log effort and log size: defect counts scale sub-linearly with module size
    return pd.DataFrame({"log_e": np.log1p(df["E"]), "log_loc": np.log1p(df["loc"]),
                         "log_vg": np.log1p(df["v(g)"])}, index=df.index)


PROPOSALS = [
    ("complexity_density", complexity_density),
    ("comment_ratios", comment_ratios),
    ("structure_ratios", structure_ratios),
    ("operand_reuse", operand_reuse),
    ("halstead_estimate_gap", halstead_estimate_gap),
    ("log_size_effort", log_size_effort),
]
