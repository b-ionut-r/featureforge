"""Domain features for tic_tac_toe, written from the schema and description only."""
import numpy as np
import pandas as pd

SQ = ["top-left-square", "top-middle-square", "top-right-square",
      "middle-left-square", "middle-middle-square", "middle-right-square",
      "bottom-left-square", "bottom-middle-square", "bottom-right-square"]
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _board(df):
    return df[SQ].fillna("b").astype(str).apply(lambda s: s.str.strip().str.lower())


def x_lines(df):
    # the target concept is literally "x has a three-in-a-row": count x's completed lines
    b = _board(df)
    return sum(((b[SQ[i]] == "x") & (b[SQ[j]] == "x") & (b[SQ[k]] == "x")).astype(float) for i, j, k in LINES)


def o_lines(df):
    # o completed lines (o winning means x did not)
    b = _board(df)
    return sum(((b[SQ[i]] == "o") & (b[SQ[j]] == "o") & (b[SQ[k]] == "o")).astype(float) for i, j, k in LINES)


def piece_counts(df):
    # number of x, o and blanks: tells who moved last and whether the board is full (draw)
    b = _board(df)
    return pd.DataFrame({"n_x": (b == "x").sum(axis=1).astype(float), "n_o": (b == "o").sum(axis=1).astype(float),
                         "n_b": (b == "b").sum(axis=1).astype(float)})


def x_two_open(df):
    # lines with two x and no o: near-wins
    b = _board(df)
    out = 0
    for i, j, k in LINES:
        t = b[[SQ[i], SQ[j], SQ[k]]]
        out = out + (((t == "x").sum(axis=1) == 2) & ((t == "o").sum(axis=1) == 0)).astype(float)
    return out


def x_strategic(df):
    # x on center and corners: the strongest squares
    b = _board(df)
    return pd.DataFrame({"x_center": (b[SQ[4]] == "x").astype(float),
                         "x_corners": sum((b[SQ[i]] == "x").astype(float) for i in (0, 2, 6, 8))})


PROPOSALS = [("x_lines", x_lines), ("o_lines", o_lines), ("piece_counts", piece_counts), ("x_two_open", x_two_open),
             ("x_strategic", x_strategic)]
