"""Domain features for fried, written from the schema and description only."""
import numpy as np
import pandas as pd

# PMLB "fried" is Friedman's synthetic benchmark #1: inputs uniform on [0, 1]; the response is known to be
# 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5 + noise, and X6..X10 are pure noise.


def sin_pi_x1x2(df):
    # the multiplicative sine term of the Friedman function: trees approximate it poorly with axis splits
    return np.sin(np.pi * df["X1"].astype(float) * df["X2"].astype(float))


def x3_centered_sq(df):
    # quadratic bowl around X3 = 0.5 in the Friedman function
    return (df["X3"].astype(float) - 0.5) ** 2


def friedman_linear_part(df):
    # the additive linear part 10*X4 + 5*X5, as one direction a tree cannot form with axis splits
    return 10 * df["X4"].astype(float) + 5 * df["X5"].astype(float)


def x1_times_x2(df):
    # plain pairwise interaction of the first two inputs
    return df["X1"].astype(float) * df["X2"].astype(float)


def friedman_structural(df):
    # full noise-free Friedman #1 signal built from the inputs
    return (10 * sin_pi_x1x2(df) + 20 * x3_centered_sq(df) + friedman_linear_part(df))


PROPOSALS = [
    ("friedman_structural", friedman_structural),
    ("sin_pi_x1x2", sin_pi_x1x2),
    ("x3_centered_sq", x3_centered_sq),
    ("friedman_linear_part", friedman_linear_part),
    ("x1_times_x2", x1_times_x2),
]
