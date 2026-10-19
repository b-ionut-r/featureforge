"""Domain features for jungle_chess, written from the schema and description only."""
import numpy as np
import pandas as pd

# Board: 7 files (0-6) x 9 ranks (0-8); dens at (file 3, rank 0) and (file 3, rank 8);
# river squares: ranks 3-5, files 1,2,4,5. Strength 0=rat ... 7=elephant (0-indexed).


def _w(df):
    return (df["white_piece0_strength"].astype(float), df["white_piece0_file"].astype(float),
            df["white_piece0_rank"].astype(float))


def _b(df):
    return (df["black_piece0_strength"].astype(float), df["black_piece0_file"].astype(float),
            df["black_piece0_rank"].astype(float))


def capture_relations(df):
    # who can capture whom: higher-or-equal strength captures, except rat beats elephant and elephant can't take rat
    ws, _, _ = _w(df)
    bs, _, _ = _b(df)
    w_can = ((ws >= bs) & ~((ws == 7) & (bs == 0))) | ((ws == 0) & (bs == 7))
    b_can = ((bs >= ws) & ~((bs == 7) & (ws == 0))) | ((bs == 0) & (ws == 7))
    return pd.DataFrame({"w_can_capture": w_can.astype(float), "b_can_capture": b_can.astype(float),
                         "strength_diff": ws - bs}, index=df.index)


def den_race(df):
    # endgames with two pieces are races to the enemy den: distance to each den for both pieces
    _, wf, wr = _w(df)
    _, bf, br = _b(df)
    out = pd.DataFrame(index=df.index)
    out["w_to_den8"] = (wf - 3).abs() + (8 - wr)
    out["w_to_den0"] = (wf - 3).abs() + wr
    out["b_to_den0"] = (bf - 3).abs() + br
    out["b_to_den8"] = (bf - 3).abs() + (8 - br)
    out["race_w8_b0"] = out["w_to_den8"] - out["b_to_den0"]
    out["race_w0_b8"] = out["w_to_den0"] - out["b_to_den8"]
    return out


def piece_distance(df):
    # distance between the pieces and its parity decide whether a capture/escape is possible with tempo
    _, wf, wr = _w(df)
    _, bf, br = _b(df)
    man = (wf - bf).abs() + (wr - br).abs()
    return pd.DataFrame({"manhattan": man, "parity": (man % 2), "dfile": wf - bf, "drank": wr - br,
                         "chebyshev": np.maximum((wf - bf).abs(), (wr - br).abs())}, index=df.index)


def river_and_traps(df):
    # only rats swim (and can't capture from water); lion/tiger jump the river; traps near dens neutralise pieces
    _, wf, wr = _w(df)
    _, bf, br = _b(df)

    def water(f, r):
        return (r.between(3, 5) & f.isin([1, 2, 4, 5])).astype(float)

    def trap(f, r):
        return (((f.isin([2, 4])) & r.isin([0, 8])) | ((f == 3) & r.isin([1, 7]))).astype(float)
    return pd.DataFrame({"w_in_water": water(wf, wr), "b_in_water": water(bf, br),
                         "w_on_trap": trap(wf, wr), "b_on_trap": trap(bf, br)}, index=df.index)


def same_line(df):
    # pieces on the same file or rank: direct confrontation / jump lines across the river
    _, wf, wr = _w(df)
    _, bf, br = _b(df)
    return pd.DataFrame({"same_file": (wf == bf).astype(float), "same_rank": (wr == br).astype(float),
                         "adjacent": (((wf - bf).abs() + (wr - br).abs()) == 1).astype(float)}, index=df.index)


PROPOSALS = [
    ("capture_relations", capture_relations),
    ("den_race", den_race),
    ("piece_distance", piece_distance),
    ("river_and_traps", river_and_traps),
    ("same_line", same_line),
]
