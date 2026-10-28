"""Domain features for optdigits, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: input1..input64 are assumed to be the 8x8 block-count image in row-major order
# (input1..input8 = top row), as in the standard UCI optdigits file.


def _img(df):
    X = df[[f"input{i}" for i in range(1, 65)]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    return X.reshape(len(df), 8, 8).astype(float)


_r = np.arange(8)[None, :, None].astype(float)
_c = np.arange(8)[None, None, :].astype(float)


def ink_and_centroid(df):
    # total ink and centre of mass: e.g. 1 has little ink, 7 is top-heavy, 6 bottom-heavy
    I = _img(df)
    m = I.sum(axis=(1, 2))
    mz = np.where(m == 0, np.nan, m)
    return pd.DataFrame({"ink": m, "cy": (I * _r).sum(axis=(1, 2)) / mz, "cx": (I * _c).sum(axis=(1, 2)) / mz},
                        index=df.index)


def second_moments(df):
    # central second moments: spread, elongation and slant of the stroke mass
    I = _img(df)
    m = np.where(I.sum(axis=(1, 2)) == 0, np.nan, I.sum(axis=(1, 2)))
    cy = (I * _r).sum(axis=(1, 2)) / m
    cx = (I * _c).sum(axis=(1, 2)) / m
    dy = _r - cy[:, None, None]
    dx = _c - cx[:, None, None]
    myy = (I * dy ** 2).sum(axis=(1, 2)) / m
    mxx = (I * dx ** 2).sum(axis=(1, 2)) / m
    mxy = (I * dx * dy).sum(axis=(1, 2)) / m
    return pd.DataFrame({"mxx": mxx, "myy": myy, "mxy": mxy, "aspect": myy / np.where(mxx == 0, np.nan, mxx)},
                        index=df.index)


def symmetry(df):
    # left-right and top-bottom mirror differences: 0/8 symmetric, 2/5/7 not
    I = _img(df)
    return pd.DataFrame({"lr_asym": np.abs(I - I[:, :, ::-1]).sum(axis=(1, 2)),
                         "tb_asym": np.abs(I - I[:, ::-1, :]).sum(axis=(1, 2))}, index=df.index)


def quadrants(df):
    # ink per quadrant: coarse layout separating 4/7/9 and 2/5
    I = _img(df)
    return pd.DataFrame({"q_tl": I[:, :4, :4].sum(axis=(1, 2)), "q_tr": I[:, :4, 4:].sum(axis=(1, 2)),
                         "q_bl": I[:, 4:, :4].sum(axis=(1, 2)), "q_br": I[:, 4:, 4:].sum(axis=(1, 2))}, index=df.index)


def center_hole(df):
    # ink in central 2x2 vs surrounding ring: loops (0) leave the centre empty, 1/8 fill it
    I = _img(df)
    centre = I[:, 3:5, 3:5].sum(axis=(1, 2))
    ring = I[:, 1:7, 1:7].sum(axis=(1, 2)) - I[:, 2:6, 2:6].sum(axis=(1, 2))
    return pd.DataFrame({"centre_ink": centre, "ring_ink": ring, "centre_over_ring": centre / np.where(ring == 0, np.nan, ring)},
                        index=df.index)


def stroke_crossings(df):
    # number of ink/no-ink transitions along middle rows and columns: counts strokes crossed (0 gives 2, 1 gives 1)
    B = (_img(df) > 4).astype(int)
    h = np.abs(np.diff(B[:, 2:6, :], axis=2)).sum(axis=2)
    v = np.abs(np.diff(B[:, :, 2:6], axis=1)).sum(axis=1)
    return pd.DataFrame({"hcross_mean": h.mean(axis=1), "vcross_mean": v.mean(axis=1),
                         "hcross_max": h.max(axis=1), "vcross_max": v.max(axis=1)}, index=df.index)


def profiles(df):
    # row and column ink profiles (projections) collapsed to 16 numbers
    I = _img(df)
    rows = I.sum(axis=2)
    cols = I.sum(axis=1)
    out = {f"row{k}": rows[:, k] for k in range(8)}
    out.update({f"col{k}": cols[:, k] for k in range(8)})
    return pd.DataFrame(out, index=df.index)


PROPOSALS = [("ink_and_centroid", ink_and_centroid), ("second_moments", second_moments), ("symmetry", symmetry),
             ("quadrants", quadrants), ("center_hole", center_hole), ("stroke_crossings", stroke_crossings),
             ("profiles", profiles)]
