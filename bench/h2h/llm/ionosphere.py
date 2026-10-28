"""Domain features for ionosphere, written from the schema and description only.

a03..a34 are read as 16 consecutive (real, imaginary) autocorrelation pairs per the description
(2 attributes per pulse number); a01/a02 are leading flags.
"""
import numpy as np
import pandas as pd


def _pairs(df):
    cols = ["a%02d" % i for i in range(3, 35)]
    a = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return a[:, 0::2], a[:, 1::2]


def magnitude_stats(df):
    # |ACF| per lag: structured returns keep coherent energy across lags, pass-through returns are noise
    re, im = _pairs(df)
    m = np.sqrt(re ** 2 + im ** 2)
    return pd.DataFrame({"mag_mean": m.mean(1), "mag_std": m.std(1), "mag_first4": m[:, :4].mean(1),
                         "mag_last4": m[:, -4:].mean(1)}, index=df.index)


def magnitude_decay(df):
    # decay of autocorrelation magnitude with lag (slope): coherent targets decorrelate smoothly
    re, im = _pairs(df)
    m = np.sqrt(re ** 2 + im ** 2)
    x = np.arange(m.shape[1]) - (m.shape[1] - 1) / 2
    return pd.Series((m * x).sum(1) / (x ** 2).sum(), index=df.index)


def phase_smoothness(df):
    # phase of the complex ACF should rotate smoothly (Doppler) for real echoes; jumpy phase signals noise
    re, im = _pairs(df)
    ph = np.arctan2(im, re)
    d = np.angle(np.exp(1j * np.diff(ph, axis=1)))
    return pd.DataFrame({"phase_step_absmean": np.abs(d).mean(1), "phase_step_std": d.std(1),
                         "phase_step_mean": d.mean(1)}, index=df.index)


def series_roughness(df):
    # lag-to-lag jumps in real and imaginary parts: smooth vs erratic autocorrelation sequences
    re, im = _pairs(df)
    return pd.DataFrame({"re_rough": np.abs(np.diff(re, axis=1)).mean(1),
                         "im_rough": np.abs(np.diff(im, axis=1)).mean(1)}, index=df.index)


def saturation_counts(df):
    # values clipped at exactly +-1 or exactly 0 indicate saturated/empty returns
    re, im = _pairs(df)
    a = np.hstack([re, im])
    return pd.DataFrame({"n_sat": (np.abs(a) >= 0.99999).sum(1).astype(float),
                         "n_zero": (a == 0).sum(1).astype(float)}, index=df.index)


def real_mean(df):
    # mean real part across lags and its relation to first-pulse flag a01
    re, im = _pairs(df)
    return pd.DataFrame({"re_mean": re.mean(1), "im_mean": im.mean(1),
                         "a01_x_re": pd.to_numeric(df["a01"], errors="coerce").to_numpy(dtype=float) * re.mean(1)},
                        index=df.index)


PROPOSALS = [("magnitude_stats", magnitude_stats), ("magnitude_decay", magnitude_decay),
             ("phase_smoothness", phase_smoothness), ("series_roughness", series_roughness),
             ("saturation_counts", saturation_counts), ("real_mean", real_mean)]
