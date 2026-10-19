"""Domain features for sonar, written from the schema and description only.

attribute_1..attribute_60 are energies in increasing frequency bands of a rising chirp
(higher bands also integrated later in time), so they are treated as an ordered spectrum.
"""
import numpy as np
import pandas as pd


def _spec(df):
    return df[["attribute_%d" % i for i in range(1, 61)]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def band_energies(df):
    # energy in low / mid / high frequency thirds: metal cylinders vs rocks reflect bands differently
    s = _spec(df)
    return pd.DataFrame({"low": s[:, :20].mean(1), "mid": s[:, 20:40].mean(1), "high": s[:, 40:].mean(1)},
                        index=df.index)


def band_ratios(df):
    # spectral tilt: mid/low and high/mid energy ratios, independent of overall echo strength
    s = _spec(df)
    lo, mi, hi = s[:, :20].mean(1), s[:, 20:40].mean(1), s[:, 40:].mean(1)
    lo[lo == 0] = np.nan
    mi[mi == 0] = np.nan
    return pd.DataFrame({"mid_over_low": mi / lo, "high_over_mid": hi / mi}, index=df.index)


def spectral_centroid(df):
    # energy-weighted mean band index and spread: where in the chirp the echo is strongest
    s = _spec(df)
    k = np.arange(1, 61)
    tot = s.sum(1)
    tot[tot == 0] = np.nan
    c = (s * k).sum(1) / tot
    spread = np.sqrt((s * (k - c[:, None]) ** 2).sum(1) / tot)
    return pd.DataFrame({"centroid": c, "spread": spread}, index=df.index)


def peak_info(df):
    # location and height of the spectral peak relative to mean (resonance of a hollow metal cylinder)
    s = _spec(df)
    m = s.mean(1)
    m[m == 0] = np.nan
    return pd.DataFrame({"peak_band": s.argmax(1).astype(float), "peak_over_mean": s.max(1) / m}, index=df.index)


def roughness(df):
    # band-to-band variation: ringing metal objects give jagged spectra, rough rocks smoother/diffuse
    s = _spec(df)
    d = np.abs(np.diff(s, axis=1))
    m = s.mean(1)
    m[m == 0] = np.nan
    return pd.DataFrame({"rough": d.mean(1), "rough_rel": d.mean(1) / m}, index=df.index)


def flatness(df):
    # spectral flatness (geometric/arithmetic mean): tonal vs noise-like returns
    s = np.clip(_spec(df), 1e-4, None)
    return pd.Series(np.exp(np.log(s).mean(1)) / s.mean(1), index=df.index)


def total_energy(df):
    # total echo energy (log): target strength
    return pd.Series(np.log(_spec(df).sum(1) + 1e-6), index=df.index)


PROPOSALS = [("band_energies", band_energies), ("band_ratios", band_ratios), ("spectral_centroid", spectral_centroid),
             ("peak_info", peak_info), ("roughness", roughness), ("flatness", flatness),
             ("total_energy", total_energy)]
