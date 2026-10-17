"""Domain features for cylinder_bands, written from the schema and description only."""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def _s(df, c):
    return df[c].astype(object).astype(str).str.strip().str.lower()


def ink_solvent_ratio(df):
    # ink-to-solvent balance of the ink mix: too thin/thick ink changes how it sits on the cylinder
    ink, sol = _f(df, "ink_pct"), _f(df, "solvent_pct")
    return pd.DataFrame({"ink_over_solvent": ink / sol.replace(0, np.nan),
                         "mix_total": ink + sol + _f(df, "varnish_pct")})


def viscosity_per_temp(df):
    # viscosity relative to ink temperature: warm ink should be thinner; mismatch suggests off-spec ink
    return pd.DataFrame({"visc_over_temp": _f(df, "viscosity") / _f(df, "ink_temperature").replace(0, np.nan),
                         "visc_x_speed": _f(df, "viscosity") * _f(df, "press_speed") / 1000.0})


def blade_load(df):
    # doctor-blade pressure relative to press speed and roller hardness: wiping action per unit of throughput
    return pd.DataFrame({"blade_per_speed": _f(df, "blade_pressure") / _f(df, "press_speed").replace(0, np.nan) * 1000,
                         "blade_per_durometer": _f(df, "blade_pressure") / _f(df, "roller_durometer").replace(0, np.nan)})


def esa_power(df):
    # electrostatic assist usage: whether ESA is used at all (voltage or amperage set) to pull ink from cells
    v, a = _f(df, "ESA_Voltage"), _f(df, "ESA_Amperage")
    return pd.DataFrame({"esa_any": ((v.fillna(0) > 0) | (a.fillna(0) > 0)).astype(float),
                         "esa_on": (v.fillna(0) > 0).astype(float)})


def plating_quality(df):
    # chrome plating bath conditions: current density x anode ratio, chrome content deviation from nominal 100
    cd, asr, cr = _f(df, "current_density"), _f(df, "anode_space_ratio"), _f(df, "chrome_content")
    return pd.DataFrame({"cd_x_anode": cd * asr / 100.0, "chrome_dev": (cr - 100).abs()})


def hardener_wax_ratio(df):
    # hardener vs wax additives in the ink: tradeoff between film hardness and slip
    return _f(df, "hardener") / _f(df, "wax").replace(0, np.nan)


def humidity_temp(df):
    # pressroom humidity and ink temperature: drying conditions; humid+cool slows solvent evaporation
    return pd.DataFrame({"humid_over_temp": _f(df, "humifity") / _f(df, "ink_temperature").replace(0, np.nan)})


def paper_ink_match(df):
    # coated/uncoated paper vs ink type mismatch and paper roughness x caliper (surface the cylinder prints on)
    p, i = _s(df, "paper_type"), _s(df, "ink_type")
    return pd.DataFrame({"paper_ink_mismatch": (p != i).astype(float),
                         "uncoated_paper": (p == "uncoated").astype(float),
                         "rough_x_caliper": _f(df, "roughness") * _f(df, "caliper")})


def binary_flags(df):
    # yes/no process switches mapped to numbers; count of missing process measurements (sloppy logging of a job)
    out = {}
    for c in ["grain_screened", "proof_on_ctd_ink", "direct_steam", "type_on_cylinder"]:
        s = _s(df, c)
        out[c + "_yes"] = np.where(s == "yes", 1.0, np.where(s == "no", 0.0, np.nan))
    num = ["proof_cut", "viscosity", "roughness", "blade_pressure", "varnish_pct", "press_speed", "ink_pct",
           "solvent_pct", "ESA_Voltage", "ESA_Amperage", "wax", "hardener", "roller_durometer", "anode_space_ratio"]
    out["n_missing"] = pd.DataFrame({c: _f(df, c) for c in num}).isna().sum(axis=1).astype(float)
    return pd.DataFrame(out, index=df.index)


def job_era(df):
    # job number is roughly sequential in time; coarse era captures process changes over the years
    return _f(df, "job_number") / 1000.0


PROPOSALS = [("ink_solvent_ratio", ink_solvent_ratio), ("viscosity_per_temp", viscosity_per_temp),
             ("blade_load", blade_load), ("esa_power", esa_power), ("plating_quality", plating_quality),
             ("hardener_wax_ratio", hardener_wax_ratio), ("humidity_temp", humidity_temp),
             ("paper_ink_match", paper_ink_match), ("binary_flags", binary_flags), ("job_era", job_era)]
