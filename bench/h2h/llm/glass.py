"""Domain features for glass, written from the schema and description only."""
import numpy as np
import pandas as pd


def alkali_earth(df):
    # network modifiers: alkali (Na+K) flux vs alkaline-earth (Ca+Mg) stabilisers define the glass recipe
    alk = df["Na"] + df["K"]
    ae = df["Ca"] + df["Mg"]
    return pd.DataFrame({"alkali": alk, "alk_earth": ae, "alk_over_ae": alk / ae.replace(0, np.nan)})


def former_ratios(df):
    # network formers Si and Al relative to modifiers: containers are Al-rich, window glass Si/Ca balanced
    mod = df["Na"] + df["K"] + df["Ca"] + df["Mg"] + df["Ba"]
    return pd.DataFrame({"si_over_mod": df["Si"] / mod.replace(0, np.nan),
                         "al_over_si": df["Al"] / df["Si"].replace(0, np.nan)})


def marker_presence(df):
    # trace markers: Mg near zero = non-float glass; Ba = headlamp glass; Fe = cheap/container glass; K
    # OUTSIDE KNOWLEDGE: float window glass has ~3.5-4% MgO; headlamp glass contains barium
    return pd.DataFrame({"mg_zero": (df["Mg"] < 0.5).astype(float), "ba_present": (df["Ba"] > 0).astype(float),
                         "fe_present": (df["Fe"] > 0).astype(float), "k_zero": (df["K"] == 0).astype(float)})


def ca_mg_ratio(df):
    # dolomite (Mg) vs limestone (Ca) source; Ca/Mg separates float from non-float / tableware glass
    return df["Ca"] / (df["Mg"] + 0.1)


def ri_residual(df):
    # refractive index rises with Ca and heavy oxides; residual RI after Ca reflects processing/thermal history
    return pd.DataFrame({"ri_per_ca": (df["RI"] - 1.5) * 1000 / df["Ca"].replace(0, np.nan),
                         "ri_minus_ca": (df["RI"] - 1.5) * 1000 - df["Ca"]})


def oxide_total(df):
    # oxide sum (should be ~100%): shortfall hints at unmeasured components (e.g. B2O3, Pb in tableware)
    return 100.0 - df[["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]].sum(axis=1)


PROPOSALS = [("alkali_earth", alkali_earth), ("former_ratios", former_ratios), ("marker_presence", marker_presence),
             ("ca_mg_ratio", ca_mg_ratio), ("ri_residual", ri_residual), ("oxide_total", oxide_total)]
