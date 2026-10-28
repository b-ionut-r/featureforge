"""Domain features for forest_fires, written from the schema and description only."""
import numpy as np
import pandas as pd

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_DAYS = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}


def _month(df):
    return df["month"].astype(str).str.lower().map(_MONTHS).astype(float)


# Seasonality: month as cyclic coordinates plus a summer fire-season flag.
def month_cyclic(df):
    m = _month(df)
    return pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "fire_season": m.isin([7, 8, 9]).astype(float),
    }, index=df.index)


# Weekends bring more visitors to the park, hence more human ignitions.
def weekend(df):
    d = df["day"].astype(str).str.lower().map(_DAYS).astype(float)
    return (d >= 6).astype(float)


# Buildup Index and Fire Weather Index from the Canadian FWI system.
# OUTSIDE KNOWLEDGE: Van Wagner (1987) BUI and FWI equations from DMC, DC and ISI.
def bui_fwi(df):
    dmc = df["DMC"].astype(float)
    dc = df["DC"].astype(float)
    isi = df["ISI"].astype(float)
    denom = (dmc + 0.4 * dc).replace(0, np.nan)
    bui_a = 0.8 * dmc * dc / denom
    bui_b = dmc - (1 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7)
    bui = np.where(dmc <= 0.4 * dc, bui_a, bui_b)
    bui = pd.Series(bui, index=df.index).clip(lower=0).fillna(0)
    fd = np.where(bui <= 80, 0.626 * bui ** 0.809 + 2,
                  1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    b = 0.1 * isi * fd
    with np.errstate(divide="ignore", invalid="ignore"):
        fwi = np.where(b > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1e-9))) ** 0.647), b)
    return pd.DataFrame({"BUI": bui, "FWI": fwi}, index=df.index)


# Vapour pressure deficit: dryness of air from temperature and humidity.
# OUTSIDE KNOWLEDGE: Tetens formula for saturation vapour pressure.
def vpd(df):
    es = 0.6108 * np.exp(17.27 * df["temp"] / (df["temp"] + 237.3))
    return es * (1 - df["RH"] / 100.0)


# Wind-driven spread under dry fine fuels.
def wind_x_ffmc(df):
    return df["wind"] * df["FFMC"]


# Any rain recently wets surface fuels.
def rain_flag(df):
    return (df["rain"] > 0).astype(float)


# Long-tailed drought codes on log scale.
def log_moisture_codes(df):
    return pd.DataFrame({
        "log_DMC": np.log1p(df["DMC"].clip(lower=0)),
        "log_ISI": np.log1p(df["ISI"].clip(lower=0)),
        "ffmc_exp": np.exp((df["FFMC"] - 100) / 10.0),
    }, index=df.index)


# Spatial cell of the park map (fire-prone zones), as a single location id.
def grid_cell(df):
    return (df["X"].astype(float) * 10 + df["Y"].astype(float))


PROPOSALS = [
    ("month_cyclic", month_cyclic),
    ("weekend", weekend),
    ("bui_fwi", bui_fwi),
    ("vpd", vpd),
    ("wind_x_ffmc", wind_x_ffmc),
    ("rain_flag", rain_flag),
    ("log_moisture_codes", log_moisture_codes),
    ("grid_cell", grid_cell),
]
