"""Domain features for t02, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the schema matches the Montesinho forest-fires table: f_1/f_2 grid X/Y, f_3 month,
# f_4 weekday, f_5 FFMC, f_6 DMC, f_7 DC, f_8 ISI (Canadian Fire Weather Index components), f_9 temperature (C),
# f_10 relative humidity (%), f_11 wind (km/h), f_12 rain (mm).

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_DAYS = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}


# Season: cyclic month encoding and peak fire season flag (Aug/Sep).
def season(df):
    m = df["f_3"].astype(object).astype(str).str.lower().map(_MONTHS).astype(float)
    return pd.DataFrame({
        "month_num": m,
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "peak_season": m.isin([8, 9]).astype(float),
    })


# Human activity: weekends bring more visitors and ignitions.
def weekday(df):
    d = df["f_4"].astype(object).astype(str).str.lower().map(_DAYS).astype(float)
    return pd.DataFrame({"day_num": d, "weekend": d.isin([5, 6, 7]).astype(float)})


# OUTSIDE KNOWLEDGE: Canadian FWI system: BUI from DMC and DC, FWI from ISI and BUI (Van Wagner 1987).
# Buildup Index and Fire Weather Index: fuel available for combustion combined with spread rate.
def fwi_system(df):
    dmc = pd.to_numeric(df["f_6"], errors="coerce").clip(lower=0)
    dc = pd.to_numeric(df["f_7"], errors="coerce").clip(lower=0)
    isi = pd.to_numeric(df["f_8"], errors="coerce").clip(lower=0)
    denom = (dmc + 0.4 * dc).replace(0, np.nan)
    bui_a = 0.8 * dmc * dc / denom
    bui_b = dmc - (1 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7)
    bui = pd.Series(np.where(dmc <= 0.4 * dc, bui_a, bui_b), index=df.index).clip(lower=0).fillna(0)
    fd = pd.Series(np.where(bui <= 80, 0.626 * bui ** 0.809 + 2, 1000 / (25 + 108.64 * np.exp(-0.023 * bui))),
                   index=df.index)
    b = 0.1 * isi * fd
    fwi = pd.Series(np.where(b > 1, np.exp(2.72 * (0.434 * np.log(b.clip(lower=1e-9))) ** 0.647), b),
                    index=df.index)
    dsr = 0.0272 * fwi ** 1.77
    return pd.DataFrame({"bui": bui, "fwi": fwi, "dsr": dsr})


# OUTSIDE KNOWLEDGE: FFMC to fine-fuel moisture content conversion from the FWI system.
# Fine fuel moisture content: inverse of FFMC, drives ignition probability.
def fuel_moisture(df):
    ffmc = pd.to_numeric(df["f_5"], errors="coerce")
    mc = 147.2 * (101 - ffmc) / (59.5 + ffmc).replace(0, np.nan)
    return pd.DataFrame({"fine_fuel_mc": mc, "log_fine_fuel_mc": np.log1p(mc.clip(lower=0))})


# Weather dryness: vapour pressure deficit from temperature and humidity, plus wind-driven drying.
def weather(df):
    t = pd.to_numeric(df["f_9"], errors="coerce")
    rh = pd.to_numeric(df["f_10"], errors="coerce")
    w = pd.to_numeric(df["f_11"], errors="coerce")
    rain = pd.to_numeric(df["f_12"], errors="coerce")
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    vpd = es * (1 - rh / 100.0)
    return pd.DataFrame({
        "vpd": vpd,
        "vpd_x_wind": vpd * w,
        "temp_minus_rh": t - rh,
        "rained": (rain > 0).astype(float),
    })


# Location: grid cell id and log-ish transforms of the drought codes (skewed).
def location_and_logs(df):
    x = pd.to_numeric(df["f_1"], errors="coerce")
    y = pd.to_numeric(df["f_2"], errors="coerce")
    cell = x * 10 + y
    return pd.DataFrame({
        "cell_count": cell.map(cell.value_counts()).astype(float),
        "log_dmc": np.log1p(pd.to_numeric(df["f_6"], errors="coerce").clip(lower=0)),
        "log_isi": np.log1p(pd.to_numeric(df["f_8"], errors="coerce").clip(lower=0)),
    })


PROPOSALS = [
    ("season", season),
    ("weekday", weekday),
    ("fwi_system", fwi_system),
    ("fuel_moisture", fuel_moisture),
    ("weather", weather),
    ("location_and_logs", location_and_logs),
]
