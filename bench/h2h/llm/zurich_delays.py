"""Domain features for zurich_delays, written from the schema and description only."""
import numpy as np
import pandas as pd


def _k(df, c):
    return df[c].astype(str).where(df[c].notna(), "NA")


def stop_position_proxy(df):
    # delays accumulate along a route; stops served by many lines are central hubs (congested downtown)
    s = _k(df, "stop_id")
    lines = _k(df, "line_number").groupby(s).transform("nunique").astype(float)
    return pd.DataFrame({"stop_n_lines": lines, "stop_n": s.groupby(s).transform("size").astype(float)})


def line_stop_dir_count(df):
    # frequency of the (line, direction, stop) combination in the sample: busy, frequently-served segments
    k = _k(df, "line_number") + "_" + _k(df, "direction") + "_" + _k(df, "stop_id")
    return k.groupby(k).transform("size").astype(float)


def rush_hour(df):
    # peak-hour congestion on weekdays (7-9, 16-19) is the main delay driver for road-bound buses/trolleys
    m = pd.to_numeric(df["dayminute"], errors="coerce")
    wd = pd.to_numeric(df["weekday"], errors="coerce")
    peak = (((m >= 420) & (m < 540)) | ((m >= 960) & (m < 1140))).astype(float)
    weekday = (wd <= 5).astype(float)
    return pd.DataFrame({"peak": peak, "peak_weekday": peak * weekday,
                         "tod_sin": np.sin(2 * np.pi * m / 1440), "tod_cos": np.cos(2 * np.pi * m / 1440)})


def vehicle_peak(df):
    # buses suffer road traffic far more than trams with own right-of-way; interact type with peak time
    m = pd.to_numeric(df["dayminute"], errors="coerce")
    road = df["vehicle_type"].astype(str).isin(["Bus", "Trolley"]).astype(float)
    peak = (((m >= 420) & (m < 540)) | ((m >= 960) & (m < 1140))).astype(float)
    return road * peak + road


def weather_bad(df):
    # rain and wind slow boarding/traffic; near-saturation (temp - dew point small) means fog/drizzle
    return pd.DataFrame({"dewpoint_spread": df["temp"] - df["dew_point"],
                         "gust_factor": df["windspeed_max"] / df["windspeed_avg"].replace(0, np.nan),
                         "rain_x_road": (df["precipitation"] > 0).astype(float)
                         * df["vehicle_type"].astype(str).isin(["Bus", "Trolley"]).astype(float)})


def line_n_stops(df):
    # line length proxy (distinct stops on the line): longer lines accumulate more delay
    ln = _k(df, "line_number")
    return _k(df, "stop_id").groupby(ln).transform("nunique").astype(float)


def line_time_rel(df):
    # time of day relative to the line's typical service time: late-evening trips of daytime lines
    m = pd.to_numeric(df["dayminute"], errors="coerce")
    ln = _k(df, "line_number")
    return m - m.groupby(ln).transform("mean")


PROPOSALS = [("rush_hour", rush_hour), ("vehicle_peak", vehicle_peak), ("stop_position_proxy", stop_position_proxy),
             ("line_stop_dir_count", line_stop_dir_count), ("line_n_stops", line_n_stops), ("weather_bad", weather_bad),
             ("line_time_rel", line_time_rel)]
