"""Domain features for bike_sharing, written from the schema and description only."""
import numpy as np
import pandas as pd


def _bool(s):
    return s.astype("string").str.strip().str.lower().isin(["true", "1", "yes"]).astype(float)


def hour_workingday(df):
    # commuter peaks (8h, 17-18h) on working days vs midday leisure peak on weekends: encode the joint pattern
    return pd.to_numeric(df["hour"], errors="coerce").astype(float) + 24 * _bool(df["workingday"])


def commute_peak(df):
    # explicit rush-hour flag on working days, the dominant demand driver
    h = pd.to_numeric(df["hour"], errors="coerce")
    return (_bool(df["workingday"]) * (h.isin([7, 8, 9, 16, 17, 18, 19])).astype(float))


def time_index(df):
    # system growth trend: ridership grew steadily over the two years
    return pd.to_numeric(df["year"], errors="coerce").astype(float) * 12 + pd.to_numeric(df["month"], errors="coerce")


def hour_cyclic(df):
    # hour of day is circular (23h next to 0h)
    h = pd.to_numeric(df["hour"], errors="coerce").astype(float)
    return pd.DataFrame({"sin": np.sin(2 * np.pi * h / 24), "cos": np.cos(2 * np.pi * h / 24)})


def comfort_index(df):
    # riding comfort: warm but not hot, low humidity, low wind; penalise deviation from ~25C
    t = df["temp"].astype(float)
    return -(t - 25).abs() - 20 * df["humidity"].astype(float) - 0.3 * df["windspeed"].astype(float)


def bad_weather_x_hour(df):
    # rain hurts most at peak leisure/commute hours; severity rank times daytime
    sev = df["weather"].astype("string").str.strip().map({"clear": 0, "misty": 1, "rain": 2, "heavy_rain": 3}).astype(float)
    h = pd.to_numeric(df["hour"], errors="coerce").astype(float)
    return sev * ((h >= 7) & (h <= 20)).astype(float)


def feel_minus_temp(df):
    # apparent vs actual temperature: humidity/wind discomfort
    return df["feel_temp"].astype(float) - df["temp"].astype(float)


def night(df):
    # hours with near-zero demand
    h = pd.to_numeric(df["hour"], errors="coerce")
    return ((h <= 5)).astype(float)


PROPOSALS = [("hour_workingday", hour_workingday), ("commute_peak", commute_peak), ("time_index", time_index),
             ("comfort_index", comfort_index), ("bad_weather_x_hour", bad_weather_x_hour), ("hour_cyclic", hour_cyclic),
             ("feel_minus_temp", feel_minus_temp), ("night", night)]
