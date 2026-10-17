"""Domain features for nyc_taxi_tip, written from the schema and description only."""
import numpy as np
import pandas as pd

# Green taxi trips, December 2016 (Dec 1 2016 was a Thursday). Tips scale with fare, which scales with time/distance.


def _num(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def _duration(df):
    pu = _num(df, "lpep_pickup_datetime_day") * 1440 + _num(df, "lpep_pickup_datetime_hour") * 60 + \
        _num(df, "lpep_pickup_datetime_minute")
    do = _num(df, "lpep_dropoff_datetime_day") * 1440 + _num(df, "lpep_dropoff_datetime_hour") * 60 + \
        _num(df, "lpep_dropoff_datetime_minute")
    d = do - pu
    d = d.where(d >= -1, d + 31 * 1440)  # month wrap (Dec 31 -> Jan 1)
    return d.where(d.between(0, 600))


def trip_duration(df):
    # trip minutes ~ metered fare; the tip is roughly a percentage of the fare
    return _duration(df)


def route_stats(df):
    # route typical duration and popularity: PU-DO pair is a proxy for distance (fare) and airport/Manhattan trips
    pair = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    dur = _duration(df)
    return pd.DataFrame({"route_count": pair.groupby(pair).transform("size").astype(float),
                         "route_mean_dur": dur.groupby(pair).transform("mean"),
                         "dur_minus_route": dur - dur.groupby(pair).transform("mean"),
                         "same_zone": (df["PULocationID"].astype(str) == df["DOLocationID"].astype(str)).astype(float)},
                        index=df.index)


def zone_popularity(df):
    # pickup/dropoff zone volume: busy commercial zones vs outer-borough residential zones tip differently
    pu = df["PULocationID"].astype(str)
    do = df["DOLocationID"].astype(str)
    return pd.DataFrame({"pu_count": pu.groupby(pu).transform("size").astype(float),
                         "do_count": do.groupby(do).transform("size").astype(float),
                         "do_mean_dur": _duration(df).groupby(do).transform("mean")}, index=df.index)


def calendar(df):
    # weekday, weekend, night and holiday-season effects on tipping behaviour
    day = _num(df, "lpep_pickup_datetime_day")
    hour = _num(df, "lpep_pickup_datetime_hour")
    dow = (day - 1 + 3) % 7  # Monday=0; Dec 1 2016 = Thursday
    return pd.DataFrame({"dow": dow, "weekend": (dow >= 5).astype(float),
                         "night": ((hour >= 20) | (hour < 6)).astype(float),
                         "holiday": day.isin([24, 25, 26, 31]).astype(float),
                         "hour_frac": hour + _num(df, "lpep_pickup_datetime_minute") / 60.0}, index=df.index)


def surcharges_total(df):
    # fixed surcharges + tolls: tolls mean bridge/airport trips (long, higher fare)
    total = _num(df, "extra").fillna(0) + _num(df, "mta_tax").fillna(0) + \
        _num(df, "improvement_surcharge").fillna(0) + _num(df, "tolls_amount").fillna(0)
    return pd.DataFrame({"surcharges_total": total, "has_toll": (_num(df, "tolls_amount") > 0).astype(float)},
                        index=df.index)


def duration_per_passenger(df):
    # long group trips (shared fare) vs long solo trips
    return _duration(df) / _num(df, "passenger_count").clip(lower=1)


PROPOSALS = [
    ("trip_duration", trip_duration),
    ("route_stats", route_stats),
    ("zone_popularity", zone_popularity),
    ("surcharges_total", surcharges_total),
    ("calendar", calendar),
    ("duration_per_passenger", duration_per_passenger),
]
