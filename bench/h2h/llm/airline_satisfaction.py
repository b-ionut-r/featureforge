"""Domain features for airline_satisfaction, written from the schema and description only."""
import numpy as np
import pandas as pd

RATINGS = ["Seatcomfort", "DepartureArrivaltimeconvenient", "Foodanddrink", "Gatelocation", "Inflightwifiservice",
           "Inflightentertainment", "Onlinesupport", "EaseofOnlinebooking", "Onboardservice", "Legroomservice",
           "Baggagehandling", "Checkinservice", "Cleanliness", "Onlineboarding"]
INFLIGHT = ["Seatcomfort", "Foodanddrink", "Inflightentertainment", "Onboardservice", "Legroomservice", "Cleanliness"]
ONLINE = ["Inflightwifiservice", "Onlinesupport", "EaseofOnlinebooking", "Onlineboarding"]


def _r(df, cols):
    # 0 means "not applicable / not rated" so treat it as missing
    return df[cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)


def mean_rating(df):
    # overall satisfaction is driven by the average service experience
    return _r(df, RATINGS).mean(axis=1)


def inflight_mean(df):
    # in-flight comfort (seat, entertainment, food, legroom) is what passengers weigh most
    return _r(df, INFLIGHT).mean(axis=1)


def online_mean(df):
    # digital touchpoints form a separate experience factor
    return _r(df, ONLINE).mean(axis=1)


def min_rating(df):
    # one terrible aspect can sink overall satisfaction (peak-end / weakest-link effect)
    return _r(df, RATINGS).min(axis=1)


def n_top_ratings(df):
    # count of 5-star aspects: delighters drive "satisfied" answers
    return (df[RATINGS].apply(pd.to_numeric, errors="coerce") == 5).sum(axis=1).astype(float)


def n_zero_ratings(df):
    # number of skipped/N.A. aspects reflects engagement and a distinct response style
    return (df[RATINGS].apply(pd.to_numeric, errors="coerce") == 0).sum(axis=1).astype(float)


def rating_std(df):
    # inconsistent ratings indicate a mixed experience vs. uniformly happy/unhappy passengers
    return _r(df, RATINGS).std(axis=1)


def total_delay_per_km(df):
    # a delay hurts more on a short flight than on a long haul
    d = df["DepartureDelayinMinutes"].astype(float) + df["ArrivalDelayinMinutes"].astype(float).fillna(0)
    return d / df["FlightDistance"].astype(float).replace(0, np.nan)


def delay_recovered(df):
    # time made up in the air (departure minus arrival delay) softens the delay experience
    return df["DepartureDelayinMinutes"].astype(float) - df["ArrivalDelayinMinutes"].astype(float)


PROPOSALS = [("inflight_mean", inflight_mean), ("mean_rating", mean_rating), ("n_top_ratings", n_top_ratings),
             ("min_rating", min_rating), ("online_mean", online_mean), ("n_zero_ratings", n_zero_ratings),
             ("rating_std", rating_std), ("total_delay_per_km", total_delay_per_km),
             ("delay_recovered", delay_recovered)]
