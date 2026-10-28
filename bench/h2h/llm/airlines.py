"""Domain features for airlines, written from the schema and description only."""
import numpy as np
import pandas as pd


def _route(df):
    return df["AirportFrom"].astype(str) + "_" + df["AirportTo"].astype(str)


def length_vs_route(df):
    # schedule padding: a flight scheduled longer than usual for its route has slack to absorb delays
    L = df["Length"].astype(float)
    return L - L.groupby(_route(df)).transform("median")


def dep_hour(df):
    # Time is minutes after midnight; delays cascade through the day so late departures are delayed more
    return (df["Time"].astype(float) // 60).clip(0, 23)


def arrival_time(df):
    # scheduled arrival minute of day; evening arrivals hit congested banks at hubs
    return (df["Time"].astype(float) + df["Length"].astype(float)) % 1440


def origin_hour_congestion(df):
    # number of departures from the same airport in the same hour: congestion causes ground delays
    h = (df["Time"].astype(float) // 60)
    return df.groupby([df["AirportFrom"].astype(str), h])["Time"].transform("size").astype(float)


def route_count(df):
    # busy trunk routes (shuttles) differ from thin regional routes in delay propensity
    r = _route(df)
    return r.groupby(r).transform("size").astype(float)


def airline_flight_legs(df):
    # same airline+flight number appearing many times = multi-leg/high-frequency rotation, delays propagate
    k = df["Airline"].astype(str) + "_" + df["Flight"].astype(str)
    return k.groupby(k).transform("size").astype(float)


def flight_number_rank_in_airline(df):
    # high flight numbers are typically regional/express partner flights; rank within carrier normalises across carriers
    f = pd.to_numeric(df["Flight"], errors="coerce")
    return f.groupby(df["Airline"].astype(str)).rank(pct=True)


def dest_count(df):
    # hub destinations (high traffic) get arrival metering and holding delays
    a = df["AirportTo"].astype(str)
    return a.groupby(a).transform("size").astype(float)


def airline_time_rel(df):
    # departure time relative to the carrier's typical schedule (late in that airline's rotation)
    t = df["Time"].astype(float)
    return t - t.groupby(df["Airline"].astype(str)).transform("mean")


PROPOSALS = [("dep_hour", dep_hour), ("length_vs_route", length_vs_route),
             ("origin_hour_congestion", origin_hour_congestion), ("airline_flight_legs", airline_flight_legs),
             ("arrival_time", arrival_time), ("flight_number_rank_in_airline", flight_number_rank_in_airline),
             ("route_count", route_count), ("dest_count", dest_count), ("airline_time_rel", airline_time_rel)]
