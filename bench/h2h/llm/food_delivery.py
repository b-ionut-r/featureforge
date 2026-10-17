"""Domain features for food_delivery, written from the schema and description only."""
import numpy as np
import pandas as pd


def _coords(df):
    rl = df["Restaurant_latitude"].astype(float).abs()
    ro = df["Restaurant_longitude"].astype(float).abs()
    dl = df["Delivery_location_latitude"].astype(float).abs()
    do = df["Delivery_location_longitude"].astype(float).abs()
    return rl, ro, dl, do


def haversine_km(df):
    # great-circle distance restaurant -> customer: travel time scales with distance
    rl, ro, dl, do = _coords(df)
    lat1, lon1, lat2, lon2 = map(np.radians, [rl, ro, dl, do])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return pd.Series(2 * 6371.0 * np.arcsin(np.sqrt(a.clip(0, 1))), index=df.index, dtype=float)


def manhattan_km(df):
    # grid-street distance approximation in km
    rl, ro, dl, do = _coords(df)
    return (dl - rl).abs() * 111.0 + (do - ro).abs() * 111.0 * np.cos(np.radians(dl))


def _id_parts(df):
    s = df["Delivery_person_ID"].astype(str).str.strip()
    city = s.str.extract(r"^([A-Z]+?)RES")[0]
    rest = pd.to_numeric(s.str.extract(r"RES0*(\d+)DEL")[0], errors="coerce")
    dlv = pd.to_numeric(s.str.extract(r"DEL0*(\d+)$")[0], errors="coerce")
    return city, rest, dlv


def city_code(df):
    # city parsed from the courier id (e.g. RANCHI, VAD, JAP): city traffic sets baseline time
    city, _, _ = _id_parts(df)
    codes = {c: i for i, c in enumerate(sorted(city.dropna().unique()))}
    return city.map(codes).astype(float)


def courier_id_number(df):
    # restaurant number and courier slot parsed from the id
    _, rest, dlv = _id_parts(df)
    return pd.DataFrame({"id_rest_no": rest, "id_del_no": dlv}, index=df.index).astype(float)


def courier_order_count(df):
    # how many orders this courier handled: busy couriers are experienced or overloaded
    k = df["Delivery_person_ID"].astype(str)
    return k.map(k.value_counts()).astype(float)


def rating_vs_city(df):
    # courier rating relative to couriers in the same city
    city, _, _ = _id_parts(df)
    r = df["Delivery_person_Ratings"].astype(float)
    return r - r.groupby(city.fillna("?")).transform("mean")


def dist_per_vehicle(df):
    # distance scaled by vehicle speed class (bicycle slow, motorcycle fast)
    speed = df["Type_of_vehicle"].astype(str).str.strip().map(
        {"motorcycle": 1.0, "scooter": 0.85, "electric_scooter": 0.8, "bicycle": 0.5}).fillna(0.85)
    return haversine_km(df) / speed


def restaurant_order_count(df):
    # orders from the same restaurant location: busy kitchens delay pickup
    k = df["Restaurant_latitude"].round(4).astype(str) + "," + df["Restaurant_longitude"].round(4).astype(str)
    return k.map(k.value_counts()).astype(float)


def low_rating_x_age(df):
    # older couriers with poor ratings are slower; ratings and age jointly proxy reliability
    return (5.0 - df["Delivery_person_Ratings"].astype(float)) * df["Delivery_person_Age"].astype(float)


PROPOSALS = [
    ("haversine_km", haversine_km),
    ("city_code", city_code),
    ("dist_per_vehicle", dist_per_vehicle),
    ("rating_vs_city", rating_vs_city),
    ("courier_order_count", courier_order_count),
    ("manhattan_km", manhattan_km),
    ("courier_id_number", courier_id_number),
    ("restaurant_order_count", restaurant_order_count),
    ("low_rating_x_age", low_rating_x_age),
]
