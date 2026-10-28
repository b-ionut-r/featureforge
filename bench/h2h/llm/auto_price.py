"""Domain features for auto_price, written from the schema and description only."""
import numpy as np
import pandas as pd


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def power_to_weight(df):
    # performance: horsepower per curb weight (sports/luxury cars score high)
    return _n(df, "horsepower") / _n(df, "curb-weight").replace(0, np.nan)


def specific_output(df):
    # horsepower per unit engine size: engine tuning/technology level
    return _n(df, "horsepower") / _n(df, "engine-size").replace(0, np.nan)


def body_volume(df):
    # exterior box volume and footprint: overall car size class
    return pd.DataFrame({"box_volume": _n(df, "length") * _n(df, "width") * _n(df, "height") / 1e3,
                         "footprint": _n(df, "length") * _n(df, "width")})


def fuel_consumption(df):
    # fuel use per distance (1/mpg) is linear in vehicle mass/power; average of city & highway
    c, h = _n(df, "city-mpg").replace(0, np.nan), _n(df, "highway-mpg").replace(0, np.nan)
    return pd.DataFrame({"gal_per_100mi": 100 * (0.55 / c + 0.45 / h), "hwy_minus_city": h - c})


def bore_stroke(df):
    # bore/stroke ratio: oversquare engines rev higher (performance)
    return _n(df, "bore") / _n(df, "stroke").replace(0, np.nan)


def diesel_flag(df):
    # OUTSIDE KNOWLEDGE: diesel engines have compression ratios ~ >15 (fuel type not in table)
    return (_n(df, "compression-ratio") > 15).astype(float)


def weight_density(df):
    # curb weight per volume / per length: build quality & materials (heavier luxury builds)
    return pd.DataFrame({"wt_per_len": _n(df, "curb-weight") / _n(df, "length").replace(0, np.nan),
                         "wt_per_vol": _n(df, "curb-weight") / (_n(df, "length") * _n(df, "width") * _n(df, "height") / 1e3).replace(0, np.nan)})


def power_rpm(df):
    # torque proxy: hp / rpm (hp = torque*rpm/5252)
    # OUTSIDE KNOWLEDGE: hp = torque(lb-ft) * rpm / 5252
    return 5252.0 * _n(df, "horsepower") / _n(df, "peak-rpm").replace(0, np.nan)


def risk_vs_losses(df):
    # insurance risk symbol and normalized losses: combined risk index
    s = _n(df, "symboling")
    return pd.DataFrame({"symboling_num": s, "losses_x_risk": _n(df, "normalized-losses") * (s + 3)})


PROPOSALS = [("power_to_weight", power_to_weight), ("specific_output", specific_output), ("body_volume", body_volume),
             ("fuel_consumption", fuel_consumption), ("bore_stroke", bore_stroke), ("diesel_flag", diesel_flag),
             ("weight_density", weight_density), ("power_rpm", power_rpm), ("risk_vs_losses", risk_vs_losses)]
