"""Domain features for t01, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the columns look like the US College Scorecard: f_6 admission rate, f_7..f_9 SAT section
# midpoints, f_10..f_12 ACT midpoints, f_14 SAT average, f_15 undergraduate enrollment, f_16..f_20 race shares,
# f_21/f_22 net price (public/private), f_23/f_24 in/out-of-state tuition, f_27/f_28 retention rates,
# f_38/f_39 shares (e.g. Pell, part-time), f_40 family income, f_41..f_44 earnings/debt-like dollar amounts.


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# Selectivity: combine SAT/ACT/admission into one academic-preparation score (ACT converted roughly to SAT scale).
def selectivity(df):
    sat = _num(df["f_14"])
    act = _num(df["f_10"])
    act_as_sat = 400 + (act - 1) * (1200 / 35.0)
    score = sat.fillna(act_as_sat)
    adm = _num(df["f_6"])
    return pd.DataFrame({
        "test_score_combined": score,
        "admit_rate_logit": np.log((adm.clip(0.001, 0.999)) / (1 - adm.clip(0.001, 0.999))),
        "has_test_scores": score.notna().astype(float),
    })


# Price structure: net price either sector, out-of-state premium, net price vs tuition (aid generosity).
def price_features(df):
    net = _num(df["f_21"]).fillna(_num(df["f_22"]))
    tin = _num(df["f_23"])
    tout = _num(df["f_24"])
    return pd.DataFrame({
        "net_price": net,
        "out_state_premium": tout - tin,
        "out_in_ratio": tout / tin.replace(0, np.nan),
        "net_to_tuition": net / tout.replace(0, np.nan),
        "log_net_price": np.log1p(net.clip(lower=0)),
    })


# Scale and resources: log enrollment, spending per student relative to tuition.
def scale_resources(df):
    ug = _num(df["f_15"])
    f25 = _num(df["f_25"])
    f26 = _num(df["f_26"])
    tin = _num(df["f_23"])
    return pd.DataFrame({
        "log_enroll": np.log1p(ug.clip(lower=0)),
        "log_f25": np.log1p(f25.clip(lower=0)),
        "f26_to_f25": f26 / f25.replace(0, np.nan),
        "f25_to_tuition": f25 / tin.replace(0, np.nan),
    })


# Student body composition: diversity (Herfindahl) and dominant group share over the race-share columns.
def composition(df):
    shares = df[["f_16", "f_17", "f_18", "f_19", "f_20"]].apply(_num)
    tot = shares.sum(axis=1)
    hhi = (shares ** 2).sum(axis=1)
    return pd.DataFrame({
        "race_hhi": hhi,
        "race_max_share": shares.max(axis=1),
        "race_other_share": (1 - tot).clip(lower=0),
    })


# Ordered credential levels: predominant and highest degree, plus control type and gender policy.
def institution_type(df):
    lvl = {"Non-degree": 0, "Certificate": 1, "Associate": 2, "Bachelors": 3, "Graduate": 4}
    pred = df["f_29"].astype(object).map(lvl).astype(float)
    high = df["f_30"].astype(object).map(lvl).astype(float)
    ctrl = df["f_31"].astype(object)
    return pd.DataFrame({
        "pred_degree": pred,
        "high_degree": high,
        "degree_gap": high - pred,
        "is_public": (ctrl == "Public").astype(float),
        "is_forprofit": (ctrl == "Private for-profit").astype(float),
        "single_sex": (df["f_33"].astype(object) != "COED").astype(float),
        "religious": df["f_37"].notna().astype(float),
    })


# Carnegie text decoded: research intensity, residential character, size, selectivity keywords.
def carnegie(df):
    c = df["f_34"].astype(object).fillna("").astype(str)
    p = df["f_35"].astype(object).fillna("").astype(str)
    s = df["f_36"].astype(object).fillna("").astype(str)
    research = np.where(c.str.contains("very high research", case=False), 3,
               np.where(c.str.contains("high research", case=False), 2,
               np.where(c.str.contains("Doctoral|Research", case=False), 1, 0)))
    resid = np.where(s.str.contains("highly residential", case=False), 2,
            np.where(s.str.contains("primarily residential", case=False), 1,
            np.where(s.str.contains("nonresidential", case=False), 0, np.nan)))
    sel = np.where(p.str.contains("more selective", case=False), 2,
          np.where(p.str.contains("selective", case=False), 1,
          np.where(p.str.contains("inclusive", case=False), 0, np.nan)))
    return pd.DataFrame({
        "research_level": research.astype(float),
        "arts_sciences": c.str.contains("Arts & Sciences", case=False).astype(float),
        "residential_level": resid.astype(float),
        "carnegie_selectivity": sel.astype(float),
        "full_time_profile": p.str.contains("Full-time|full-time", case=True).astype(float),
    })


# Earnings/debt-like dollar columns: relative growth and ratios to family income.
def outcome_money(df):
    a, b, c, d = (_num(df[k]) for k in ["f_41", "f_42", "f_43", "f_44"])
    inc = _num(df["f_40"])
    return pd.DataFrame({
        "f43_minus_f41": c - a,
        "f44_minus_f42": d - b,
        "f41_to_f42": a / b.replace(0, np.nan),
        "f43_to_f44": c / d.replace(0, np.nan),
        "f43_to_income": c / inc.replace(0, np.nan),
        "log_income": np.log1p(inc.clip(lower=0)),
    })


# Retention and student mix: average retention, Pell-like share times income.
def retention_mix(df):
    r1 = _num(df["f_27"])
    r2 = _num(df["f_28"])
    s38 = _num(df["f_38"])
    s39 = _num(df["f_39"])
    return pd.DataFrame({
        "retention_mean": pd.concat([r1, r2], axis=1).mean(axis=1),
        "retention_gap": r1 - r2,
        "f38_x_f39": s38 * s39,
        "f38_minus_f39": s38 - s39,
    })


# Location: state frequency and zip-code+4 indicator (large institutions often have zip+4).
def location(df):
    st = df["f_2"].astype(object)
    city = df["f_1"].astype(object).astype(str) + "|" + st.astype(str)
    return pd.DataFrame({
        "state_count": st.map(st.value_counts()).astype(float),
        "city_count": city.map(city.value_counts()).astype(float),
        "zip_plus4": df["f_3"].astype(str).str.contains("-").astype(float),
        "lat": _num(df["f_4"]),
        "lon": _num(df["f_5"]),
    })


PROPOSALS = [
    ("selectivity", selectivity),
    ("price_features", price_features),
    ("scale_resources", scale_resources),
    ("composition", composition),
    ("institution_type", institution_type),
    ("carnegie", carnegie),
    ("outcome_money", outcome_money),
    ("retention_mix", retention_mix),
    ("location", location),
]
