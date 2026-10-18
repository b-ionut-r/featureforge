"""Domain features for ieee, written from the schema and description only."""
import numpy as np
import pandas as pd


def _key(df, cols):
    # Combine columns into one integer group id (NaN kept as its own level).
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "_" + df[c].astype(str)
    return pd.Series(pd.factorize(s)[0], index=df.index)


def _uid(df):
    # Card holder proxy: card1 + addr1 + the day the account/card was first seen (day - D1).
    day = np.floor(df["TransactionDT"] / 86400.0)
    d1n = (day - df["D1"]).astype(str)
    s = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + d1n
    return pd.Series(pd.factorize(s)[0], index=df.index)


def _num(s):
    # Numeric view of a column: numbers as float, anything else (str/category) factorized to codes.
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    codes = pd.Series(pd.factorize(s.astype(str))[0], index=s.index).astype(float)
    return codes.where(s.notna(), np.nan)


# Calendar position of the transaction: hour of day and day of week (fraud concentrates at odd hours).
def time_of_day(df):
    t = df["TransactionDT"]
    out = pd.DataFrame(index=df.index)
    out["hour"] = np.floor(t / 3600.0) % 24
    out["weekday"] = np.floor(t / 86400.0) % 7
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


# Amount shape: log, cents part, number of decimals (foreign-currency conversions produce odd decimals).
def amount_shape(df):
    a = df["TransactionAmt"]
    out = pd.DataFrame(index=df.index)
    out["amt_log"] = np.log1p(a.clip(lower=0))
    cents = np.round((a - np.floor(a)) * 1000).astype(float)
    out["amt_cents"] = cents / 1000.0
    out["amt_is_round"] = (cents == 0).astype(float)
    s = a.round(3)
    out["amt_n_decimals"] = ((np.round(s * 1000) % 10 != 0).astype(int)
                             + (np.round(s * 1000) % 100 != 0).astype(int)
                             + (np.round(s * 1000) % 1000 != 0).astype(int)).astype(float)
    return out


# D columns are day counters that grow with time; subtracting the transaction day turns them into
# fixed "anchor dates" per card/account.
# OUTSIDE KNOWLEDGE: day - D1/D10/D15 normalisation was a key trick in the IEEE-CIS winning solution.
def d_normalised(df):
    day = np.floor(df["TransactionDT"] / 86400.0)
    out = pd.DataFrame(index=df.index)
    for c in ["D1", "D2", "D3", "D4", "D10", "D11", "D15"]:
        out[c + "_anchor"] = day - df[c]
    out["D1_minus_D15"] = df["D1"] - df["D15"]
    out["D4_minus_D10"] = df["D4"] - df["D10"]
    return out


# Card holder identity: transactions grouped by the reconstructed client id, with amount and
# behaviour statistics (label-free aggregations over train+test).
# OUTSIDE KNOWLEDGE: uid = card1 + addr1 + (day - D1) is the known client decoding from the IEEE-CIS winners.
def client_uid_aggs(df):
    uid = _uid(df)
    a = df["TransactionAmt"]
    g = a.groupby(uid)
    out = pd.DataFrame(index=df.index)
    out["uid_count"] = uid.map(uid.value_counts()).astype(float)
    out["uid_amt_mean"] = g.transform("mean")
    out["uid_amt_std"] = g.transform("std")
    out["uid_amt_rel"] = a / out["uid_amt_mean"].replace(0, np.nan)
    for c in ["P_emaildomain", "dist1", "id_19", "id_20", "id_13", "DeviceInfo", "C13", "V307"]:
        out["uid_nunique_" + c] = _num(df[c]).groupby(uid).transform("nunique").astype(float)
    m4 = df["M4"].astype(str).map({"M0": 0.0, "M1": 1.0, "M2": 2.0})
    for name, v in [("C1", _num(df["C1"])), ("C13", _num(df["C13"])), ("C14", _num(df["C14"])),
                    ("D15", _num(df["D15"])), ("M4", m4.astype(float)), ("dist1", _num(df["dist1"]))]:
        out["uid_mean_" + name] = v.groupby(uid).transform("mean")
    dt = df["TransactionDT"].astype(float)
    out["uid_span_days"] = (dt.groupby(uid).transform("max") - dt.groupby(uid).transform("min")) / 86400.0
    return out


# Frequency encodings of card / address / email fields: rare cards and combinations signal fraud rings
# or unusual issuers.
def frequency_encodings(df):
    out = pd.DataFrame(index=df.index)
    for c in ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain",
              "DeviceInfo", "id_30", "id_31", "id_33", "dist1"]:
        k = df[c].astype(str)
        out["freq_" + c] = k.map(k.value_counts()).astype(float)
    for cols in [["card1", "addr1"], ["card1", "addr1", "P_emaildomain"], ["card1", "card2", "card3", "card5"],
                 ["card1", "ProductCD"]]:
        k = _key(df, cols)
        out["freq_" + "_".join(cols)] = k.map(k.value_counts()).astype(float)
    return out


# Amount relative to its card and product: how unusual this payment is for the card.
def amount_vs_card(df):
    a = df["TransactionAmt"]
    out = pd.DataFrame(index=df.index)
    for cols in [["card1"], ["card1", "addr1"], ["ProductCD"], ["card4", "card6"]]:
        k = _key(df, cols)
        m = a.groupby(k).transform("mean").replace(0, np.nan)
        s = a.groupby(k).transform("std").replace(0, np.nan)
        name = "_".join(cols)
        out["amt_div_mean_" + name] = a / m
        out["amt_z_" + name] = (a - m) / s
    return out


# Match flags (name/address checks) and email consistency between purchaser and recipient.
def match_and_email(df):
    out = pd.DataFrame(index=df.index)
    tf = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
    m = df[tf].astype(str)
    out["m_true"] = (m == "T").sum(axis=1)
    out["m_false"] = (m == "F").sum(axis=1)
    out["m_missing"] = df[tf].isna().sum(axis=1)
    out["M4_level"] = df["M4"].astype(str).map({"M0": 0, "M1": 1, "M2": 2}).astype(float)
    pe = df["P_emaildomain"].astype(str)
    re = df["R_emaildomain"].astype(str)
    out["email_same"] = ((pe == re) & df["P_emaildomain"].notna()).astype(float)
    out["email_p_missing"] = df["P_emaildomain"].isna().astype(float)
    out["email_anonymous"] = (pe.str.contains("anonymous") | re.str.contains("anonymous")).astype(float)
    out["email_p_tld_non_com"] = (df["P_emaildomain"].notna() & ~pe.str.endswith(".com")).astype(float)
    out["email_r_protonmail"] = re.str.contains("protonmail").astype(float)
    return out


# Device / identity hints: browser and OS families and versions, screen size, proxies, identity presence.
def device_identity(df):
    out = pd.DataFrame(index=df.index)
    b = df["id_31"].astype(str).str.lower()
    o = df["id_30"].astype(str).str.lower()
    out["has_identity"] = df["id_01"].notna().astype(float)
    out["browser_version"] = pd.to_numeric(b.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    out["browser_chrome"] = b.str.contains("chrome").astype(float)
    out["browser_generic"] = b.str.contains("generic").astype(float)
    out["os_windows"] = o.str.contains("windows").astype(float)
    out["os_ios"] = o.str.contains("ios|mac").astype(float)
    out["os_android"] = o.str.contains("android").astype(float)
    res = df["id_33"].astype(str).str.split("x", n=1, expand=True)
    if res.shape[1] == 2:
        w = pd.to_numeric(res[0], errors="coerce")
        h = pd.to_numeric(res[1], errors="coerce")
    else:
        w = pd.Series(np.nan, index=df.index)
        h = pd.Series(np.nan, index=df.index)
    out["screen_pixels"] = w * h
    out["screen_aspect"] = w / h.replace(0, np.nan)
    out["proxy"] = df["id_23"].notna().astype(float)
    out["mobile"] = (df["DeviceType"].astype(str) == "mobile").astype(float)
    out["id_missing_count"] = df[[c for c in df.columns if c.startswith("id_")]].isna().sum(axis=1)
    return out


# Count-feature summaries: totals and ratios over the C counters (addresses/cards linked to the payer).
def c_counts(df):
    cs = ["C%d" % i for i in range(1, 15)]
    out = pd.DataFrame(index=df.index)
    out["C_sum"] = df[cs].sum(axis=1)
    out["C_max"] = df[cs].max(axis=1)
    out["C_nonzero"] = (df[cs] > 0).sum(axis=1)
    out["C1_div_C13"] = df["C1"] / df["C13"].replace(0, np.nan)
    out["C14_div_C1"] = df["C14"] / df["C1"].replace(0, np.nan)
    out["C5_div_C1"] = df["C5"] / df["C1"].replace(0, np.nan)
    return out


PROPOSALS = [
    ("time_of_day", time_of_day),
    ("amount_shape", amount_shape),
    ("d_normalised", d_normalised),
    ("client_uid_aggs", client_uid_aggs),
    ("frequency_encodings", frequency_encodings),
    ("amount_vs_card", amount_vs_card),
    ("match_and_email", match_and_email),
    ("device_identity", device_identity),
    ("c_counts", c_counts),
]
