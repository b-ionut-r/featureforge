"""Domain features for medical_charges, written from the schema and description only."""
import numpy as np
import pandas as pd

# Medicare pays a DRG weight x hospital wage/teaching adjustment; hospitals bill
# covered charges = their own chargemaster markup x resource use. So payment ~ charges / hospital markup.


def _s(df, c):
    return df[c].astype(object).fillna("__nan__").astype(str)


def _logc(df):
    return np.log(pd.to_numeric(df["average_covered_charges"], errors="coerce").clip(lower=1))


def _drg_level(df):
    lc = _logc(df)
    return lc.groupby(_s(df, "drg_definition")).transform("median")


def provider_markup(df):
    # a hospital's chargemaster markup: its charges relative to the national median for the same DRG
    lc = _logc(df)
    resid = lc - _drg_level(df)
    p = _s(df, "provider_id")
    cnt = resid.groupby(p).transform("size")
    mk = resid.groupby(p).transform("mean")
    hrr = resid.groupby(_s(df, "hospital_referral_region_(hrr)_description")).transform("mean")
    # shrink providers with few rows toward their region
    mk_shrunk = (mk * cnt + hrr * 3) / (cnt + 3)
    return pd.DataFrame({"provider_markup": mk_shrunk, "hrr_markup": hrr,
                         "state_markup": resid.groupby(_s(df, "provider_state")).transform("mean")}, index=df.index)


def charges_net_of_markup(df):
    # charges deflated by the hospital's markup ~ resource use the payment formula actually reimburses
    lc = _logc(df)
    resid = lc - _drg_level(df)
    p = _s(df, "provider_id")
    cnt = resid.groupby(p).transform("size")
    mk = resid.groupby(p).transform("mean")
    hrr = resid.groupby(_s(df, "hospital_referral_region_(hrr)_description")).transform("mean")
    mk_shrunk = (mk * cnt + hrr * 3) / (cnt + 3)
    return lc - mk_shrunk


def drg_weight_proxy(df):
    # DRG relative weight proxy: national median (log) charges for the DRG orders DRGs by cost
    return _drg_level(df)


def drg_severity(df):
    # complication tier parsed from the DRG title: W MCC > W CC (or W CC/MCC) > W/O CC/MCC
    t = _s(df, "drg_definition").str.upper()
    sev = np.where(t.str.contains(r"W MCC"), 2.0,
                   np.where(t.str.contains(r"W CC"), 1.0, np.where(t.str.contains(r"W/O"), 0.0, np.nan)))
    code = pd.to_numeric(t.str.extract(r"^(\d+)")[0], errors="coerce")
    return pd.DataFrame({"drg_severity": sev, "drg_code": code.astype(float)}, index=df.index)


def provider_volume(df):
    # hospital size proxies: number of DRGs billed and total discharges; large teaching hospitals get add-ons
    p = _s(df, "provider_id")
    disc = pd.to_numeric(df["total_discharges"], errors="coerce")
    return pd.DataFrame({"prov_n_drgs": p.groupby(p).transform("size").astype(float),
                         "prov_total_disc": disc.groupby(p).transform("sum"),
                         "disc_vs_drg_mean": disc / disc.groupby(_s(df, "drg_definition")).transform("mean")},
                        index=df.index)


def charges_vs_drg(df):
    # this row's charges relative to the DRG's median: high outliers attract outlier payments
    return _logc(df) - _drg_level(df)


PROPOSALS = [
    ("charges_net_of_markup", charges_net_of_markup),
    ("drg_weight_proxy", drg_weight_proxy),
    ("provider_markup", provider_markup),
    ("charges_vs_drg", charges_vs_drg),
    ("drg_severity", drg_severity),
    ("provider_volume", provider_volume),
]
