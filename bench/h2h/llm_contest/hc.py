"""Domain features for hc, written from the schema and description only."""
import numpy as np
import pandas as pd


# Affordability of the loan: annuity burden on income, loan size vs income, implied loan term.
# OUTSIDE KNOWLEDGE: credit/annuity (approximate term in months/years) is a known strong Home Credit feature.
def loan_affordability(df):
    inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    cred = df["AMT_CREDIT"].replace(0, np.nan)
    ann = df["AMT_ANNUITY"].replace(0, np.nan)
    goods = df["AMT_GOODS_PRICE"].replace(0, np.nan)
    fam = df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["annuity_to_income"] = df["AMT_ANNUITY"] / inc
    out["credit_to_income"] = df["AMT_CREDIT"] / inc
    out["credit_to_annuity"] = df["AMT_CREDIT"] / ann
    out["credit_to_goods"] = df["AMT_CREDIT"] / goods
    out["goods_minus_credit"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    out["income_per_person"] = df["AMT_INCOME_TOTAL"] / fam
    out["annuity_per_person_income"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / fam).replace(0, np.nan)
    out["goods_to_income"] = df["AMT_GOODS_PRICE"] / inc
    return out


# External credit scores are the strongest risk signals; combine them (mean, product, weighted, spread).
# OUTSIDE KNOWLEDGE: EXT_SOURCE mean/product/weighted sums were top features in the Home Credit competition.
def ext_source_combo(df):
    e = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out = pd.DataFrame(index=df.index)
    out["ext_mean"] = e.mean(axis=1)
    out["ext_min"] = e.min(axis=1)
    out["ext_max"] = e.max(axis=1)
    out["ext_std"] = e.std(axis=1)
    out["ext_prod"] = e["EXT_SOURCE_1"] * e["EXT_SOURCE_2"] * e["EXT_SOURCE_3"]
    out["ext_weighted"] = (2 * e["EXT_SOURCE_1"].fillna(e.mean(axis=1))
                           + 3 * e["EXT_SOURCE_2"].fillna(e.mean(axis=1))
                           + 4 * e["EXT_SOURCE_3"].fillna(e.mean(axis=1))) / 9.0
    out["ext_n_missing"] = e.isna().sum(axis=1)
    out["ext_mean_x_credit_term"] = out["ext_mean"] * (df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan))
    return out


# Life-stage and stability: age, employment length (365243 is a sentinel for not employed), recency of
# registration/ID/phone changes relative to age.
# OUTSIDE KNOWLEDGE: DAYS_EMPLOYED == 365243 is a known placeholder (pensioners/unemployed).
def time_stability(df):
    age = -df["DAYS_BIRTH"]
    emp_raw = df["DAYS_EMPLOYED"]
    sentinel = (emp_raw == 365243)
    emp = (-emp_raw).where(~sentinel, np.nan)
    agep = age.replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["age_years"] = age / 365.25
    out["employed_years"] = emp / 365.25
    out["employed_sentinel"] = sentinel.astype(float)
    out["employed_to_age"] = emp / agep
    out["registration_to_age"] = (-df["DAYS_REGISTRATION"]) / agep
    out["id_publish_to_age"] = (-df["DAYS_ID_PUBLISH"]) / agep
    out["phone_change_to_age"] = (-df["DAYS_LAST_PHONE_CHANGE"]) / agep
    out["phone_change_to_employed"] = (-df["DAYS_LAST_PHONE_CHANGE"]) / emp.replace(0, np.nan)
    out["car_age_to_age"] = df["OWN_CAR_AGE"] / (age / 365.25).replace(0, np.nan)
    out["age_at_employment_start"] = (age - emp) / 365.25
    out["credit_term_end_age"] = out["age_years"] + (df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)) / 12.0
    return out


# Documents provided, contact channels, and address mismatches as counts over related flag groups.
def flag_counts(df):
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    contact = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    mism = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]
    out = pd.DataFrame(index=df.index)
    out["n_documents"] = df[doc_cols].sum(axis=1)
    out["n_documents_excl3"] = df[[c for c in doc_cols if c != "FLAG_DOCUMENT_3"]].sum(axis=1)
    out["n_contact"] = df[contact].sum(axis=1)
    out["n_addr_mismatch"] = df[mism].sum(axis=1)
    out["n_city_mismatch"] = df[mism[3:]].sum(axis=1)
    return out


# Credit bureau enquiry intensity: total enquiries and the share that happened recently (credit hunger).
def bureau_enquiries(df):
    cols = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
    tot = df[cols].sum(axis=1, min_count=1)
    recent = df[cols[:4]].sum(axis=1, min_count=1)
    out = pd.DataFrame(index=df.index)
    out["req_total"] = tot
    out["req_recent_month"] = recent
    out["req_recent_share"] = recent / tot.replace(0, np.nan)
    out["req_missing"] = df[cols].isna().all(axis=1).astype(float)
    return out


# Social circle default rates: share of observed acquaintances who defaulted.
def social_circle(df):
    out = pd.DataFrame(index=df.index)
    out["def30_rate"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / df["OBS_30_CNT_SOCIAL_CIRCLE"].replace(0, np.nan)
    out["def60_rate"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / df["OBS_60_CNT_SOCIAL_CIRCLE"].replace(0, np.nan)
    out["def_any"] = ((df["DEF_30_CNT_SOCIAL_CIRCLE"] + df["DEF_60_CNT_SOCIAL_CIRCLE"]) > 0).astype(float)
    return out


# Ordered socio-economic levels: education rank, family, children share, region ratings, ownership.
def socio_ordinal(df):
    edu_map = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
               "Higher education": 3, "Academic degree": 4}
    out = pd.DataFrame(index=df.index)
    out["education_rank"] = df["NAME_EDUCATION_TYPE"].astype(str).map(edu_map).astype(float)
    out["children_share"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    out["adults"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
    out["region_rating_sum"] = df["REGION_RATING_CLIENT"] + df["REGION_RATING_CLIENT_W_CITY"]
    out["owns_car_and_realty"] = ((df["FLAG_OWN_CAR"].astype(str) == "Y").astype(int)
                                  + (df["FLAG_OWN_REALTY"].astype(str) == "Y").astype(int))
    out["is_male"] = (df["CODE_GENDER"].astype(str) == "M").astype(float)
    out["revolving"] = (df["NAME_CONTRACT_TYPE"].astype(str) == "Revolving loans").astype(float)
    return out


# Income relative to peers: applicant income and credit vs the median of their occupation/organisation
# (label-free group statistics over train+test).
def peer_relative(df):
    out = pd.DataFrame(index=df.index)
    inc = df["AMT_INCOME_TOTAL"]
    for g in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE"]:
        key = df[g].astype(str)
        med = inc.groupby(key).transform("median").replace(0, np.nan)
        out["income_vs_" + g] = inc / med
    key = df["NAME_EDUCATION_TYPE"].astype(str) + "_" + df["OCCUPATION_TYPE"].astype(str)
    ext2 = df["EXT_SOURCE_2"]
    out["ext2_vs_edu_occ"] = ext2 - ext2.groupby(key).transform("mean")
    cr = df["AMT_CREDIT"] / inc.replace(0, np.nan)
    out["credit_income_vs_org"] = cr / cr.groupby(df["ORGANIZATION_TYPE"].astype(str)).transform("median").replace(0, np.nan)
    return out


# Housing quality summary over the building-description block (avg columns) and its missingness.
def housing_summary(df):
    avg_cols = [c for c in df.columns if c.endswith("_AVG")]
    out = pd.DataFrame(index=df.index)
    out["housing_n_missing"] = df[avg_cols].isna().sum(axis=1)
    out["housing_avg_mean"] = df[avg_cols].mean(axis=1)
    out["living_to_total_area"] = df["LIVINGAREA_AVG"] / df["TOTALAREA_MODE"].replace(0, np.nan)
    out["area_per_apartment"] = df["TOTALAREA_MODE"] / df["APARTMENTS_AVG"].replace(0, np.nan)
    return out


PROPOSALS = [
    ("loan_affordability", loan_affordability),
    ("ext_source_combo", ext_source_combo),
    ("time_stability", time_stability),
    ("flag_counts", flag_counts),
    ("bureau_enquiries", bureau_enquiries),
    ("social_circle", social_circle),
    ("socio_ordinal", socio_ordinal),
    ("peer_relative", peer_relative),
    ("housing_summary", housing_summary),
]
