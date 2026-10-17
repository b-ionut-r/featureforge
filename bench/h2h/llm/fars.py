"""Domain features for fars, written from the schema and description only."""
import numpy as np
import pandas as pd

DRUG_RES = ["DRUG_TEST_RESULTS_(1_of_3)", "DRUG_TEST_RESULTS_(2_of_3)", "DRUG_TEST_RESULTS_(3_of_3)"]
DRUG_TYPE = ["DRUG_TEST_TYPE", "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_TYPE_(3_of_3)"]


def bac_value(df):
    # alcohol test result as BAC (hundredths); codes >= 95 are not-tested/unknown placeholders
    v = df["ALCOHOL_TEST_RESULT"].astype(float)
    return v.where(v < 95)


def drug_positive_count(df):
    # number of drug tests with a substance found (codes 0/1 = none/negative, 995+ = unknown)
    r = df[DRUG_RES].astype(float)
    return ((r > 1) & (r < 995)).sum(axis=1).astype(float)


def role_seat_combo_freq(df):
    # how common this person-type / seat combination is: unusual positions (cargo, outside) are deadly
    k = df["PERSON_TYPE"].astype(int) * 1000 + df["SEATING_POSITION"].astype(int)
    return k.map(k.value_counts()).astype(float)


def restraint_airbag_combo_freq(df):
    # frequency of the restraint/airbag combination: protection level is the key survival factor
    k = df["RESTRAINT_SYSTEM-USE"].astype(int) * 100 + df["AIR_BAG_AVAILABILITY/DEPLOYMENT"].astype(int)
    return k.map(k.value_counts()).astype(float)


def age_risk(df):
    # frailty: injury severity rises steeply with age; codes >= 99 mean unknown
    a = df["AGE"].astype(float).where(lambda x: x < 99)
    return pd.DataFrame({"age_clean": a, "age_sq_over_100": a ** 2 / 100.0}, index=df.index)


def age_vs_person_type(df):
    # age relative to others in the same role (driver, passenger, pedestrian)
    a = df["AGE"].astype(float).where(lambda x: x < 99)
    return a - a.groupby(df["PERSON_TYPE"]).transform("mean")


def ejected_any(df):
    # any ejection path recorded combined with ejection code
    return ((df["EJECTION"].astype(float) > 0).astype(float)
            + (df["EJECTION_PATH"].astype(float) > 0).astype(float))


def n_drug_tests(df):
    # number of drug tests administered: tests are typically done on fatalities
    return (df[DRUG_TYPE].astype(float) > 0).sum(axis=1).astype(float)


def state_count(df):
    # crash-volume of the state in the data (rural vs urban states differ in severity)
    return df.groupby("CASE_STATE")["CASE_STATE"].transform("size").astype(float)


PROPOSALS = [
    ("role_seat_combo_freq", role_seat_combo_freq),
    ("restraint_airbag_combo_freq", restraint_airbag_combo_freq),
    ("n_drug_tests", n_drug_tests),
    ("bac_value", bac_value),
    ("age_risk", age_risk),
    ("drug_positive_count", drug_positive_count),
    ("ejected_any", ejected_any),
    ("age_vs_person_type", age_vs_person_type),
    ("state_count", state_count),
]
