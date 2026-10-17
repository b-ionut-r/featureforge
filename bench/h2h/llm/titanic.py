"""Domain features for titanic, written from the schema and description only."""
import numpy as np
import pandas as pd


def _ticket(df):
    return df["Ticket"].fillna("NA").astype(str).str.strip()


def fare_per_person(df):
    # Fare is per ticket, shared by everyone on it; per-person fare is the true wealth/class signal
    t = _ticket(df)
    n = t.groupby(t).transform("size")
    return pd.to_numeric(df["Fare"], errors="coerce") / n


def family_size(df):
    # family size: singles and very large families survived less, small families more
    fs = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return pd.DataFrame({"family_size": fs.astype(float), "is_alone": (fs == 1).astype(float),
                         "large_family": (fs >= 5).astype(float)})


def ticket_group_size(df):
    # passengers travelling on the same ticket (friends/servants included), beyond the declared family
    t = _ticket(df)
    return t.groupby(t).transform("size").astype(float)


def cabin_deck(df):
    # deck letter: upper decks (A-E) were closer to the boats; missing cabin itself flags 3rd class / crew records
    c = df["Cabin"].fillna("").astype(str)
    deck = c.str[:1].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 0})
    return pd.DataFrame({"deck": deck.astype(float), "has_cabin": (c != "").astype(float),
                         "n_cabins": c.str.split().str.len().where(c != "", 0).astype(float)})


def sex_class(df):
    # "women and children first", strongly modulated by class: combined sex x class code
    female = (df["Sex"].astype(str).str.lower() == "female").astype(float)
    return female * 10 + pd.to_numeric(df["Pclass"], errors="coerce")


def child_flag(df):
    # children were prioritised for lifeboats; Parch>0 & missing age is often a child
    age = pd.to_numeric(df["Age"], errors="coerce")
    return pd.DataFrame({"is_child": (age < 14).astype(float).where(age.notna()),
                         "age_x_class": age * pd.to_numeric(df["Pclass"], errors="coerce")})


def ticket_prefix(df):
    # ticket prefix (PC, CA, SOTON, STON...) encodes booking office/line; numeric part encodes issue series
    t = _ticket(df)
    pre = t.str.replace(r"[./\s]", "", regex=True).str.extract(r"^([A-Za-z]+)", expand=False).fillna("NONE").str.upper()
    num = pd.to_numeric(t.str.extract(r"(\d+)$", expand=False), errors="coerce")
    return pd.DataFrame({"prefix_code": pre.map({"NONE": 0, "PC": 1, "CA": 2, "A": 3, "SOTONOQ": 4, "STONO": 5,
                                                 "WC": 6, "SCPARIS": 7, "SOC": 8, "FCC": 9, "LINE": 10}).fillna(11).astype(float),
                         "ticket_num_len": num.astype(str).str.len().where(num.notna()).astype(float)})


def fare_rel_class(df):
    # fare relative to class median per person: luxury within class
    fpp = fare_per_person(df)
    return fpp / fpp.groupby(df["Pclass"]).transform("median").replace(0, np.nan)


PROPOSALS = [("sex_class", sex_class), ("fare_per_person", fare_per_person), ("family_size", family_size),
             ("ticket_group_size", ticket_group_size), ("cabin_deck", cabin_deck), ("child_flag", child_flag),
             ("ticket_prefix", ticket_prefix), ("fare_rel_class", fare_rel_class)]
