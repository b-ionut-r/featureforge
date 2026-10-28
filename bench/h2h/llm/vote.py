"""Domain features for vote, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 1984 US House roll calls; the "conservative/Reagan-administration" position on each bill
# (y = support) is taken from general knowledge of these votes, not from the data.
CONS_YES = ["physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "education-spending",
            "superfund-right-to-sue", "crime"]
CONS_NO = ["adoption-of-the-budget-resolution", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile",
           "duty-free-exports", "handicapped-infants", "synfuels-corporation-cutback"]
ALL = CONS_YES + CONS_NO + ["water-project-cost-sharing", "immigration", "export-administration-act-south-africa"]


def _yn(df, c):
    return df[c].astype(object).map({"y": 1.0, "n": 0.0}).astype(float)


def vote_counts(df):
    # overall voting activity: number of yes, no and abstentions (absences cluster by party/role)
    v = pd.DataFrame({c: _yn(df, c) for c in ALL})
    return pd.DataFrame({"n_yes": (v == 1).sum(axis=1).astype(float), "n_no": (v == 0).sum(axis=1).astype(float),
                         "n_missing": v.isna().sum(axis=1).astype(float)})


def conservative_score(df):
    # agreement with the conservative position across partisan bills (missing treated as neutral 0.5)
    # OUTSIDE KNOWLEDGE: party-line direction of each 1984 vote
    s = sum(_yn(df, c).fillna(0.5) for c in CONS_YES) + sum((1 - _yn(df, c)).fillna(0.5) for c in CONS_NO)
    return s / float(len(CONS_YES) + len(CONS_NO))


def defense_hawk(df):
    # foreign-policy / defence block: Contras aid, MX missile, El Salvador aid, against satellite test ban
    s = (_yn(df, "aid-to-nicaraguan-contras").fillna(0.5) + _yn(df, "mx-missile").fillna(0.5)
         + _yn(df, "el-salvador-aid").fillna(0.5) + (1 - _yn(df, "anti-satellite-test-ban")).fillna(0.5))
    return s / 4.0


def fiscal_social(df):
    # domestic block: physician fee freeze and budget resolution (fiscal), religion in schools/crime (social)
    fiscal = _yn(df, "physician-fee-freeze").fillna(0.5) - _yn(df, "adoption-of-the-budget-resolution").fillna(0.5)
    social = (_yn(df, "religious-groups-in-schools").fillna(0.5) + _yn(df, "crime").fillna(0.5)) / 2.0
    return pd.DataFrame({"fiscal_cons": fiscal, "social_cons": social})


def key_votes(df):
    # physician fee freeze was the single most party-line vote; contras aid second
    return pd.DataFrame({"fee_freeze": _yn(df, "physician-fee-freeze"),
                         "fee_freeze_x_budget": _yn(df, "physician-fee-freeze").fillna(0.5)
                         * (1 - _yn(df, "adoption-of-the-budget-resolution").fillna(0.5))})


def consistency(df):
    # how strongly the member's votes line up with either bloc (|score-0.5|): defectors / moderates vs loyalists
    return (conservative_score(df) - 0.5).abs()


PROPOSALS = [("vote_counts", vote_counts), ("conservative_score", conservative_score), ("defense_hawk", defense_hawk),
             ("fiscal_social", fiscal_social), ("key_votes", key_votes), ("consistency", consistency)]
