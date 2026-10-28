"""Domain features for spambase, written from the schema and description only.

Column order follows the UCI Spambase layout: 0-47 word frequencies (make, address, all, 3d, our, over,
remove, internet, order, mail, receive, will, people, report, addresses, free, business, email, you,
credit, your, font, 000, money, hp, hpl, george, 650, lab, labs, telnet, 857, data, 415, 85, technology,
1999, parts, pm, direct, cs, meeting, original, project, re, edu, table, conference), 48-53 char
frequencies (; ( [ ! $ #), 54-56 capital-run stats (average, longest, total)."""
import numpy as np
import pandas as pd


def _c(df, i):
    return pd.to_numeric(df[str(i)] if str(i) in df.columns else df[i], errors="coerce")


def _sum(df, idx):
    return sum(_c(df, i) for i in idx)


def spam_word_score(df):
    # total frequency of classic spam vocabulary: remove, free, money, 000, credit, business, order, receive, $, !
    return _sum(df, [6, 15, 23, 22, 19, 16, 8, 10, 7, 39]) + _c(df, 52) + _c(df, 51)


def ham_word_score(df):
    # total frequency of words typical of the HP-labs recipient's genuine mail (hp, hpl, george, lab, edu, meeting...)
    return _sum(df, [24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 40, 41, 43, 44, 45, 47])


def spam_ham_ratio(df):
    # log-odds-like contrast between spammy and hammy vocabulary
    return np.log1p(spam_word_score(df)) - np.log1p(ham_word_score(df))


def caps_total_per_avg(df):
    # number of capital runs (total / average length): shouting spread across the email
    return _c(df, 56) / _c(df, 54).replace(0, np.nan)


def caps_longest_share(df):
    # longest capital run relative to total capitals: one SHOUTED headline vs scattered acronyms
    return _c(df, 55) / _c(df, 56).replace(0, np.nan)


def nonzero_words(df):
    # vocabulary breadth over the 48 tracked words
    return sum((_c(df, i) > 0).astype(float) for i in range(48))


def money_punct(df):
    # '$' and '!' together are a hallmark of commercial spam
    return _c(df, 52) * _c(df, 51)


def you_your(df):
    # second-person addressing (you + your) typical of marketing copy
    return _c(df, 18) + _c(df, 20)


PROPOSALS = [("spam_ham_ratio", spam_ham_ratio), ("spam_word_score", spam_word_score), ("ham_word_score", ham_word_score),
             ("caps_total_per_avg", caps_total_per_avg), ("money_punct", money_punct), ("caps_longest_share", caps_longest_share),
             ("nonzero_words", nonzero_words), ("you_your", you_your)]
