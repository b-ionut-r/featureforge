"""Domain features for phishing, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: in the UCI phishing-websites features doc, each feature is coded 1 = legitimate,
# 0 = suspicious, -1 = phishing; groups below follow that doc's address-bar / abnormal / HTML-JS / domain split.
URL_COLS = ["having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", "double_slash_redirecting",
            "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length", "Favicon",
            "port", "HTTPS_token"]
ABNORMAL_COLS = ["Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL"]
HTMLJS_COLS = ["Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe"]
DOMAIN_COLS = ["age_of_domain", "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
               "Statistical_report"]
ALL = URL_COLS + ABNORMAL_COLS + HTMLJS_COLS + DOMAIN_COLS


def _X(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").astype(float)


def group_scores(df):
    # mean legitimacy score within each feature family: URL tricks, page content, scripts, domain reputation
    return pd.DataFrame({"url_score": _X(df, URL_COLS).mean(axis=1), "abnormal_score": _X(df, ABNORMAL_COLS).mean(axis=1),
                         "htmljs_score": _X(df, HTMLJS_COLS).mean(axis=1), "domain_score": _X(df, DOMAIN_COLS).mean(axis=1)})


def red_flag_counts(df):
    # number of features in the "phishing" (-1) and "suspicious" (0) states overall
    X = _X(df, ALL)
    return pd.DataFrame({"n_neg": (X == -1).sum(axis=1), "n_zero": (X == 0).sum(axis=1), "n_pos": (X == 1).sum(axis=1)})


def ssl_and_anchor(df):
    # strongest content signals together: valid SSL and anchors pointing to own domain
    X = _X(df, ["SSLfinal_State", "URL_of_Anchor", "Links_in_tags"])
    return pd.DataFrame({"ssl_plus_anchor": X["SSLfinal_State"] + X["URL_of_Anchor"],
                         "ssl_x_anchor": X["SSLfinal_State"] * X["URL_of_Anchor"],
                         "anchor_plus_links": X["URL_of_Anchor"] + X["Links_in_tags"]})


def reputation(df):
    # independent reputation evidence: traffic, page rank, index, inbound links, domain age
    X = _X(df, ["web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page", "age_of_domain", "DNSRecord"])
    return pd.DataFrame({"reputation_sum": X.sum(axis=1), "young_and_unranked":
                         ((X["age_of_domain"] < 0) & (X["Page_Rank"] < 0)).astype(float)})


def deceptive_url(df):
    # URL-obfuscation tricks: IP host, @, shortener, '-' in domain, many subdomains, https token in name
    X = _X(df, ["having_IP_Address", "having_At_Symbol", "Shortining_Service", "Prefix_Suffix", "having_Sub_Domain",
                "HTTPS_token", "double_slash_redirecting"])
    return (X < 0).sum(axis=1).astype(float)


def ssl_vs_subdomain(df):
    # a site with no trusted SSL and many subdomains is a classic phishing pattern
    X = _X(df, ["SSLfinal_State", "having_Sub_Domain", "Prefix_Suffix"])
    return pd.DataFrame({"ssl_x_sub": X["SSLfinal_State"] * X["having_Sub_Domain"],
                         "ssl_x_prefix": X["SSLfinal_State"] * X["Prefix_Suffix"]})


PROPOSALS = [("group_scores", group_scores), ("red_flag_counts", red_flag_counts), ("ssl_and_anchor", ssl_and_anchor),
             ("reputation", reputation), ("deceptive_url", deceptive_url), ("ssl_vs_subdomain", ssl_vs_subdomain)]
