"""Domain features for machine_cpu, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: UCI CPU performance columns: MYCT machine cycle time (ns), MMIN/MMAX main memory (KB),
# CACH cache (KB), CHMIN/CHMAX I/O channels; the task predicts relative performance.


def clock_rate(df):
    # cycle time in ns -> clock frequency (MHz); performance scales with frequency, not period
    return pd.DataFrame({"mhz": 1000.0 / df["MYCT"].replace(0, np.nan), "log_myct": np.log(df["MYCT"].clip(lower=1))})


def memory(df):
    # memory capacity: average configured memory and log scale (performance roughly log-linear in memory)
    avg = (df["MMIN"] + df["MMAX"]) / 2.0
    return pd.DataFrame({"mem_avg": avg, "log_mmax": np.log1p(df["MMAX"]), "log_mmin": np.log1p(df["MMIN"]),
                         "mem_range_ratio": df["MMAX"] / df["MMIN"].replace(0, np.nan)})


def throughput_proxy(df):
    # memory bandwidth-ish proxy: memory size times clock frequency (bigger, faster machines)
    mhz = 1000.0 / df["MYCT"].replace(0, np.nan)
    return pd.DataFrame({"mmax_x_mhz": df["MMAX"] * mhz / 1000.0, "mavg_x_mhz": (df["MMIN"] + df["MMAX"]) / 2.0 * mhz / 1000.0})


def cache_features(df):
    # cache hides memory latency: cache per clock cycle and per memory size
    return pd.DataFrame({"cach_per_myct": df["CACH"] / df["MYCT"].replace(0, np.nan),
                         "log_cach": np.log1p(df["CACH"]), "cach_per_mem": df["CACH"] / df["MMAX"].replace(0, np.nan) * 1000})


def channels(df):
    # I/O channels: capacity of the I/O subsystem, scaled by clock
    ch = (df["CHMIN"].astype(float) + df["CHMAX"].astype(float)) / 2.0
    return pd.DataFrame({"ch_avg": ch, "ch_x_mhz": ch * 1000.0 / df["MYCT"].replace(0, np.nan)})


def ein_like(df):
    # multiplicative performance model: log(perf) ~ sum of logs of clock, memory, cache, channels
    return (np.log(1000.0 / df["MYCT"].clip(lower=1)) + np.log1p((df["MMIN"] + df["MMAX"]) / 2.0)
            + 0.5 * np.log1p(df["CACH"]) + 0.3 * np.log1p(df["CHMAX"].astype(float)))


PROPOSALS = [("clock_rate", clock_rate), ("memory", memory), ("throughput_proxy", throughput_proxy),
             ("cache_features", cache_features), ("channels", channels), ("ein_like", ein_like)]
