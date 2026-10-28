"""Domain features for cpu_act, written from the schema and description only."""
import numpy as np
import pandas as pd


def _div(a, b):
    return a.astype(float) / b.astype(float).replace(0, np.nan)


def log_freeswap(df):
    # swap almost exhausted means the box is thrashing, so user CPU share collapses
    return np.log1p(df["freeswap"].clip(lower=0).astype(float))


def log_freemem(df):
    # free memory on a log scale; low free memory goes with paging and lower user CPU
    return np.log1p(df["freemem"].clip(lower=0).astype(float))


def paging_activity(df):
    # total pages moved in/out and scanned per second: system time spent on memory management
    return (df["pgin"] + df["ppgin"] + df["pgout"] + df["ppgout"] + df["pgfree"] + df["pgscan"]).astype(float)


def syscalls_per_runnable(df):
    # system calls per process in the run queue: kernel-bound load vs user computation
    return _div(df["scall"], df["runqsz"])


def bytes_per_read_call(df):
    # bytes transferred per read/write call: big buffered I/O vs many small kernel calls
    return pd.DataFrame({"rchar_per_sread": _div(df["rchar"], df["sread"]),
                         "wchar_per_swrite": _div(df["wchar"], df["swrite"])}, index=df.index)


def process_creation(df):
    # fork+exec rate: process churn costs system time
    return (df["fork"] + df["exec"]).astype(float)


def faults_per_fork(df):
    # page faults per created process; high values mean memory pressure rather than new work
    return _div(df["pflt"] + df["vflt"], df["fork"] + df["exec"])


def total_syscall_io(df):
    # all read/write syscalls plus logical I/O: I/O-bound load lowers user-mode CPU
    return (df["sread"] + df["swrite"] + df["lread"] + df["lwrite"]).astype(float)


PROPOSALS = [
    ("log_freeswap", log_freeswap),
    ("paging_activity", paging_activity),
    ("syscalls_per_runnable", syscalls_per_runnable),
    ("faults_per_fork", faults_per_fork),
    ("process_creation", process_creation),
    ("total_syscall_io", total_syscall_io),
    ("bytes_per_read_call", bytes_per_read_call),
    ("log_freemem", log_freemem),
]
