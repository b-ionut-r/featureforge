"""Work clock and memory model of the search.

Every budget decision is charged in seconds of a 4-core reference machine computed from row and
column counts, never from the wall clock, so the same data gives the same features on any machine.
The memory side prices a candidate's working set before it is built (``_spec_bytes``).
"""
from __future__ import annotations

from featureforge.specs.neighbors import KNNClassDist, KNNTarget, Projection
from featureforge.specs.numeric import LinResid
from featureforge.specs.text import CrossLinearOOF, TextLinearOOF, TextSVD


# Work clock. Every budget decision (how many candidates get built, which families are cut, how
# many prefixes and rounds run, whether the search shrinks its rows) counts work, not wall-clock
# seconds: timed decisions picked different features on the same data from run to run (Home
# Credit kept 12 features or none at one commit, as 9-11 neighbour candidates fitted in the time)
# and would pick differently again on a faster or busier machine. Work is charged in seconds of
# the 4-core reference container, from row and column counts only, so ``time_budget`` keeps its
# meaning there. Measured on Home Credit, IEEE-CIS, Santander and Porto Seguro builds:
_CV_CELL_S = 3e-7       # prior for one 5-fold CV fit, per search row and model column (before it runs)
_TREE_CELL_S = 1.6e-9   # one boosting iteration, per training row and model column (Home Credit 320 and 120
                        # columns, 80k-197k rows, and a 100-column regression all measured 1.5-2.0e-9)
_TREE_SPLIT_S = 1.1e-6  # one boosting iteration, per model column and leaf: split finding over a column's bins, which
                        # no row count pays for on small tables (600 x 3,000: 11 s per 5-fold fit charged as 0.3 s, a
                        # 20 s budget ran 210 s). Leaves grow with rows up to 31 (about one per 80 training rows).
                        # Measured 3.1 / 14.5 / 36 us per iteration and column at 300 / 1,000 / 3,000 rows; from
                        # about 21k rows the row term is the larger and charges are as before.


def _tree_col_s(n_rows: int) -> float:
    """Work of one boosting iteration on one model column of ``n_rows`` training rows."""
    return max(_TREE_CELL_S * n_rows, _TREE_SPLIT_S * min(31.0, n_rows / 80))


_ROW_PASS_S = 2e-7      # building and screening one candidate of unit cost, per row (thread time)
_PAIR_CELL_S = 2e-7     # pair scan, per subsample row and candidate column
_KEY_ROW_S = 1.5e-7     # entity-anchor test, per sampled row and key count
_SETUP_COL_S = 5e-1     # column screens before the search (codes, keys, label echoes), per column of a
                        # table of 300k rows or more (Home Credit and IEEE-CIS both took 0.6 s a column)
_WU_ROW_SHARE = 0.05    # a label-free candidate fitted over all known rows costs its search rows plus this
                        # share of those rows (IEEE-CIS: 1,698 group statistics over 1.1M rows, 0.09 s each)
_REF_THREADS = 4        # candidate building runs on this many threads there
_CLASS_CELL_S = 5e-8    # multiclass only: one residual-probe column or interaction grid, per row and class.
                        # Each class adds two histogram passes and its share of the softmax loss (4-5.5e-8
                        # measured at 12 and 100 classes); a 100-output target map is probed column by
                        # column, so it costs classes squared. Binary and regression costs are in the
                        # family constants.
_SCAN_SHARE = 0.125     # multiclass interaction scan: at most this share of the work left (rows subsampled)
# Candidate cost in row passes (thread time per row / _ROW_PASS_S), by family.
_SPEC_WORK = {
    "Arith": 1, "Digits": 1, "DatePart": 1, "Rotate": 1, "GroupStat": 2, "StrNum": 2, "GeoPair": 2,
    "TargetEnc": 3, "KeyBinTE": 3, "MixedTE": 3, "MixedCount": 3, "BinnedPairTE": 4, "Count": 5, "RowStat": 5,
    "SlotBag": 5, "SlotTE": 5, "TextStats": 5, "EntityLag": 10, "GroupSeq": 10, "LaggedTargetMean": 10,
    "EventRecency": 14, "LinResid": 60, "LinearOOF": 60, "Projection": 60, "FamilyCount": 400,
    "TextSVD": 500, "CrossLinearOOF": 1000, "TextLinearOOF": 1000, "KNNClassDist": 1300, "ClientProfile": 1300,
    "KNNTarget": 1900,
}
_SPEC_WORK_DEFAULT = 20


# Resident memory beyond the data frames (the interpreter, numpy, pandas, scikit-learn, LightGBM,
# Arrow's pool after reading the files): 0.63-0.73 GB on an Avazu-shaped table (_mem_rows).
_BASE_RSS = 0.8e9
# A candidate's working arrays while it is built, per row it reads: keys' codes, values and the group
# statistics' temporaries, about eight float64 arrays (_materialize starts fewer at once when memory is short).
_SPEC_BYTES_PER_ROW = 64


def _spec_bytes(s, rows: int, wm: int | None = None, text_nnz: dict | None = None) -> int:
    """A candidate's working memory while it is built over ``rows`` rows: keys' codes, values and group
    statistics' temporaries (_SPEC_BYTES_PER_ROW); a matrix of its columns in about four float64 copies
    (linear residuals, projections, neighbours: 513 and 776 bytes per row measured over 16 and 24 columns);
    for neighbour queries also scikit-learn's distance blocks (``wm`` bytes, its working memory, at most every
    row against every row) and a fold's query rows' neighbours; for a target-dependent candidate of several
    outputs (a class each), its out-of-fold outputs, a fold's statistics gathered per row and its transform, a
    float64 rows x classes array each; for text models, their sparse matrices, from ``text_nnz`` (each text
    column's distinct words and word pairs, and character n-grams, a row)."""
    b = _SPEC_BYTES_PER_ROW * rows
    n_out = getattr(s, "n_out", 1) or 1
    if s.target_dep and n_out > 1:
        # (Expedia's 100 classes under a 524 MB limit: a target encoder priced at 64 bytes a row, admitted
        # alone, killed it)
        b += 24 * n_out * rows
    if text_nnz and isinstance(s, (TextSVD, TextLinearOOF)):
        # Measured on Mercari's and a wine-review table's descriptions (41 to 73 nonzeros a row): 34 to 54 bytes a
        # nonzero for the linear model (vocabularies, the rows' matrix, its blocks stacked, the folds' row
        # takes), 30 to 37 for the SVD's fit on up to max_fit_rows rows. (MCTS under 3.5 GB: three of them at
        # once, priced at 64 bytes a row, killed it.)
        for c in (s.text_cols if isinstance(s, TextLinearOOF) else s.parents):
            words, chars = text_nnz.get(c, (0.0, 0.0))
            if isinstance(s, TextLinearOOF):
                b += int(56 * (words + (chars if s.chars else 0.0)) * rows)
            else:
                b += int(40 * words * min(rows, s.max_fit_rows))
    if isinstance(s, CrossLinearOOF):
        # Its codes (int64) and one-hot designs for each key and key pair, a fold's at a time: 40 to 49 bytes a row
        # and block measured over 200k and 400k rows of 4 and 6 keys (1 KB a row with 6 keys, priced at 64 bytes).
        k = len(s.parents)
        b += 56 * (k + (k * (k - 1) // 2 if s.pairs else 0)) * rows
    if type(s).__name__ == "EntityLag":
        # Each key's codes, the combined key and the sorted positions: 0.51 GB with one key and 1.68 GB with
        # three over PaySim's 6.4M rows (two killed it at 14 GB, priced at 64 bytes a row).
        b += 72 * (len(s.parents) - 1) * rows
    if isinstance(s, (LinResid, KNNTarget)):  # (projections are neighbour encoders' kin)
        b += 32 * len(s.parents) * rows
    if isinstance(s, (KNNTarget, KNNClassDist)) and not isinstance(s, Projection):
        b += min(_sk_working_bytes() if wm is None else wm, 8 * rows * rows) + 24 * rows * max(s.ks) // 4  # (a fold's query rows at a time)
    if type(s).__name__ == "ClientProfile":
        # Its key strings (each key's text, about 80 bytes a row as Python strings, and their joins) and one value
        # column's grouping at a time (the value, codes, sort order and statistics). Measured over 5.04M rows of 2
        # keys and 25 value columns: 1.5 GB with 50 rows a client, 3.0 GB with one (6.1 to 6.2 GB when every
        # column was grouped at once, priced at 0.65 GB without this term).
        b += (80 * (2 * len(s.keys) + 1) + 200) * rows
    return b


def _sk_working_bytes() -> int:
    """scikit-learn's working memory (bytes): the distance blocks a neighbour query holds at a time."""
    try:
        import sklearn
        return int(sklearn.get_config().get("working_memory", 1024)) << 20
    except Exception:
        return 1 << 30
