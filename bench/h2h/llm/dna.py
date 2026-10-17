"""Domain features for dna, written from the schema and description only."""
import numpy as np
import pandas as pd

# Per the description: 60 nucleotide positions x 3 indicators (A->100, C->010, G->001, T->000), so position p
# (1-based) occupies columns A{3(p-1)}, A{3(p-1)+1}, A{3(p-1)+2}; the junction sits mid-sequence (between 30 and 31).


def _seq(df):
    X = df[[f"A{i}" for i in range(180)]].apply(pd.to_numeric, errors="coerce").fillna(0).values.reshape(len(df), 60, 3)
    a, c, g = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    t = 1 - a - c - g
    return a, c, g, t


def _base(df):
    a, c, g, t = _seq(df)
    return a, c, g, t


def donor_motif(df):
    # OUTSIDE KNOWLEDGE: exon->intron (donor) sites start the intron with GT, consensus (exon)AG|GTRAGT(intron)
    a, c, g, t = _base(df)
    i = 30  # 0-based index of position 31, first base after the junction
    gt = g[:, i] * t[:, i + 1]
    score = (g[:, i - 1] + a[:, i - 2] + g[:, i] + t[:, i + 1] + (a[:, i + 2] + g[:, i + 2]) + a[:, i + 3]
             + g[:, i + 4] + t[:, i + 5])
    return pd.DataFrame({"donor_GT": gt, "donor_score": score}, index=df.index)


def acceptor_motif(df):
    # OUTSIDE KNOWLEDGE: intron->exon (acceptor) sites end the intron with AG, preceded by a pyrimidine-rich tract
    a, c, g, t = _base(df)
    i = 29  # 0-based index of position 30, last base before the junction
    ag = a[:, i - 1] * g[:, i]
    pyr = (c[:, 10:28] + t[:, 10:28]).sum(axis=1)
    return pd.DataFrame({"acceptor_AG": ag, "pyrimidine_tract": pyr, "acc_score": ag * pyr}, index=df.index)


def junction_bases(df):
    # identity of the two bases on each side of the junction as dinucleotide-flags (GT right, AG left)
    a, c, g, t = _base(df)
    return pd.DataFrame({"left_AG": a[:, 28] * g[:, 29], "right_GT": g[:, 30] * t[:, 31],
                         "right_GC": g[:, 30] * c[:, 31], "left_G": g[:, 29], "right_G": g[:, 30]}, index=df.index)


def composition_sides(df):
    # base composition left vs right of the junction: exons are GC-richer than introns, so the sign shows the orientation
    a, c, g, t = _base(df)
    gc = c + g
    pyr = c + t
    return pd.DataFrame({"gc_left": gc[:, :30].mean(axis=1), "gc_right": gc[:, 30:].mean(axis=1),
                         "gc_diff": gc[:, 30:].mean(axis=1) - gc[:, :30].mean(axis=1),
                         "pyr_left": pyr[:, :30].mean(axis=1), "pyr_right": pyr[:, 30:].mean(axis=1)}, index=df.index)


def near_junction_composition(df):
    # counts of each base in the 20 positions centred on the junction (the description's hint region)
    a, c, g, t = _base(df)
    s = slice(20, 40)
    return pd.DataFrame({"nA": a[:, s].sum(axis=1), "nC": c[:, s].sum(axis=1), "nG": g[:, s].sum(axis=1),
                         "nT": t[:, s].sum(axis=1)}, index=df.index)


def stop_codons(df):
    # OUTSIDE KNOWLEDGE: stop codons TAA/TAG/TGA are depleted in coding exons, so counts per side indicate exon vs intron
    a, c, g, t = _base(df)
    stop = t[:, :-2] * ((a[:, 1:-1] * a[:, 2:]) + (a[:, 1:-1] * g[:, 2:]) + (g[:, 1:-1] * a[:, 2:]))
    return pd.DataFrame({"stops_left": stop[:, :28].sum(axis=1), "stops_right": stop[:, 30:].sum(axis=1)},
                        index=df.index)


PROPOSALS = [("donor_motif", donor_motif), ("acceptor_motif", acceptor_motif), ("junction_bases", junction_bases),
             ("composition_sides", composition_sides), ("near_junction_composition", near_junction_composition),
             ("stop_codons", stop_codons)]
