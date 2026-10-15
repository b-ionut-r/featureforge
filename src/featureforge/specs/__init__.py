"""Candidate feature families the search builds, screens and keeps (each a ``Spec``)."""
from featureforge.specs.base import Spec
from featureforge.specs.dates import DatePart, EventRecency
from featureforge.specs.encoding import (BinnedPairTE, Count, FamilyCount, KeyBinTE, MixedCount, MixedTE, SlotBag,
                                         SlotTE, TargetEnc)
from featureforge.specs.groups import EntityLag, GroupSeq, GroupStat, LaggedTargetMean
from featureforge.specs.neighbors import KNNClassDist, KNNTarget, LinearOOF, Projection
from featureforge.specs.numeric import Arith, Digits, GeoPair, LinResid, Rotate, RowStat, StrNum
from featureforge.specs.text import CrossLinearOOF, TextLinearOOF, TextStats, TextSVD

__all__ = ["Spec", "DatePart", "EventRecency", "BinnedPairTE", "Count", "FamilyCount", "KeyBinTE", "MixedCount",
           "MixedTE", "SlotBag", "SlotTE", "TargetEnc", "EntityLag", "GroupSeq", "GroupStat", "LaggedTargetMean",
           "KNNClassDist", "KNNTarget", "LinearOOF", "Projection", "Arith", "Digits", "GeoPair", "LinResid", "Rotate",
           "RowStat", "StrNum", "CrossLinearOOF", "TextLinearOOF", "TextStats", "TextSVD"]
