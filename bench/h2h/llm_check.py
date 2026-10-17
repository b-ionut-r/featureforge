"""Check that the proposals in llm/<name>.py run on the dataset: prints each proposal's columns, missing
share and spread. It never trains a model or reads the target, so writing proposals stays blind."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import llm_features  # noqa: E402
from datasets import load  # noqa: E402

for name in sys.argv[1:]:
    X, _, task, _ = load(name)
    props = llm_features(name, X)
    print(f"{name}: {len(props)} proposals ran")
    for label, v in props:
        for c in v.columns:
            s = v[c]
            print(f"  {c}: missing {s.isna().mean():.0%}, unique {s.nunique()}, min {np.nanmin(s) if s.notna().any() else 'nan'}, max {np.nanmax(s) if s.notna().any() else 'nan'}")
