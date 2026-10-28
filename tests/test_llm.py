"""The optional Anthropic API step, tested with recorded responses: no network, no key."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from featureforge import build_features
from featureforge import llm as L

GOOD = '''Here is the file.
```python
"""Domain features for shop, written from the schema and description only."""
import numpy as np
import pandas as pd


def price_per_unit(df):
    # spend divided by units bought is the unit price
    return df["spend"] / df["units"].replace(0, np.nan)


PROPOSALS = [("price_per_unit", price_per_unit)]
```'''

BROKEN = GOOD.replace('df["units"]', 'df["unitz"]')


class Recorded:
    """Stands in for anthropic.Anthropic(): replays recorded replies and keeps the requests."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.requests.append(kw)
        msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.replies.pop(0))])

        class S:
            def __enter__(s):
                return SimpleNamespace(get_final_message=lambda: msg)

            def __exit__(s, *a):
                return False
        return S()


def _data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    units = rng.integers(1, 20, n)
    price = rng.uniform(1, 50, n)
    X = pd.DataFrame({"units": units, "spend": units * price, "noise": rng.normal(size=n)})
    y = (price + rng.normal(scale=2, size=n) > 25).astype(int)
    return X, y


def test_prompt_is_blind_and_file_written(tmp_path):
    X, y = _data()
    rec = Recorded(GOOD)
    out = L.write(X.iloc[:1000], X.iloc[1000:], "binary", tmp_path / "p.py", name="shop", client=rec)
    assert out and out.read_text().startswith('"""Domain features for shop')
    prompt = rec.requests[0]["messages"][0]["content"]
    assert "`spend`" in prompt and "target: (hidden)" in prompt
    assert rec.requests[0]["model"] == L.MODEL


def test_crashing_file_goes_back_for_repair(tmp_path):
    X, _ = _data()
    rec = Recorded(BROKEN, GOOD)
    out = L.write(X.iloc[:1000], X.iloc[1000:], "binary", tmp_path / "p.py", client=rec)
    assert len(rec.requests) == 2 and "unitz" in rec.requests[1]["messages"][-1]["content"]
    assert "unitz" not in out.read_text()


def test_service_failure_means_no_proposals(tmp_path):
    X, _ = _data()

    class Down:
        beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: (_ for _ in ()).throw(OSError("down"))))
    assert L.write(X.iloc[:1000], X.iloc[1000:], "binary", tmp_path / "p.py", client=Down()) is None


def test_no_key_leaves_output_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(L, "_client", lambda: (_ for _ in ()).throw(AssertionError("called the API")))
    X, y = _data()
    tr = X.iloc[:1000].assign(y=y[:1000])
    a = build_features(tr, X.iloc[1000:], "y", budget=20, out_dir=tmp_path / "a")
    b = build_features(tr, X.iloc[1000:], "y", budget=20, out_dir=tmp_path / "b", llm="off")
    assert pd.read_parquet(a[0]).equals(pd.read_parquet(b[0]))
    assert not (tmp_path / "a" / L.FILE_NAME).exists()


def test_key_set_writes_checks_and_reuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    rec = Recorded(GOOD)
    monkeypatch.setattr(L, "_client", lambda: rec)
    X, y = _data()
    tr = X.iloc[:1000].assign(y=y[:1000])
    p, _ = build_features(tr, X.iloc[1000:], "y", budget=20, out_dir=tmp_path)
    assert (tmp_path / L.FILE_NAME).exists() and len(rec.requests) == 1
    assert "proposal price_per_unit (1 columns)" in capsys.readouterr().out  # judged by the usual check
    build_features(tr, X.iloc[1000:], "y", budget=20, out_dir=tmp_path)  # reuses the saved file
    assert len(rec.requests) == 1


def test_placeholder_names_without_description_skip_the_call(tmp_path):
    assert not L.informative(["f_1", "f_2", "V3", "feature_4", "x5"])
    assert L.informative(["f_1", "f_2"], description="sensor readings from a turbine")
    assert L.informative(["units", "spend", "f_3"])
    X, _ = _data()
    X.columns = ["feature_1", "feature_2", "feature_3"]
    rec = Recorded(GOOD)
    assert L.write(X.iloc[:1000], X.iloc[1000:], "binary", tmp_path / "p.py", client=rec) is None
    assert rec.requests == []
