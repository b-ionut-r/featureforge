"""Domain proposals written by Claude through the Anthropic API, from the table's schema only.

Optional. When ``ANTHROPIC_API_KEY`` is set (or ``llm="on"``) and no proposals file is given,
``build_features`` asks Claude for a proposals file before the search starts. Claude sees only what a person
reading the data dictionary would: the task, the description you pass, each column's type, unique count,
missing share and a few example values, and 8 sample rows with the target removed. It never sees the target,
a score or a model run. The file it writes then goes through the same cross-validated check as any proposals
file (``featureforge.proposals``): each proposal is kept only if it beats the search's output on held-out
folds, and the kept set must hold again on new folds. A useless or harmful proposal costs runtime, not loss.

The written file is saved as ``<out_dir>/proposals_llm.py`` and reused when present, so a rerun reads the
same proposals and builds the same features. Delete it to ask again.

The code Claude writes is executed on your data, like any proposals file. Read it if that matters to you.
"""
from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

MODEL = "claude-opus-5-5"
FILE_NAME = "proposals_llm.py"

SYSTEM = """You write domain feature proposals for a tabular prediction task, blind: from the table's schema only.
You see the task, the data's description, the columns (type, unique count, missing share, example values) and 8
sample rows with the target removed. You see no target values and no scores.

Write one Python file:
- a docstring "Domain features for <name>, written from the schema and description only.", then imports of numpy
  as np and pandas as pd (nothing else);
- 3-10 functions `fn(df)` that each take the raw table (training rows then test rows, the columns listed, without
  the target) and return a pd.Series or pd.DataFrame of numbers with one row per input row, in the same order;
- `PROPOSALS = [("label", fn), ...]` at the end.
Rules:
- Use only columns that exist, by their exact names. Categorical columns may be strings or categories: map them to
  numbers when meaningful (ordered levels to ordered numbers).
- Be robust to missing values and division by zero (use .replace(0, np.nan)).
- Keep functions vectorised (no Python loops or row-wise apply): the table may have millions of rows.
- Functions must be row-wise or label-free; never use any target.
- Prefer features grounded in what the columns mean: domain formulas, ratios, differences, semantically
  meaningful combinations, counts over groups of related flags, summaries over groups of related columns (for
  ordered numeric columns such as a signal or spectrum, its shape). If columns are anonymous, write a few generic
  but sensible ones and say so in the docstring.
- Give each function a one-line comment on the domain reasoning. If you use knowledge from outside the schema
  (known formulas, known dataset decodings), add a comment line starting "# OUTSIDE KNOWLEDGE:" saying what.
A cross-validated check later keeps only the proposals that help, so a sensible proposal that does not help costs
little; one that crashes is lost.
Reply with the whole file in one ```python code block and nothing else."""


def schema(raw: pd.DataFrame, task: str, name: str = "data", description: Optional[str] = None) -> str:
    """The prompt's view of the table: what a person reading its data dictionary would see, no target."""
    L = [f"# {name}", f"task: {task}; target: (hidden); rows: {len(raw)}", "", "## Description",
         (description or "(none given)")[:3000], "", "## Columns"]
    for c in raw.columns:
        v = raw[c]
        ex = [str(x) for x in v.dropna().unique()[:5]]
        L.append(f"- `{c}`: {v.dtype}, {v.nunique()} unique, {100 * v.isna().mean():.0f}% missing, e.g. {ex}")
    L += ["", "## 8 sample rows (target omitted)", "```",
          raw.sample(min(8, len(raw)), random_state=0).to_string(max_cols=60)[:6000], "```"]
    return "\n".join(L)


def _code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else text


def check(code: str, sample: pd.DataFrame) -> list:
    """Problems running the file on sample rows: [] when every proposal returns one number per row."""
    ns: dict = {"__name__": "ff_proposals_check"}
    try:
        exec(compile(code, FILE_NAME, "exec"), ns)  # noqa: S102  the user opted in to running proposals
    except Exception:  # noqa: BLE001
        return [f"the file fails to load:\n{traceback.format_exc(limit=3)}"]
    props = ns.get("PROPOSALS")
    if not props:
        return ["the file defines no PROPOSALS list"]
    errs = []
    for item in props:
        try:
            label, fn = item
            v = fn(sample.copy())
            v = v.to_frame() if isinstance(v, pd.Series) else pd.DataFrame(v)
            if len(v) != len(sample):
                errs.append(f"{label}: returned {len(v)} rows for {len(sample)}")
            elif not v.apply(pd.to_numeric, errors="coerce").notna().any().any():
                errs.append(f"{label}: returned nothing numeric")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{item[0] if isinstance(item, tuple) and item else item}: {type(e).__name__}: {e}"[:400])
    return errs


def _client():
    import anthropic  # optional dependency: pip install featureforge[llm]
    return anthropic.Anthropic()


def _ask(client, model: str, messages: list) -> str:
    # fallbacks="default": if a safety check declines, the API reruns the request on a fallback model
    with client.beta.messages.stream(model=model, max_tokens=32000, system=SYSTEM, messages=messages,
                                     thinking={"type": "adaptive"}, output_config={"effort": "high"},
                                     betas=["server-side-fallback-2026-07-01"], fallbacks="default") as s:
        msg = s.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("the model declined")
    return "".join(b.text for b in msg.content if b.type == "text")


def write(raw_tr: pd.DataFrame, raw_te: pd.DataFrame, task: str, out_path, *, name: str = "data",
          description: Optional[str] = None, model: Optional[str] = None, client=None, repairs: int = 2):
    """Ask Claude for a proposals file and write it to ``out_path``; returns the path, or None on failure.

    The file is run on 200 sample rows; when a proposal crashes, the errors go back to Claude for a corrected
    file, up to ``repairs`` times. Proposals still broken after that are dropped later by ``proposals.compute``.
    """
    out_path = Path(out_path)
    try:
        client = client or _client()
        model = model or os.environ.get("FEATUREFORGE_LLM_MODEL") or MODEL
        allx = pd.concat([raw_tr, raw_te], ignore_index=True)
        sample = allx.sample(min(200, len(allx)), random_state=0).reset_index(drop=True)
        messages = [{"role": "user", "content": schema(raw_tr, task, name, description)}]
        text = _ask(client, model, messages)
        code, errs = _code(text), None
        for _ in range(repairs + 1):
            errs = check(code, sample)
            if not errs or _ == repairs:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Running the file on sample rows gave these errors:\n"
                          + "\n".join(errs) + "\nReply with the whole corrected file in one ```python block."}]
            text = _ask(client, model, messages)
            code = _code(text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code)
        print(f"llm proposals: {model} wrote {out_path}" + (f" ({len(errs)} still failing)" if errs else ""),
              flush=True)
        return out_path
    except Exception as e:  # noqa: BLE001  an outside service: any failure means no proposals, never a crash
        print(f"llm proposals: skipped ({repr(e)[:200]})", flush=True)
        return None


def wanted(mode: str) -> bool:
    """``"auto"``: on when an Anthropic API key is set; ``"on"`` / ``"off"`` force it."""
    if mode == "off":
        return False
    return mode == "on" or bool(os.environ.get("ANTHROPIC_API_KEY"))
