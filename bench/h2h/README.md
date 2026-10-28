# Head-to-head bench

Every feature tool through one fixed LightGBM judge on the same held-out rows. Results and protocol:
[docs/benchmark.md](../../docs/benchmark.md).

```bash
python run.py --datasets papers,suite,openml,fresh --seeds 0 --arms raw,featureforge,featuretools,openfe,autofeat
python report.py results/results.jsonl --arms featureforge,featuretools,openfe
H2H_TIME_SPLIT=1 python run.py --datasets bike_sharing,kick --root ~/h2h_runs_time --log results/results_time.jsonl
python contest_check.py split --case hc && python contest_check.py arm --case hc --label new --src ../../src
```

- `datasets.py`: the tables (groups `suite`, `openml`, `papers`, `fresh`, `kaggle`), cached under `$H2H_DATA`.
- `arms.py`: one arm per subprocess: `featureforge` (source from `$H2H_FF_SRC`), `ff:<label>` (a frozen copy
  under `$FF_SNAP/<label>/src`), `openfe`, `autofeat`, `featuretools`, `llm`, `llmff:<label>`.
  OpenFE, AutoFeat and Featuretools need pandas 2 (`H2H_COMP_PY` points at that environment's python).
- `run.py`: splits, caps each arm's wall clock, judges, appends to `results/*.jsonl` (resumable);
  `--shuffle` is the shuffled-label control.
- `llm/`: the CAAFE-style proposals, one file per table, written blind from the schema.
- `contest_check.py`: Kaggle contests rebuilt from their files, two source trees compared.
