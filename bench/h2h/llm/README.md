# LLM-proposed features (CAAFE-style arm)

One file per dataset. Each was written by Claude acting as the LLM in CAAFE (Hollmann et al., NeurIPS 2023):
it read only the dataset's description, column names, types, unique counts and 8 sample rows (no target,
no scores, no model runs), then wrote 3-10 feature functions. `arms.py llm` runs them on train + test rows
and, as CAAFE does, keeps a proposal only when it lowers the 4-fold cross-validated loss on the training rows.

Where a proposal relies on outside knowledge rather than the schema, the writer flagged it:
- **Contest or generator knowledge** (reported separately, since it is closer to a hand-tuned answer):
  `fried` (the Friedman #1 formula), `porto_seguro` (decoded `ps_reg_03`/`ps_car` integers from Kaggle
  forums), `mercedes` (X314/X315/X118 from the contest), `kdd_appetency`/`kdd_upselling` (Var126 segments).
- **Public column meanings** for anonymous columns: `bank_marketing`, `cmc`, `spambase`, `nomao` (UCI docs),
  `jungle_chess` (game rules), `wind` (assumed target station).
- `tic_tac_toe`: `x_lines` counts completed rows of x, which is close to the label's definition but is a pure
  function of the board.
