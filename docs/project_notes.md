# Project Notes

## Current status

- Main notebook preserved in `notebooks/ForexPredictML.ipynb`.
- Clean script pipeline available in `src/forex_predict_clean.py`.
- Temporal split logic is used in notebook tuning to reduce test leakage.

## Recommended next improvements

1. Move repeated notebook logic to reusable functions in `src/`.
2. Add unit tests for feature creation and split logic.
3. Add a small CLI to choose symbol and horizon from command line.
4. Add `results/` folder with saved metrics (CSV/JSON).
5. Add walk-forward validation for stronger time-series evaluation.

## Suggested branch strategy

- `main`: stable experiments and docs.
- `feature/notebook-cleanup`: notebook simplification.
- `feature/model-validation`: robust evaluation upgrades.
