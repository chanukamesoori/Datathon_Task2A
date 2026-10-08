# Datathon Task 2A forecasting

Custom local Python workflow for ten-week depot/brand total and Fresh chilled order-volume forecasting. Includes strict order-date target construction, leakage-safe origin features, expanding chronological validation, an independent final holdout, final retraining and submission verification.

Models compared include naive, moving-average and seasonal baselines, Ridge, Random Forest, HistGradientBoosting, CatBoost and LightGBM. Model selection uses earlier validation windows only; the final holdout is reserved for independent evaluation. No pretrained predictor or AutoML library is used.

## Local reproduction

Use Python 3.14. Place the authorized competition CSVs in `data/`: `deliveries_train.csv`, `task1_test_inputs.csv`, `calendar.csv`, `task2a_test_inputs.csv`, and `submission_task2a.csv`. Data is intentionally not included.

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m src.pipeline
& '.\.venv\Scripts\python.exe' -m src.verification
& '.\.venv\Scripts\python.exe' scripts/execute_notebook.py
```

The notebook can also be run in VS Code by choosing the workspace Python environment and Run All. It runs the complete local workflow. Models and reports are generated locally in ignored `models/` and `outputs/` folders. Predictions preserve template identifiers and order and enforce nonnegative volume, chilled volume no greater than total, and exact zero chilled volume for non-Fresh brands.

## Repository contents

- `src/`: preparation, features, estimators, evaluation, plotting, orchestration and verification.
- `notebooks/Task2A_Complete_ML_Forecasting.ipynb`: source-only notebook with no execution outputs or dataset-specific findings.
- `scripts/execute_notebook.py`: fresh-kernel execution using a workspace-local environment.
- `docs/architecture.md` and `docs/preprocessing.md`: general architecture and methodology.
- `requirements.txt`: tested package versions.

## Confidentiality and disclosure

This public repository excludes competition data, derived tables, predictions, trained model artifacts, figures, execution outputs and dataset-specific results. Keep locally generated results and executed notebooks private. Publication of such material requires organizer authorization.

AI assistance helped create and debug the custom source and documentation. The team must review its work and provide an accurate AI-tool disclosure. Review the organizer's restriction on fully automated end-to-end modelling tools before submission. This repository covers Task 2A only.
