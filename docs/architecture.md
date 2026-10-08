# Local forecasting architecture

```mermaid
flowchart TD
    A[Historical order CSVs] --> C[Validate IDs, fields, dates and coverage]
    B[Supplied daily calendar] --> C
    C --> D[Weekly depot and brand total and chilled demand]
    D --> E[Reserve final ten complete weeks]
    E --> F[Expanding chronological validation]
    B --> G[Known future calendar and seasonality]
    F --> H[Origin-only lags, rolling statistics and normalization]
    G --> H
    H --> I[Baselines, Ridge, RF, HGB, CatBoost, LightGBM]
    I --> J[Validation selection and ensemble comparison]
    J --> K[Freeze selection and evaluate holdout once]
    K --> L[Retrain selected models on all permitted history]
    L --> M[Save fitted model and preprocessing artifact]
    M --> N[Local batch inference for ten future weeks]
    N --> O[Enforce volume constraints and validate template]
    O --> P[Private competition submission CSV]
```

Proposed deployment: a local batch command loads the trusted artifact and supplied forecast-input CSV, generates predictions with its stored history and known calendar, and exports validated results. No external hosting is needed. New observations or calendar periods require an explicit refresh and retraining; the saved artifact supports its specified ten-week window.
