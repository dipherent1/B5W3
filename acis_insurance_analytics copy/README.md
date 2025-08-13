# ACIS Insurance Analytics — South Africa (2014–2015)

This repository contains a **complete, reproducible pipeline** for:
- Task 1: Git/GitHub setup, CI, EDA & Stats
- Task 2: Data Version Control (DVC)
- Task 3: A/B-style hypothesis testing on portfolio risk & margin
- Task 4: Predictive modeling for claim severity and risk‑based premium

> ⚠️ Replace `data/raw/claims.csv` with your actual dataset (Feb 2014–Aug 2015). See **Data Layout** below.

## Quick Start

```bash
# 1) Create and activate a virtual environment (example: venv)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put your dataset here
#    data/raw/claims.csv  (or update configs/config.yaml)

# 4) Run basic EDA (plots + stats go to reports/)
python -m src.eda.eda --config configs/config.yaml

# 5) Run hypothesis tests (results written to reports/hypothesis_report.json)
python -m src.tests.hypothesis_tests --config configs/config.yaml

# 6) Train models (severity + probability + premium)
python -m src.models.train_all --config configs/config.yaml
```

## Project Structure

```
acis_insurance_analytics/
├─ configs/
│  └─ config.yaml
├─ data/
│  ├─ raw/               # Put the provided CSV here
│  ├─ intermediate/
│  └─ processed/
├─ dvc.yaml              # Example DVC pipeline
├─ models/               # Saved models and artifacts (created by scripts)
├─ notebooks/            # Optional notebooks
├─ reports/
│  ├─ figures/
│  └─ hypothesis_report.json
├─ src/
│  ├─ data/
│  │  └─ load_data.py
│  ├─ eda/
│  │  └─ eda.py
│  ├─ features/
│  │  └─ preprocess.py
│  ├─ models/
│  │  ├─ train_all.py
│  │  ├─ claim_severity.py
│  │  ├─ claim_probability.py
│  │  ├─ premium_model.py
│  │  └─ utils.py
│  └─ tests/
│     └─ hypothesis_tests.py
├─ .github/workflows/ci.yml
├─ .gitignore
├─ Makefile
├─ requirements.txt
└─ pyproject.toml
```

## DVC (Task 2)

```bash
# Initialize DVC (run once)
dvc init

# Configure a local remote (example path — change as needed)
dvc remote add -d localstorage /path/to/local/storage

# Track your data
dvc add data/raw/claims.csv

# Commit the .dvc file and dvc.lock
git add data/raw/claims.csv.dvc .gitignore dvc.lock .dvc/config
git commit -m "Track raw data with DVC"

# Push data to local remote
dvc push
```

`dvc.yaml` is provided to define pipeline stages for **eda**, **features**, and **models**.
You can run the whole pipeline with:

```bash
dvc repro
```

## Branching & CI (Task 1)

- Create branch `task-1` for initial EDA & stats, `task-2` for DVC, `task-3` for hypothesis tests, `task-4` for modeling.
- Open Pull Requests to merge into `main`.
- GitHub Actions runs lint + unit checks.

## Hypotheses (Task 3)

- H₀: No risk differences across provinces.
- H₀: No risk differences between zip codes.
- H₀: No margin differences between zip codes.
- H₀: No risk differences between Women and Men.

**Risk** metrics:
- **Claim Frequency**: proportion of policies with `TotalClaims > 0`.
- **Claim Severity**: average `TotalClaims` conditional on `TotalClaims > 0`.
- **Margin**: `TotalPremium - TotalClaims`.

We use **t-tests**, **Mann–Whitney U**, **Levene**, and **chi-squared** where appropriate and write an audit‑ready JSON report.

## Modeling (Task 4)

- **Claim Severity** (regression, claims > 0): LinearRegression, RandomForestRegressor, XGBoost; metrics: RMSE, R².
- **Claim Probability** (classification): LogisticRegression, RandomForestClassifier, XGBoost; metrics: ROC‑AUC, F1.
- **Risk‑Based Premium**: `Premium = p(claim) * E[severity] + expense_loading + profit_margin` (configurable).
- **Interpretability**: SHAP summary for best model; top features saved as artifacts.

## Data Layout

Expected CSV columns (subset):
```
UnderwrittenCoverID, PolicyID, TransactionMonth, IsVATRegistered, Citizenship, LegalType, Title,
Language, Bank, AccountType, MaritalStatus, Gender, Country, Province, PostalCode, MainCrestaZone,
SubCrestaZone, ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity,
Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice,
CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet,
SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup,
Section, Product, StatutoryClass, StatutoryRiskType, TotalPremium, TotalClaims
```

If your columns differ, adjust `configs/config.yaml` accordingly.

## Reports

- `reports/figures/` — loss ratio heatmaps, distributions, time trends (3+ creative plots).
- `reports/hypothesis_report.json` — p-values and decisions for each test.
- `models/` — trained models, SHAP plots, and feature importances.

## License

MIT
