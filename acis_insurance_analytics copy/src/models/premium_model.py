import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import joblib

from src.data.load_data import load_config, load_raw_df
from src.features.preprocess import preprocess

def compute_risk_based_premium(prob, severity, expense_loading, profit_margin):
    risk_cost = prob * severity
    return risk_cost * (1 + expense_loading + profit_margin)

def main(config_path: str):
    config = load_config(config_path)
    df = load_raw_df(config)
    dfp = preprocess(df, config)

    models_dir = Path(config["paths"]["models_dir"])
    prob_model = joblib.load(models_dir / "probability_model.joblib")
    sev_model  = joblib.load(models_dir / "severity_model.joblib")

    # Use overlapping feature set that both models can handle (we'll pass full; pipelines handle columns)
    X = dfp[[c for c in dfp.columns if c not in ["TotalClaims","HasClaim"]]]

    prob = prob_model.predict_proba(X)[:,1]
    # Expected severity for *all*, but better to pass best estimator directly
    sev_pred = np.maximum(sev_model.predict(X), 0)

    p = compute_risk_based_premium(
        prob, sev_pred, config["premium_framework"]["expense_loading"], config["premium_framework"]["profit_margin"]
    )
    out = pd.DataFrame({
        "UnderwrittenCoverID": dfp.get("UnderwrittenCoverID", pd.Series(range(len(prob)))),
        "PredictedClaimProbability": prob,
        "PredictedSeverity": sev_pred,
        "RiskBasedPremium": p
    })
    outpath = models_dir / "risk_based_premium_predictions.parquet"
    out.to_parquet(outpath, index=False)
    print(f"[premium] Saved predictions to {outpath}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
