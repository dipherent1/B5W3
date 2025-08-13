import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
import yaml

from src.data.load_data import load_config, load_raw_df
from src.features.preprocess import preprocess
from src.models.utils import build_preprocessor, FeatureConfig, split_features

def train_and_eval(X, y, config, models_dir: Path):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["modeling"]["test_size"], random_state=config["modeling"]["random_state"]
    )
    fcfg = FeatureConfig(
        numeric_impute_strategy=config["modeling"]["numeric_impute_strategy"],
        categorical_impute_strategy=config["modeling"]["categorical_impute_strategy"],
    )
    pre = build_preprocessor(X_train, fcfg)

    candidates = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "xgb": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist"),
    }

    metrics = {}
    best_name, best_model, best_rmse = None, None, 1e18
    for name, est in candidates.items():
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, pred, squared=False)
        r2 = r2_score(y_test, pred)
        metrics[name] = {"rmse": float(rmse), "r2": float(r2)}
        if rmse < best_rmse:
            best_rmse, best_name, best_model = rmse, name, pipe

    # Save metrics
    (models_dir).mkdir(parents=True, exist_ok=True)
    with open(models_dir / "severity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # SHAP (tree-based preferred). If best is linear, skip to avoid heavy calc.
    shap_summary_path = None
    try:
        if best_name in ["rf", "xgb"]:
            # Use a sample to speed up
            import random
            idx = np.random.choice(np.arange(X_test.shape[0]), size=min(1000, X_test.shape[0]), replace=False)
            # Transform features
            Xt = best_model.named_steps["pre"].transform(X_test.iloc[idx])
            model = best_model.named_steps["model"]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(Xt)
            shap.summary_plot(shap_values, Xt, show=False)
            shap_summary_path = models_dir / "severity_shap_summary.png"
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(shap_summary_path); plt.close()
    except Exception as e:
        print(f"[warn] SHAP computation skipped: {e}")

    return best_model, {"best": best_name, "metrics": metrics, "shap": str(shap_summary_path) if shap_summary_path else None}

def main(config_path: str):
    config = load_config(config_path)
    df = load_raw_df(config)
    dfp = preprocess(df, config)
    # Subset to claims > 0
    df_claims = dfp[dfp["TotalClaims"] > 0].copy()
    # Targets & features
    target = "TotalClaims"
    keep = config["features"]["owner"] + config["features"]["location"] + config["features"]["car"] + config["features"]["plan"]
    keep = [c for c in keep if c in df_claims.columns]
    X = df_claims[keep]
    y = df_claims[target]

    models_dir = Path(config["paths"]["models_dir"])
    model, info = train_and_eval(X, y, config, models_dir)
    # Persist model
    import joblib
    joblib.dump(model, models_dir / "severity_model.joblib")
    with open(models_dir / "severity_best.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[severity] Best model: {info['best']} | metrics={info['metrics'][info['best']]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
