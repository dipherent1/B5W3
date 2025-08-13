import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import yaml

from src.data.load_data import load_config, load_raw_df
from src.features.preprocess import preprocess
from src.models.utils import build_preprocessor, FeatureConfig

def train_and_eval(X, y, config, models_dir: Path):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["modeling"]["test_size"], random_state=config["modeling"]["random_state"], stratify=y
    )
    fcfg = FeatureConfig(
        numeric_impute_strategy=config["modeling"]["numeric_impute_strategy"],
        categorical_impute_strategy=config["modeling"]["categorical_impute_strategy"],
    )
    pre = build_preprocessor(X_train, fcfg)

    from sklearn.pipeline import Pipeline
    candidates = {
        "logreg": Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000, n_jobs=None))]),
        "rf": Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))]),
        "xgb": Pipeline([("pre", pre), ("model", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                                               subsample=0.8, colsample_bytree=0.8,
                                                               random_state=42, n_jobs=-1, tree_method="hist"))]),
    }

    metrics = {}
    best_name, best_model, best_auc = None, None, -1.0
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:,1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, prob)
        f1 = f1_score(y_test, pred)
        metrics[name] = {"roc_auc": float(auc), "f1": float(f1)}
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, pipe

    (models_dir).mkdir(parents=True, exist_ok=True)
    with open(models_dir / "probability_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return best_model, {"best": best_name, "metrics": metrics}

def main(config_path: str):
    config = load_config(config_path)
    df = load_raw_df(config)
    dfp = preprocess(df, config)

    target = "HasClaim"
    keep = config["features"]["owner"] + config["features"]["location"] + config["features"]["car"] + config["features"]["plan"]
    keep = [c for c in keep if c in dfp.columns]
    X = dfp[keep]
    y = dfp[target].astype(int)

    models_dir = Path(config["paths"]["models_dir"])
    import joblib
    model, info = train_and_eval(X, y, config, models_dir)
    joblib.dump(model, models_dir / "probability_model.joblib")
    with open(models_dir / "probability_best.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[probability] Best model: {info['best']} | metrics={info['metrics'][info['best']]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
