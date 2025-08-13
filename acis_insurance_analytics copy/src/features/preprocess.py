import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from src.data.load_data import load_config, load_raw_df

def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Basic cleaning
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Clip extreme negatives on claims/premium (if any)
    for c in ["TotalClaims", "TotalPremium"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)

    # Create helper targets
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    # Fill NA for categorical quickly (will also be handled in pipeline)
    df = df.fillna({"Province": "Unknown", "PostalCode": "Unknown"})

    return df

def main(config_path: str):
    config = load_config(config_path)
    df = load_raw_df(config)
    dfp = preprocess(df, config)

    outdir = Path(config["paths"]["processed_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "features.parquet"
    dfp.to_parquet(outpath, index=False)
    print(f"[features] Saved: {outpath}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
