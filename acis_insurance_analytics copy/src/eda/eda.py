import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.data.load_data import load_config, load_raw_df

plt.rcParams["figure.figsize"] = (9, 6)

def loss_ratio(df):
    return (df["TotalClaims"].sum() / np.maximum(df["TotalPremium"].sum(), 1e-9))

def eda(df: pd.DataFrame, config: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    figs = outdir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Basic summaries
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "overall_loss_ratio": float(loss_ratio(df)),
        "date_min": str(df[config["columns"]["date"]].min()),
        "date_max": str(df[config["columns"]["date"]].max()),
    }
    with open(outdir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Distribution plots
    for col in ["TotalPremium", "TotalClaims", "CustomValueEstimate"]:
        if col in df.columns:
            plt.figure()
            df[col].dropna().hist(bins=50)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col); plt.ylabel("Count")
            plt.tight_layout(); plt.savefig(figs / f"dist_{col}.png"); plt.close()

    # Outliers via boxplots
    for col in ["TotalPremium", "TotalClaims"]:
        if col in df.columns:
            plt.figure()
            df[[col]].boxplot()
            plt.title(f"Boxplot of {col}")
            plt.tight_layout(); plt.savefig(figs / f"box_{col}.png"); plt.close()

    # Temporal trend of claims & premium
    date_col = config["columns"]["date"]
    if date_col in df.columns:
        monthly = (
            df.dropna(subset=[date_col])
              .assign(month=lambda d: d[date_col].dt.to_period("M").dt.to_timestamp())
              .groupby("month")[["TotalPremium","TotalClaims"]].sum()
              .reset_index()
        )
        if not monthly.empty:
            plt.figure()
            plt.plot(monthly["month"], monthly["TotalPremium"], label="TotalPremium")
            plt.plot(monthly["month"], monthly["TotalClaims"], label="TotalClaims")
            plt.title("Monthly Premium vs Claims")
            plt.xlabel("Month"); plt.ylabel("Amount")
            plt.legend(); plt.tight_layout()
            plt.savefig(figs / "trend_premium_claims.png"); plt.close()

    # Loss ratio by Province / VehicleType / Gender
    for facet in [config["columns"]["geo_province"], config["columns"]["vehicle_type"], config["columns"]["gender"]]:
        if facet in df.columns:
            grp = df.groupby(facet)[["TotalPremium", "TotalClaims"]].sum()
            grp["LossRatio"] = grp["TotalClaims"] / grp["TotalPremium"].replace(0, np.nan)
            grp = grp.sort_values("LossRatio", ascending=False).head(20)
            plt.figure()
            plt.bar(grp.index.astype(str), grp["LossRatio"])
            plt.title(f"Loss Ratio by {facet}")
            plt.xticks(rotation=45, ha="right"); plt.ylabel("Loss Ratio")
            plt.tight_layout(); plt.savefig(figs / f"lossratio_by_{facet}.png"); plt.close()

    # Creative plot 1: Heatmap Province x VehicleType (avg loss ratio)
    if all(c in df.columns for c in [config["columns"]["geo_province"], config["columns"]["vehicle_type"]]):
        pivot = (df.groupby([config["columns"]["geo_province"], config["columns"]["vehicle_type"]])[["TotalPremium","TotalClaims"]]
                   .sum())
        pivot = (pivot["TotalClaims"] / pivot["TotalPremium"]).unstack().replace([np.inf, -np.inf], np.nan)
        pivot = pivot.fillna(0)
        plt.figure()
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="Loss Ratio")
        plt.title("Loss Ratio Heatmap: Province x VehicleType")
        plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
        plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(figs / "heatmap_province_vehicle.png"); plt.close()

    # Creative plot 2: Top/Bottom makes by mean claims
    if "Make" in df.columns:
        mk = df.groupby("Make")["TotalClaims"].mean().sort_values(ascending=False)
        top = mk.head(10); bot = mk.tail(10)
        plt.figure()
        plt.bar(top.index.astype(str), top.values)
        plt.title("Top 10 Makes by Mean Claim Severity")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(figs / "top_makes_claims.png"); plt.close()

        plt.figure()
        plt.bar(bot.index.astype(str), bot.values)
        plt.title("Bottom 10 Makes by Mean Claim Severity")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(figs / "bottom_makes_claims.png"); plt.close()

    # Creative plot 3: ZipCode correlation scatter (monthly deltas)
    zip_col = config["columns"]["geo_zip"]
    if all(c in df.columns for c in [zip_col, date_col]):
        tmp = df.dropna(subset=[date_col]).assign(month=lambda d: d[date_col].dt.to_period("M").dt.to_timestamp())
        agg = tmp.groupby([zip_col, "month"])[["TotalPremium","TotalClaims"]].sum().reset_index()
        agg = agg.sort_values(["PostalCode", "month"])
        agg["dPrem"] = agg.groupby(zip_col)["TotalPremium"].diff()
        agg["dClaim"] = agg.groupby(zip_col)["TotalClaims"].diff()
        agg = agg.dropna(subset=["dPrem","dClaim"])
        if not agg.empty:
            plt.figure()
            plt.scatter(agg["dPrem"], agg["dClaim"], alpha=0.5)
            plt.title("Zip-level monthly change: ΔPremium vs ΔClaims")
            plt.xlabel("Δ TotalPremium"); plt.ylabel("Δ TotalClaims")
            plt.tight_layout(); plt.savefig(figs / "delta_scatter_zip.png"); plt.close()

def main(config_path: str):
    config = load_config(config_path)
    df = load_raw_df(config)
    outdir = Path(config["paths"]["reports_dir"])
    eda(df, config, outdir)
    print("[eda] Completed. Figures saved to reports/figures")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
