import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from src.data.load_data import load_config, load_raw_df

ALPHA = 0.05

def claim_frequency(df):
    return (df["TotalClaims"] > 0).mean()

def claim_severity(df):
    has = df["TotalClaims"] > 0
    if has.sum() == 0:
        return np.nan
    return df.loc[has, "TotalClaims"].mean()

def margin(df):
    return (df["TotalPremium"] - df["TotalClaims"]).mean()

def test_group_diff_numeric(a, b, equal_var=False):
    # Use Levene to test variance equality
    lev = stats.levene(a, b, center="median", nan_policy="omit")
    t = stats.ttest_ind(a, b, equal_var=(lev.pvalue >= ALPHA), nan_policy="omit")
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"levene_p": lev.pvalue, "ttest_p": t.pvalue, "mannwhitney_p": mw.pvalue}

def chi2_frequency(groups):
    # groups: dict[label] -> binary array (HasClaim)
    obs = np.array([ [ (v==1).sum(), (v==0).sum() ] for v in groups.values() ])
    chi2, p, dof, _ = stats.chi2_contingency(obs)
    return {"chi2": chi2, "p": p, "dof": dof, "table": obs.tolist()}

def run_tests(df: pd.DataFrame, config: dict):
    results = {}

    # Normalize columns
    df = df.copy()
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
    df["Margin"]   = df["TotalPremium"] - df["TotalClaims"]

    # 1) Provinces: risk difference (claim frequency)
    if config["columns"]["geo_province"] in df.columns:
        prov = df[config["columns"]["geo_province"]]
        groups = {k: g["HasClaim"].values for k,g in df.groupby(prov)}
        if len(groups) >= 2:
            results["risk_across_provinces_freq"] = chi2_frequency(groups)

        # Severity (among claimants) via ANOVA/Kruskal
        sev_groups = {k: g.loc[g["HasClaim"]==1, "TotalClaims"].values for k,g in df.groupby(prov)}
        sev_groups = {k: v for k,v in sev_groups.items() if len(v) > 1}
        if len(sev_groups) >= 2:
            arrays = list(sev_groups.values())
            kw = stats.kruskal(*arrays)
            results["risk_across_provinces_severity"] = {"kruskal_H": kw.statistic, "p": kw.pvalue}

    # 2) Zip codes: risk difference (frequency) + margin
    if config["columns"]["geo_zip"] in df.columns:
        zipc = df[config["columns"]["geo_zip"]].astype(str)
        groups = {k: g["HasClaim"].values for k,g in df.groupby(zipc)}
        if len(groups) >= 2:
            results["risk_across_zip_freq"] = chi2_frequency(groups)

        # Margin differences across zips (ANOVA/Kruskal)
        m_groups = {k: g["Margin"].values for k,g in df.groupby(zipc)}
        # use Kruskal (robust to non-normal)
        arrays = list(m_groups.values())[:20]  # limit to 20 groups to keep compute manageable
        if len(arrays) >= 2:
            kw = stats.kruskal(*arrays)
            results["margin_across_zip"] = {"kruskal_H": kw.statistic, "p": kw.pvalue, "note": "subset of zip groups (max 20) for compute"}

    # 3) Gender risk differences
    if "Gender" in df.columns:
        ggroups = {k: g["HasClaim"].values for k,g in df.groupby("Gender") if k in ["Male","Female"]}
        if len(ggroups) == 2:
            results["risk_gender_freq"] = chi2_frequency(ggroups)
        # Severity: compare Male vs Female severity (ttest/MW)
        s_m = df.loc[(df["Gender"]=="Male") & (df["HasClaim"]==1), "TotalClaims"].values
        s_f = df.loc[(df["Gender"]=="Female") & (df["HasClaim"]==1), "TotalClaims"].values
        if len(s_m) > 5 and len(s_f) > 5:
            results["risk_gender_severity"] = test_group_diff_numeric(s_m, s_f)

    # Decisions
    decisions = {}
    # Province frequency
    if "risk_across_provinces_freq" in results:
        decisions["H0_provinces_no_risk_diff"] = results["risk_across_provinces_freq"]["p"] >= 0.05
    if "risk_across_zip_freq" in results:
        decisions["H0_zip_no_risk_diff"] = results["risk_across_zip_freq"]["p"] >= 0.05
    if "margin_across_zip" in results:
        decisions["H0_zip_no_margin_diff"] = results["margin_across_zip"]["p"] >= 0.05
    if "risk_gender_freq" in results:
        decisions["H0_gender_no_risk_diff"] = results["risk_gender_freq"]["p"] >= 0.05

    return {"alpha": 0.05, "results": results, "decisions_fail_to_reject": decisions}

def main(config_path: str):
    config = load_config(config_path)
    df = load_raw_df(config)
    report = run_tests(df, config)

    repath = Path(config["paths"]["reports_dir"]) / "hypothesis_report.json"
    repath.parent.mkdir(parents=True, exist_ok=True)
    with open(repath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[tests] Wrote hypothesis report to {repath}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
