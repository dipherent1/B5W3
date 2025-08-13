import argparse, json
from pathlib import Path
from src.data.load_data import load_config
from src.models.claim_probability import main as train_prob
from src.models.claim_severity import main as train_sev
from src.models.premium_model import main as compute_prem

def main(config_path: str):
    # Train severity & probability models, then compute premiums
    train_sev(config_path)
    train_prob(config_path)
    compute_prem(config_path)
    print("[train_all] Completed modeling pipeline.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
