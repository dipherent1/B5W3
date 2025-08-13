import pandas as pd
import yaml
from pathlib import Path

def load_config(path: str | Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_raw_df(config):
    csv_path = Path(config["paths"]["raw_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path} (update configs/config.yaml)")
    df = pd.read_csv(csv_path)
    # Ensure date parsing
    date_col = config["columns"]["date"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df
