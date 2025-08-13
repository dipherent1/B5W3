from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    numeric_impute_strategy: str
    categorical_impute_strategy: str

def split_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target].copy()
    X = df.drop(columns=[target])
    return X, y

def build_preprocessor(X: pd.DataFrame, cfg: FeatureConfig) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg.numeric_impute_strategy)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg.categorical_impute_strategy)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre
