import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from disk."""
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset before encoding."""
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).astype('category')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    logger.debug("Cleaned data types:\n%s", df.dtypes)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing transformer for categorical variables."""
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Churn']

    logger.info("Categorical columns: %s", categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'ohe',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                categorical_cols,
            )
        ],
        remainder='passthrough',
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract clean feature names after preprocessing."""
    raw_names = preprocessor.get_feature_names_out()
    cleaned_names = [name.replace('ohe__', '').replace('remainder__', '') for name in raw_names]
    return list(cleaned_names)


def preprocess_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Clean the data and apply preprocessing to features."""
    df_clean = clean_data(df)
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    if preprocessor is None:
        preprocessor = build_preprocessor(X)
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)

    feature_names = get_feature_names(preprocessor)
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    logger.info("Preprocessed feature matrix shape: %s", X_df.shape)
    return X_df, y, preprocessor


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def describe_class_balance(y: pd.Series) -> None:
    """Log class balance information and detect imbalance."""
    counts = y.value_counts(normalize=True)
    imbalance_ratio = counts.max() / counts.min()
    logger.info("Class balance: %s", counts.to_dict())
    logger.info("Imbalance ratio: %.2f", imbalance_ratio)
    if imbalance_ratio > 1.5:
        logger.warning("Detected class imbalance. Using class weights for training.")
