import sys
from pathlib import Path
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Configuration (paths relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "data.csv"
MODELS_DIR = REPO_ROOT / "models"

def find_label_column(df: pd.DataFrame):
    candidates = ['label', 'class', 'gesture', 'target', 'y', 'annotation', 'label_id', 'sign']
    for c in candidates:
        if c in df.columns:
            return c
    # prefer object / categorical columns with reasonable cardinality
    for col in df.columns[::-1]:  # try last columns first (common pattern)
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    # fallback: last column
    return df.columns[-1]

def prepare_features(df: pd.DataFrame, label_col: str, max_cat_unique=50):
    # Separate target
    y = df[label_col].copy()
    X = df.drop(columns=[label_col])

    # Inspect and handle categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    dropped_cols = []
    for col in cat_cols:
        uniq = X[col].nunique(dropna=True)
        if uniq == 0:
            dropped_cols.append(col)
        elif uniq <= max_cat_unique:
            # expand to dummies (drop first to avoid collinearity)
            X = pd.get_dummies(X, columns=[col], prefix=[col], drop_first=True)
        else:
            # too many categories -> drop (could log alternative treatments)
            dropped_cols.append(col)
            X = X.drop(columns=[col])

    # Keep only numeric columns after encoding
    numeric_X = X.select_dtypes(include=[np.number])
    non_numeric_remaining = set(X.columns) - set(numeric_X.columns)
    if non_numeric_remaining:
        # Drop any leftover non-numeric columns
        numeric_X = numeric_X.copy()
        numeric_X = numeric_X  # explicit

    return numeric_X, y, dropped_cols

def main():
    if not DATA_FILE.exists():
        print(f"Error: data file not found at {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")
    if df.empty:
        print("Error: dataset is empty.")
        sys.exit(1)

    print("Columns and dtypes:")
    print(df.dtypes.to_string())

    label_col = find_label_column(df)
    if label_col not in df.columns:
        print("Error: could not determine label column. Columns available:", df.columns.tolist())
        sys.exit(1)

    print(f"Using '{label_col}' as label column.")

    # Drop rows where label is missing
    before = len(df)
    df = df.dropna(subset=[label_col])
    if len(df) != before:
        print(f"Dropped {before - len(df)} rows with missing label.")

    X, y, dropped = prepare_features(df, label_col)
    if dropped:
        print(f"Dropped non-usable categorical columns: {dropped}")

    if X.shape[1] == 0:
        print("Error: no numeric features left after preprocessing. Inspect your CSV.")
        sys.exit(1)

    # Encode target if non-numeric
    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))
        print(f"Label encoding applied. Classes: {label_encoder.classes_.tolist()}")
    else:
        y_encoded = y.values

    # Impute missing feature values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if len(np.unique(y_encoded))>1 else None)

    # Train
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n--- Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"rf_model_{timestamp}.joblib"
    artifacts = {
        "model": clf,
        "imputer": imputer,
        "feature_columns": X.columns.tolist(),
        "label_encoder": label_encoder,  # may be None if target numeric
        "label_column": label_col,
    }
    joblib.dump(artifacts, model_file)
    print(f"Saved model and preprocessing artifacts to {model_file}")

if __name__ == "__main__":
    main()
