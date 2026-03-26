import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_squared_error
)
import xgboost as xgb
import lightgbm as lgb
import pickle

def detect_task(y: pd.Series) -> str:
    """Auto-detect if task is classification or regression."""
    if y.dtype == object or y.nunique() <= 10:
        return "classification"
    return "regression"

def preprocess(df: pd.DataFrame):
    """Encode categoricals, fill nulls, handle infinity, split X and y."""
    import numpy as np

    # Replace infinity values with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with too many nulls
    df = df.dropna(thresh=len(df) * 0.5, axis=1)

    # Fill remaining nulls with median
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def train_models(df: pd.DataFrame):
    """Train 4 models, return results and best model."""
    X, y = preprocess(df)
    task = detect_task(y)

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
        }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained[name] = model

        if task == "classification":
            score = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            results.append({
                "Model": name,
                "Accuracy": round(score, 4),
                "F1 Score": round(f1, 4)
            })
        else:
            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)
            results.append({
                "Model": name,
                "R² Score": round(r2, 4),
                "RMSE": round(rmse, 4)
            })

    results_df = pd.DataFrame(results)

    # Pick best model
    sort_col = "Accuracy" if task == "classification" else "R² Score"
    best_name = results_df.sort_values(sort_col, ascending=False).iloc[0]["Model"]
    best_model = trained[best_name]

    return results_df, best_model, best_name, task, X_train, X_test, list(X.columns)