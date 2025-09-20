import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

# Load dataset
df = pd.read_csv("winequality-red-selected-missing.csv")

# Clean column names
df.columns = [c.strip() for c in df.columns]

# Create binary target
df["quality_good"] = (df["quality"] >= 7).astype(int)

# Select numeric features
features = df.drop(columns=["quality", "quality_good"]).select_dtypes(include=np.number).columns.tolist()
X = df[features]
y = df["quality_good"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), features)
])

# Model
base_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)

# Pipeline
pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", model)
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_proba = pipeline.predict_proba(X_test)[:, 1]
metrics = {
    "roc_auc": roc_auc_score(y_test, y_proba),
    "f1": f1_score(y_test, y_proba > 0.5),
    "brier": brier_score_loss(y_test, y_proba),
    "features": features
}

# Save artifacts
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(pipeline, "artifacts/wine_quality_pipeline.joblib")
with open("artifacts/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open("artifacts/schema.json", "w") as f:
    json.dump({"features": features}, f, indent=2)

print("Training complete. Artifacts saved in /artifacts")
