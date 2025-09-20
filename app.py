import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Predictor")
st.caption("Enter wine attributes to predict if it's premium quality (≥ 7)")

# Load model and schema
pipe = joblib.load("artifacts/wine_quality_pipeline.joblib")
with open("artifacts/schema.json") as f:
    schema = json.load(f)
features = schema["features"]

# Input form
st.subheader("🔬 Single Sample Prediction")
inputs = {}
for feat in features:
    inputs[feat] = st.number_input(feat, value=0.0)

threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    proba = pipe.predict_proba(df)[0][1]
    label = "✅ Good Quality" if proba >= threshold else "❌ Not Good"
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {proba:.2f}")
    st.progress(int(proba * 100))

# Batch scoring
st.subheader("📁 Batch CSV Prediction")
file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        proba = pipe.predict_proba(df[features])[:, 1]
        df["prob_good"] = proba
        df["pred_good"] = (proba >= threshold).astype(int)
        st.write(df.head())
        st.download_button("Download Results", df.to_csv(index=False).encode(), "wine_predictions.csv")
