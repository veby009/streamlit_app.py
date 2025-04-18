import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cyber Threat Detection", layout="wide")

st.title("🛡️ Cyber Threat Detection with Isolation Forest")
st.markdown("This Streamlit dashboard detects anomalies in enterprise network traffic using unsupervised learning.")

# Upload dataset
uploaded_file = st.file_uploader("Upload cleaned network data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded")

    if 'label' in df.columns:
        label_present = True
        y = df['label']
        X = df.drop(columns=['label'])
    else:
        label_present = False
        X = df.copy()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    iso_model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    preds = iso_model.fit_predict(X_scaled)
    df['anomaly_score'] = preds
    df['predicted_label'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # Show stats
    st.subheader("📊 Data Overview")
    st.write(df.head())

    st.subheader("🧮 Anomaly Detection Summary")
    st.write(df['predicted_label'].value_counts().rename(index={0: "Normal", 1: "Anomaly"}))

    # Plot
    st.subheader("📉 Anomaly Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='anomaly_score', bins=30, hue='predicted_label', palette='coolwarm', ax=ax)
    st.pyplot(fig)

    # Optional: Compare to actual labels
    if label_present:
        from sklearn.metrics import classification_report, confusion_matrix
        st.subheader("🧾 Evaluation vs Ground Truth")
        st.text(classification_report(y, df['predicted_label'], target_names=["Normal", "Attack"]))

    # Download option
    st.download_button("⬇️ Download Results CSV", df.to_csv(index=False), file_name="anomaly_results.csv")

else:
    st.warning("👈 Please upload a CSV to begin.")
