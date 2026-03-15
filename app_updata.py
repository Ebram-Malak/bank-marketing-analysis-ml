import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import subprocess


# =========================
# Paths
# =========================
PROJECT_DIR = Path(r"https://github.com/Ebram-Malak/bank-marketing-analysis-ml")
DATA_PATH = Path(r"bank_customers_train.csv")
MODEL_PATH = PROJECT_DIR / "blob/main/final_model.pkl"

IMAGE_EXTENSIONS =  {".png", ".jpg", ".jpeg", ".webp", ".jfif"}

# =========================
# Loaders
# =========================
@st.cache_resource
def load_artifact(path: Path):
    artifact = joblib.load(path)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        threshold = artifact.get("threshold", 0.5)
        positive_label = artifact.get("positive_label", "yes")
        negative_label = artifact.get("negative_label", "no")
    else:
        model = artifact
        threshold = 0.5
        positive_label = "yes"
        negative_label = "no"

    return model, float(threshold), positive_label, negative_label


@st.cache_data
def load_dataset(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def get_project_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def predict_with_threshold(model, input_df, threshold, positive_label="yes", negative_label="no"):
    classes = list(model.classes_)
    pos_idx = classes.index(positive_label) if positive_label in classes else 1
    proba_yes = model.predict_proba(input_df)[:, pos_idx][0]
    pred_label = positive_label if proba_yes >= threshold else negative_label
    return pred_label, proba_yes


# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# Load resources
# =========================
model_loaded = False
load_error = None

try:
    model, threshold, positive_label, negative_label = load_artifact(MODEL_PATH)
    model_loaded = True
except Exception as e:
    load_error = str(e)

df = load_dataset(DATA_PATH)
project_images = get_project_images(PROJECT_DIR)

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Feature Explanation", "Prediction"]
)

# =========================
# PAGE 1: Overview
# =========================
if page == "Project Overview":
    st.title("🏦 Bank Term Deposit Subscription Prediction")

    st.markdown(
        """
        This application predicts whether a bank client is likely to subscribe to a **term deposit**
        based on client profile, campaign information, and economic indicators.
        """
    )

    if project_images:
        st.image(
            [str(img) for img in project_images[:3]],
            caption=[img.name for img in project_images[:3]],
            use_container_width=True
        )
    else:
        st.info(f"No images found in: {PROJECT_DIR}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Project Folder", str(PROJECT_DIR))
    with c2:
        st.metric("Dataset Loaded", "Yes" if df is not None else "No")
    with c3:
        st.metric("Threshold", f"{threshold:.3f}" if model_loaded else "N/A")

    if df is not None and "y" in df.columns:
        rows, cols = df.shape
        yes_count = int((df["y"] == "yes").sum())
        no_count = int((df["y"] == "no").sum())

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Rows", f"{rows:,}")
        with m2:
            st.metric("Features", cols - 1)
        with m3:
            st.metric("Subscribed (yes)", f"{yes_count:,}")
        with m4:
            st.metric("Not Subscribed (no)", f"{no_count:,}")

    st.markdown(
        """
        ### Project Objective
        Predict whether a client will subscribe to a bank term deposit.

        ### What was done
        - Data cleaning and preprocessing
        - Missing value handling
        - Categorical encoding and numerical scaling
        - Model training and hyperparameter tuning
        - Threshold optimization for better precision/recall trade-off

        ### Important Note
        The model uses an **optimized threshold**, not just the default 0.5.
        """
    )

    if not model_loaded:
        st.error("Model could not be loaded. Make sure `final_model.pkl` exists in the project folder.")
        st.code(load_error)

# =========================
# PAGE 2: Features
# =========================
elif page == "Feature Explanation":
    st.title("📘 Feature Description")

    feature_info = {
        "age": "Client age.",
        "job": "Client job type.",
        "marital": "Marital status.",
        "education": "Education level.",
        "default": "Whether the client has credit in default.",
        "housing": "Whether the client has a housing loan.",
        "loan": "Whether the client has a personal loan.",
        "contact": "Contact communication type.",
        "month": "Month of last contact.",
        "day_of_week": "Day of week of last contact.",
        "duration": "Last contact duration in seconds.",
        "campaign": "Number of contacts during this campaign.",
        "pdays": "Days since previous contact.",
        "previous": "Number of previous contacts.",
        "poutcome": "Outcome of previous campaign.",
        "emp.var.rate": "Employment variation rate.",
        "cons.price.idx": "Consumer price index.",
        "cons.conf.idx": "Consumer confidence index.",
        "euribor3m": "Euribor 3-month rate.",
        "nr.employed": "Number of employees indicator."
    }

    df_features = pd.DataFrame(
        [{"Feature": k, "Description": v} for k, v in feature_info.items()]
    )
    st.dataframe(df_features, use_container_width=True, hide_index=True)

# =========================
# PAGE 3: Prediction
# =========================
elif page == "Prediction":
    st.title("🤖 Predict Client Subscription")

    if not model_loaded:
        st.error("Model file is missing or could not be loaded.")
        st.code(load_error)
        st.stop()

    st.caption(f"Using optimized threshold = {threshold:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)

        job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician", "unemployed"
            ]
        )

        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])

        education = st.selectbox(
            "Education",
            [
                "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                "professional.course", "university.degree"
            ]
        )

        default = st.selectbox("Credit in Default", ["no", "yes"])
        housing = st.selectbox("Housing Loan", ["no", "yes"])
        loan = st.selectbox("Personal Loan", ["no", "yes"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])

        month = st.selectbox(
            "Last Contact Month",
            ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        )

        day_of_week = st.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])

    with col2:
        duration = st.number_input(
            "Last Contact Duration (seconds)",
            min_value=0, max_value=5000, value=180, step=10
        )

        campaign = st.number_input(
            "Campaign Contacts",
            min_value=1, max_value=100, value=2, step=1
        )

        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=0, max_value=999, value=999, step=1
        )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0, max_value=50, value=0, step=1
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["nonexistent", "failure", "success"]
        )

        cons_price_idx = st.number_input(
            "Consumer Price Index",
            min_value=90.0, max_value=100.0, value=93.2, step=0.01
        )

        cons_conf_idx = st.number_input(
            "Consumer Confidence Index",
            min_value=-60.0, max_value=-20.0, value=-40.0, step=0.1
        )

        euribor3m = st.number_input(
            "Euribor 3M",
            min_value=0.0, max_value=6.0, value=4.8, step=0.001, format="%.3f"
        )

    if st.button("🎯 Predict"):

        # Feature Engineering
        pdays_group = "recent_contact" if pdays <= 49 else "old_contact"

        if age <= 35:
            age_group = "young_age"
        elif age <= 60:
            age_group = "middle_age"
        else:
            age_group = "senior_age"

        campaign_group = (
            "very_low" if campaign == 1 else
            "low" if campaign == 2 else
            "medium" if campaign == 3 else
            "medium_high" if campaign == 4 else
            "high" if campaign == 5 else
            "very_high"
        )

        previous_group = "many_previous" if previous > 1 else "few_previous"

        # Final columns sent to the model
        input_df = pd.DataFrame([{
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "poutcome": poutcome,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "pdays_group": pdays_group,
            "age_group": age_group,
            "campaign_group": campaign_group,
            "previous_group": previous_group
        }])

        pred_label, proba_yes = predict_with_threshold(
            model,
            input_df,
            threshold,
            positive_label=positive_label,
            negative_label=negative_label
        )

        proba_no = 1 - proba_yes

        if pred_label == positive_label:
            st.success(f"Prediction: {pred_label.upper()} ✅")
        else:
            st.warning(f"Prediction: {pred_label.upper()} ⚠️")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Probability of YES", f"{proba_yes:.3f}")
        with m2:
            st.metric("Probability of NO", f"{proba_no:.3f}")
        with m3:
            st.metric("Applied Threshold", f"{threshold:.3f}")

        st.write("### Engineered Features Used")
        st.write(f"- age_group: {age_group}")
        st.write(f"- pdays_group: {pdays_group}")
        st.write(f"- campaign_group: {campaign_group}")
        st.write(f"- previous_group: {previous_group}")

app_path = Path(__file__).with_name("app_updata.py")

subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(app_path)
])

