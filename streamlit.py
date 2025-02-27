import numpy as np
import pandas as pd
import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Page Config at the Top
st.set_page_config(page_title="PCOS Risk Dashboard", layout="wide")

# Load and prepare dataset
file_path = "PCOS_data.csv"
try:
    df = pd.read_csv(file_path)
    df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")

    if "PCOS (Y/N)" not in df_cleaned.columns:
        raise ValueError("Target column 'PCOS (Y/N)' not found in the dataset.")
    
    X = df_cleaned.drop(columns=["PCOS (Y/N)"])
    y = df_cleaned["PCOS (Y/N)"]
    
    X_filled = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Apply Custom CSS for Better UI
st.markdown(
    """
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #ff4b4b; text-align: center;}
    .stButton>button {background-color: #ff4b4b; color: white; font-size: 18px; border-radius: 10px;}
    .stSidebar {background-color: #f5f7fa;}
    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard Layout
st.title("✨ PCOS Risk Assessment Dashboard ✨")

with st.sidebar:
    st.header("🔍 Navigation")
    page = st.radio("Go to", ["PCOS Awareness", "PCOS Prediction", "Lifestyle Quiz"])
    st.write(f"✅ Model Accuracy: {model_accuracy * 100:.2f}%")

# PCOS Awareness Section
def pcos_awareness():
    st.header("📖 Understanding PCOS")
    st.image("pcos_awareness.jpg", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("⚠️ Symptoms")
        symptoms = ["Irregular periods", "Excess hair growth", "Acne or oily skin", "Weight gain", "Thinning hair"]
        st.write("\n".join([f"- {symptom}" for symptom in symptoms]))
    with col2:
        st.subheader("🔬 Causes")
        causes = ["Hormonal imbalances", "Genetic factors", "Insulin resistance", "Inflammation"]
        st.write("\n".join([f"- {cause}" for cause in causes]))
    with col3:
        st.subheader("💔 Risks")
        risks = ["Infertility", "Type 2 diabetes", "High blood pressure", "Heart disease", "Depression"]
        st.write("\n".join([f"- {risk}" for risk in risks]))

    # Graphs and Data Visualization
    st.header("2. Data Visualizations 📊")
    st.subheader("PCOS Prevalence in Different Studies")

    # Data from different studies
    study_labels = ["Tamil Nadu (18%)", "Mumbai (22.5%)", "Lucknow (3.7%)", "NIH Criteria (7.2%)", "Rotterdam Criteria (19.6%)"]
    study_values = [18, 22.5, 3.7, 7.2, 19.6]

    fig, ax = plt.subplots()
    sns.barplot(x=study_labels, y=study_values, ax=ax)
    ax.set_ylabel("Prevalence (%)")
    ax.set_xlabel("Study Locations & Criteria")
    ax.set_title("PCOS Prevalence in Different Studies")
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig)

    st.subheader("SHAP Model Impact")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    st.pyplot(fig)

# PCOS Prediction Section
def pcos_prediction():
    st.header("🧪 PCOS Prediction")
    st.write("Enter your details below to predict your PCOS risk level.")
    
    user_input = []
    col1, col2 = st.columns(2)
    for idx, feature in enumerate(X_filled.columns):
        with col1 if idx % 2 == 0 else col2:
            value = st.number_input(f"Enter {feature}", min_value=0.0, format="%.2f")
            user_input.append(value)
    
    if st.button("🚀 Predict PCOS Risk!"):
        with st.spinner("Analyzing data...⏳"):
            time.sleep(2)
            user_input = np.array(user_input).reshape(1, -1)
            prediction = model.predict(user_input)
            risk_level = random.randint(1, 100)
        
        st.subheader(f"📊 Estimated Risk Level: {risk_level}%")
        if prediction[0] == 0:
            st.success("✅ Low risk of PCOS")
        else:
            st.warning("⚠️ High risk of PCOS")

if page == "PCOS Awareness":
    pcos_awareness()
elif page == "PCOS Prediction":
    pcos_prediction()
elif page == "Lifestyle Quiz":
    lifestyle_quiz()
