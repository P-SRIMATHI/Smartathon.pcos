import numpy as np
import pandas as pd
import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    st.sidebar.write(f"✅ Model Accuracy: {model_accuracy * 100:.2f}%")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Interactive PCOS Information Section
def pcos_awareness():
    st.title("🔍 Understanding PCOS")
    st.markdown("## What is PCOS?")
    st.write("PCOS (Polycystic Ovary Syndrome) is a common hormonal disorder affecting people with ovaries.")
    
    st.markdown("### Symptoms of PCOS")
    symptoms = ["Irregular periods", "Excess hair growth", "Acne or oily skin", "Weight gain", "Thinning hair"]
    st.write("\n".join([f"- {symptom}" for symptom in symptoms]))
    
    st.markdown("### Causes of PCOS")
    causes = ["Hormonal imbalances", "Genetic factors", "Insulin resistance", "Inflammation"]
    st.write("\n".join([f"- {cause}" for cause in causes]))
    
    st.markdown("### Risks of PCOS")
    risks = ["Infertility", "Type 2 diabetes", "High blood pressure", "Heart disease", "Depression"]
    st.write("\n".join([f"- {risk}" for risk in risks]))
    
    if st.button("Proceed to PCOS Risk Prediction ➡️"):
        st.session_state['page'] = 'prediction'

# PCOS Risk Prediction
def pcos_prediction():
    st.title("🧪 PCOS Prediction")
    user_input = []
    progress_bar = st.progress(0)
    
    for idx, feature in enumerate(X_filled.columns):
        value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
        user_input.append(value)
        progress_bar.progress((idx + 1) / len(X_filled.columns))
    
    if st.button("Predict PCOS Risk!"):
        with st.spinner("Analyzing data..."):
            time.sleep(2)
            user_input = np.array(user_input).reshape(1, -1)
            prediction = model.predict(user_input)
            risk_level = random.randint(1, 100)
        
        if prediction[0] == 0:
            st.success(f"✅ Low risk of PCOS. Estimated risk level: {risk_level}%")
        else:
            st.warning(f"⚠️ High risk of PCOS. Estimated risk level: {risk_level}%")
        
        if st.button("Take the PCOS Lifestyle Quiz ➡️"):
            st.session_state['page'] = 'quiz'

# PCOS Lifestyle Quiz
def personality_quiz():
    st.title("🩺 PCOS Lifestyle Risk Assessment")
    questions = {
        "How often do you exercise?": {"Daily": 0, "3-5 times a week": 10, "1-2 times a week": 20, "Rarely": 30},
        "How would you rate your diet?": {"Excellent": 0, "Good": 10, "Average": 20, "Poor": 30},
        "Do you have irregular periods?": {"Never": 0, "Occasionally": 10, "Often": 20, "Always": 30},
        "How stressed are you daily?": {"Not at all": 0, "Mildly": 10, "Moderately": 20, "Highly stressed": 30},
        "How many hours do you sleep?": {"More than 8": 0, "7-8 hours": 10, "5-6 hours": 20, "Less than 5": 30}
    }
    
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, list(options.keys()), index=0)
        score += options[answer]
    
    st.subheader(f"📊 Your Risk Score: **{score}**")
    if score < 40:
        st.success("✅ Low risk! Keep up the healthy habits!")
    elif score < 70:
        st.warning("⚠️ Moderate risk! Consider improving lifestyle choices.")
    else:
        st.error("🚨 High risk! Seek medical advice.")
    
    if st.button("Back to Home 🏠"):
        st.session_state['page'] = 'awareness'

# Main Navigation
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'awareness'
    
    if st.session_state['page'] == 'awareness':
        pcos_awareness()
    elif st.session_state['page'] == 'prediction':
        pcos_prediction()
    elif st.session_state['page'] == 'quiz':
        personality_quiz()

if __name__ == "__main__":
    main()
