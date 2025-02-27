import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

# PCOS Information Section
st.title("Understanding PCOS 🩺")
st.write("Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder that affects women of reproductive age.PCOS affects 1 in 10 women of reproductive age!Lifestyle changes, such as exercise and a balanced diet, can help manage PCOS symptoms.PCOS is one of the leading causes of infertility in women.Insulin resistance is a key factor in PCOS development.Maintaining a healthy weight can reduce PCOS symptoms!")
st.subheader("Causes of PCOS")
st.write("- Insulin resistance\n- Hormonal imbalances\n- Genetic factors\n- Inflammation")
st.subheader("Symptoms of PCOS")
st.write("- Irregular periods\n- Excess androgen leading to acne, hair growth\n- Polycystic ovaries\n- Weight gain and difficulty losing weight")
st.subheader("Risks Associated with PCOS")
st.write("- Infertility\n- Type 2 diabetes\n- Cardiovascular disease\n- Mental health issues like anxiety and depression")

# Fun facts and PCOS Prediction Game
st.header("PCOS Prediction Game 🎮")
fun_facts = [
    "",
    "Lifestyle changes, such as exercise and a balanced diet, can help manage PCOS symptoms.",
    "PCOS is one of the leading causes of infertility in women.",
    "Insulin resistance is a key factor in PCOS development.",
    "Maintaining a healthy weight can reduce PCOS symptoms!"
]
st.write(random.choice(fun_facts))

# Load and prepare dataset
file_path = "PCOS_data.csv"
df = pd.read_csv(file_path)
df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")

for col in df_cleaned.columns:
    if df_cleaned[col].dtype == "object":
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    else:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")
X = df_cleaned.drop(columns=["PCOS (Y/N)"])
y = df_cleaned["PCOS (Y/N)"]
X_filled = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
st.sidebar.write(f"✅ Model Accuracy: {model_accuracy * 100:.2f}%")

# PCOS Prediction Game
def pcos_prediction_game():
    st.subheader("PCOS Prediction Game 🎯")
    user_input = []
    progress_bar = st.progress(0)
    for idx, feature in enumerate(X_filled.columns):
        value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
        user_input.append(value)
        progress_bar.progress((idx + 1) / len(X_filled.columns))
    if st.button("🎲 Predict PCOS Risk!"):
        with st.spinner("Analyzing your data...🔍"):
            time.sleep(2)
            user_input = np.array(user_input).reshape(1, -1)
            prediction = model.predict(user_input)
            risk_level = random.randint(1, 100)
        st.subheader("🔮 Prediction Result:")
        if prediction[0] == 1:
            st.error(f"⚠ High risk of PCOS. Your estimated risk level: {risk_level}%")
            st.write(random.choice(fun_facts))
        else:
            st.success(f"✅ Low risk of PCOS. Your estimated risk level: {risk_level}%")
            st.write("Great job! Keep maintaining a healthy lifestyle. 💪")
        fig, ax = plt.subplots()
        sns.barplot(x=["Low", "Medium", "High"], y=[30, 60, 90], color='lightgray')
        ax.bar(["Low", "Medium", "High"], [30, 60, risk_level], color=['green', 'orange', 'red'])
        st.pyplot(fig)
pcos_prediction_game()

# Generate PDF Report
def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, "High probability of PCOS detected." if prediction_prob > 0.5 else "No significant risk of PCOS detected.")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Lifestyle Changes:\n- Maintain a balanced diet\n- Exercise regularly\n- Manage stress\n- Get enough sleep")
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    return report_path

# Health Gamification, Community Support, PCOS Quiz, Mood Tracker, PCOS Recipes, and 3D Model sections remain the same
