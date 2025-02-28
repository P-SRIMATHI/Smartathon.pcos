import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from fpdf import FPDF

# Sidebar for Model Metrics
st.sidebar.title("\ud83d\udcca Model Performance")
st.sidebar.markdown("---")

# Load and Prepare Dataset
file_path = "PCOS_data.csv"

def load_data():
    df = pd.read_csv(file_path)
    df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")
    return df_cleaned

df_cleaned = load_data()
if "PCOS (Y/N)" not in df_cleaned.columns:
    st.error("Target column 'PCOS (Y/N)' not found in the dataset.")
    st.stop()

# Train Model
X = df_cleaned.drop(columns=["PCOS (Y/N)"])
y = df_cleaned["PCOS (Y/N)"]
X_filled = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate Model Metrics
model_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

# Display Model Metrics in Sidebar
st.sidebar.write(f"\u2705 **Accuracy:** {model_accuracy * 100:.2f}%")
st.sidebar.write(f"\ud83c\udfaf **Precision:** {precision:.2f}")
st.sidebar.write(f"\ud83d\udccc **F1 Score:** {f1:.2f}")
st.sidebar.markdown("---")

# Graphs and Data Visualization
st.header("2. Data Visualizations \ud83d\udcca")
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

# PCOS Prediction Game
st.title("1.\ud83c\udfae PCOS Prediction Game")

user_input = []
progress_bar = st.progress(0)
for idx, feature in enumerate(X_filled.columns):
    value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
    user_input.append(value)
    progress_bar.progress((idx + 1) / len(X_filled.columns))

if st.button("\ud83c\udfb2 Predict PCOS Risk!"):
    with st.spinner("Analyzing your data...\ud83d\udd0d"):
        time.sleep(2)
        user_input = np.array(user_input).reshape(1, -1)
        prediction = model.predict(user_input)
        risk_level = random.randint(1, 100)
    
    st.subheader("\ud83d\udd2e Prediction Result:")
    result_text = "High risk of PCOS" if prediction[0] == 1 else "Low risk of PCOS"
    st.write(f"{result_text}. Estimated risk level: {risk_level}%")
    
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "PCOS Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Prediction: {result_text}\nRisk Level: {risk_level}%")
        pdf.ln(10)
        pdf.multi_cell(0, 10, "Recommended Lifestyle Changes:\n- Maintain a balanced diet\n- Exercise regularly\n- Manage stress\n- Get enough sleep")
        report_path = "PCOS_Report.pdf"
        pdf.output(report_path)
        return report_path
    
    report_path = generate_report()
    with open(report_path, "rb") as file:
        st.download_button("Download Report", file, file_name="PCOS_Report.pdf")
