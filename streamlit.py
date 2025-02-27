import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fpdf import FPDF

# PCOS Introduction Section
st.title("Understanding PCOS (Polycystic Ovary Syndrome)")

st.markdown("""
## 📌 What is PCOS?
PCOS (Polycystic Ovary Syndrome) is a hormonal disorder that affects women of reproductive age. 
It can lead to irregular menstrual cycles, excessive hair growth, acne, and infertility.

## ⚠ Symptoms of PCOS:
- Irregular periods or no periods at all
- Excess androgen (leading to excess facial or body hair)
- Polycystic ovaries (enlarged ovaries with fluid-filled sacs)
- Weight gain or difficulty losing weight
- Thinning hair or hair loss
- Acne or oily skin

## 🔍 Causes of PCOS:
The exact cause of PCOS is unknown, but common factors include:
- **Hormonal imbalances** (such as high levels of insulin and androgens)
- **Genetics** (family history may increase risk)
- **Insulin resistance** (leads to weight gain and metabolic issues)
- **Inflammation** (chronic low-grade inflammation may contribute)

---
""")

st.write("### Now, let's explore the PCOS Health Dashboard! 🚀")

# Load and prepare dataset
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

X = df_cleaned.drop(columns=["PCOS (Y/N)"])
y = df_cleaned["PCOS (Y/N)"]
X_filled = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Dashboard layout
st.title("PCOS Health Dashboard")
st.sidebar.write(f"✅ Model Accuracy: {model_accuracy * 100:.2f}%")

# The rest of your code continues from here...
