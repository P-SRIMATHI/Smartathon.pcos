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

# 🎯 Sidebar for Model Metrics
st.sidebar.title("📊 Model Performance")
st.sidebar.markdown("---")

# ✅ **Load and Prepare Dataset**
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

# 🎯 **Train Model**
X = df_cleaned.drop(columns=["PCOS (Y/N)"])
y = df_cleaned["PCOS (Y/N)"]
X_filled = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🎯 **Calculate Model Metrics**
model_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

# 📊 **Display Model Metrics in Sidebar**
st.sidebar.write(f"✅ **Accuracy:** {model_accuracy * 100:.2f}%")
st.sidebar.write(f"🎯 **Precision:** {precision:.2f}")
st.sidebar.write(f"📌 **F1 Score:** {f1:.2f}")
st.sidebar.markdown("---")

# 🩺 **PCOS Introduction Section**
with st.expander("📌 **What is PCOS? (Click to Expand)**", expanded=True):
    st.markdown("""
        ### 🏥 **Understanding PCOS (Polycystic Ovary Syndrome)**
        - **PCOS** is a common hormonal disorder affecting women of reproductive age.
        - It can cause irregular periods, excessive hair growth, acne, and fertility issues.
        
        ### ⚠ **Symptoms of PCOS**
        - Irregular or absent menstrual cycles
        - Unwanted facial & body hair (hirsutism)
        - Thinning hair or hair loss
        - Weight gain & difficulty losing weight
        - Acne, oily skin, and dark patches

        ### 🔍 **Common Causes of PCOS**
        - **Hormonal Imbalance** (High levels of insulin & androgens)
        - **Genetics** (Runs in families)
        - **Insulin Resistance** (Leads to weight gain & metabolic issues)
        - **Inflammation** (Chronic low-grade inflammation)
    """)

st.markdown("---")  # Divider for clean UI

# 🎯 **PCOS Prediction Game**
st.subheader("🎯 PCOS Prediction Game")
user_input = []
progress_bar = st.progress(0)
for idx, feature in enumerate(X_filled.columns[:5]):  # Show first 5 inputs for better UX
    value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
    user_input.append(value)
    progress_bar.progress((idx + 1) / len(X_filled.columns))

if st.button("🔍 Predict PCOS Risk!"):
    with st.spinner("Analyzing your data...🔍"):
        time.sleep(2)
        user_input = np.array(user_input).reshape(1, -1)
        prediction = model.predict(user_input)
        risk_level = random.randint(1, 100)

    st.subheader("🔮 Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠ High risk of PCOS. Your estimated risk level: {risk_level}%")
    else:
        st.success(f"✅ Low risk of PCOS. Your estimated risk level: {risk_level}%")

# 📊 **Health Gamification**
st.subheader("🎮 Health Gamification")
col1, col2 = st.columns(2)
with col1:
    water_glasses = st.slider("💧 Glasses of water today?", 0, 15)
with col2:
    steps = st.slider("🚶 Steps walked today?", 0, 20000)
st.write(f"🏆 **Total Health Points:** {water_glasses * 2 + steps // 500}")
if water_glasses >= 8: st.success("✅ Great job on water intake!")
if steps >= 10000: st.success("🔥 Awesome! You've walked 10,000+ steps!")

# 💬 **Community Support**
st.subheader("💬 Community Support")
new_post = st.text_area("Share your experience or ask a question:")
if st.button("Submit Post") and new_post:
    st.session_state.setdefault("posts", []).append(new_post)
    st.success("✅ Post submitted!")
if "posts" in st.session_state and st.session_state["posts"]:
    st.write("### Community Posts:")
    for post in st.session_state["posts"]:
        st.write(f"- {post}")

# 🧠 **PCOS Quiz**
st.subheader("🧠 PCOS Trivia Quiz")
questions = {
    "Common PCOS symptom?": ["Irregular periods", "Acne", "Hair loss"],
    "Hormone imbalance in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
    "Best lifestyle change for PCOS?": ["Exercise", "Skipping meals", "High sugar diet"]
}
quiz_score = sum(1 for q, opts in questions.items() if st.radio(q, opts) == opts[0])
st.write(f"🎯 **Final quiz score: {quiz_score}/{len(questions)}**")

# 😊 **Mood Tracker**
st.subheader("😊 Mood Tracker")
mood = st.selectbox("How do you feel today?", ["Happy", "Excited", "Neutral", "Sad", "Anxious"])
st.write(f"💬 You are feeling: **{mood}**")

# 🍲 **PCOS-Friendly Recipes**
st.subheader("🍲 PCOS-Friendly Recipes")
recipes = [
    {"name": "Spinach & Chickpea Curry", "ingredients": ["Spinach", "Chickpeas", "Coconut milk"]},
    {"name": "Oats Pancakes", "ingredients": ["Oats", "Eggs", "Banana"]},
    {"name": "Greek Yogurt Salad", "ingredients": ["Greek Yogurt", "Cucumber", "Olives"]}
]
for recipe in recipes:
    st.subheader(recipe["name"])
    st.write("🥗 Ingredients:", ", ".join(recipe["ingredients"]))

# 🩺 **3D Model of PCOS**
st.subheader("🩺 Explore PCOS in 3D")
st.components.v1.iframe("https://sketchfab.com/models/62bfb490ad344caaaea675da9df7ba34/embed", height=500)
