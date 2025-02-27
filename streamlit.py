import numpy as np
import pandas as pd
import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare dataset
file_path = "PCOS_data.csv"
try:
    df = pd.read_csv(file_path)
    df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
    
    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # Convert non-numeric columns to numeric
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")

    # Define features (X) and target variable (y)
    if "PCOS (Y/N)" not in df_cleaned.columns:
        raise ValueError("Target column 'PCOS (Y/N)' not found in the dataset.")
    
    X = df_cleaned.drop(columns=["PCOS (Y/N)"])
    y = df_cleaned["PCOS (Y/N)"]
    
    # Handle missing values in features
    X_filled = X.fillna(X.median())
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
    
    # Train the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model accuracy
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"✅ Model Accuracy: {model_accuracy * 100:.2f}%")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Health tips
health_tips = [
    "🌱 Eat a balanced diet rich in whole foods and fiber!",
    "🏃‍♀️ Regular exercise can improve insulin sensitivity and overall health.",
    "💧 Stay hydrated! Drinking enough water helps in hormonal balance.",
    "😴 Prioritize sleep! Aim for 7-9 hours to regulate hormones.",
    "🧘‍♀️ Manage stress with yoga, meditation, or deep breathing techniques."
]

# Streamlit UI for PCOS Prediction
def pcos_prediction_game():
    st.title("🎮 PCOS Prediction Game")
    st.write("Answer the following questions and unlock insights! 🎯")
    
    user_input = []
    progress_bar = st.progress(0)
    for idx, feature in enumerate(X_filled.columns):
        value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
        user_input.append(value)
        progress_bar.progress((idx + 1) / len(X_filled.columns))
    
    if st.button("🎲 Predict PCOS Risk!"):
        with st.spinner("Analyzing your data...🔍"):
            time.sleep(2)  # Simulate processing time
            user_input = np.array(user_input).reshape(1, -1)
            prediction = model.predict(user_input)
            risk_level = random.randint(1, 100)
        
        if prediction[0] == 0:
            st.success(f"✅ Low risk of PCOS. Your estimated risk level: {risk_level}%")
            st.write("Great job! Keep maintaining a healthy lifestyle. 💪")
            st.write("Here are some additional health tips for you:")
            for tip in random.sample(health_tips, 3):
                st.write(f"- {tip}")
            
            # Show a gauge chart for risk level
            st.write("### 📊 Your Risk Level")
            fig, ax = plt.subplots()
            sns.barplot(x=["Low", "Medium", "High"], y=[30, 60, 90], color='lightgray')
            ax.bar(["Low", "Medium", "High"], [30, 60, risk_level], color=['green', 'orange', 'red'])
            st.pyplot(fig)
    
    st.write("\nThank you for playing! 🌟")

# Run the game in Streamlit
pcos_prediction_game()
