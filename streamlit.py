import streamlit as st
import random
import time
import plotly.graph_objects as go

def risk_meter(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "PCOS Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if score > 70 else "orange" if score > 40 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    return fig

def personality_quiz():
    st.title("🩺 PCOS Lifestyle Risk Assessment")
    
    questions = {
        "How often do you exercise?": ["Rarely", "1-2 times a week", "3-5 times a week", "Daily"],
        "How would you rate your diet?": ["Poor", "Average", "Good", "Excellent"],
        "Do you have irregular menstrual cycles?": ["Never", "Occasionally", "Often", "Always"],
        "How stressed do you feel daily?": ["Not at all", "Mildly", "Moderately", "Highly stressed"],
        "How many hours of sleep do you get per night?": ["Less than 5", "5-6 hours", "7-8 hours", "More than 8"]
    }
    
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, options, index=1)
        score += options.index(answer) * 25  # Assign risk scores dynamically
        st.progress(score // len(questions))
        time.sleep(0.5)  # Simulate progress
    
    return score

def get_recommendations(score):
    if score < 40:
        return "Great job! Keep up your healthy habits! 🌟"
    elif score < 70:
        return "You're doing well, but there's room for improvement. Consider more balanced meals and exercise! 🏋️‍♀️"
    else:
        return "Your lifestyle suggests a higher risk. Consult a specialist and make small, sustainable changes! ❤️"

def get_motivational_message():
    messages = [
        "Your health journey starts with small steps! 🚶‍♀️",
        "Consistency is key to a healthier you! 🔑",
        "Healthy habits today mean a better future! 🌱",
        "Your body loves when you take care of it! ❤️"
    ]
    return random.choice(messages)

def main():
    score = personality_quiz()
    st.subheader("📊 Your PCOS Risk Score: " + str(score))
    st.plotly_chart(risk_meter(score))
    
    st.markdown(f"### 💡 {get_recommendations(score)}")
    st.success(get_motivational_message())
    
    if score < 40:
        st.balloons()
    elif score < 70:
        st.snow()
    else:
        st.error("⚠️ Consider lifestyle changes and consult a doctor!")

if __name__ == "__main__":
    main()
