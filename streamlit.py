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
    st.markdown("#### Answer these questions to assess your risk level.")
    
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
        time.sleep(0.3)  # Smooth progress animation
    
    return score

def get_recommendations(score):
    if score < 40:
        return "✅ You're doing great! Keep maintaining a balanced lifestyle."
    elif score < 70:
        return "⚠️ Consider improving your diet and exercise habits to lower risk."
    else:
        return "🚨 High risk detected! Consult a healthcare provider and adopt healthier habits."

def get_motivational_message():
    messages = [
        "🌟 Every step towards a healthier you is a victory!",
        "🏆 Small changes today lead to a healthier tomorrow!",
        "💖 Your health matters—take care of yourself!",
        "🔥 Keep pushing forward, your body will thank you!"
    ]
    return random.choice(messages)

def main():
    score = personality_quiz()
    st.subheader(f"📊 Your PCOS Risk Score: **{score}**")
    st.plotly_chart(risk_meter(score))
    
    st.markdown(f"### 💡 {get_recommendations(score)}")
    st.success(get_motivational_message())
    
    if score < 40:
        st.balloons()
    elif score < 70:
        st.snow()
    else:
        st.warning("⚠️ Consider lifestyle changes and consult a doctor!")
        st.error("🚑 Immediate action is recommended!")

if __name__ == "__main__":
    main()
