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

def spin_the_wheel():
    tips = [
        "Drink 8 glasses of water today! 💧", 
        "Do 10 minutes of stretching! 🧘", 
        "Try a sugar-free day! 🍎", 
        "Take a deep-breathing break! 😌"
    ]
    return random.choice(tips)

def personality_quiz():
    st.title("🩺 PCOS Lifestyle Risk Assessment")
    st.markdown("#### Answer these questions to assess your risk level.")
    
    questions = {
        "How often do you exercise?": {"Daily": 0, "3-5 times a week": 10, "1-2 times a week": 20, "Rarely": 30},
        "How would you rate your diet?": {"Excellent": 0, "Good": 10, "Average": 20, "Poor": 30},
        "Do you have irregular menstrual cycles?": {"Never": 0, "Occasionally": 10, "Often": 20, "Always": 30},
        "How stressed do you feel daily?": {"Not at all": 0, "Mildly": 10, "Moderately": 20, "Highly stressed": 30},
        "How many hours of sleep do you get per night?": {"More than 8": 0, "7-8 hours": 10, "5-6 hours": 20, "Less than 5": 30}
    }
    
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, list(options.keys()), index=0)
        score += options[answer]  # Assign appropriate risk scores
        st.progress(score // (len(questions) * 3))  # Normalize progress bar
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
    
    if st.button("🎡 Spin the Wheel for a Health Tip!"):
        st.write(spin_the_wheel())
    
if __name__ == "__main__":
    main()
