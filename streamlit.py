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

def community_forum():
    st.title("🌍 PCOS Community Forum")
    st.markdown("### Share experiences, tips, and support with others!")
    
    if "posts" not in st.session_state:
        st.session_state.posts = []
        st.session_state.upvotes = {}
    
    with st.form("new_post"):
        user_name = st.text_input("Your Name (or leave blank for anonymous):")
        user_message = st.text_area("Share your experience or ask a question:")
        submit_button = st.form_submit_button("Post")
        
        if submit_button and user_message:
            user_name = user_name if user_name else "Anonymous"
            post_id = len(st.session_state.posts)
            st.session_state.posts.append((post_id, user_name, user_message))
            st.session_state.upvotes[post_id] = 0
            st.success("✅ Post shared successfully!")
    
    st.markdown("---")
    
    if st.session_state.posts:
        for post_id, name, message in reversed(st.session_state.posts):
            st.markdown(f"**{name}:** {message}")
            col1, col2 = st.columns([1, 6])
            with col1:
                if st.button(f"👍 {st.session_state.upvotes[post_id]}", key=f"upvote_{post_id}"):
                    st.session_state.upvotes[post_id] += 1
            with col2:
                st.markdown("---")

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
        score += options[answer]
        st.progress(score // (len(questions) * 3))
        time.sleep(0.3)
    
    return score

def get_recommendations(score):
    if score < 40:
        return "✅ You're doing great! Keep maintaining a balanced lifestyle."
    elif score < 70:
        return "⚠️ Consider improving your diet and exercise habits to lower risk."
    else:
        return "🚨 High risk detected! Consult a healthcare provider and adopt healthier habits."

def get_personalized_plan(score):
    if score < 40:
        return "🥗 Healthy Diet: Continue balanced meals with fruits, veggies, and lean proteins.\n🏋️‍♀️ Exercise: Maintain your routine with 30 min daily workouts."
    elif score < 70:
        return "🥗 Diet Tip: Reduce processed foods and add more fiber-rich meals.\n🏋️‍♀️ Exercise: Try strength training and yoga for better hormone balance."
    else:
        return "🚨 High Risk Alert: \n🥗 Focus on low-glycemic foods, whole grains, and healthy fats.\n🏋️‍♀️ Regular Exercise: Daily 30-45 min workouts with cardio and strength training recommended."

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
    st.markdown(f"### 📅 Personalized Diet & Exercise Plan:\n{get_personalized_plan(score)}")
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
    
    st.markdown("---")
    community_forum()
    
if __name__ == "__main__":
    main()
