
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# -----------------------------------
# TRAINING DATA
# -----------------------------------

data = {
    "study_hours": [
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        2,3,4,5,6,7,8,9,
        1,3,5,7,2,4,6,8
    ],

    "attendance": [
        55,60,65,70,75,80,85,90,
        50,62,68,72,78,82,88,95,
        58,66,73,79,84,89,93,96,
        52,67,76,91,61,74,87,94
    ],

    "previous_score": [
        35,42,48,55,61,68,74,82,
        30,40,46,57,64,71,79,88,
        38,50,58,65,72,77,85,92,
        33,49,63,81,44,59,73,86
    ],

    "topics_completed": [
        20,25,30,40,45,55,60,75,
        15,22,35,42,50,58,68,80,
        18,32,40,48,56,65,72,85,
        12,38,52,78,28,46,63,76
    ],

    "final_score": [
        38,45,52,60,66,73,79,87,
        34,43,50,62,69,76,84,91,
        41,54,61,69,76,82,89,95,
        36,53,68,86,47,63,78,90
    ]
}

df = pd.DataFrame(data)


# -----------------------------------
# TRAIN AI
# -----------------------------------

X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "topics_completed"
    ]
]

y = df["final_score"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AI Study Analyser",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------------
# HEADER
# -----------------------------------

st.title("🎓 AI Study Analyser")

st.write(
    "Analyse your study habits and get an AI-powered "
    "performance prediction and personalized recommendations."
)

st.divider()


# -----------------------------------
# INPUTS
# -----------------------------------

st.subheader("📚 Enter Your Study Information")

study_hours = st.slider(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

attendance = st.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.slider(
    "Previous Exam Score (%)",
    min_value=0,
    max_value=100,
    value=60
)

topics_completed = st.slider(
    "Topics Completed (%)",
    min_value=0,
    max_value=100,
    value=50
)


# -----------------------------------
# ANALYSE BUTTON
# -----------------------------------

if st.button(
    "🤖 Analyse My Performance",
    use_container_width=True
):

    student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_score": [previous_score],
        "topics_completed": [topics_completed]
    })

    predicted_score = model.predict(student)[0]

    predicted_score = max(
        0,
        min(100, predicted_score)
    )

    # Performance
    if predicted_score >= 85:
        performance = "Excellent 🌟"
    elif predicted_score >= 70:
        performance = "Good 👍"
    elif predicted_score >= 50:
        performance = "Average 📖"
    else:
        performance = "Needs Improvement ⚠️"


    # -----------------------------------
    # RESULT
    # -----------------------------------

    st.divider()

    st.subheader("📊 AI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Score",
            f"{predicted_score:.1f}%"
        )

    with col2:
        st.metric(
            "Performance",
            performance
        )


    # -----------------------------------
    # STRENGTHS
    # -----------------------------------

    st.subheader("💪 Strengths")

    strengths = []

    if study_hours >= 5:
        strengths.append(
            "You have a strong daily study routine."
        )

    if attendance >= 75:
        strengths.append(
            "Your attendance is good."
        )

    if previous_score >= 75:
        strengths.append(
            "Your previous academic performance is strong."
        )

    if topics_completed >= 75:
        strengths.append(
            "You have completed most of your syllabus."
        )

    if not strengths:
        strengths.append(
            "You have started tracking your study performance."
        )

    for strength in strengths:
        st.success(strength)


    # -----------------------------------
    # AREAS TO IMPROVE
    # -----------------------------------

    st.subheader("⚠️ Areas to Improve")

    improvements = []

    if study_hours < 2:
        improvements.append(
            "Increase your daily study time."
        )

    if attendance < 75:
        improvements.append(
            "Try to improve your attendance."
        )

    if previous_score < 50:
        improvements.append(
            "Revise fundamental concepts from previous topics."
        )

    if topics_completed < 50:
        improvements.append(
            "Complete more of your syllabus."
        )

    if not improvements:
        improvements.append(
            "Maintain your current study habits and focus on revision."
        )

    for item in improvements:
        st.warning(item)


    # -----------------------------------
    # STUDY PLAN
    # -----------------------------------

    st.subheader("📅 AI Recommended Study Plan")

    if predicted_score < 50:
        st.write(
            "📌 Focus heavily on fundamentals and weak topics."
        )
        st.write(
            "📌 Study at least 3 hours daily."
        )
        st.write(
            "📌 Practice previous examination questions."
        )

    elif predicted_score < 70:
        st.write(
            "📌 Study 2–3 focused hours every day."
        )
        st.write(
            "📌 Spend extra time on difficult topics."
        )
        st.write(
            "📌 Take a short practice test every week."
        )

    else:
        st.write(
            "📌 Maintain your current study routine."
        )
        st.write(
            "📌 Focus on revision and problem solving."
        )
        st.write(
            "📌 Practice mock examinations regularly."
        )


    # -----------------------------------
    # PROGRESS BAR
    # -----------------------------------

    st.subheader("📈 Predicted Performance")

    st.progress(
        int(predicted_score)
    )

    st.caption(
        "Prediction generated using a Random Forest machine-learning model."
    )
