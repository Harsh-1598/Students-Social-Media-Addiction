import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"deployment\addiction_model.pkl")

st.title("Student Social Media Addiction Predictor")

Age = st.number_input("Age", 10, 100)
Gender = st.selectbox("Gender", ["Male", "Female"])
Academic_Level = st.selectbox("Academic Level", ["Undergraduate", "Postgraduate"])
Avg_Daily_Usage_Hours = st.slider("Daily Usage Hours", 0, 24)
Most_Used_Platform = st.selectbox("Most Used Platform",
                                  ["Instagram","TikTok","Facebook","Snapchat","Twitter"])
Affects_Academic_Performance = st.selectbox("Affects Academic Performance", ["Yes","No"])
Sleep_Hours_Per_Night = st.slider("Sleep Hours", 0, 12)
Mental_Health_Score = st.slider("Mental Health Score", 0, 10)
Relationship_Status = st.selectbox("Relationship Status",
                                   ["Single","In Relationship","Complicated"])
Conflicts_Over_Social_Media = st.number_input("Conflicts Due To Social Media", 0, 20)

if st.button("Predict Addiction Level"):

    new_student = pd.DataFrame({
        "Age": Age,
        "Gender": Gender,
        "Academic_Level": Academic_Level,
        "Avg_Daily_Usage_Hours": Avg_Daily_Usage_Hours,
        "Most_Used_Platform": Most_Used_Platform,
        "Affects_Academic_Performance": Affects_Academic_Performance,
        "Sleep_Hours_Per_Night": Sleep_Hours_Per_Night,
        "Mental_Health_Score": Mental_Health_Score,
        "Relationship_Status": Relationship_Status,
        "Conflicts_Over_Social_Media": Conflicts_Over_Social_Media
    })

    raw = model.predict(new_student)[0]
    raw = min(max(raw,2),9)

    percent = (raw-2)/7 * 100
    percent = min(max(percent,0),100)

    st.success(f"Addiction Score: {percent:.2f}%")
