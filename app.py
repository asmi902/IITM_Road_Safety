import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data and create model
data = pd.DataFrame({
    'module_score': [70, 80, 90],
    'interactions': [15, 20, 30],
    'progress': [0.5, 0.7, 0.9],
    'final_score': [1, 1, 0]
})

X = data[['module_score', 'interactions', 'progress']]
y = data['final_score']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def adaptive_questioning(user_score):
    if user_score < 60:
        return "Easy question"
    elif 60 <= user_score < 80:
        return "Medium question"
    else:
        return "Hard question"

def get_feedback(module_score):
    if module_score < 50:
        return "Review basics."
    elif 50 <= module_score < 75:
        return "Room for improvement."
    else:
        return "Great job!"

# Streamlit App
st.title("AI-Powered Adaptive Learning Assistant")

# User input
user_score = st.slider("User Module Score", 0, 100, 50)
interactions = st.number_input("Number of Interactions", min_value=0, step=1)
progress = st.slider("Progress", 0.0, 1.0, 0.5)

# Prediction
prediction = model.predict([[user_score, interactions, progress]])
st.write("Predicted Outcome:", "Pass" if prediction[0] == 1 else "Fail")

# Adaptive Question
st.write("Adaptive Question:", adaptive_questioning(user_score))

# Feedback
st.write("Personalized Feedback:", get_feedback(user_score))
