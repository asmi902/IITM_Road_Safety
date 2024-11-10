# Streamlit and other libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Title and description
st.title("Adaptive Learning Performance Prediction")
st.write("""
This app uses a machine learning model to predict a learner's final score based on engagement, module scores, and progress.
""")

# Load or train the model
# Load or train the model
@st.cache_resource
def train_model():
    # Sample data (replace with actual dataset)
    data = pd.DataFrame({
        'module_score': [70, 80, 90, 50, 60, 85, 55, 40, 90, 95],
        'interactions': [15, 20, 30, 5, 10, 25, 7, 12, 35, 40],
        'progress': [0.5, 0.7, 0.9, 0.2, 0.3, 0.8, 0.3, 0.4, 1.0, 1.0],
        'final_score': [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]  # 1=pass, 0=fail
    })

    # Feature engineering
    data['interaction_rate'] = data['interactions'] / (data['progress'] + 0.01)
    data['progress_score_ratio'] = data['module_score'] / (data['progress'] + 0.01)

    X = data[['module_score', 'interactions', 'progress', 'interaction_rate', 'progress_score_ratio']]
    y = data['final_score']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline and Grid Search
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    # Change cv to 3
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Model evaluation
    best_model = grid_search.best_estimator_

    # Save model to disk
    joblib.dump(best_model, 'optimized_performance_model.pkl')

    # Return model and test data for further use
    return best_model, X_test, y_test


# Load trained model and data
model, X_test, y_test = train_model()

# Display model evaluation results
st.subheader("Model Evaluation Metrics")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

st.write("**Accuracy:**", accuracy)
st.write("**ROC-AUC Score:**", roc_auc)
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))
st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))

# Plot feature importance
st.subheader("Feature Importance")
feature_importances = model.named_steps['rf'].feature_importances_
features = X_test.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importance_df)
st.pyplot()

# User input for predictions
st.subheader("Predict Learner's Performance")
module_score = st.slider("Module Score", min_value=0, max_value=100, value=70)
interactions = st.slider("Interactions", min_value=0, max_value=50, value=15)
progress = st.slider("Progress", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Calculate engineered features for prediction
interaction_rate = interactions / (progress + 0.01)
progress_score_ratio = module_score / (progress + 0.01)

# Make prediction based on user input
user_input = pd.DataFrame([[module_score, interactions, progress, interaction_rate, progress_score_ratio]],
                          columns=['module_score', 'interactions', 'progress', 'interaction_rate', 'progress_score_ratio'])

prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)[0]

st.subheader("Prediction Results")
st.write("**Predicted Final Score (Pass=1 / Fail=0):**", int(prediction[0]))
st.write("**Probability of Passing:**", round(prediction_proba[1] * 100, 2), "%")
st.write("**Probability of Failing:**", round(prediction_proba[0] * 100, 2), "%")

# Download model
if st.button("Download Trained Model"):
    with open("optimized_performance_model.pkl", "rb") as file:
        btn = st.download_button(
            label="Download Model",
            data=file,
            file_name="optimized_performance_model.pkl",
            mime="application/octet-stream"
        )
