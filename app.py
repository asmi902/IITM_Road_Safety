import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Adaptive Learning Performance Prediction")
st.write("""
This app uses a machine learning model to predict a learner's final score based on engagement, module scores, and progress.
""")

@st.cache_resource
def train_model():
    data = pd.DataFrame({
        'module_score': [70, 80, 90, 50, 60, 85, 55, 40, 90, 95],
        'interactions': [15, 20, 30, 5, 10, 25, 7, 12, 35, 40],
        'progress': [0.5, 0.7, 0.9, 0.2, 0.3, 0.8, 0.3, 0.4, 1.0, 1.0],
        'final_score': [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    })

    data['interaction_rate'] = data['interactions'] / (data['progress'] + 0.01)
    data['progress_score_ratio'] = data['module_score'] / (data['progress'] + 0.01)

    X = data[['module_score', 'interactions', 'progress', 'interaction_rate', 'progress_score_ratio']]
    y = data['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'optimized_performance_model.pkl')

    return best_model, X_test, y_test

model, X_test, y_test = train_model()

# Display model evaluation results
st.subheader("Model Evaluation Metrics")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)  # Use probabilities for ROC AUC score

st.write("**Accuracy:**", accuracy)
st.write("**ROC-AUC Score:**", roc_auc)
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))
st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))


st.subheader("Feature Importance")
feature_importances = model.named_steps['rf'].feature_importances_
features = X_test.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
st.pyplot(plt.gcf())
