# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading the model
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data (Replace with actual dataset if available)
data = pd.DataFrame({
    'module_score': [70, 80, 90, 50, 60, 85, 55, 40, 90, 95],
    'interactions': [15, 20, 30, 5, 10, 25, 7, 12, 35, 40],
    'progress': [0.5, 0.7, 0.9, 0.2, 0.3, 0.8, 0.3, 0.4, 1.0, 1.0],
    'final_score': [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]  # 1=pass, 0=fail
})

# Feature Engineering: Create new features like interaction rate, progress rate
data['interaction_rate'] = data['interactions'] / (data['progress'] + 0.01)
data['progress_score_ratio'] = data['module_score'] / (data['progress'] + 0.01)

# Define features and target
X = data[['module_score', 'interactions', 'progress', 'interaction_rate', 'progress_score_ratio']]
y = data['final_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a pipeline with StandardScaler and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestClassifier(random_state=42))
])

# Set up the hyperparameter grid for tuning
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Cross-Validation Score
cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Score:", np.mean(cv_score))

# Model Performance on Test Set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Additional Evaluation Metrics
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score:", roc_auc)

# Feature Importance Analysis
feature_importances = best_model.named_steps['rf'].feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in RandomForest Model')
plt.show()

# Save the best model to disk
joblib.dump(best_model, 'optimized_performance_model.pkl')
