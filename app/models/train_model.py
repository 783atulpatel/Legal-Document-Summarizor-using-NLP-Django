import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def train_churn_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Train a churn prediction model
    
    Parameters:
    X_train (numpy.ndarray): Training features
    X_test (numpy.ndarray): Testing features
    y_train (numpy.ndarray): Training target
    y_test (numpy.ndarray): Testing target
    feature_names (list): List of feature names
    
    Returns:
    tuple: (model, accuracy, feature_importance)
    """
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    # Get feature importance
    importances = model.feature_importances_
    feature_importance = [(feature, importance) for feature, importance in zip(feature_names, importances)]
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Important Features:")
    for feature, importance in feature_importance[:10]:
        print(f"{feature}: {importance:.4f}")
    
    return model, accuracy, feature_importance

if __name__ == "__main__":
    # Test the model training
    from app.utils.data_generator import generate_data
    from app.utils.data_processor import preprocess_data
    
    df = generate_data(1000)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    model, accuracy, feature_importance = train_churn_model(X_train, X_test, y_train, y_test, feature_names)
